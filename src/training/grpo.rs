//! Group Relative Policy Optimization (GRPO) trainer.
//!
//! Implements the core RL algorithm from the SkillRL paper (Equation 9):
//!
//!   J(theta) = E[ 1/G * sum_i min(rho_i * A_i, clip(rho_i, 1-eps, 1+eps) * A_i)
//!              - beta * D_KL(pi_theta || pi_ref) ]
//!
//! where:
//!   - rho_i = pi_theta(tau_i | d, S_g, S_ret) / pi_old(tau_i | d, S_g, S_ret)
//!   - A_i  = (R_i - mean(R)) / std(R)  (group-relative advantage)
//!   - beta is the KL divergence coefficient
//!   - G is the group size

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::config::{EvolutionConfig, RlConfig};
use crate::model::LlmClient;

use super::advantage::{clip_ratio, compute_group_advantages, compute_importance_ratio};

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A single training sample with all information needed for a GRPO update.
///
/// Each sample corresponds to one trajectory tau_i within a group of G
/// trajectories sampled for the same task prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpoSample {
    /// Unique identifier for the trajectory this sample was derived from.
    pub trajectory_id: String,
    /// Human-readable task description (d).
    pub task_description: String,
    /// The full prompt including injected skills (d + S_g + S_ret).
    pub prompt: String,
    /// The agent's response / trajectory (tau).
    pub completion: String,
    /// Binary reward: 0 (failure) or 1 (success).
    pub reward: f64,
    /// Log probability under the current policy: log pi_theta(tau | d, S_g, S_ret).
    pub current_log_prob: f64,
    /// Log probability under the old (sampling) policy: log pi_old(tau | d, S_g, S_ret).
    pub old_log_prob: f64,
    /// Log probability under the reference policy: log pi_ref(tau | d, S_g, S_ret).
    /// Used for the KL divergence penalty.
    pub ref_log_prob: f64,
}

/// The result of a single GRPO optimization step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpoStepResult {
    /// The mean clipped surrogate policy loss (negated for gradient descent).
    pub policy_loss: f64,
    /// The mean KL divergence from the reference policy.
    pub kl_divergence: f64,
    /// Total loss = -mean(clipped_objective) + beta * KL.
    pub total_loss: f64,
    /// Mean advantage across all samples in the step.
    pub mean_advantage: f64,
    /// Mean importance sampling ratio across all samples.
    pub mean_ratio: f64,
    /// Fraction of ratios that were clipped by the epsilon bound.
    pub clip_fraction: f64,
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// The GRPO trainer computes the policy gradient objective for a batch of
/// trajectory groups and dispatches weight updates to the model server.
pub struct GrpoTrainer {
    config: RlConfig,
}

impl GrpoTrainer {
    /// Create a new GRPO trainer with the given RL configuration.
    pub fn new(config: RlConfig) -> Self {
        Self { config }
    }

    /// Compute the GRPO loss for a single group of G samples.
    ///
    /// # Algorithm
    ///
    /// 1. Extract rewards and compute group-relative advantages:
    ///      A_i = (R_i - mean(R)) / std(R)
    /// 2. Compute importance ratios:
    ///      rho_i = exp(log pi_theta - log pi_old)
    /// 3. Compute the clipped surrogate objective for each sample:
    ///      L_i = min(rho_i * A_i, clip(rho_i, 1-eps, 1+eps) * A_i)
    /// 4. Compute the KL divergence penalty:
    ///      D_KL = mean(log pi_theta - log pi_ref)
    /// 5. Combine:
    ///      total_loss = -mean(L_i) + beta * D_KL
    ///
    /// # Errors
    ///
    /// Returns an error if the group is empty.
    pub fn compute_grpo_loss(&self, group: &[GrpoSample]) -> Result<GrpoStepResult> {
        if group.is_empty() {
            bail!("Cannot compute GRPO loss for an empty group");
        }

        let g = group.len() as f64;
        let epsilon = self.config.clip_epsilon;
        let beta = self.config.kl_coeff;

        // Step 1: Extract rewards and compute group-relative advantages.
        let rewards: Vec<f64> = group.iter().map(|s| s.reward).collect();
        let advantages = compute_group_advantages(&rewards);

        // Step 2-3: For each sample, compute ratio, clipped objective, and KL term.
        let mut total_clipped_obj = 0.0;
        let mut total_kl = 0.0;
        let mut total_ratio = 0.0;
        let mut num_clipped = 0usize;

        for (i, sample) in group.iter().enumerate() {
            let advantage = advantages[i];

            // Importance sampling ratio.
            let ratio = compute_importance_ratio(sample.current_log_prob, sample.old_log_prob);
            total_ratio += ratio;

            // Unclipped surrogate.
            let surr_unclipped = ratio * advantage;

            // Clipped surrogate.
            let clipped = clip_ratio(ratio, epsilon);
            let surr_clipped = clipped * advantage;

            // Check if this ratio was actually clipped.
            if (clipped - ratio).abs() > 1e-10 {
                num_clipped += 1;
            }

            // PPO-clip objective: take the pessimistic (minimum) bound.
            let objective = surr_unclipped.min(surr_clipped);
            total_clipped_obj += objective;

            // KL divergence contribution: log pi_theta - log pi_ref.
            // This is the per-sample approximation of D_KL(pi_theta || pi_ref).
            let kl_sample = sample.current_log_prob - sample.ref_log_prob;
            total_kl += kl_sample;
        }

        // Average over the group.
        let mean_clipped_obj = total_clipped_obj / g;
        let mean_kl = total_kl / g;
        let mean_ratio = total_ratio / g;
        let mean_advantage = advantages.iter().sum::<f64>() / g;
        let clip_fraction = num_clipped as f64 / g;

        // Total loss: we negate the objective because optimizers minimize.
        // J(theta) = mean(clipped_obj) - beta * D_KL
        // loss     = -J(theta) = -mean(clipped_obj) + beta * D_KL
        let policy_loss = -mean_clipped_obj;
        let total_loss = policy_loss + beta * mean_kl;

        debug!(
            policy_loss,
            kl = mean_kl,
            total_loss,
            mean_ratio,
            clip_fraction,
            "GRPO loss computed for group of {} samples",
            group.len()
        );

        Ok(GrpoStepResult {
            policy_loss,
            kl_divergence: mean_kl,
            total_loss,
            mean_advantage,
            mean_ratio,
            clip_fraction,
        })
    }

    /// Perform a full training step over a batch of groups.
    ///
    /// Each inner `Vec<GrpoSample>` is one group of G trajectories sampled for
    /// the same task prompt. The losses are averaged across all groups, and the
    /// result is sent to the model server via a POST to `{base_url}/train`.
    ///
    /// In a production setup this endpoint would apply the gradient update to the
    /// model weights. Here we compute the loss analytically and delegate the
    /// actual weight update to the training backend.
    pub async fn train_step(
        &self,
        batch: &[Vec<GrpoSample>],
        policy_client: &LlmClient,
        model_id: &str,
    ) -> Result<GrpoStepResult> {
        if batch.is_empty() {
            bail!("Cannot run a GRPO train step on an empty batch");
        }

        // Compute per-group losses and aggregate.
        let mut agg_policy_loss = 0.0;
        let mut agg_kl = 0.0;
        let mut agg_total_loss = 0.0;
        let mut agg_mean_advantage = 0.0;
        let mut agg_mean_ratio = 0.0;
        let mut agg_clip_fraction = 0.0;
        let num_groups = batch.len() as f64;

        for group in batch {
            let result = self.compute_grpo_loss(group)?;
            agg_policy_loss += result.policy_loss;
            agg_kl += result.kl_divergence;
            agg_total_loss += result.total_loss;
            agg_mean_advantage += result.mean_advantage;
            agg_mean_ratio += result.mean_ratio;
            agg_clip_fraction += result.clip_fraction;
        }

        let step_result = GrpoStepResult {
            policy_loss: agg_policy_loss / num_groups,
            kl_divergence: agg_kl / num_groups,
            total_loss: agg_total_loss / num_groups,
            mean_advantage: agg_mean_advantage / num_groups,
            mean_ratio: agg_mean_ratio / num_groups,
            clip_fraction: agg_clip_fraction / num_groups,
        };

        info!(
            total_loss = step_result.total_loss,
            kl = step_result.kl_divergence,
            clip_frac = step_result.clip_fraction,
            num_groups = batch.len(),
            "Sending GRPO update to model server"
        );

        // Dispatch the training update to the model server.
        // The server is expected to accept a JSON payload with the loss
        // information and apply the corresponding gradient step.
        let train_payload = serde_json::json!({
            "model": model_id,
            "loss": step_result.total_loss,
            "policy_loss": step_result.policy_loss,
            "kl_divergence": step_result.kl_divergence,
            "learning_rate": self.config.learning_rate,
            "clip_epsilon": self.config.clip_epsilon,
            "kl_coeff": self.config.kl_coeff,
            "batch_size": batch.len(),
            // Include the full batch so the server can recompute gradients if needed.
            "groups": batch,
        });

        let resp = policy_client
            .http
            .post(format!("{}/train", policy_client.api_base))
            .bearer_auth(&policy_client.api_key)
            .json(&train_payload)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            warn!(status = %status, body = %text, "Training endpoint returned error");
            bail!("Training endpoint returned {status}: {text}");
        }

        info!("GRPO train step completed successfully");
        Ok(step_result)
    }

    /// Determine whether the skill bank should be evolved at this training step.
    ///
    /// Returns `true` every `config.validation_interval` steps (1-indexed),
    /// triggering a validation + potential skill evolution cycle.
    pub fn should_update_skills(&self, step: usize, config: &EvolutionConfig) -> bool {
        if config.validation_interval == 0 {
            return false;
        }
        step > 0 && step % config.validation_interval == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_rl_config() -> RlConfig {
        RlConfig {
            learning_rate: 1e-6,
            batch_size: 64,
            group_size: 8,
            kl_coeff: 0.01,
            clip_epsilon: 0.2,
            invalid_action_penalty: 0.1,
            max_prompt_length: 6000,
            max_response_length: 1024,
            training_epochs: 150,
        }
    }

    fn make_sample(reward: f64, current_lp: f64, old_lp: f64, ref_lp: f64) -> GrpoSample {
        GrpoSample {
            trajectory_id: uuid::Uuid::new_v4().to_string(),
            task_description: "test task".into(),
            prompt: "test prompt".into(),
            completion: "test completion".into(),
            reward,
            current_log_prob: current_lp,
            old_log_prob: old_lp,
            ref_log_prob: ref_lp,
        }
    }

    // ------------------------------------------------------------------
    // compute_grpo_loss
    // ------------------------------------------------------------------

    #[test]
    fn test_grpo_loss_basic() {
        let trainer = GrpoTrainer::new(default_rl_config());

        // Group of 4: two successes, two failures.
        // Same log probs for current/old (ratio = 1), same ref (KL ~ 0).
        let group = vec![
            make_sample(0.0, -2.0, -2.0, -2.0),
            make_sample(1.0, -2.0, -2.0, -2.0),
            make_sample(0.0, -2.0, -2.0, -2.0),
            make_sample(1.0, -2.0, -2.0, -2.0),
        ];

        let result = trainer.compute_grpo_loss(&group).unwrap();

        // When ratio = 1 and KL = 0:
        // advantages = [-1, 1, -1, 1] (z-scores of [0,1,0,1])
        // clipped_obj = 1*A_i for each (since clip(1, 0.8, 1.2) = 1)
        // mean(obj) = mean([-1, 1, -1, 1]) = 0
        // policy_loss = -0 = 0
        assert!(result.policy_loss.abs() < 1e-9);
        assert!(result.kl_divergence.abs() < 1e-9);
        assert!(result.total_loss.abs() < 1e-9);
        assert!((result.mean_ratio - 1.0).abs() < 1e-9);
        assert!(result.clip_fraction.abs() < 1e-9);
    }

    #[test]
    fn test_grpo_loss_with_kl() {
        let trainer = GrpoTrainer::new(default_rl_config());

        // Current policy has drifted from reference (current_lp != ref_lp).
        let group = vec![
            make_sample(0.0, -1.5, -2.0, -2.0),
            make_sample(1.0, -1.5, -2.0, -2.0),
            make_sample(0.0, -1.5, -2.0, -2.0),
            make_sample(1.0, -1.5, -2.0, -2.0),
        ];

        let result = trainer.compute_grpo_loss(&group).unwrap();

        // KL = mean(current_lp - ref_lp) = mean(-1.5 - (-2.0)) = 0.5
        assert!((result.kl_divergence - 0.5).abs() < 1e-9);
        // total_loss includes beta * KL = 0.01 * 0.5 = 0.005
        assert!(result.total_loss > result.policy_loss);
    }

    #[test]
    fn test_grpo_loss_clipping() {
        let config = RlConfig {
            clip_epsilon: 0.2,
            ..default_rl_config()
        };
        let trainer = GrpoTrainer::new(config);

        // Large ratio difference: current_lp = -1.0, old_lp = -3.0
        // ratio = exp(2.0) ~ 7.389, way outside [0.8, 1.2]
        let group = vec![
            make_sample(0.0, -1.0, -3.0, -2.0),
            make_sample(1.0, -1.0, -3.0, -2.0),
        ];

        let result = trainer.compute_grpo_loss(&group).unwrap();

        // All ratios should be clipped.
        assert!((result.clip_fraction - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_grpo_loss_all_same_reward() {
        let trainer = GrpoTrainer::new(default_rl_config());

        // All rewards are 1.0 -> advantages are all 0 -> policy loss = 0.
        let group = vec![
            make_sample(1.0, -2.0, -2.0, -2.0),
            make_sample(1.0, -2.0, -2.0, -2.0),
            make_sample(1.0, -2.0, -2.0, -2.0),
            make_sample(1.0, -2.0, -2.0, -2.0),
        ];

        let result = trainer.compute_grpo_loss(&group).unwrap();
        assert!(result.policy_loss.abs() < 1e-9);
        assert!(result.mean_advantage.abs() < 1e-9);
    }

    #[test]
    fn test_grpo_loss_empty_group() {
        let trainer = GrpoTrainer::new(default_rl_config());
        let result = trainer.compute_grpo_loss(&[]);
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // should_update_skills
    // ------------------------------------------------------------------

    #[test]
    fn test_should_update_skills() {
        let trainer = GrpoTrainer::new(default_rl_config());
        let evo_config = EvolutionConfig {
            validation_interval: 5,
            max_new_skills: 3,
            evolution_threshold: 0.4,
            max_analysis_deep: 10,
            max_analysis_shallow: 5,
        };

        assert!(!trainer.should_update_skills(0, &evo_config));
        assert!(!trainer.should_update_skills(1, &evo_config));
        assert!(!trainer.should_update_skills(4, &evo_config));
        assert!(trainer.should_update_skills(5, &evo_config));
        assert!(trainer.should_update_skills(10, &evo_config));
        assert!(!trainer.should_update_skills(7, &evo_config));
    }

    #[test]
    fn test_should_update_skills_zero_interval() {
        let trainer = GrpoTrainer::new(default_rl_config());
        let evo_config = EvolutionConfig {
            validation_interval: 0,
            max_new_skills: 3,
            evolution_threshold: 0.4,
            max_analysis_deep: 10,
            max_analysis_shallow: 5,
        };

        // Should never update when interval is 0.
        assert!(!trainer.should_update_skills(0, &evo_config));
        assert!(!trainer.should_update_skills(5, &evo_config));
    }
}
