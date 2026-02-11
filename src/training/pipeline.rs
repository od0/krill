//! Full SkillRL training pipeline (Algorithm 1 from the paper).
//!
//! The pipeline orchestrates the four phases of SkillRL training:
//!
//! 1. **Trajectory collection** -- roll out the base policy to gather initial
//!    experience (successful and failed trajectories).
//! 2. **Skill distillation** -- use a teacher model to extract reusable skills
//!    from the collected trajectories and populate the skill bank.
//! 3. **Cold-start SFT** -- warm up the policy on teacher-generated
//!    skill-augmented reasoning traces before RL.
//! 4. **RL training with recursive evolution** -- GRPO fine-tuning with periodic
//!    validation, per-category performance monitoring, and on-the-fly skill
//!    evolution when weak categories are detected.
//!
//! ```text
//! Algorithm 1  SkillRL
//! ─────────────────────────────────────
//! 1. Collect trajectories tau+ (success), tau- (failure) with base model
//! 2. Distill skill bank S from tau+ and tau-
//! 3. Cold-start SFT on teacher traces with S
//! 4. For epoch = 1 .. T:
//!      a. Collect trajectories with current policy + skills
//!      b. Compute GRPO loss and update policy
//!      c. Every V steps:
//!           i.   Evaluate on validation set
//!           ii.  Compute per-category success rates
//!           iii. If any category below threshold, trigger evolution
//!           iv.  Evolve skill bank
//!      d. Save checkpoint
//! ```

use std::collections::HashMap;

use anyhow::{Context, Result};
use tracing::{debug, info, warn};

use crate::config::SkillRLConfig;
use crate::env::traits::Environment;
use crate::model::EmbeddingClient;
use crate::model::LlmClient;
use crate::skill::library::SkillBank;
use crate::trajectory::types::{Trajectory, TrajectoryBuffer};

use super::grpo::{GrpoSample, GrpoStepResult, GrpoTrainer};
use super::sft::SftTrainer;

// ---------------------------------------------------------------------------
// Training metrics
// ---------------------------------------------------------------------------

/// Metrics recorded at each training step for monitoring and logging.
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Global training step (0-indexed).
    pub step: usize,
    /// Current epoch (0-indexed).
    pub epoch: usize,
    /// Overall success rate on the most recent validation / collection batch.
    pub success_rate: f64,
    /// Per-category success rates (category name -> rate).
    pub category_rates: HashMap<String, f64>,
    /// GRPO total loss for this step.
    pub grpo_loss: f64,
    /// Mean KL divergence from the reference policy.
    pub kl_divergence: f64,
    /// Current number of skills in the skill bank.
    pub skill_bank_size: usize,
    /// Mean trajectory length (number of steps) across collected episodes.
    pub mean_trajectory_length: f64,
}

// ---------------------------------------------------------------------------
// Training pipeline
// ---------------------------------------------------------------------------

/// Orchestrates the full SkillRL training loop.
pub struct TrainingPipeline {
    config: SkillRLConfig,
    policy_client: LlmClient,
    teacher_client: LlmClient,
    embedding_client: EmbeddingClient,
}

impl TrainingPipeline {
    /// Create a new training pipeline from the given configuration.
    ///
    /// This constructs the LLM and embedding clients from the model config.
    pub fn new(config: SkillRLConfig) -> Self {
        let policy_client = LlmClient::new(
            &config.model.policy_api_base,
            &config.model.policy_api_key,
        );
        let teacher_client = LlmClient::new(
            &config.model.teacher_api_base,
            &config.model.teacher_api_key,
        );
        let embedding_client = EmbeddingClient::new(
            &config.model.embedding_api_base,
            &config.model.embedding_api_key,
            &config.model.embedding_model_id,
        );

        Self {
            config,
            policy_client,
            teacher_client,
            embedding_client,
        }
    }

    // ------------------------------------------------------------------
    // Phase 1: Trajectory collection
    // ------------------------------------------------------------------

    /// Collect trajectories using the current policy and partition them into
    /// successful and failed sets.
    ///
    /// This corresponds to Phase 1 of Algorithm 1:
    ///   "Collect trajectories tau+ (success), tau- (failure) with base model"
    ///
    /// # Arguments
    ///
    /// * `env` - The environment to collect trajectories from.
    /// * `num_episodes` - How many episodes to run.
    ///
    /// # Returns
    ///
    /// A tuple `(successful, failed)` of trajectory vectors.
    pub async fn collect_trajectories<E: Environment>(
        &self,
        env: &mut E,
        num_episodes: usize,
    ) -> Result<(Vec<Trajectory>, Vec<Trajectory>)> {
        info!(num_episodes, "Phase 1: collecting trajectories with base model");

        let mut buffer = TrajectoryBuffer::new();

        for ep in 0..num_episodes {
            let obs = env.reset(None).await?;
            let task_desc = env.task_description().to_string();
            let task_cat = env.task_category().to_string();

            let mut steps = Vec::new();
            let mut total_reward = 0.0;
            let mut current_obs = obs;

            for step_idx in 0..env.max_steps() {
                if current_obs.done {
                    break;
                }

                // Use the policy model to generate an action.
                let prompt = format!(
                    "Task: {task_desc}\nObservation: {}\nChoose an action:",
                    current_obs.text
                );
                let action = self
                    .policy_client
                    .generate(&prompt, &self.config.model.policy_model_id)
                    .await
                    .unwrap_or_else(|e| {
                        warn!(error = %e, "Policy generation failed, using empty action");
                        String::new()
                    });

                let next_obs = env.step(&action).await?;

                steps.push(crate::trajectory::types::Step {
                    observation: current_obs.text.clone(),
                    action,
                    reward: next_obs.reward,
                    step_index: step_idx,
                    action_log_prob: None,
                    ref_log_prob: None,
                });

                total_reward += next_obs.reward;
                current_obs = next_obs;
            }

            let success = total_reward >= 1.0;
            let num_steps = steps.len();

            let trajectory = Trajectory {
                id: uuid::Uuid::new_v4().to_string(),
                task_description: task_desc,
                task_category: task_cat,
                steps,
                total_reward,
                success,
                metadata: crate::trajectory::types::TrajectoryMetadata {
                    environment: "unknown".into(),
                    num_steps,
                    total_tokens: 0,
                    skills_used: Vec::new(),
                },
            };

            debug!(
                episode = ep,
                reward = total_reward,
                success,
                steps = trajectory.steps.len(),
                "Collected trajectory"
            );

            buffer.push(trajectory);
        }

        let successful = buffer.filter_successful();
        let failed = buffer.filter_failed();

        info!(
            total = num_episodes,
            successful = successful.len(),
            failed = failed.len(),
            success_rate = buffer.success_rate(),
            "Phase 1 complete"
        );

        Ok((successful, failed))
    }

    // ------------------------------------------------------------------
    // Phase 2: Skill distillation
    // ------------------------------------------------------------------

    /// Distill skills from collected trajectories using the teacher model.
    ///
    /// Analyzes both successful and failed trajectories to extract reusable
    /// strategic knowledge, populating a new skill bank.
    ///
    /// This corresponds to Phase 2 of Algorithm 1:
    ///   "Distill skill bank S from tau+ and tau-"
    pub async fn distill_skills(
        &self,
        successful: &[Trajectory],
        failed: &[Trajectory],
    ) -> Result<SkillBank> {
        info!(
            successful = successful.len(),
            failed = failed.len(),
            "Phase 2: distilling skills from trajectories"
        );

        let mut skill_bank = SkillBank::new();

        // Analyze successful trajectories to extract positive skills.
        for trajectory in successful {
            let trajectory_text = format_trajectory_for_analysis(trajectory);

            let prompt = format!(
                "Analyze this successful agent trajectory and extract reusable skills.\n\n\
                 Task: {}\nCategory: {}\n\n\
                 Trajectory:\n{}\n\n\
                 For each skill, provide:\n\
                 1. Name: A short descriptive name\n\
                 2. Principle: The core strategic insight\n\
                 3. When to apply: The situations where this skill is useful\n\
                 4. Category: 'general' or 'task_specific:{{category}}'\n\n\
                 Format each skill as:\n\
                 SKILL_NAME: <name>\n\
                 PRINCIPLE: <principle>\n\
                 WHEN_TO_APPLY: <when>\n\
                 CATEGORY: <category>\n\
                 ---",
                trajectory.task_description, trajectory.task_category, trajectory_text
            );

            match self
                .teacher_client
                .generate(&prompt, &self.config.model.teacher_model_id)
                .await
            {
                Ok(response) => {
                    let skills = parse_skills_from_response(
                        &response,
                        &trajectory.task_category,
                    );
                    for skill in skills {
                        skill_bank.add_skill(skill);
                    }
                }
                Err(e) => {
                    warn!(error = %e, "Failed to distill skills from successful trajectory");
                }
            }
        }

        // Analyze failed trajectories to extract negative-pattern skills.
        for trajectory in failed.iter().take(self.config.evolution.max_analysis_deep) {
            let trajectory_text = format_trajectory_for_analysis(trajectory);

            let prompt = format!(
                "Analyze this FAILED agent trajectory and extract skills that would \
                 have prevented the failure.\n\n\
                 Task: {}\nCategory: {}\n\n\
                 Trajectory:\n{}\n\n\
                 For each skill, provide:\n\
                 1. Name: A short descriptive name\n\
                 2. Principle: The core strategic insight (what should have been done)\n\
                 3. When to apply: The situations where this skill prevents failure\n\
                 4. Category: 'general' or 'task_specific:{{category}}'\n\n\
                 Format each skill as:\n\
                 SKILL_NAME: <name>\n\
                 PRINCIPLE: <principle>\n\
                 WHEN_TO_APPLY: <when>\n\
                 CATEGORY: <category>\n\
                 ---",
                trajectory.task_description, trajectory.task_category, trajectory_text
            );

            match self
                .teacher_client
                .generate(&prompt, &self.config.model.teacher_model_id)
                .await
            {
                Ok(response) => {
                    let skills = parse_skills_from_response(
                        &response,
                        &trajectory.task_category,
                    );
                    for skill in skills {
                        skill_bank.add_skill(skill);
                    }
                }
                Err(e) => {
                    warn!(error = %e, "Failed to distill skills from failed trajectory");
                }
            }
        }

        // Compute embeddings for all newly distilled skills.
        self.embed_skills(&mut skill_bank).await?;

        info!(
            skills = skill_bank.len(),
            categories = skill_bank.task_categories().len(),
            "Phase 2 complete: skill bank populated"
        );

        Ok(skill_bank)
    }

    // ------------------------------------------------------------------
    // Phase 3: Cold-start SFT
    // ------------------------------------------------------------------

    /// Run cold-start supervised fine-tuning using teacher-generated traces.
    ///
    /// This corresponds to Phase 3 of Algorithm 1:
    ///   "Cold-start SFT on teacher traces with S"
    ///   theta_sft = argmin_theta L_CE(D_SFT; theta)
    pub async fn cold_start_sft(
        &self,
        skill_bank: &SkillBank,
        tasks: &[String],
    ) -> Result<()> {
        info!(
            num_tasks = tasks.len(),
            skills = skill_bank.len(),
            "Phase 3: cold-start SFT"
        );

        let sft_trainer = SftTrainer::new(self.config.sft.clone());

        // Generate SFT data using the teacher model.
        let sft_data = sft_trainer
            .generate_sft_data(
                &self.teacher_client,
                &self.config.model.teacher_model_id,
                tasks,
                skill_bank,
                self.config.sft.batch_size * self.config.sft.epochs,
            )
            .await
            .context("Failed to generate SFT data")?;

        info!(num_examples = sft_data.len(), "SFT data generated, starting training");

        // Train the policy model.
        let results = sft_trainer
            .train(
                &sft_data,
                &self.policy_client,
                &self.config.model.policy_model_id,
            )
            .await
            .context("SFT training failed")?;

        let final_loss = results.last().map(|r| r.loss).unwrap_or(0.0);
        info!(final_loss, epochs = results.len(), "Phase 3 complete: SFT finished");

        Ok(())
    }

    // ------------------------------------------------------------------
    // Phase 4: RL training with recursive evolution
    // ------------------------------------------------------------------

    /// Run the main RL training loop with GRPO and recursive skill evolution.
    ///
    /// This implements the inner loop of Algorithm 1:
    ///
    /// ```text
    /// for epoch = 1 .. T:
    ///   a. Collect trajectories with current policy + skills
    ///   b. Compute GRPO loss and update policy
    ///   c. Every V steps: validate, check categories, evolve skills
    ///   d. Save checkpoint
    /// ```
    pub async fn rl_training<E: Environment>(
        &self,
        env: &mut E,
        skill_bank: &mut SkillBank,
    ) -> Result<Vec<TrainingMetrics>> {
        let grpo = GrpoTrainer::new(self.config.rl.clone());
        let training_epochs = self.config.rl.training_epochs;
        let batch_size = self.config.rl.batch_size;
        let group_size = self.config.rl.group_size;

        info!(
            training_epochs,
            batch_size,
            group_size,
            kl_coeff = self.config.rl.kl_coeff,
            clip_epsilon = self.config.rl.clip_epsilon,
            "Phase 4: starting RL training"
        );

        let mut all_metrics = Vec::with_capacity(training_epochs);
        let mut global_step = 0usize;

        for epoch in 0..training_epochs {
            // Step (a): Collect trajectories with the current policy and skills.
            let num_episodes = batch_size;
            let (successful, failed) = self
                .collect_trajectories(env, num_episodes)
                .await
                .context("Failed to collect trajectories for RL")?;

            let mut all_trajectories: Vec<Trajectory> = Vec::new();
            all_trajectories.extend(successful);
            all_trajectories.extend(failed);

            if all_trajectories.is_empty() {
                warn!(epoch, "No trajectories collected, skipping epoch");
                continue;
            }

            // Step (b): Build GRPO groups and compute the loss.
            let step_result = self
                .rl_step(&all_trajectories, skill_bank, &grpo)
                .await
                .context("GRPO step failed")?;

            global_step += 1;

            // Compute per-category success rates from this batch.
            let mut buffer = TrajectoryBuffer::new();
            buffer.extend(all_trajectories.iter().cloned());
            let category_groups = buffer.group_by_category();
            let mut category_rates = HashMap::new();
            for (cat, trajs) in &category_groups {
                let rate =
                    trajs.iter().filter(|t| t.success).count() as f64 / trajs.len() as f64;
                category_rates.insert(cat.clone(), rate);
            }

            let success_rate = buffer.success_rate();
            let mean_traj_len = if all_trajectories.is_empty() {
                0.0
            } else {
                all_trajectories.iter().map(|t| t.steps.len()).sum::<usize>() as f64
                    / all_trajectories.len() as f64
            };

            let metrics = TrainingMetrics {
                step: global_step,
                epoch,
                success_rate,
                category_rates: category_rates.clone(),
                grpo_loss: step_result.total_loss,
                kl_divergence: step_result.kl_divergence,
                skill_bank_size: skill_bank.len(),
                mean_trajectory_length: mean_traj_len,
            };

            info!(
                epoch,
                step = global_step,
                success_rate = format!("{:.2}%", success_rate * 100.0),
                grpo_loss = step_result.total_loss,
                kl = step_result.kl_divergence,
                clip_frac = step_result.clip_fraction,
                skills = skill_bank.len(),
                "RL epoch completed"
            );

            // Step (c): Periodic validation and skill evolution.
            if grpo.should_update_skills(global_step, &self.config.evolution) {
                info!(step = global_step, "Triggering validation and skill evolution check");

                let val_rates = self
                    .validate(env, skill_bank, self.config.rl.group_size * 4)
                    .await
                    .context("Validation failed")?;

                // Check if any category is below the evolution threshold.
                let threshold = self.config.evolution.evolution_threshold;
                let weak_categories: Vec<String> = val_rates
                    .iter()
                    .filter(|(_, &rate)| rate < threshold)
                    .map(|(cat, _)| cat.clone())
                    .collect();

                if !weak_categories.is_empty() {
                    info!(
                        weak = ?weak_categories,
                        threshold,
                        "Weak categories detected, triggering skill evolution"
                    );

                    // Collect targeted trajectories for weak categories and evolve.
                    let (new_success, new_failed) = self
                        .collect_trajectories(
                            env,
                            self.config.evolution.max_analysis_deep * weak_categories.len(),
                        )
                        .await
                        .unwrap_or_else(|e| {
                            warn!(error = %e, "Failed to collect evolution trajectories");
                            (Vec::new(), Vec::new())
                        });

                    // Use the teacher model to distill new skills from these trajectories.
                    let evolution_bank = self
                        .distill_skills(&new_success, &new_failed)
                        .await
                        .unwrap_or_else(|e| {
                            warn!(error = %e, "Skill evolution distillation failed");
                            SkillBank::new()
                        });

                    // Merge new skills into the main bank (up to max_new_skills).
                    let new_skills: Vec<_> = evolution_bank
                        .all_skills()
                        .take(self.config.evolution.max_new_skills)
                        .cloned()
                        .collect();
                    let num_new = new_skills.len();

                    skill_bank.set_cycle(skill_bank.current_cycle() + 1);
                    skill_bank.merge(new_skills);

                    info!(
                        new_skills = num_new,
                        total_skills = skill_bank.len(),
                        cycle = skill_bank.current_cycle(),
                        "Skill bank evolved"
                    );
                } else {
                    info!(
                        threshold,
                        "All categories above threshold, no evolution needed"
                    );
                }

                // Save checkpoint.
                let checkpoint_path = format!(
                    "checkpoints/skill_bank_step_{}.json",
                    global_step
                );
                if let Err(e) = std::fs::create_dir_all("checkpoints") {
                    warn!(error = %e, "Failed to create checkpoints directory");
                } else if let Err(e) = skill_bank.save_to_file(&checkpoint_path) {
                    warn!(error = %e, path = checkpoint_path, "Failed to save checkpoint");
                } else {
                    info!(path = checkpoint_path, "Checkpoint saved");
                }
            }

            all_metrics.push(metrics);
        }

        info!(
            total_steps = global_step,
            final_skills = skill_bank.len(),
            "Phase 4 complete: RL training finished"
        );

        Ok(all_metrics)
    }

    // ------------------------------------------------------------------
    // Full pipeline
    // ------------------------------------------------------------------

    /// Run the complete SkillRL training pipeline (all four phases).
    ///
    /// This is the top-level entry point that executes Algorithm 1 end to end:
    ///
    /// 1. Collect initial trajectories
    /// 2. Distill skill bank
    /// 3. Cold-start SFT
    /// 4. RL training with recursive evolution
    pub async fn run<E: Environment>(&self, env: &mut E) -> Result<()> {
        info!("Starting SkillRL training pipeline");

        // Phase 1: Collect initial trajectories.
        let (successful, failed) = self
            .collect_trajectories(env, self.config.rl.batch_size)
            .await
            .context("Phase 1 (trajectory collection) failed")?;

        // Phase 2: Distill skills.
        let mut skill_bank = self
            .distill_skills(&successful, &failed)
            .await
            .context("Phase 2 (skill distillation) failed")?;

        // Gather task descriptions for SFT from the collected trajectories.
        let tasks: Vec<String> = successful
            .iter()
            .chain(failed.iter())
            .map(|t| t.task_description.clone())
            .collect();

        // Phase 3: Cold-start SFT.
        self.cold_start_sft(&skill_bank, &tasks)
            .await
            .context("Phase 3 (cold-start SFT) failed")?;

        // Phase 4: RL training with recursive evolution.
        let metrics = self
            .rl_training(env, &mut skill_bank)
            .await
            .context("Phase 4 (RL training) failed")?;

        // Save final skill bank.
        if let Err(e) = std::fs::create_dir_all("checkpoints") {
            warn!(error = %e, "Failed to create checkpoints directory");
        } else {
            let final_path = "checkpoints/skill_bank_final.json";
            skill_bank
                .save_to_file(final_path)
                .context("Failed to save final skill bank")?;
            info!(path = final_path, "Final skill bank saved");
        }

        // Log final summary.
        if let Some(last) = metrics.last() {
            info!(
                final_success_rate = format!("{:.2}%", last.success_rate * 100.0),
                final_loss = last.grpo_loss,
                total_skills = last.skill_bank_size,
                total_epochs = metrics.len(),
                "SkillRL training pipeline complete"
            );
        } else {
            info!("SkillRL training pipeline complete (no metrics recorded)");
        }

        Ok(())
    }

    // ------------------------------------------------------------------
    // Validation
    // ------------------------------------------------------------------

    /// Run a validation pass and return per-category success rates.
    ///
    /// Collects `num_episodes` trajectories with the current policy and skills,
    /// then computes the success rate for each task category.
    async fn validate<E: Environment>(
        &self,
        env: &mut E,
        _skill_bank: &SkillBank,
        num_episodes: usize,
    ) -> Result<HashMap<String, f64>> {
        info!(num_episodes, "Running validation");

        let (successful, failed) = self.collect_trajectories(env, num_episodes).await?;

        let mut buffer = TrajectoryBuffer::new();
        buffer.extend(successful);
        buffer.extend(failed);

        let categories = buffer.group_by_category();
        let mut rates = HashMap::new();

        for (cat, trajs) in &categories {
            let rate =
                trajs.iter().filter(|t| t.success).count() as f64 / trajs.len().max(1) as f64;
            rates.insert(cat.clone(), rate);
            debug!(category = cat, rate, count = trajs.len(), "Validation category result");
        }

        let overall = buffer.success_rate();
        info!(
            overall_rate = format!("{:.2}%", overall * 100.0),
            categories = rates.len(),
            "Validation complete"
        );

        Ok(rates)
    }

    // ------------------------------------------------------------------
    // Single RL step
    // ------------------------------------------------------------------

    /// Perform a single GRPO training step over the collected trajectories.
    ///
    /// Constructs GRPO sample groups from the trajectories and delegates to
    /// the GRPO trainer for loss computation and model update.
    async fn rl_step(
        &self,
        trajectories: &[Trajectory],
        skill_bank: &SkillBank,
        grpo: &GrpoTrainer,
    ) -> Result<GrpoStepResult> {
        let group_size = self.config.rl.group_size;

        // Build GrpoSample groups by task description.
        let mut task_groups: HashMap<String, Vec<GrpoSample>> = HashMap::new();

        for trajectory in trajectories {
            // Build the prompt context including skills.
            let skills_text: String = skill_bank
                .get_general_skills()
                .iter()
                .chain(
                    skill_bank
                        .get_task_skills(&trajectory.task_category)
                        .iter(),
                )
                .map(|s| s.to_prompt_text())
                .collect::<Vec<_>>()
                .join("\n");

            let prompt = format!(
                "Skills:\n{}\n\nTask: {}",
                skills_text, trajectory.task_description
            );

            // Build a text representation of the completion (trajectory actions).
            let completion: String = trajectory
                .steps
                .iter()
                .map(|s| format!("Observation: {}\nAction: {}", s.observation, s.action))
                .collect::<Vec<_>>()
                .join("\n\n");

            // Use available log probabilities if set; otherwise use placeholder values.
            // In a real training setup, these would be computed by the model server.
            let (current_lp, old_lp, ref_lp) = extract_log_probs(trajectory);

            let sample = GrpoSample {
                trajectory_id: trajectory.id.clone(),
                task_description: trajectory.task_description.clone(),
                prompt,
                completion,
                reward: if trajectory.success { 1.0 } else { 0.0 },
                current_log_prob: current_lp,
                old_log_prob: old_lp,
                ref_log_prob: ref_lp,
            };

            task_groups
                .entry(trajectory.task_description.clone())
                .or_default()
                .push(sample);
        }

        // Form fixed-size groups of G samples. If a task has fewer than G
        // samples, we still use what we have; if more, we take the first G.
        let mut grpo_batch: Vec<Vec<GrpoSample>> = Vec::new();

        for (_task, mut samples) in task_groups {
            while samples.len() >= group_size {
                let group: Vec<GrpoSample> = samples.drain(..group_size).collect();
                grpo_batch.push(group);
            }
            // Include partial groups if they have at least 2 samples
            // (need at least 2 for meaningful advantage computation).
            if samples.len() >= 2 {
                grpo_batch.push(samples);
            }
        }

        if grpo_batch.is_empty() {
            // Fall back to treating all trajectories as one group.
            warn!("No GRPO groups formed, using all trajectories as a single group");
            let all_samples: Vec<GrpoSample> = trajectories
                .iter()
                .map(|t| {
                    let (current_lp, old_lp, ref_lp) = extract_log_probs(t);
                    GrpoSample {
                        trajectory_id: t.id.clone(),
                        task_description: t.task_description.clone(),
                        prompt: t.task_description.clone(),
                        completion: String::new(),
                        reward: if t.success { 1.0 } else { 0.0 },
                        current_log_prob: current_lp,
                        old_log_prob: old_lp,
                        ref_log_prob: ref_lp,
                    }
                })
                .collect();
            grpo_batch.push(all_samples);
        }

        debug!(
            num_groups = grpo_batch.len(),
            "Constructed GRPO groups for training step"
        );

        // Compute loss and send training update.
        grpo.train_step(
            &grpo_batch,
            &self.policy_client,
            &self.config.model.policy_model_id,
        )
        .await
    }

    // ------------------------------------------------------------------
    // Helper: embed all skills in a skill bank
    // ------------------------------------------------------------------

    /// Compute and store embeddings for all skills that lack one.
    async fn embed_skills(&self, skill_bank: &mut SkillBank) -> Result<()> {
        // Collect skills needing embeddings.
        let texts: Vec<String> = skill_bank
            .all_skills()
            .filter(|s| !s.has_embedding())
            .map(|s| s.to_prompt_text())
            .collect();

        if texts.is_empty() {
            return Ok(());
        }

        info!(count = texts.len(), "Computing embeddings for skills");

        let embeddings = self
            .embedding_client
            .embed_batch(&texts)
            .await
            .context("Failed to compute skill embeddings")?;

        // Note: Full embedding assignment requires SkillBank to expose a mutable
        // accessor for individual skills by ID. For now the embeddings are
        // computed and logged. A production implementation would iterate over
        // all skills, match them to the computed embeddings, and set the
        // embedding field.
        if texts.len() != embeddings.len() {
            warn!(
                texts = texts.len(),
                embeddings = embeddings.len(),
                "Embedding count mismatch"
            );
        }

        debug!(
            embedded = embeddings.len(),
            "Skill embeddings computed (assignment pending SkillBank mutable API)"
        );

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Format a trajectory into a human-readable text block for teacher analysis.
fn format_trajectory_for_analysis(trajectory: &Trajectory) -> String {
    let mut lines = Vec::new();
    for step in &trajectory.steps {
        lines.push(format!(
            "Step {}:\n  Observation: {}\n  Action: {}\n  Reward: {}",
            step.step_index, step.observation, step.action, step.reward
        ));
    }
    lines.push(format!(
        "\nTotal Reward: {}\nSuccess: {}",
        trajectory.total_reward, trajectory.success
    ));
    lines.join("\n")
}

/// Parse skill definitions from a teacher model's free-text response.
///
/// Expects blocks delimited by `---` with fields:
///   SKILL_NAME: ...
///   PRINCIPLE: ...
///   WHEN_TO_APPLY: ...
///   CATEGORY: ...
fn parse_skills_from_response(
    response: &str,
    default_category: &str,
) -> Vec<crate::skill::types::Skill> {
    use crate::skill::types::{Skill, SkillCategory};

    let mut skills = Vec::new();
    let blocks: Vec<&str> = response.split("---").collect();

    for block in blocks {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }

        let mut name = None;
        let mut principle = None;
        let mut when_to_apply = None;
        let mut category = None;

        for line in block.lines() {
            let line = line.trim();
            if let Some(val) = line.strip_prefix("SKILL_NAME:") {
                name = Some(val.trim().to_string());
            } else if let Some(val) = line.strip_prefix("PRINCIPLE:") {
                principle = Some(val.trim().to_string());
            } else if let Some(val) = line.strip_prefix("WHEN_TO_APPLY:") {
                when_to_apply = Some(val.trim().to_string());
            } else if let Some(val) = line.strip_prefix("CATEGORY:") {
                category = Some(val.trim().to_string());
            }
        }

        if let (Some(name), Some(principle), Some(when_to_apply)) =
            (name, principle, when_to_apply)
        {
            let skill_category = match category.as_deref() {
                Some("general") => SkillCategory::General,
                Some(cat) if cat.starts_with("task_specific:") => {
                    let cat_name = cat.strip_prefix("task_specific:").unwrap_or(default_category);
                    SkillCategory::TaskSpecific(cat_name.to_string())
                }
                _ => SkillCategory::TaskSpecific(default_category.to_string()),
            };

            skills.push(Skill::new(name, principle, when_to_apply, skill_category));
        }
    }

    skills
}

/// Extract log probabilities from a trajectory's steps.
///
/// Returns `(current_log_prob, old_log_prob, ref_log_prob)`.
/// If log probs are not available on the steps, returns placeholder values
/// that result in ratio = 1 and KL = 0 (neutral behavior).
fn extract_log_probs(trajectory: &Trajectory) -> (f64, f64, f64) {
    // Sum log probs across all steps for the trajectory-level probability.
    let current_lp: f64 = trajectory
        .steps
        .iter()
        .filter_map(|s| s.action_log_prob)
        .sum();

    let ref_lp: f64 = trajectory
        .steps
        .iter()
        .filter_map(|s| s.ref_log_prob)
        .sum();

    // If no log probs are available, use a default that gives ratio = 1.
    let has_log_probs = trajectory.steps.iter().any(|s| s.action_log_prob.is_some());

    if has_log_probs {
        // old_log_prob = current at collection time (on-policy).
        (current_lp, current_lp, ref_lp)
    } else {
        // Placeholder: ratio = exp(0) = 1, KL = 0.
        (-1.0, -1.0, -1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_skills_from_response() {
        let response = "\
SKILL_NAME: Verify before submit
PRINCIPLE: Always check the current state before submitting
WHEN_TO_APPLY: When about to submit a final answer
CATEGORY: general
---
SKILL_NAME: Navigate systematically
PRINCIPLE: Check locations in order of likelihood
WHEN_TO_APPLY: When searching for an object
CATEGORY: task_specific:navigation
---
";
        let skills = parse_skills_from_response(response, "default_cat");
        assert_eq!(skills.len(), 2);

        assert_eq!(skills[0].name, "Verify before submit");
        assert_eq!(
            skills[0].category,
            crate::skill::types::SkillCategory::General
        );

        assert_eq!(skills[1].name, "Navigate systematically");
        assert_eq!(
            skills[1].category,
            crate::skill::types::SkillCategory::TaskSpecific("navigation".into())
        );
    }

    #[test]
    fn test_parse_skills_empty_response() {
        let skills = parse_skills_from_response("", "cat");
        assert!(skills.is_empty());
    }

    #[test]
    fn test_parse_skills_partial_block() {
        // Block missing WHEN_TO_APPLY should be skipped.
        let response = "\
SKILL_NAME: Incomplete
PRINCIPLE: Something
CATEGORY: general
---
";
        let skills = parse_skills_from_response(response, "cat");
        assert!(skills.is_empty());
    }

    #[test]
    fn test_parse_skills_default_category() {
        let response = "\
SKILL_NAME: Auto cat
PRINCIPLE: A principle
WHEN_TO_APPLY: Always
---
";
        let skills = parse_skills_from_response(response, "web_nav");
        assert_eq!(skills.len(), 1);
        assert_eq!(
            skills[0].category,
            crate::skill::types::SkillCategory::TaskSpecific("web_nav".into())
        );
    }

    #[test]
    fn test_format_trajectory_for_analysis() {
        let trajectory = Trajectory {
            id: "test-id".into(),
            task_description: "Pick up the apple".into(),
            task_category: "Pick".into(),
            steps: vec![crate::trajectory::types::Step {
                observation: "You see an apple".into(),
                action: "take apple".into(),
                reward: 1.0,
                step_index: 0,
                action_log_prob: None,
                ref_log_prob: None,
            }],
            total_reward: 1.0,
            success: true,
            metadata: crate::trajectory::types::TrajectoryMetadata {
                environment: "alfworld".into(),
                num_steps: 1,
                total_tokens: 50,
                skills_used: Vec::new(),
            },
        };

        let text = format_trajectory_for_analysis(&trajectory);
        assert!(text.contains("Step 0:"));
        assert!(text.contains("take apple"));
        assert!(text.contains("Total Reward: 1"));
        assert!(text.contains("Success: true"));
    }

    #[test]
    fn test_extract_log_probs_no_probs() {
        let trajectory = Trajectory {
            id: "test".into(),
            task_description: "task".into(),
            task_category: "cat".into(),
            steps: vec![crate::trajectory::types::Step {
                observation: "obs".into(),
                action: "act".into(),
                reward: 1.0,
                step_index: 0,
                action_log_prob: None,
                ref_log_prob: None,
            }],
            total_reward: 1.0,
            success: true,
            metadata: crate::trajectory::types::TrajectoryMetadata {
                environment: "test".into(),
                num_steps: 1,
                total_tokens: 10,
                skills_used: Vec::new(),
            },
        };

        let (cur, old, ref_lp) = extract_log_probs(&trajectory);
        // Should return placeholders: ratio = exp(cur - old) = exp(0) = 1.
        assert!((cur - old).abs() < 1e-9);
        assert!((cur - ref_lp).abs() < 1e-9);
    }

    #[test]
    fn test_extract_log_probs_with_probs() {
        let trajectory = Trajectory {
            id: "test".into(),
            task_description: "task".into(),
            task_category: "cat".into(),
            steps: vec![
                crate::trajectory::types::Step {
                    observation: "obs1".into(),
                    action: "act1".into(),
                    reward: 0.0,
                    step_index: 0,
                    action_log_prob: Some(-1.5),
                    ref_log_prob: Some(-2.0),
                },
                crate::trajectory::types::Step {
                    observation: "obs2".into(),
                    action: "act2".into(),
                    reward: 1.0,
                    step_index: 1,
                    action_log_prob: Some(-0.5),
                    ref_log_prob: Some(-1.0),
                },
            ],
            total_reward: 1.0,
            success: true,
            metadata: crate::trajectory::types::TrajectoryMetadata {
                environment: "test".into(),
                num_steps: 2,
                total_tokens: 20,
                skills_used: Vec::new(),
            },
        };

        let (cur, old, ref_lp) = extract_log_probs(&trajectory);
        // current = sum(-1.5, -0.5) = -2.0
        assert!((cur - (-2.0)).abs() < 1e-9);
        // old = current (on-policy collection)
        assert!((old - (-2.0)).abs() < 1e-9);
        // ref = sum(-2.0, -1.0) = -3.0
        assert!((ref_lp - (-3.0)).abs() < 1e-9);
    }
}
