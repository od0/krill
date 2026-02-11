//! Cold-start Supervised Fine-Tuning (SFT) trainer.
//!
//! Before RL training begins, the policy model is warmed up on teacher-generated
//! skill-augmented reasoning traces:
//!
//!   D_SFT = {(d_i, S_i, tau*_i)}_{i=1}^N
//!   theta_sft = argmin_theta L_CE(D_SFT; theta)
//!
//! The teacher model (e.g., o3) generates optimal trajectories that demonstrate
//! how to leverage the distilled skills, giving the student policy a strong
//! starting point for GRPO fine-tuning.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::config::SftConfig;
use crate::model::LlmClient;
use crate::skill::library::SkillBank;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A single SFT training example consisting of a task, its skill context, and
/// the teacher-generated target trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftExample {
    /// The natural-language task description (d_i).
    pub task_description: String,
    /// Formatted skill context injected into the prompt (S_i = S_g union S_ret).
    pub skills_context: String,
    /// The teacher-generated optimal trajectory (tau*_i).
    pub target_trajectory: String,
}

/// The result of a single SFT training step / epoch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftStepResult {
    /// Cross-entropy loss for this step.
    pub loss: f64,
    /// Number of examples in the step.
    pub num_examples: usize,
    /// The epoch index (0-based).
    pub epoch: usize,
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// Supervised fine-tuning trainer for the cold-start phase.
pub struct SftTrainer {
    config: SftConfig,
}

impl SftTrainer {
    /// Create a new SFT trainer with the given configuration.
    pub fn new(config: SftConfig) -> Self {
        Self { config }
    }

    /// Use the teacher model to generate SFT training data.
    ///
    /// For each task, the teacher model is prompted with the task description and
    /// available skills, and asked to produce an optimal reasoning trace that
    /// demonstrates effective skill usage.
    ///
    /// # Arguments
    ///
    /// * `teacher_client` - Client for the teacher model API.
    /// * `teacher_model` - Model identifier for the teacher (e.g., "o3").
    /// * `tasks` - A list of task descriptions to generate traces for.
    /// * `skill_bank` - The current skill bank to draw skill context from.
    /// * `num_examples` - Maximum number of examples to generate.
    ///
    /// # Returns
    ///
    /// A vector of `SftExample`s, one per successfully generated trace.
    pub async fn generate_sft_data(
        &self,
        teacher_client: &LlmClient,
        teacher_model: &str,
        tasks: &[String],
        skill_bank: &SkillBank,
        num_examples: usize,
    ) -> Result<Vec<SftExample>> {
        if tasks.is_empty() {
            bail!("Cannot generate SFT data with an empty task list");
        }

        info!(
            num_tasks = tasks.len(),
            num_examples,
            teacher_model,
            "Generating SFT data from teacher model"
        );

        // Build a shared skills context string from the skill bank.
        // In a full implementation this would use embedding-based retrieval per task;
        // for SFT data generation we include all general skills plus a sample of
        // task-specific skills.
        let general_skills_text: String = skill_bank
            .get_general_skills()
            .iter()
            .map(|s| s.to_prompt_text())
            .collect::<Vec<_>>()
            .join("\n");

        let mut examples = Vec::with_capacity(num_examples);
        let tasks_to_use = if tasks.len() > num_examples {
            &tasks[..num_examples]
        } else {
            tasks
        };

        for (i, task) in tasks_to_use.iter().enumerate() {
            if examples.len() >= num_examples {
                break;
            }

            // Build per-task skill context (general + any task-specific skills).
            let mut skills_context = general_skills_text.clone();

            // Attempt to find task-specific skills by scanning category names.
            // In a production system this would use the embedding retriever.
            for category in skill_bank.task_categories() {
                if task.to_lowercase().contains(&category.to_lowercase()) {
                    let task_skills_text: String = skill_bank
                        .get_task_skills(&category)
                        .iter()
                        .map(|s| s.to_prompt_text())
                        .collect::<Vec<_>>()
                        .join("\n");
                    if !task_skills_text.is_empty() {
                        skills_context.push('\n');
                        skills_context.push_str(&task_skills_text);
                    }
                }
            }

            let prompt = format!(
                "You are an expert agent solving tasks step by step.\n\n\
                 ## Available Skills\n\
                 {skills_context}\n\n\
                 ## Task\n\
                 {task}\n\n\
                 ## Instructions\n\
                 Produce an optimal step-by-step reasoning trace that solves this task.\n\
                 Demonstrate how to effectively use the available skills.\n\
                 Format your response as a sequence of (Thought, Action, Observation) steps.\n\
                 Be thorough but concise."
            );

            let system_msg = "You are a teacher model generating optimal reasoning traces \
                              for training a smaller student model. Your traces should \
                              demonstrate clear, strategic thinking that leverages the \
                              provided skills.";

            match teacher_client
                .generate_with_system(&prompt, system_msg, teacher_model)
                .await
            {
                Ok(trajectory) => {
                    debug!(
                        task_index = i,
                        trajectory_len = trajectory.len(),
                        "Generated SFT example"
                    );
                    examples.push(SftExample {
                        task_description: task.clone(),
                        skills_context: skills_context.clone(),
                        target_trajectory: trajectory,
                    });
                }
                Err(e) => {
                    warn!(
                        task_index = i,
                        error = %e,
                        "Failed to generate SFT example, skipping task"
                    );
                    // Continue with next task rather than aborting.
                }
            }
        }

        info!(
            generated = examples.len(),
            requested = num_examples,
            "SFT data generation complete"
        );

        if examples.is_empty() {
            bail!("Failed to generate any SFT examples");
        }

        Ok(examples)
    }

    /// Run supervised fine-tuning for the configured number of epochs.
    ///
    /// Each epoch iterates over all examples in batches, sending the training
    /// data to the model server's training endpoint.
    ///
    /// # Arguments
    ///
    /// * `examples` - The SFT training examples.
    /// * `policy_client` - Client for the policy model API / training server.
    /// * `model_id` - The model identifier to fine-tune.
    ///
    /// # Returns
    ///
    /// A vector of `SftStepResult`s, one per epoch.
    pub async fn train(
        &self,
        examples: &[SftExample],
        policy_client: &LlmClient,
        model_id: &str,
    ) -> Result<Vec<SftStepResult>> {
        if examples.is_empty() {
            bail!("Cannot run SFT with an empty example set");
        }

        info!(
            num_examples = examples.len(),
            epochs = self.config.epochs,
            batch_size = self.config.batch_size,
            learning_rate = self.config.learning_rate,
            "Starting SFT training"
        );

        let mut results = Vec::with_capacity(self.config.epochs);

        for epoch in 0..self.config.epochs {
            let batches = self.create_training_batch(examples, self.config.batch_size);
            let mut epoch_loss = 0.0;
            let mut epoch_examples = 0;

            for (batch_idx, batch) in batches.iter().enumerate() {
                // Build the training payload for this batch.
                let training_data: Vec<serde_json::Value> = batch
                    .iter()
                    .map(|ex| {
                        serde_json::json!({
                            "messages": [
                                {
                                    "role": "system",
                                    "content": format!(
                                        "You are a task-solving agent. Use the following skills:\n{}",
                                        ex.skills_context
                                    )
                                },
                                {
                                    "role": "user",
                                    "content": ex.task_description
                                },
                                {
                                    "role": "assistant",
                                    "content": ex.target_trajectory
                                }
                            ]
                        })
                    })
                    .collect();

                let payload = serde_json::json!({
                    "model": model_id,
                    "training_data": training_data,
                    "learning_rate": self.config.learning_rate,
                    "epoch": epoch,
                    "batch_index": batch_idx,
                });

                let resp = policy_client
                    .http
                    .post(format!("{}/train", policy_client.api_base))
                    .bearer_auth(&policy_client.api_key)
                    .json(&payload)
                    .send()
                    .await?;

                let status = resp.status();
                if !status.is_success() {
                    let text = resp.text().await.unwrap_or_default();
                    warn!(
                        epoch,
                        batch_idx,
                        status = %status,
                        "SFT training batch failed: {text}"
                    );
                    bail!("SFT training endpoint returned {status}: {text}");
                }

                // Parse the loss from the server response.
                let resp_json: serde_json::Value = resp.json().await?;
                let batch_loss = resp_json["loss"].as_f64().unwrap_or(0.0);
                epoch_loss += batch_loss * batch.len() as f64;
                epoch_examples += batch.len();

                debug!(
                    epoch,
                    batch_idx,
                    batch_size = batch.len(),
                    batch_loss,
                    "SFT batch completed"
                );
            }

            let avg_loss = if epoch_examples > 0 {
                epoch_loss / epoch_examples as f64
            } else {
                0.0
            };

            info!(epoch, avg_loss, num_examples = epoch_examples, "SFT epoch completed");

            results.push(SftStepResult {
                loss: avg_loss,
                num_examples: epoch_examples,
                epoch,
            });
        }

        info!(
            final_loss = results.last().map(|r| r.loss).unwrap_or(0.0),
            "SFT training complete"
        );

        Ok(results)
    }

    /// Split a set of examples into fixed-size batches.
    ///
    /// The last batch may be smaller than `batch_size` if the examples do not
    /// divide evenly.
    pub fn create_training_batch<'a>(
        &self,
        examples: &'a [SftExample],
        batch_size: usize,
    ) -> Vec<Vec<&'a SftExample>> {
        if batch_size == 0 || examples.is_empty() {
            return Vec::new();
        }

        examples
            .chunks(batch_size)
            .map(|chunk| chunk.iter().collect())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_sft_config() -> SftConfig {
        SftConfig {
            learning_rate: 1e-4,
            batch_size: 16,
            epochs: 3,
        }
    }

    fn make_example(task: &str) -> SftExample {
        SftExample {
            task_description: task.to_string(),
            skills_context: "skill1\nskill2".to_string(),
            target_trajectory: "step1\nstep2\nstep3".to_string(),
        }
    }

    #[test]
    fn test_create_training_batch_even() {
        let trainer = SftTrainer::new(default_sft_config());
        let examples: Vec<SftExample> = (0..8).map(|i| make_example(&format!("task_{i}"))).collect();

        let batches = trainer.create_training_batch(&examples, 4);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 4);
        assert_eq!(batches[1].len(), 4);
    }

    #[test]
    fn test_create_training_batch_uneven() {
        let trainer = SftTrainer::new(default_sft_config());
        let examples: Vec<SftExample> = (0..5).map(|i| make_example(&format!("task_{i}"))).collect();

        let batches = trainer.create_training_batch(&examples, 3);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[1].len(), 2);
    }

    #[test]
    fn test_create_training_batch_single() {
        let trainer = SftTrainer::new(default_sft_config());
        let examples = vec![make_example("task_0")];

        let batches = trainer.create_training_batch(&examples, 10);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 1);
    }

    #[test]
    fn test_create_training_batch_empty() {
        let trainer = SftTrainer::new(default_sft_config());
        let batches = trainer.create_training_batch(&[], 4);
        assert!(batches.is_empty());
    }

    #[test]
    fn test_create_training_batch_zero_size() {
        let trainer = SftTrainer::new(default_sft_config());
        let examples = vec![make_example("task_0")];
        let batches = trainer.create_training_batch(&examples, 0);
        assert!(batches.is_empty());
    }

    #[test]
    fn test_create_training_batch_preserves_order() {
        let trainer = SftTrainer::new(default_sft_config());
        let examples: Vec<SftExample> = (0..6).map(|i| make_example(&format!("task_{i}"))).collect();

        let batches = trainer.create_training_batch(&examples, 2);
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0][0].task_description, "task_0");
        assert_eq!(batches[0][1].task_description, "task_1");
        assert_eq!(batches[1][0].task_description, "task_2");
        assert_eq!(batches[2][1].task_description, "task_5");
    }
}
