//! The SkillRL agent: skill-augmented policy that retrieves relevant skills,
//! constructs a prompt, and generates actions via an LLM.
//!
//! Implements the core policy:
//!
//!   `a_t ~ pi_theta(a_t | o_<=t, d, S_g, S_ret)`
//!
//! where `S_g` are general skills (always included), and `S_ret` are the top-K
//! task-specific skills retrieved by embedding similarity.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::config::{SkillRLConfig, SkillRetrievalConfig};
use crate::model::api::LlmClient;
use crate::model::embedding::EmbeddingClient;
use crate::model::prompt::skill_augmented_action_prompt;
use crate::skill::library::SkillBank;
use crate::skill::retrieval::SkillRetriever;
use crate::trajectory::types::Step;

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

/// The result of a single agent action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionOutput {
    /// The chosen action string.
    pub action: String,
    /// The chain-of-thought reasoning that preceded the action.
    pub reasoning: String,
    /// IDs of skills that were retrieved and injected into the prompt.
    pub skills_retrieved: Vec<String>,
}

/// An action output together with its log probability under the current policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionOutputWithLogProb {
    /// The action output.
    pub output: ActionOutput,
    /// `log pi_theta(action | context)` -- used for GRPO importance ratio.
    pub log_prob: f64,
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

/// The SkillRL agent that uses a skill-augmented prompt to drive an LLM policy.
pub struct SkillAgent {
    /// LLM client for the policy model.
    policy_client: LlmClient,
    /// Model identifier for the policy (e.g. `"Qwen/Qwen2.5-7B-Instruct"`).
    policy_model_id: String,
    /// Embedding client for computing task and skill embeddings.
    embedding_client: EmbeddingClient,
    /// The hierarchical skill bank.
    skill_bank: SkillBank,
    /// Retrieval configuration (top_k, similarity_threshold).
    retrieval_config: SkillRetrievalConfig,
}

impl SkillAgent {
    /// Create a new agent from the global configuration and an initial skill bank.
    pub fn new(config: &SkillRLConfig, skill_bank: SkillBank) -> Self {
        let policy_client = LlmClient::new(
            &config.model.policy_api_base,
            &config.model.policy_api_key,
        );
        let embedding_client = EmbeddingClient::new(
            &config.model.embedding_api_base,
            &config.model.embedding_api_key,
            &config.model.embedding_model_id,
        );

        Self {
            policy_client,
            policy_model_id: config.model.policy_model_id.clone(),
            embedding_client,
            skill_bank,
            retrieval_config: config.skill_retrieval.clone(),
        }
    }

    // ------------------------------------------------------------------
    // Core action methods
    // ------------------------------------------------------------------

    /// Generate an action for the current state.
    ///
    /// This is the main inference method implementing:
    ///
    ///   `a_t ~ pi_theta(a_t | o_<=t, d, S_g, S_ret)`
    ///
    /// Steps:
    /// 1. Embed the task description.
    /// 2. Retrieve the top-K relevant skills via cosine similarity.
    /// 3. Construct the skill-augmented prompt (general + retrieved skills).
    /// 4. Call the policy LLM to generate a response.
    /// 5. Parse the response to extract the action and reasoning.
    pub async fn act(
        &self,
        task_description: &str,
        observation_history: &[Step],
        task_category: &str,
    ) -> Result<ActionOutput> {
        // 1. Retrieve skills.
        let (general_skills, retrieved_skills) = self
            .retrieve_skills(task_description, task_category)
            .await?;

        let retrieved_ids: Vec<String> = retrieved_skills.iter().map(|s| s.id.clone()).collect();

        // 2. Format observation history.
        let obs_text = format_observation_history(observation_history);

        // 3. Build the prompt.
        let messages = skill_augmented_action_prompt(
            task_description,
            &obs_text,
            &general_skills,
            &retrieved_skills,
        );

        // 4. Call the policy model.
        let response = self
            .policy_client
            .chat_completion(
                &self.policy_model_id,
                &messages,
                0.7, // temperature
                1024, // max_tokens
            )
            .await
            .context("policy model chat completion failed")?;

        let raw_text = response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        // 5. Parse the response.
        let (reasoning, action) = parse_action_response(&raw_text);

        info!(
            action = %action,
            skills_retrieved = retrieved_ids.len(),
            "agent produced action"
        );

        Ok(ActionOutput {
            action,
            reasoning,
            skills_retrieved: retrieved_ids,
        })
    }

    /// Generate an action and also return its log probability under the
    /// current policy (needed for GRPO importance ratio computation).
    pub async fn act_with_logprob(
        &self,
        task_description: &str,
        observation_history: &[Step],
        task_category: &str,
    ) -> Result<ActionOutputWithLogProb> {
        // 1. Retrieve skills.
        let (general_skills, retrieved_skills) = self
            .retrieve_skills(task_description, task_category)
            .await?;

        let retrieved_ids: Vec<String> = retrieved_skills.iter().map(|s| s.id.clone()).collect();

        // 2. Format observation history.
        let obs_text = format_observation_history(observation_history);

        // 3. Build the prompt.
        let messages = skill_augmented_action_prompt(
            task_description,
            &obs_text,
            &general_skills,
            &retrieved_skills,
        );

        // 4. Call the policy model with logprobs.
        let response = self
            .policy_client
            .chat_completion_with_logprobs(
                &self.policy_model_id,
                &messages,
                0.7,
                1024,
            )
            .await
            .context("policy model chat completion (with logprobs) failed")?;

        let raw_text = response
            .response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        let log_prob = response.total_log_prob;

        // 5. Parse the response.
        let (reasoning, action) = parse_action_response(&raw_text);

        info!(
            action = %action,
            log_prob,
            skills_retrieved = retrieved_ids.len(),
            "agent produced action with logprob"
        );

        Ok(ActionOutputWithLogProb {
            output: ActionOutput {
                action,
                reasoning,
                skills_retrieved: retrieved_ids,
            },
            log_prob,
        })
    }

    // ------------------------------------------------------------------
    // Skill bank management
    // ------------------------------------------------------------------

    /// Replace the agent's skill bank (e.g. after an evolution cycle).
    pub fn update_skill_bank(&mut self, bank: SkillBank) {
        info!(
            old_skills = self.skill_bank.len(),
            new_skills = bank.len(),
            "updating skill bank"
        );
        self.skill_bank = bank;
    }

    /// Return a reference to the current skill bank.
    pub fn skill_bank(&self) -> &SkillBank {
        &self.skill_bank
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Retrieve general and task-specific skills for the current task.
    ///
    /// General skills are always returned in full.  Task-specific skills are
    /// selected by embedding similarity using `SkillRetriever`.
    async fn retrieve_skills(
        &self,
        task_description: &str,
        task_category: &str,
    ) -> Result<(Vec<crate::skill::types::Skill>, Vec<crate::skill::types::Skill>)> {
        let general_skills = self.skill_bank.get_general_skills().to_vec();

        // Get task-specific candidates.
        let task_candidates = self.skill_bank.get_task_skills(task_category);

        if task_candidates.is_empty() {
            debug!(task_category, "no task-specific skills available for category");
            return Ok((general_skills, Vec::new()));
        }

        // Embed the task description for similarity matching.
        let task_embedding = self
            .embedding_client
            .embed(task_description)
            .await
            .context("failed to embed task description")?;

        if task_embedding.is_empty() {
            warn!("task embedding is empty; skipping skill retrieval");
            return Ok((general_skills, Vec::new()));
        }

        // Use the retriever to find top-K skills above the similarity threshold.
        let retriever = SkillRetriever::retrieve(
            &task_embedding,
            task_category,
            &self.skill_bank,
            &self.retrieval_config,
        );

        let retrieved_skills: Vec<crate::skill::types::Skill> =
            retriever.into_iter().cloned().collect();

        debug!(
            task_category,
            retrieved = retrieved_skills.len(),
            candidates = task_candidates.len(),
            "retrieved task-specific skills"
        );

        Ok((general_skills, retrieved_skills))
    }
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

/// Format a slice of [`Step`]s into a human-readable observation history string.
fn format_observation_history(steps: &[Step]) -> String {
    if steps.is_empty() {
        return "(no observations yet)".to_string();
    }

    steps
        .iter()
        .map(|step| {
            format!(
                "Step {}: Observation: {}\n        Action: {} (reward: {:.2})",
                step.step_index, step.observation, step.action, step.reward
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parse the LLM response to extract reasoning and action.
///
/// Looks for a line matching `Action: <action>`.  Everything before that line
/// is treated as chain-of-thought reasoning.  If no `Action:` line is found,
/// the entire response is returned as the action (with an empty reasoning
/// string) so that the agent can still operate even if the model does not
/// follow the expected format.
fn parse_action_response(response: &str) -> (String, String) {
    // Search for the last occurrence of "Action:" to handle cases where the
    // model mentions "Action:" in its reasoning.
    let lines: Vec<&str> = response.lines().collect();

    let mut action_line_idx = None;
    for (i, line) in lines.iter().enumerate().rev() {
        let trimmed = line.trim();
        if trimmed.starts_with("Action:") || trimmed.starts_with("action:") {
            action_line_idx = Some(i);
            break;
        }
    }

    match action_line_idx {
        Some(idx) => {
            let reasoning = lines[..idx].join("\n").trim().to_string();
            let action_line = lines[idx].trim();

            // Extract the action text after "Action:" (case-insensitive prefix).
            let action = if let Some(rest) = action_line.strip_prefix("Action:") {
                rest.trim().to_string()
            } else if let Some(rest) = action_line.strip_prefix("action:") {
                rest.trim().to_string()
            } else {
                action_line.to_string()
            };

            (reasoning, action)
        }
        None => {
            // No "Action:" line found -- use the whole response as the action.
            warn!("no 'Action:' line found in LLM response; using full response as action");
            (String::new(), response.trim().to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_action_response_with_reasoning() {
        let response = "I should look for the ball.\nLet me check the table.\n\nAction: go to table 1";
        let (reasoning, action) = parse_action_response(response);
        assert_eq!(action, "go to table 1");
        assert!(reasoning.contains("look for the ball"));
    }

    #[test]
    fn test_parse_action_response_action_only() {
        let response = "Action: pick up mug";
        let (reasoning, action) = parse_action_response(response);
        assert_eq!(action, "pick up mug");
        assert!(reasoning.is_empty());
    }

    #[test]
    fn test_parse_action_response_no_action_line() {
        let response = "I will pick up the mug.";
        let (reasoning, action) = parse_action_response(response);
        assert!(reasoning.is_empty());
        assert_eq!(action, "I will pick up the mug.");
    }

    #[test]
    fn test_parse_action_response_multiple_action_lines() {
        let response =
            "I considered Action: look around\nBut instead:\n\nAction: go to table 2";
        let (reasoning, action) = parse_action_response(response);
        assert_eq!(action, "go to table 2");
        // Reasoning should include the earlier mention.
        assert!(reasoning.contains("Action: look around"));
    }

    #[test]
    fn test_parse_action_response_case_insensitive() {
        let response = "Thinking...\n\naction: open fridge 1";
        let (reasoning, action) = parse_action_response(response);
        assert_eq!(action, "open fridge 1");
        assert!(reasoning.contains("Thinking"));
    }

    #[test]
    fn test_format_observation_history_empty() {
        let result = format_observation_history(&[]);
        assert_eq!(result, "(no observations yet)");
    }

    #[test]
    fn test_format_observation_history() {
        let steps = vec![
            Step {
                observation: "You see a room.".into(),
                action: "look".into(),
                reward: 0.0,
                step_index: 0,
                action_log_prob: None,
                ref_log_prob: None,
            },
            Step {
                observation: "You see a table.".into(),
                action: "go to table 1".into(),
                reward: 0.5,
                step_index: 1,
                action_log_prob: None,
                ref_log_prob: None,
            },
        ];
        let result = format_observation_history(&steps);
        assert!(result.contains("Step 0"));
        assert!(result.contains("Step 1"));
        assert!(result.contains("You see a room."));
        assert!(result.contains("go to table 1"));
    }
}
