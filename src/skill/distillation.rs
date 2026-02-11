//! Skill distillation from agent trajectories using a teacher model.
//!
//! Implements the core distillation operations from the SkillRL paper:
//!
//! - **Success distillation**: `s+ = M_T(tau+, d)` -- extract strategic patterns from
//!   successful trajectories.
//! - **Failure distillation**: `s- = M_T(tau-, d)` -- synthesize lessons from failed
//!   trajectories.
//! - **Initial library construction**: combine success and failure skills into a
//!   hierarchical [`SkillBank`] with both general and task-specific tiers.

use anyhow::{Context, Result};
use tracing;

use super::library::SkillBank;
use super::types::{Skill, SkillCategory};
use crate::model::LlmClient;
use crate::trajectory::Trajectory;

// ---------------------------------------------------------------------------
// Prompt templates
// ---------------------------------------------------------------------------

/// System prompt for the teacher model during skill distillation.
const DISTILLATION_SYSTEM_PROMPT: &str = r#"You are an expert skill-extraction engine for autonomous agents. Your job is to analyze agent trajectories and distill reusable strategic skills.

A "skill" is a compact, actionable piece of strategic knowledge with:
- name: A short, descriptive title (3-8 words).
- principle: The core strategic insight (1-3 sentences).
- when_to_apply: A description of the situations where this skill is useful (1-2 sentences).
- category: Either "general" (domain-agnostic) or the specific task category name.

You MUST respond with a JSON array of skill objects. Do NOT include any text outside the JSON array.

Example output:
[
  {
    "name": "Verify Before Submitting",
    "principle": "Always verify the current page state matches expectations before submitting a form or answer.",
    "when_to_apply": "When you are about to perform an irreversible action like form submission.",
    "category": "general"
  }
]"#;

/// Prompt template for extracting skills from successful trajectories.
const SUCCESS_DISTILLATION_PROMPT: &str = r#"Analyze the following SUCCESSFUL agent trajectories for the task described below. Extract reusable strategic skills that explain WHY these trajectories succeeded.

Focus on:
1. Effective strategies that led to success.
2. Clever action sequences that solved sub-problems efficiently.
3. Decision patterns that avoided common pitfalls.

Task description: {task_description}

Trajectories:
{trajectories}

Extract 1-5 skills as a JSON array. Each skill should have: name, principle, when_to_apply, category.
For category, use "general" if the skill applies broadly, or "{task_category}" if it is specific to this task type."#;

/// Prompt template for synthesizing skills from failed trajectories.
const FAILURE_DISTILLATION_PROMPT: &str = r#"Analyze the following FAILED agent trajectories for the task described below. Synthesize defensive skills that would help an agent AVOID these failure modes in the future.

Focus on:
1. What went wrong and why.
2. Early warning signs the agent should have recognized.
3. Alternative strategies that could have succeeded.
4. Common traps or mistakes to avoid.

Task description: {task_description}

Trajectories:
{trajectories}

Synthesize 1-5 defensive skills as a JSON array. Each skill should have: name, principle, when_to_apply, category.
For category, use "general" if the lesson applies broadly, or "{task_category}" if it is specific to this task type."#;

/// Prompt template for building the initial hierarchical library from a mixed set.
const INITIAL_LIBRARY_PROMPT: &str = r#"You are building an initial hierarchical skill library for an autonomous agent. Analyze the following successful and failed trajectories across multiple task categories.

Your goal is to produce two sets of skills:
1. GENERAL skills: domain-agnostic strategies useful across ALL task categories.
2. TASK-SPECIFIC skills: strategies specific to each task category encountered.

Successful trajectories:
{successful}

Failed trajectories:
{failed}

Task categories observed: {categories}

Produce a JSON array of 5-15 skills. Use "general" for the category field on broadly applicable skills. For task-specific skills, set category to the exact task category name (e.g., "web_navigation", "code_editing")."#;

// ---------------------------------------------------------------------------
// SkillDistiller
// ---------------------------------------------------------------------------

/// Distills reusable strategic skills from agent trajectories using a teacher model.
///
/// The distiller sends trajectory data to a powerful teacher model (e.g., GPT-4, o3)
/// which identifies patterns and produces structured skill definitions.
pub struct SkillDistiller {
    /// The LLM client used to call the teacher model.
    client: LlmClient,
    /// The model identifier for the teacher (e.g., "o3").
    model_id: String,
}

impl SkillDistiller {
    /// Create a new distiller backed by the given client and model.
    pub fn new(client: LlmClient, model_id: impl Into<String>) -> Self {
        Self {
            client,
            model_id: model_id.into(),
        }
    }

    /// Distill strategic skills from successful trajectories.
    ///
    /// Implements `s+ = M_T(tau+, d)` from the paper.
    ///
    /// # Arguments
    ///
    /// * `trajectories` - One or more successful trajectories to analyze.
    /// * `task_description` - A natural-language description of the task.
    ///
    /// # Returns
    ///
    /// A vector of newly distilled skills.
    pub async fn distill_from_success(
        &self,
        trajectories: &[Trajectory],
        task_description: &str,
    ) -> Result<Vec<Skill>> {
        if trajectories.is_empty() {
            return Ok(Vec::new());
        }

        let task_category = &trajectories[0].task_category;
        let traj_text = format_trajectories(trajectories);

        let prompt = SUCCESS_DISTILLATION_PROMPT
            .replace("{task_description}", task_description)
            .replace("{trajectories}", &traj_text)
            .replace("{task_category}", task_category);

        tracing::info!(
            task_category,
            n_trajectories = trajectories.len(),
            "Distilling skills from successful trajectories"
        );

        let response = self
            .client
            .generate_with_system(&prompt, DISTILLATION_SYSTEM_PROMPT, &self.model_id)
            .await
            .context("Teacher model call failed during success distillation")?;

        parse_skills_from_response(&response, task_category)
    }

    /// Synthesize defensive skills from failed trajectories.
    ///
    /// Implements `s- = M_T(tau-, d)` from the paper.
    ///
    /// # Arguments
    ///
    /// * `trajectories` - One or more failed trajectories to analyze.
    /// * `task_description` - A natural-language description of the task.
    ///
    /// # Returns
    ///
    /// A vector of newly distilled defensive skills.
    pub async fn distill_from_failure(
        &self,
        trajectories: &[Trajectory],
        task_description: &str,
    ) -> Result<Vec<Skill>> {
        if trajectories.is_empty() {
            return Ok(Vec::new());
        }

        let task_category = &trajectories[0].task_category;
        let traj_text = format_trajectories(trajectories);

        let prompt = FAILURE_DISTILLATION_PROMPT
            .replace("{task_description}", task_description)
            .replace("{trajectories}", &traj_text)
            .replace("{task_category}", task_category);

        tracing::info!(
            task_category,
            n_trajectories = trajectories.len(),
            "Distilling skills from failed trajectories"
        );

        let response = self
            .client
            .generate_with_system(&prompt, DISTILLATION_SYSTEM_PROMPT, &self.model_id)
            .await
            .context("Teacher model call failed during failure distillation")?;

        parse_skills_from_response(&response, task_category)
    }

    /// Build the initial hierarchical skill library from a set of successful and failed
    /// trajectories.
    ///
    /// This is called once during Phase 1 (SFT warm-up) to bootstrap the skill bank
    /// before RL training begins.
    ///
    /// # Arguments
    ///
    /// * `successful` - Successful trajectories from the warm-up phase.
    /// * `failed` - Failed trajectories from the warm-up phase.
    ///
    /// # Returns
    ///
    /// A fully populated [`SkillBank`] with both general and task-specific skills.
    pub async fn distill_initial_library(
        &self,
        successful: &[Trajectory],
        failed: &[Trajectory],
    ) -> Result<SkillBank> {
        // Collect all observed task categories.
        let categories: Vec<String> = {
            let mut cats: Vec<String> = successful
                .iter()
                .chain(failed.iter())
                .map(|t| t.task_category.clone())
                .collect();
            cats.sort();
            cats.dedup();
            cats
        };

        let successful_text = format_trajectories(successful);
        let failed_text = format_trajectories(failed);
        let categories_text = categories.join(", ");

        let prompt = INITIAL_LIBRARY_PROMPT
            .replace("{successful}", &successful_text)
            .replace("{failed}", &failed_text)
            .replace("{categories}", &categories_text);

        tracing::info!(
            n_successful = successful.len(),
            n_failed = failed.len(),
            n_categories = categories.len(),
            "Distilling initial skill library"
        );

        let response = self
            .client
            .generate_with_system(&prompt, DISTILLATION_SYSTEM_PROMPT, &self.model_id)
            .await
            .context("Teacher model call failed during initial library distillation")?;

        // Use empty string as default category -- the parser will determine each skill's
        // actual category from the JSON response.
        let skills = parse_skills_from_response(&response, "")?;

        let mut bank = SkillBank::new();
        bank.set_cycle(0);
        bank.merge(skills);

        tracing::info!(
            total_skills = bank.len(),
            categories = ?bank.skill_count_by_category(),
            "Initial skill library distilled"
        );

        Ok(bank)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Format a slice of trajectories into a human-readable text block for prompt injection.
fn format_trajectories(trajectories: &[Trajectory]) -> String {
    let mut buf = String::new();
    for (i, traj) in trajectories.iter().enumerate() {
        buf.push_str(&format!(
            "\n--- Trajectory {} (reward: {:.2}, success: {}) ---\n",
            i + 1,
            traj.total_reward,
            traj.success
        ));
        buf.push_str(&format!("Task: {}\n", traj.task_description));
        buf.push_str(&format!("Category: {}\n", traj.task_category));
        buf.push_str("Steps:\n");
        for step in &traj.steps {
            buf.push_str(&format!(
                "  [t={}] Observation: {}\n         Action: {}\n         Reward: {:.2}\n",
                step.step_index, step.observation, step.action, step.reward
            ));
        }
    }
    buf
}

/// Raw skill structure that mirrors the JSON the teacher model is expected to produce.
#[derive(serde::Deserialize)]
struct RawSkill {
    name: String,
    principle: String,
    when_to_apply: String,
    category: String,
}

/// Parse the teacher model's JSON response into a vector of [`Skill`] objects.
///
/// The response is expected to be a JSON array of objects with fields:
/// `name`, `principle`, `when_to_apply`, `category`.
///
/// `default_task_category` is used when a skill's `category` field is not "general"
/// but also not a recognized task category name -- in that case we fall back to this.
fn parse_skills_from_response(
    response: &str,
    default_task_category: &str,
) -> Result<Vec<Skill>> {
    // The model may wrap the JSON in markdown code fences; strip them.
    let trimmed = strip_code_fences(response);

    let raw_skills: Vec<RawSkill> = serde_json::from_str(trimmed).with_context(|| {
        format!(
            "Failed to parse teacher model response as JSON skill array. Response:\n{response}"
        )
    })?;

    let skills = raw_skills
        .into_iter()
        .map(|raw| {
            let category = if raw.category.eq_ignore_ascii_case("general") {
                SkillCategory::General
            } else {
                let cat_name = if raw.category.is_empty() {
                    default_task_category.to_string()
                } else {
                    raw.category
                };
                SkillCategory::TaskSpecific(cat_name)
            };

            Skill::new(raw.name, raw.principle, raw.when_to_apply, category)
        })
        .collect();

    Ok(skills)
}

/// Strip optional markdown code fences (```json ... ``` or ``` ... ```) from the response.
fn strip_code_fences(s: &str) -> &str {
    let trimmed = s.trim();

    // Try to strip ```json ... ```
    let stripped = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .unwrap_or(trimmed);

    let stripped = stripped
        .strip_suffix("```")
        .unwrap_or(stripped);

    stripped.trim()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_skills_from_response() {
        let json = r#"[
            {
                "name": "Verify State",
                "principle": "Always check the page state before acting.",
                "when_to_apply": "Before any submission action.",
                "category": "general"
            },
            {
                "name": "Use Search Bar",
                "principle": "Prefer the search bar over manual navigation.",
                "when_to_apply": "When looking for a specific item on a page.",
                "category": "web_navigation"
            }
        ]"#;

        let skills = parse_skills_from_response(json, "web_navigation").unwrap();
        assert_eq!(skills.len(), 2);
        assert_eq!(skills[0].category, SkillCategory::General);
        assert_eq!(
            skills[1].category,
            SkillCategory::TaskSpecific("web_navigation".into())
        );
    }

    #[test]
    fn test_parse_skills_with_code_fences() {
        let json = "```json\n[\n  {\n    \"name\": \"Test\",\n    \"principle\": \"P\",\n    \"when_to_apply\": \"W\",\n    \"category\": \"general\"\n  }\n]\n```";
        let skills = parse_skills_from_response(json, "").unwrap();
        assert_eq!(skills.len(), 1);
    }

    #[test]
    fn test_strip_code_fences() {
        assert_eq!(strip_code_fences("```json\n[1,2]\n```"), "[1,2]");
        assert_eq!(strip_code_fences("```\n[1,2]\n```"), "[1,2]");
        assert_eq!(strip_code_fences("[1,2]"), "[1,2]");
    }

    #[test]
    fn test_format_trajectories() {
        let traj = Trajectory {
            id: uuid::Uuid::new_v4().to_string(),
            task_description: "Navigate to settings".into(),
            task_category: "web_navigation".into(),
            steps: vec![crate::trajectory::Step {
                observation: "Homepage loaded".into(),
                action: "click settings".into(),
                reward: 1.0,
                step_index: 0,
                action_log_prob: None,
                ref_log_prob: None,
            }],
            total_reward: 1.0,
            success: true,
            metadata: crate::trajectory::TrajectoryMetadata {
                environment: "webshop".into(),
                num_steps: 1,
                total_tokens: 0,
                skills_used: vec![],
            },
        };
        let text = format_trajectories(&[traj]);
        assert!(text.contains("Navigate to settings"));
        assert!(text.contains("click settings"));
    }
}
