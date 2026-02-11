//! Recursive skill evolution.
//!
//! Implements the evolution loop from the SkillRL paper. At each validation checkpoint
//! the system:
//!
//! 1. Evaluates per-category success rates on the validation set.
//! 2. Identifies failing categories (success rate below threshold).
//! 3. Samples representative failure trajectories via diversity-aware stratified sampling.
//! 4. Sends them to the teacher model to generate new skills: `S_new = M_T(T_val^-, SkillBank)`.
//! 5. Merges the new skills into the bank: `SkillBank <- SkillBank ∪ S_new`.

use std::collections::HashMap;

use anyhow::{Context, Result};
use tracing;

use super::library::SkillBank;
use super::types::{Skill, SkillCategory};
use crate::config::EvolutionConfig;
use crate::model::LlmClient;
use crate::trajectory::Trajectory;

// ---------------------------------------------------------------------------
// Prompt templates
// ---------------------------------------------------------------------------

const EVOLUTION_SYSTEM_PROMPT: &str = r#"You are an expert skill-evolution engine. You receive a set of agent failure trajectories alongside the agent's current skill library. Your task is to generate NEW skills that address the identified gaps.

A "skill" is a compact, actionable piece of strategic knowledge with:
- name: A short, descriptive title (3-8 words).
- principle: The core strategic insight (1-3 sentences).
- when_to_apply: A description of the situations where this skill is useful (1-2 sentences).
- category: Either "general" (domain-agnostic) or the specific task category name.

Rules:
- Do NOT duplicate skills already in the library.
- Focus on NOVEL insights that the current library lacks.
- Each skill must directly address a failure mode visible in the trajectories.

You MUST respond with a JSON array of skill objects. Do NOT include any text outside the JSON array."#;

const EVOLUTION_PROMPT: &str = r#"The agent is failing on certain task categories. Analyze the failure trajectories below and generate new skills to fill the gaps in the current skill library.

Failing categories and their success rates:
{category_rates}

Current skill library ({n_skills} skills):
{current_skills}

Failure trajectories to analyze:
{trajectories}

Generate 1-{max_new} NEW skills that address the failure patterns above. Return a JSON array."#;

// ---------------------------------------------------------------------------
// SkillEvolver
// ---------------------------------------------------------------------------

/// Drives the recursive skill evolution loop.
///
/// At each validation checkpoint, the evolver evaluates which task categories the agent
/// is struggling with, samples representative failure trajectories, and uses the teacher
/// model to generate new skills that address the identified gaps.
pub struct SkillEvolver;

impl SkillEvolver {
    /// Compute the success rate for each task category present in the given trajectories.
    ///
    /// Returns a map of `category_name -> success_rate` where success_rate is in `[0.0, 1.0]`.
    pub fn evaluate_categories(trajectories: &[Trajectory]) -> HashMap<String, f64> {
        let mut total: HashMap<String, usize> = HashMap::new();
        let mut successes: HashMap<String, usize> = HashMap::new();

        for traj in trajectories {
            *total.entry(traj.task_category.clone()).or_insert(0) += 1;
            if traj.success {
                *successes.entry(traj.task_category.clone()).or_insert(0) += 1;
            }
        }

        total
            .into_iter()
            .map(|(cat, count)| {
                let succ = *successes.get(&cat).unwrap_or(&0);
                let rate = succ as f64 / count as f64;
                (cat, rate)
            })
            .collect()
    }

    /// Select categories whose success rate falls below the given threshold.
    ///
    /// Returns a sorted vector of category names for deterministic ordering.
    pub fn select_failing_categories(
        rates: &HashMap<String, f64>,
        threshold: f64,
    ) -> Vec<String> {
        let mut failing: Vec<String> = rates
            .iter()
            .filter(|(_, &rate)| rate < threshold)
            .map(|(cat, _)| cat.clone())
            .collect();
        failing.sort();
        failing
    }

    /// Diversity-aware stratified sampling of failure trajectories.
    ///
    /// Groups failed trajectories by category, prioritises those with the lowest
    /// reward (most severe failures), and uses round-robin selection across categories
    /// to ensure diversity.
    ///
    /// # Arguments
    ///
    /// * `failed` - All failed trajectories available for sampling.
    /// * `categories` - The set of failing categories to sample from.
    /// * `max_samples` - Maximum total trajectories to return.
    ///
    /// # Returns
    ///
    /// References to up to `max_samples` trajectories, balanced across categories.
    pub fn stratified_sample<'a>(
        failed: &'a [Trajectory],
        categories: &[String],
        max_samples: usize,
    ) -> Vec<&'a Trajectory> {
        if categories.is_empty() || max_samples == 0 {
            return Vec::new();
        }

        // Group failed trajectories by category, only keeping those in the target categories.
        let mut groups: HashMap<&str, Vec<&Trajectory>> = HashMap::new();
        for traj in failed {
            if !traj.success && categories.contains(&traj.task_category) {
                groups
                    .entry(&traj.task_category)
                    .or_default()
                    .push(traj);
            }
        }

        // Within each group, sort by total_reward ascending (most severe failures first).
        for group in groups.values_mut() {
            group.sort_by(|a, b| {
                a.total_reward
                    .partial_cmp(&b.total_reward)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Round-robin selection across categories.
        let mut result: Vec<&Trajectory> = Vec::with_capacity(max_samples);
        let mut indices: HashMap<&str, usize> = HashMap::new();
        let category_order: Vec<&str> = {
            let mut cats: Vec<&str> = groups.keys().copied().collect();
            cats.sort(); // deterministic order
            cats
        };

        'outer: loop {
            let mut any_added = false;
            for &cat in &category_order {
                if result.len() >= max_samples {
                    break 'outer;
                }
                let idx = indices.entry(cat).or_insert(0);
                if let Some(group) = groups.get(cat) {
                    if *idx < group.len() {
                        result.push(group[*idx]);
                        *idx += 1;
                        any_added = true;
                    }
                }
            }
            if !any_added || result.len() >= max_samples {
                break;
            }
        }

        result
    }

    /// Run one evolution cycle: analyse failures, generate new skills, and merge them
    /// into the bank.
    ///
    /// Implements:
    ///   `S_new = M_T(T_val^-, SkillBank)`
    ///   `SkillBank <- SkillBank ∪ S_new`
    ///
    /// # Arguments
    ///
    /// * `bank` - The current skill bank (will be mutated with new skills).
    /// * `validation_trajectories` - All trajectories from the latest validation run.
    /// * `config` - Evolution hyperparameters.
    /// * `client` - The LLM client for calling the teacher model.
    /// * `model_id` - The teacher model identifier.
    ///
    /// # Returns
    ///
    /// The newly generated skills (which have already been merged into `bank`).
    pub async fn evolve(
        bank: &mut SkillBank,
        validation_trajectories: &[Trajectory],
        config: &EvolutionConfig,
        client: &LlmClient,
        model_id: &str,
    ) -> Result<Vec<Skill>> {
        // Step 1: Evaluate per-category success rates.
        let rates = Self::evaluate_categories(validation_trajectories);

        tracing::info!(
            category_rates = ?rates,
            "Evaluated per-category success rates"
        );

        // Step 2: Identify failing categories.
        let failing = Self::select_failing_categories(&rates, config.evolution_threshold);

        if failing.is_empty() {
            tracing::info!("No failing categories -- skipping evolution");
            return Ok(Vec::new());
        }

        tracing::info!(
            failing_categories = ?failing,
            threshold = config.evolution_threshold,
            "Identified failing categories for evolution"
        );

        // Step 3: Determine sampling depth based on how far below threshold we are.
        //   - If the worst category rate < threshold / 2, use deep analysis.
        //   - Otherwise, use shallow analysis.
        let worst_rate = failing
            .iter()
            .filter_map(|cat| rates.get(cat))
            .cloned()
            .fold(f64::INFINITY, f64::min);

        let max_samples = if worst_rate < config.evolution_threshold / 2.0 {
            config.max_analysis_deep
        } else {
            config.max_analysis_shallow
        };

        // Step 4: Stratified sampling of failure trajectories.
        let failed_owned: Vec<Trajectory> = validation_trajectories
            .iter()
            .filter(|t| !t.success)
            .cloned()
            .collect();

        let sampled =
            Self::stratified_sample(&failed_owned, &failing, max_samples);

        if sampled.is_empty() {
            tracing::warn!("No failed trajectories available for evolution");
            return Ok(Vec::new());
        }

        tracing::info!(
            n_sampled = sampled.len(),
            max_samples,
            "Sampled failure trajectories for analysis"
        );

        // Step 5: Build the evolution prompt.
        let category_rates_text: String = failing
            .iter()
            .map(|cat| {
                let rate = rates.get(cat).unwrap_or(&0.0);
                format!("  - {cat}: {:.1}%", rate * 100.0)
            })
            .collect::<Vec<_>>()
            .join("\n");

        let current_skills_text: String = bank
            .all_skills()
            .map(|s| format!("  - {}", s.to_prompt_text()))
            .collect::<Vec<_>>()
            .join("\n");

        let trajectories_text = format_trajectories_for_evolution(sampled.as_slice());

        let prompt = EVOLUTION_PROMPT
            .replace("{category_rates}", &category_rates_text)
            .replace("{n_skills}", &bank.len().to_string())
            .replace("{current_skills}", &current_skills_text)
            .replace("{trajectories}", &trajectories_text)
            .replace("{max_new}", &config.max_new_skills.to_string());

        // Step 6: Call the teacher model.
        let response = client
            .generate_with_system(&prompt, EVOLUTION_SYSTEM_PROMPT, model_id)
            .await
            .context("Teacher model call failed during skill evolution")?;

        // Step 7: Parse the new skills.
        let default_category = failing.first().map(|s| s.as_str()).unwrap_or("general");
        let mut new_skills = parse_evolution_response(&response, default_category)?;

        // Cap at max_new_skills.
        new_skills.truncate(config.max_new_skills);

        tracing::info!(
            n_new_skills = new_skills.len(),
            "Generated new skills via evolution"
        );

        // Step 8: Merge into bank.
        let cycle = bank.current_cycle() + 1;
        bank.set_cycle(cycle);
        bank.merge(new_skills.clone());

        tracing::info!(
            total_skills = bank.len(),
            cycle,
            "Merged evolved skills into bank"
        );

        Ok(new_skills)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Format sampled trajectories for the evolution prompt.
fn format_trajectories_for_evolution(trajectories: &[&Trajectory]) -> String {
    let mut buf = String::new();
    for (i, traj) in trajectories.iter().enumerate() {
        buf.push_str(&format!(
            "\n--- Failure {} (category: {}, reward: {:.2}) ---\n",
            i + 1,
            traj.task_category,
            traj.total_reward
        ));
        buf.push_str(&format!("Task: {}\n", traj.task_description));
        buf.push_str("Steps:\n");
        for step in &traj.steps {
            buf.push_str(&format!(
                "  [t={}] Obs: {} | Act: {} | R: {:.2}\n",
                step.step_index, step.observation, step.action, step.reward
            ));
        }
    }
    buf
}

/// Raw skill JSON shape expected from the teacher model.
#[derive(serde::Deserialize)]
struct RawSkill {
    name: String,
    principle: String,
    when_to_apply: String,
    category: String,
}

/// Parse the teacher model's evolution response into [`Skill`] objects.
fn parse_evolution_response(
    response: &str,
    default_task_category: &str,
) -> Result<Vec<Skill>> {
    let trimmed = strip_code_fences(response);

    let raw_skills: Vec<RawSkill> = serde_json::from_str(trimmed).with_context(|| {
        format!(
            "Failed to parse evolution response as JSON skill array. Response:\n{response}"
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

/// Strip optional markdown code fences from the response.
fn strip_code_fences(s: &str) -> &str {
    let trimmed = s.trim();
    let stripped = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .unwrap_or(trimmed);
    let stripped = stripped.strip_suffix("```").unwrap_or(stripped);
    stripped.trim()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::{Step, TrajectoryMetadata};

    fn make_traj(category: &str, success: bool, reward: f64) -> Trajectory {
        Trajectory {
            id: uuid::Uuid::new_v4().to_string(),
            task_description: format!("Task in {category}"),
            task_category: category.to_string(),
            steps: vec![Step {
                observation: "obs".into(),
                action: "act".into(),
                reward,
                step_index: 0,
                action_log_prob: None,
                ref_log_prob: None,
            }],
            total_reward: reward,
            success,
            metadata: TrajectoryMetadata {
                environment: "test".into(),
                num_steps: 1,
                total_tokens: 0,
                skills_used: vec![],
            },
        }
    }

    #[test]
    fn test_evaluate_categories() {
        let trajs = vec![
            make_traj("web", true, 1.0),
            make_traj("web", false, 0.0),
            make_traj("web", true, 1.0),
            make_traj("code", false, 0.0),
            make_traj("code", false, 0.0),
        ];

        let rates = SkillEvolver::evaluate_categories(&trajs);
        assert!((rates["web"] - 2.0 / 3.0).abs() < 1e-9);
        assert!((rates["code"] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_select_failing_categories() {
        let mut rates = HashMap::new();
        rates.insert("web".into(), 0.8);
        rates.insert("code".into(), 0.2);
        rates.insert("db".into(), 0.35);

        let failing = SkillEvolver::select_failing_categories(&rates, 0.4);
        assert_eq!(failing, vec!["code", "db"]);
    }

    #[test]
    fn test_select_failing_categories_none() {
        let mut rates = HashMap::new();
        rates.insert("web".into(), 0.9);
        rates.insert("code".into(), 0.5);

        let failing = SkillEvolver::select_failing_categories(&rates, 0.4);
        assert!(failing.is_empty());
    }

    #[test]
    fn test_stratified_sample_basic() {
        let trajs = vec![
            make_traj("web", false, 0.1),
            make_traj("web", false, 0.3),
            make_traj("code", false, 0.0),
            make_traj("code", false, 0.2),
            make_traj("db", false, 0.05),
        ];
        let categories = vec!["web".into(), "code".into()];

        let sampled = SkillEvolver::stratified_sample(&trajs, &categories, 3);

        // Round-robin: code(0.0), web(0.1), code(0.2) -- but order depends on sort.
        assert_eq!(sampled.len(), 3);

        // Verify all sampled are from target categories.
        for t in &sampled {
            assert!(
                t.task_category == "web" || t.task_category == "code",
                "Unexpected category: {}",
                t.task_category
            );
        }
    }

    #[test]
    fn test_stratified_sample_empty() {
        let trajs = vec![make_traj("web", true, 1.0)]; // success, should be skipped
        let categories = vec!["web".into()];

        let sampled = SkillEvolver::stratified_sample(&trajs, &categories, 5);
        assert!(sampled.is_empty());
    }

    #[test]
    fn test_stratified_sample_severity_ordering() {
        let trajs = vec![
            make_traj("web", false, 0.5), // less severe
            make_traj("web", false, 0.1), // more severe
            make_traj("web", false, 0.9), // least severe
        ];
        let categories = vec!["web".into()];

        let sampled = SkillEvolver::stratified_sample(&trajs, &categories, 2);
        assert_eq!(sampled.len(), 2);
        // Most severe first.
        assert!(sampled[0].total_reward <= sampled[1].total_reward);
    }

    #[test]
    fn test_parse_evolution_response() {
        let json = r#"[
            {
                "name": "Check Element Visibility",
                "principle": "Before clicking, verify the target element is visible.",
                "when_to_apply": "When interacting with dynamic web pages.",
                "category": "web_navigation"
            }
        ]"#;

        let skills = parse_evolution_response(json, "web_navigation").unwrap();
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "Check Element Visibility");
        assert_eq!(
            skills[0].category,
            SkillCategory::TaskSpecific("web_navigation".into())
        );
    }
}
