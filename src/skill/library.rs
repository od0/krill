//! Hierarchical skill bank for organising and persisting skills.
//!
//! The [`SkillBank`] stores skills in a two-tier hierarchy:
//! 1. **General skills** -- domain-agnostic strategies.
//! 2. **Task-specific skills** -- keyed by task category name.
//!
//! It also tracks the evolution history so we can audit when skills were added.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::types::{Skill, SkillCategory};

/// A record of when a skill was added to the bank.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillHistoryEntry {
    /// The skill ID that was added.
    pub skill_id: String,
    /// UTC timestamp of insertion.
    pub added_at: DateTime<Utc>,
    /// Which evolution cycle added this skill (0 = initial distillation).
    pub evolution_cycle: usize,
}

/// A hierarchical skill bank that organises skills into general and task-specific pools.
///
/// Corresponds to the paper's `SkillBank` which is updated across evolution cycles:
///   `SkillBank <- SkillBank ∪ S_new`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillBank {
    /// Domain-agnostic skills applicable to any task.
    general_skills: Vec<Skill>,
    /// Task-specific skills keyed by category name.
    task_specific_skills: HashMap<String, Vec<Skill>>,
    /// Chronological record of all skill insertions.
    history: Vec<SkillHistoryEntry>,
    /// Counter tracking the current evolution cycle (incremented externally).
    #[serde(default)]
    current_cycle: usize,
}

impl SkillBank {
    /// Create an empty skill bank.
    pub fn new() -> Self {
        Self {
            general_skills: Vec::new(),
            task_specific_skills: HashMap::new(),
            history: Vec::new(),
            current_cycle: 0,
        }
    }

    // ------------------------------------------------------------------
    // Mutation
    // ------------------------------------------------------------------

    /// Add a single skill to the bank, filing it under the correct tier.
    ///
    /// A history entry is recorded with the current timestamp and evolution cycle.
    pub fn add_skill(&mut self, skill: Skill) {
        let entry = SkillHistoryEntry {
            skill_id: skill.id.clone(),
            added_at: Utc::now(),
            evolution_cycle: self.current_cycle,
        };
        self.history.push(entry);

        match &skill.category {
            SkillCategory::General => {
                self.general_skills.push(skill);
            }
            SkillCategory::TaskSpecific(cat) => {
                self.task_specific_skills
                    .entry(cat.clone())
                    .or_default()
                    .push(skill);
            }
        }
    }

    /// Merge a batch of new skills into the bank.
    ///
    /// This is the `SkillBank <- SkillBank ∪ S_new` operation from the paper.
    pub fn merge(&mut self, new_skills: Vec<Skill>) {
        for skill in new_skills {
            self.add_skill(skill);
        }
    }

    /// Set the current evolution cycle index (used for history tracking).
    pub fn set_cycle(&mut self, cycle: usize) {
        self.current_cycle = cycle;
    }

    /// Return the current evolution cycle.
    pub fn current_cycle(&self) -> usize {
        self.current_cycle
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// Return a slice of all general (domain-agnostic) skills.
    pub fn get_general_skills(&self) -> &[Skill] {
        &self.general_skills
    }

    /// Return the task-specific skills for a given category, or an empty slice if none exist.
    pub fn get_task_skills(&self, category: &str) -> &[Skill] {
        self.task_specific_skills
            .get(category)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Return an iterator over all skills (general + every task-specific pool).
    pub fn all_skills(&self) -> impl Iterator<Item = &Skill> {
        self.general_skills
            .iter()
            .chain(self.task_specific_skills.values().flatten())
    }

    /// Total number of skills in the bank.
    pub fn len(&self) -> usize {
        self.general_skills.len()
            + self
                .task_specific_skills
                .values()
                .map(|v| v.len())
                .sum::<usize>()
    }

    /// Returns `true` if the bank contains no skills.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return a map of category name -> skill count.
    ///
    /// The general pool is keyed as `"general"`.
    pub fn skill_count_by_category(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        counts.insert("general".to_string(), self.general_skills.len());
        for (cat, skills) in &self.task_specific_skills {
            counts.insert(cat.clone(), skills.len());
        }
        counts
    }

    /// Return the full evolution history.
    pub fn history(&self) -> &[SkillHistoryEntry] {
        &self.history
    }

    /// Return all task category names that have at least one skill.
    pub fn task_categories(&self) -> Vec<String> {
        self.task_specific_skills.keys().cloned().collect()
    }

    // ------------------------------------------------------------------
    // Persistence
    // ------------------------------------------------------------------

    /// Serialize the skill bank to a JSON file at the given path.
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .context("Failed to serialize SkillBank to JSON")?;
        std::fs::write(path.as_ref(), json)
            .with_context(|| format!("Failed to write SkillBank to {}", path.as_ref().display()))?;
        tracing::info!(
            path = %path.as_ref().display(),
            skills = self.len(),
            "Saved skill bank"
        );
        Ok(())
    }

    /// Deserialize a skill bank from a JSON file.
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self> {
        let data = std::fs::read_to_string(path.as_ref()).with_context(|| {
            format!(
                "Failed to read SkillBank from {}",
                path.as_ref().display()
            )
        })?;
        let bank: Self =
            serde_json::from_str(&data).context("Failed to deserialize SkillBank JSON")?;
        tracing::info!(
            path = %path.as_ref().display(),
            skills = bank.len(),
            "Loaded skill bank"
        );
        Ok(bank)
    }
}

impl Default for SkillBank {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skill::types::SkillCategory;

    fn make_skill(name: &str, category: SkillCategory) -> Skill {
        Skill::new(name, "principle", "when", category)
    }

    #[test]
    fn test_add_and_retrieve() {
        let mut bank = SkillBank::new();
        bank.add_skill(make_skill("general_1", SkillCategory::General));
        bank.add_skill(make_skill(
            "web_1",
            SkillCategory::TaskSpecific("web".into()),
        ));
        bank.add_skill(make_skill(
            "web_2",
            SkillCategory::TaskSpecific("web".into()),
        ));
        bank.add_skill(make_skill(
            "code_1",
            SkillCategory::TaskSpecific("code".into()),
        ));

        assert_eq!(bank.len(), 4);
        assert_eq!(bank.get_general_skills().len(), 1);
        assert_eq!(bank.get_task_skills("web").len(), 2);
        assert_eq!(bank.get_task_skills("code").len(), 1);
        assert_eq!(bank.get_task_skills("nonexistent").len(), 0);
    }

    #[test]
    fn test_merge() {
        let mut bank = SkillBank::new();
        let skills = vec![
            make_skill("a", SkillCategory::General),
            make_skill("b", SkillCategory::TaskSpecific("web".into())),
        ];
        bank.merge(skills);
        assert_eq!(bank.len(), 2);
    }

    #[test]
    fn test_skill_count_by_category() {
        let mut bank = SkillBank::new();
        bank.add_skill(make_skill("g1", SkillCategory::General));
        bank.add_skill(make_skill("g2", SkillCategory::General));
        bank.add_skill(make_skill(
            "w1",
            SkillCategory::TaskSpecific("web".into()),
        ));

        let counts = bank.skill_count_by_category();
        assert_eq!(counts["general"], 2);
        assert_eq!(counts["web"], 1);
    }

    #[test]
    fn test_history_tracking() {
        let mut bank = SkillBank::new();
        bank.set_cycle(0);
        bank.add_skill(make_skill("initial", SkillCategory::General));
        bank.set_cycle(1);
        bank.add_skill(make_skill("evolved", SkillCategory::General));

        assert_eq!(bank.history().len(), 2);
        assert_eq!(bank.history()[0].evolution_cycle, 0);
        assert_eq!(bank.history()[1].evolution_cycle, 1);
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let mut bank = SkillBank::new();
        bank.add_skill(make_skill("test", SkillCategory::General));

        let tmp = std::env::temp_dir().join("skillrl_test_bank.json");
        bank.save_to_file(&tmp).unwrap();
        let loaded = SkillBank::load_from_file(&tmp).unwrap();
        assert_eq!(loaded.len(), bank.len());
        assert_eq!(
            loaded.get_general_skills()[0].name,
            bank.get_general_skills()[0].name
        );
        std::fs::remove_file(tmp).ok();
    }

    #[test]
    fn test_all_skills_iterator() {
        let mut bank = SkillBank::new();
        bank.add_skill(make_skill("g", SkillCategory::General));
        bank.add_skill(make_skill(
            "t",
            SkillCategory::TaskSpecific("web".into()),
        ));

        let all: Vec<_> = bank.all_skills().collect();
        assert_eq!(all.len(), 2);
    }
}
