//! Core skill data types for the SkillRL framework.
//!
//! A **skill** is a distilled piece of strategic knowledge extracted from agent trajectories
//! by a teacher model. Each skill captures *what* to do, *when* to apply it, and *why*
//! it works, in a form that can be injected into the policy model's prompt.

use serde::{Deserialize, Serialize};

/// The category a skill belongs to within the hierarchical skill bank.
///
/// Skills are organized into two tiers:
/// - **General**: domain-agnostic strategies useful across all task categories.
/// - **TaskSpecific**: strategies that apply to a particular task category (e.g.,
///   "web_navigation", "code_editing", "database_query").
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SkillCategory {
    /// A domain-agnostic skill applicable to any task.
    General,
    /// A skill specific to the named task category.
    TaskSpecific(String),
}

impl std::fmt::Display for SkillCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SkillCategory::General => write!(f, "general"),
            SkillCategory::TaskSpecific(name) => write!(f, "task_specific:{name}"),
        }
    }
}

/// A single skill in the skill bank.
///
/// Corresponds to the paper's definition:
///   s = (name, principle, when_to_apply, category)
///
/// The `embedding` field holds the vector representation produced by the embedding model,
/// used for cosine-similarity retrieval at inference time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Unique identifier (UUID v4).
    pub id: String,
    /// Short, human-readable name for the skill.
    pub name: String,
    /// The core strategic principle this skill encodes.
    pub principle: String,
    /// Description of the situations / task states where this skill should be applied.
    pub when_to_apply: String,
    /// Whether this is a general or task-specific skill.
    pub category: SkillCategory,
    /// Optional embedding vector for similarity-based retrieval.
    /// `None` until the embedding model has been called.
    #[serde(default)]
    pub embedding: Option<Vec<f64>>,
}

impl Skill {
    /// Create a new skill with a fresh UUID.
    pub fn new(
        name: impl Into<String>,
        principle: impl Into<String>,
        when_to_apply: impl Into<String>,
        category: SkillCategory,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.into(),
            principle: principle.into(),
            when_to_apply: when_to_apply.into(),
            category,
            embedding: None,
        }
    }

    /// Return a compact textual representation suitable for embedding or prompt injection.
    pub fn to_prompt_text(&self) -> String {
        format!(
            "[{}] {}: {} (Apply when: {})",
            self.category, self.name, self.principle, self.when_to_apply
        )
    }

    /// Returns `true` if the skill has an embedding vector set.
    pub fn has_embedding(&self) -> bool {
        self.embedding.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skill_creation() {
        let skill = Skill::new(
            "Verify before submit",
            "Always verify the current state before submitting an answer",
            "When you are about to submit a final answer",
            SkillCategory::General,
        );
        assert!(!skill.id.is_empty());
        assert_eq!(skill.name, "Verify before submit");
        assert_eq!(skill.category, SkillCategory::General);
        assert!(skill.embedding.is_none());
    }

    #[test]
    fn test_skill_category_display() {
        assert_eq!(SkillCategory::General.to_string(), "general");
        assert_eq!(
            SkillCategory::TaskSpecific("web_nav".into()).to_string(),
            "task_specific:web_nav"
        );
    }

    #[test]
    fn test_skill_serialization_roundtrip() {
        let skill = Skill::new(
            "Test skill",
            "A principle",
            "When testing",
            SkillCategory::TaskSpecific("testing".into()),
        );
        let json = serde_json::to_string(&skill).unwrap();
        let deserialized: Skill = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, skill.id);
        assert_eq!(deserialized.name, skill.name);
        assert_eq!(deserialized.category, skill.category);
    }
}
