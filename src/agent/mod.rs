//! Agent module: the SkillRL skill-augmented agent.
//!
//! The [`SkillAgent`] retrieves relevant skills from the [`SkillBank`],
//! constructs a skill-augmented prompt, and generates actions via an LLM
//! policy model.

pub mod agent;

// Re-export the primary types for convenient access.
pub use agent::{ActionOutput, ActionOutputWithLogProb, SkillAgent};
