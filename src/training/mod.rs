//! Training module for SkillRL.
//!
//! This module implements the full training pipeline from the SkillRL paper,
//! including GRPO (Group Relative Policy Optimization), cold-start SFT
//! (Supervised Fine-Tuning), and the recursive skill evolution loop.

pub mod advantage;
pub mod grpo;
pub mod pipeline;
pub mod sft;

pub use advantage::{clip_ratio, compute_group_advantages, compute_importance_ratio};
pub use grpo::{GrpoSample, GrpoStepResult, GrpoTrainer};
pub use pipeline::{TrainingMetrics, TrainingPipeline};
pub use sft::{SftExample, SftStepResult, SftTrainer};
