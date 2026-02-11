//! Trajectory types and collection for recording agent-environment interactions.
//!
//! This module provides:
//! - [`types::Step`], [`types::Trajectory`], [`types::TrajectoryMetadata`] -- the
//!   core data structures that capture what happened during an episode.
//! - [`types::TrajectoryBatch`] -- a group of trajectories for the same prompt
//!   (used by the GRPO advantage estimator).
//! - [`types::TrajectoryBuffer`] -- an accumulation buffer with filtering,
//!   stratified sampling, and batching helpers.
//! - [`collector::TrajectoryCollector`] -- the orchestration layer that drives
//!   agent-environment interaction loops and records trajectories.

pub mod collector;
pub mod types;

// Re-export the most commonly used items at the module level.
pub use collector::{AgentPolicy, SkillBank, SkillContext, TrajectoryCollector};
pub use types::{Step, Trajectory, TrajectoryBatch, TrajectoryBuffer, TrajectoryMetadata};
