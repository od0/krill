//! Core environment trait and shared types.
//!
//! Every task environment (ALFWorld, WebShop, ...) implements the [`Environment`]
//! trait so that the trajectory collector can interact with it uniformly.

use serde::{Deserialize, Serialize};

/// An observation returned by the environment after a reset or step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvObservation {
    /// The textual observation the agent sees.
    pub text: String,
    /// An optional explicit list of legal actions (when the env provides one).
    pub available_actions: Option<Vec<String>>,
    /// Whether the episode has terminated.
    pub done: bool,
    /// The scalar reward for the transition that produced this observation.
    pub reward: f64,
    /// Arbitrary extra information from the environment (task-specific).
    pub info: serde_json::Value,
}

/// Static configuration for an environment instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvConfig {
    /// Maximum number of interaction steps before forced termination.
    pub max_steps: usize,
    /// A short label for the environment type (e.g. "alfworld", "webshop").
    pub env_type: String,
}

/// The core environment trait.
///
/// All task environments implement this trait so that the trajectory collector
/// can drive episodes in a uniform fashion.
#[allow(async_fn_in_trait)]
pub trait Environment: Send + Sync {
    /// Reset the environment and optionally load a specific task by id.
    ///
    /// Returns the initial observation for the new episode.
    async fn reset(&mut self, task_id: Option<&str>) -> anyhow::Result<EnvObservation>;

    /// Execute an action in the environment and return the resulting observation.
    async fn step(&mut self, action: &str) -> anyhow::Result<EnvObservation>;

    /// A human-readable description of the current task.
    fn task_description(&self) -> &str;

    /// The category label for the current task (e.g. "Pick", "Clean", "search").
    fn task_category(&self) -> &str;

    /// The maximum number of steps allowed in an episode.
    fn max_steps(&self) -> usize;

    /// Whether the current episode has ended (success, failure, or truncation).
    fn is_done(&self) -> bool;
}
