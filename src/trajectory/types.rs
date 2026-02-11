//! Core trajectory data types used throughout the SkillRL pipeline.
//!
//! These types capture everything the GRPO trainer and skill evolution modules
//! need to know about an agent's interaction with an environment.

use std::collections::HashMap;

use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Single step
// ---------------------------------------------------------------------------

/// A single step within a trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    /// The textual observation the agent received.
    pub observation: String,
    /// The action the agent chose.
    pub action: String,
    /// The scalar reward for this transition.
    pub reward: f64,
    /// Zero-based index of this step within the trajectory.
    pub step_index: usize,
    /// Log probability of the action under the *current* policy (may be None
    /// until the training loop fills it in).
    pub action_log_prob: Option<f64>,
    /// Log probability of the action under the *reference* policy (for KL
    /// penalty in GRPO).
    pub ref_log_prob: Option<f64>,
}

// ---------------------------------------------------------------------------
// Trajectory metadata
// ---------------------------------------------------------------------------

/// Auxiliary metadata attached to every trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryMetadata {
    /// Which environment produced this trajectory (e.g. "alfworld", "webshop").
    pub environment: String,
    /// Number of steps in the trajectory (same as `steps.len()`).
    pub num_steps: usize,
    /// Approximate total token count consumed (prompt + generation) across all
    /// steps.
    pub total_tokens: usize,
    /// IDs of skills that were injected into the agent context for this episode.
    pub skills_used: Vec<String>,
}

// ---------------------------------------------------------------------------
// Full trajectory
// ---------------------------------------------------------------------------

/// A complete trajectory recording one episode of agent-environment interaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    /// Unique identifier (UUID v4).
    pub id: String,
    /// Human-readable description of the task the agent was solving.
    pub task_description: String,
    /// Category label for the task (e.g. "Pick", "Clean", "search").
    pub task_category: String,
    /// Ordered sequence of steps.
    pub steps: Vec<Step>,
    /// Total accumulated reward over the episode.
    pub total_reward: f64,
    /// Whether the task was completed successfully.
    pub success: bool,
    /// Extra metadata.
    pub metadata: TrajectoryMetadata,
}

// ---------------------------------------------------------------------------
// Trajectory batch (for GRPO)
// ---------------------------------------------------------------------------

/// A group of trajectories for the same task prompt, used by the GRPO
/// advantage estimator.
///
/// In GRPO, G outputs are sampled for each prompt and their rewards are used to
/// compute group-relative advantages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryBatch {
    /// The shared task description for all trajectories in the batch.
    pub task_description: String,
    /// The task category.
    pub task_category: String,
    /// The G trajectories sampled for this prompt.
    pub trajectories: Vec<Trajectory>,
}

impl TrajectoryBatch {
    /// Create a new batch for a given task.
    pub fn new(task_description: String, task_category: String) -> Self {
        Self {
            task_description,
            task_category,
            trajectories: Vec::new(),
        }
    }

    /// Number of trajectories in the group.
    pub fn group_size(&self) -> usize {
        self.trajectories.len()
    }

    /// Mean reward across the group (used for GRPO baseline).
    pub fn mean_reward(&self) -> f64 {
        if self.trajectories.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.trajectories.iter().map(|t| t.total_reward).sum();
        sum / self.trajectories.len() as f64
    }

    /// Standard deviation of rewards (used for GRPO normalisation).
    pub fn reward_std(&self) -> f64 {
        if self.trajectories.len() < 2 {
            return 1.0; // avoid division by zero
        }
        let mean = self.mean_reward();
        let var: f64 = self
            .trajectories
            .iter()
            .map(|t| (t.total_reward - mean).powi(2))
            .sum::<f64>()
            / (self.trajectories.len() as f64 - 1.0);
        var.sqrt().max(1e-8)
    }

    /// Compute the GRPO-style advantage for each trajectory in the batch.
    ///
    /// advantage_i = (r_i - mean(r)) / std(r)
    pub fn advantages(&self) -> Vec<f64> {
        let mean = self.mean_reward();
        let std = self.reward_std();
        self.trajectories
            .iter()
            .map(|t| (t.total_reward - mean) / std)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Trajectory buffer
// ---------------------------------------------------------------------------

/// A buffer for accumulating trajectories during data collection.
///
/// Supports the operations the training loop needs: batching, stratified
/// sampling, filtering by outcome, and draining for consumption.
#[derive(Debug, Clone, Default)]
pub struct TrajectoryBuffer {
    trajectories: Vec<Trajectory>,
}

impl TrajectoryBuffer {
    /// Create an empty buffer.
    pub fn new() -> Self {
        Self {
            trajectories: Vec::new(),
        }
    }

    /// Create a buffer pre-allocated for `capacity` trajectories.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            trajectories: Vec::with_capacity(capacity),
        }
    }

    /// Number of trajectories currently in the buffer.
    pub fn len(&self) -> usize {
        self.trajectories.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.trajectories.is_empty()
    }

    /// Push a single trajectory into the buffer.
    pub fn push(&mut self, trajectory: Trajectory) {
        self.trajectories.push(trajectory);
    }

    /// Extend the buffer with an iterator of trajectories.
    pub fn extend(&mut self, iter: impl IntoIterator<Item = Trajectory>) {
        self.trajectories.extend(iter);
    }

    /// Drain **all** trajectories from the buffer, leaving it empty.
    pub fn drain(&mut self) -> Vec<Trajectory> {
        std::mem::take(&mut self.trajectories)
    }

    /// Return a slice view of all trajectories.
    pub fn as_slice(&self) -> &[Trajectory] {
        &self.trajectories
    }

    /// Return only the successful trajectories (cloned).
    pub fn filter_successful(&self) -> Vec<Trajectory> {
        self.trajectories
            .iter()
            .filter(|t| t.success)
            .cloned()
            .collect()
    }

    /// Return only the failed trajectories (cloned).
    pub fn filter_failed(&self) -> Vec<Trajectory> {
        self.trajectories
            .iter()
            .filter(|t| !t.success)
            .cloned()
            .collect()
    }

    /// Group trajectories by their task category.
    pub fn group_by_category(&self) -> HashMap<String, Vec<Trajectory>> {
        let mut groups: HashMap<String, Vec<Trajectory>> = HashMap::new();
        for t in &self.trajectories {
            groups
                .entry(t.task_category.clone())
                .or_default()
                .push(t.clone());
        }
        groups
    }

    /// Sample `n` trajectories using stratified sampling across task categories.
    ///
    /// Each category contributes proportionally to its share of the buffer.
    /// If `n` exceeds the buffer size, all trajectories are returned (shuffled).
    pub fn sample_stratified(&self, n: usize) -> Vec<Trajectory> {
        if n == 0 {
            return Vec::new();
        }
        let total = self.trajectories.len();
        if n >= total {
            let mut all = self.trajectories.clone();
            let mut rng = rand::thread_rng();
            all.shuffle(&mut rng);
            return all;
        }

        let groups = self.group_by_category();
        let num_categories = groups.len();
        if num_categories == 0 {
            return Vec::new();
        }

        let mut rng = rand::thread_rng();
        let mut result: Vec<Trajectory> = Vec::with_capacity(n);

        // Compute how many samples each category should contribute.
        let mut allocations: Vec<(&String, usize)> = Vec::new();
        let mut allocated = 0usize;
        let cats: Vec<(&String, &Vec<Trajectory>)> = groups.iter().collect();

        for (i, (cat, trajs)) in cats.iter().enumerate() {
            let share = if i < cats.len() - 1 {
                // Proportional share, rounded down.
                let s = (trajs.len() as f64 / total as f64 * n as f64).floor() as usize;
                s.min(trajs.len())
            } else {
                // Last category gets the remainder.
                (n - allocated).min(trajs.len())
            };
            allocations.push((cat, share));
            allocated += share;
        }

        // If rounding left us short, top up from the largest categories.
        while allocated < n {
            let mut topped = false;
            for (cat, alloc) in allocations.iter_mut() {
                if allocated >= n {
                    break;
                }
                let available = groups[*cat].len();
                if *alloc < available {
                    *alloc += 1;
                    allocated += 1;
                    topped = true;
                }
            }
            if !topped {
                break; // cannot allocate more
            }
        }

        // Sample from each category.
        for (cat, count) in &allocations {
            let trajs = &groups[*cat];
            let mut indices: Vec<usize> = (0..trajs.len()).collect();
            indices.shuffle(&mut rng);
            for &idx in indices.iter().take(*count) {
                result.push(trajs[idx].clone());
            }
        }

        result.shuffle(&mut rng);
        result
    }

    /// Organise the buffered trajectories into [`TrajectoryBatch`]es grouped
    /// by task description.
    ///
    /// This is the grouping that GRPO expects: multiple rollouts for the same
    /// prompt.
    pub fn into_batches(self) -> Vec<TrajectoryBatch> {
        let mut map: HashMap<String, TrajectoryBatch> = HashMap::new();
        for t in self.trajectories {
            let batch = map
                .entry(t.task_description.clone())
                .or_insert_with(|| {
                    TrajectoryBatch::new(t.task_description.clone(), t.task_category.clone())
                });
            batch.trajectories.push(t);
        }
        map.into_values().collect()
    }

    /// Overall success rate across all buffered trajectories.
    pub fn success_rate(&self) -> f64 {
        if self.trajectories.is_empty() {
            return 0.0;
        }
        let successes = self.trajectories.iter().filter(|t| t.success).count();
        successes as f64 / self.trajectories.len() as f64
    }
}
