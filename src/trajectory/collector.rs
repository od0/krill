//! Trajectory collection: orchestrating agent-environment interaction loops.
//!
//! The [`TrajectoryCollector`] drives episodes by repeatedly:
//!   1. presenting the observation to the agent,
//!   2. receiving the agent's action,
//!   3. stepping the environment,
//!   4. recording the (observation, action, reward) tuple.
//!
//! It supports two modes:
//! - **Vanilla collection** via [`collect_episodes`].
//! - **Skill-augmented collection** via [`collect_with_skills`], where relevant
//!   skills from the skill bank are injected into the agent's context before
//!   each episode.

use anyhow::Result;
use tracing;
use uuid::Uuid;

use crate::env::Environment;
use crate::trajectory::types::{Step, Trajectory, TrajectoryMetadata};

// ---------------------------------------------------------------------------
// Agent trait (minimal interface the collector needs)
// ---------------------------------------------------------------------------

/// The subset of agent capabilities the collector relies on.
///
/// The full `SkillAgent` lives in `crate::agent`; this trait lets us decouple
/// the collector from that module and makes it easy to supply mock agents in
/// tests.
#[allow(async_fn_in_trait)]
pub trait AgentPolicy: Send + Sync {
    /// Given an observation and (optionally) the list of legal actions,
    /// return the action string the agent wants to take.
    async fn select_action(
        &self,
        observation: &str,
        available_actions: Option<&[String]>,
    ) -> Result<String>;

    /// Provide skill context to the agent before an episode starts.
    ///
    /// The default implementation is a no-op (for agents that don't support
    /// skill augmentation).
    async fn set_skill_context(&mut self, _skill_descriptions: &[SkillContext]) -> Result<()> {
        Ok(())
    }

    /// Clear any previously injected skill context.
    fn clear_skill_context(&mut self) {}
}

/// A lightweight description of a skill, passed to the agent for in-context
/// augmentation.
#[derive(Debug, Clone)]
pub struct SkillContext {
    pub skill_id: String,
    pub name: String,
    pub description: String,
    /// The concrete action-sequence template (may contain placeholders).
    pub body: String,
}

// ---------------------------------------------------------------------------
// Skill bank trait (minimal interface the collector needs)
// ---------------------------------------------------------------------------

/// Minimal skill-bank interface used by the collector to retrieve relevant
/// skills for a given task.
#[allow(async_fn_in_trait)]
pub trait SkillBank: Send + Sync {
    /// Retrieve the top-k skills most relevant to `task_description`.
    async fn retrieve(&self, task_description: &str, top_k: usize) -> Result<Vec<SkillContext>>;
}

// ---------------------------------------------------------------------------
// Trajectory collector
// ---------------------------------------------------------------------------

/// Orchestrates episode collection by running an agent inside an environment.
#[derive(Debug, Clone)]
pub struct TrajectoryCollector {
    /// Label for the environment type (written into trajectory metadata).
    env_label: String,
    /// How many skills to retrieve per episode (when using skill augmentation).
    skill_top_k: usize,
}

impl TrajectoryCollector {
    /// Create a new collector.
    ///
    /// * `env_label` -- a short string like `"alfworld"` or `"webshop"`.
    /// * `skill_top_k` -- number of skills to retrieve per episode (only used
    ///   by [`collect_with_skills`]).
    pub fn new(env_label: &str, skill_top_k: usize) -> Self {
        Self {
            env_label: env_label.to_string(),
            skill_top_k,
        }
    }

    /// Collect `num_episodes` trajectories without skill augmentation.
    pub async fn collect_episodes<E, A>(
        &self,
        env: &mut E,
        agent: &A,
        num_episodes: usize,
    ) -> Result<Vec<Trajectory>>
    where
        E: Environment,
        A: AgentPolicy,
    {
        let mut trajectories = Vec::with_capacity(num_episodes);

        for ep in 0..num_episodes {
            let trajectory = self.run_episode(env, agent, &[]).await?;
            tracing::info!(
                episode = ep,
                steps = trajectory.steps.len(),
                reward = trajectory.total_reward,
                success = trajectory.success,
                "collected episode"
            );
            trajectories.push(trajectory);
        }

        Ok(trajectories)
    }

    /// Collect `num_episodes` trajectories *with* skill augmentation.
    ///
    /// Before each episode the collector queries the `skill_bank` for the most
    /// relevant skills and injects them into the agent's context.
    pub async fn collect_with_skills<E, A, S>(
        &self,
        env: &mut E,
        agent: &mut A,
        skill_bank: &S,
        num_episodes: usize,
    ) -> Result<Vec<Trajectory>>
    where
        E: Environment,
        A: AgentPolicy,
        S: SkillBank,
    {
        let mut trajectories = Vec::with_capacity(num_episodes);

        for ep in 0..num_episodes {
            // Reset env first so we know the task description to query skills.
            let init_obs = env.reset(None).await?;
            let task_desc = env.task_description().to_string();

            // Retrieve and inject skills.
            let skills = skill_bank
                .retrieve(&task_desc, self.skill_top_k)
                .await
                .unwrap_or_default();
            let skill_ids: Vec<String> = skills.iter().map(|s| s.skill_id.clone()).collect();
            agent.set_skill_context(&skills).await?;

            // Run the episode (using the already-reset observation).
            let trajectory =
                self.run_episode_from_obs(env, agent, init_obs, &skill_ids).await?;

            tracing::info!(
                episode = ep,
                steps = trajectory.steps.len(),
                reward = trajectory.total_reward,
                success = trajectory.success,
                skills = skill_ids.len(),
                "collected skill-augmented episode"
            );

            agent.clear_skill_context();
            trajectories.push(trajectory);
        }

        Ok(trajectories)
    }

    // -- internal helpers ---------------------------------------------------

    /// Run a single episode from scratch (calls `env.reset` internally).
    async fn run_episode<E, A>(
        &self,
        env: &mut E,
        agent: &A,
        skill_ids: &[String],
    ) -> Result<Trajectory>
    where
        E: Environment,
        A: AgentPolicy,
    {
        let init_obs = env.reset(None).await?;
        self.run_episode_from_obs(env, agent, init_obs, skill_ids)
            .await
    }

    /// Run an episode starting from an already-obtained initial observation.
    async fn run_episode_from_obs<E, A>(
        &self,
        env: &mut E,
        agent: &A,
        init_obs: crate::env::EnvObservation,
        skill_ids: &[String],
    ) -> Result<Trajectory>
    where
        E: Environment,
        A: AgentPolicy,
    {
        let task_description = env.task_description().to_string();
        let task_category = env.task_category().to_string();
        let max_steps = env.max_steps();

        let mut steps: Vec<Step> = Vec::new();
        let mut total_reward = 0.0;
        let mut total_tokens: usize = 0;
        let mut current_obs = init_obs;

        for step_idx in 0..max_steps {
            if current_obs.done {
                break;
            }

            // Ask the agent for an action.
            let action = agent
                .select_action(
                    &current_obs.text,
                    current_obs
                        .available_actions
                        .as_deref(),
                )
                .await?;

            // Rough token estimate (4 chars ~ 1 token).
            total_tokens += (current_obs.text.len() + action.len()) / 4;

            // Step the environment.
            let next_obs = env.step(&action).await?;

            steps.push(Step {
                observation: current_obs.text.clone(),
                action: action.clone(),
                reward: next_obs.reward,
                step_index: step_idx,
                action_log_prob: None,
                ref_log_prob: None,
            });

            total_reward += next_obs.reward;
            current_obs = next_obs;
        }

        // Determine success: positive total reward, or environment signalled
        // done with reward >= 1.
        let success = total_reward >= 1.0;

        Ok(Trajectory {
            id: Uuid::new_v4().to_string(),
            task_description,
            task_category,
            steps: steps.clone(),
            total_reward,
            success,
            metadata: TrajectoryMetadata {
                environment: self.env_label.clone(),
                num_steps: steps.len(),
                total_tokens,
                skills_used: skill_ids.to_vec(),
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::alfworld::MockAlfWorldEnv;
    use crate::env::webshop::MockWebShopEnv;

    /// A trivial agent that always returns a fixed action.
    struct FixedAgent {
        action: String,
    }

    impl FixedAgent {
        fn new(action: &str) -> Self {
            Self {
                action: action.to_string(),
            }
        }
    }

    impl AgentPolicy for FixedAgent {
        async fn select_action(
            &self,
            _observation: &str,
            _available_actions: Option<&[String]>,
        ) -> Result<String> {
            Ok(self.action.clone())
        }
    }

    #[tokio::test]
    async fn collect_alfworld_episodes() {
        let mut env = MockAlfWorldEnv::new();
        let agent = FixedAgent::new("go to countertop 1");
        let collector = TrajectoryCollector::new("alfworld", 6);

        let trajectories = collector.collect_episodes(&mut env, &agent, 2).await.unwrap();

        assert_eq!(trajectories.len(), 2);

        // First episode is "Pick" with reward 1.0.
        let t0 = &trajectories[0];
        assert_eq!(t0.task_category, "Pick");
        assert!((t0.total_reward - 1.0).abs() < f64::EPSILON);
        assert!(t0.success);
        assert!(!t0.steps.is_empty());
        assert_eq!(t0.metadata.environment, "alfworld");
        assert!(t0.metadata.skills_used.is_empty());

        // Second episode is "Clean".
        let t1 = &trajectories[1];
        assert_eq!(t1.task_category, "Clean");
    }

    #[tokio::test]
    async fn collect_webshop_episodes() {
        let mut env = MockWebShopEnv::new();
        let agent = FixedAgent::new("search[headphones]");
        let collector = TrajectoryCollector::new("webshop", 6);

        let trajectories = collector.collect_episodes(&mut env, &agent, 1).await.unwrap();

        assert_eq!(trajectories.len(), 1);
        let t = &trajectories[0];
        assert_eq!(t.task_category, "search");
        assert!(t.total_reward > 0.0);
        assert_eq!(t.metadata.environment, "webshop");
    }

    #[tokio::test]
    async fn collect_with_skill_augmentation() {
        /// A dummy skill bank that always returns one skill.
        struct DummySkillBank;

        impl SkillBank for DummySkillBank {
            async fn retrieve(
                &self,
                _task_description: &str,
                _top_k: usize,
            ) -> Result<Vec<SkillContext>> {
                Ok(vec![SkillContext {
                    skill_id: "skill-001".into(),
                    name: "navigate-to-object".into(),
                    description: "Navigate to a target object by checking likely locations.".into(),
                    body: "go to {location}".into(),
                }])
            }
        }

        /// An agent that records whether skill context was set.
        struct TrackingAgent {
            has_skills: std::sync::Arc<std::sync::atomic::AtomicBool>,
        }

        impl AgentPolicy for TrackingAgent {
            async fn select_action(
                &self,
                _observation: &str,
                _available_actions: Option<&[String]>,
            ) -> Result<String> {
                Ok("go to countertop 1".into())
            }

            async fn set_skill_context(
                &mut self,
                skills: &[SkillContext],
            ) -> Result<()> {
                if !skills.is_empty() {
                    self.has_skills
                        .store(true, std::sync::atomic::Ordering::SeqCst);
                }
                Ok(())
            }

            fn clear_skill_context(&mut self) {
                self.has_skills
                    .store(false, std::sync::atomic::Ordering::SeqCst);
            }
        }

        let mut env = MockAlfWorldEnv::new();
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut agent = TrackingAgent {
            has_skills: flag.clone(),
        };
        let skill_bank = DummySkillBank;
        let collector = TrajectoryCollector::new("alfworld", 6);

        let trajectories = collector
            .collect_with_skills(&mut env, &mut agent, &skill_bank, 1)
            .await
            .unwrap();

        assert_eq!(trajectories.len(), 1);
        let t = &trajectories[0];
        assert_eq!(t.metadata.skills_used, vec!["skill-001".to_string()]);
    }
}
