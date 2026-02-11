//! ALFWorld text-based household environment.
//!
//! ALFWorld presents the agent with interactive household tasks (pick up objects,
//! clean items, heat food, etc.) described entirely in natural language.
//!
//! This module provides two implementations:
//! - [`AlfWorldEnv`] connects to a running ALFWorld server via HTTP.
//! - [`MockAlfWorldEnv`] returns canned observation sequences for testing.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing;

use super::traits::{EnvObservation, Environment};

// ---------------------------------------------------------------------------
// Task categories
// ---------------------------------------------------------------------------

/// The six task types in ALFWorld.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlfWorldTaskCategory {
    Pick,
    Look,
    Clean,
    Heat,
    Cool,
    PickTwo,
}

impl AlfWorldTaskCategory {
    /// Human-readable label.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pick => "Pick",
            Self::Look => "Look",
            Self::Clean => "Clean",
            Self::Heat => "Heat",
            Self::Cool => "Cool",
            Self::PickTwo => "PickTwo",
        }
    }

    /// Parse from a string (case-insensitive).
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "pick" => Some(Self::Pick),
            "look" => Some(Self::Look),
            "clean" => Some(Self::Clean),
            "heat" => Some(Self::Heat),
            "cool" => Some(Self::Cool),
            "picktwo" | "pick_two" | "pick two" => Some(Self::PickTwo),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP-backed ALFWorld environment
// ---------------------------------------------------------------------------

/// An ALFWorld environment that communicates with an external server via HTTP.
///
/// The server is expected to expose two endpoints:
/// - `POST {base_url}/reset`  -- body: `{"task_id": <optional string>}`
/// - `POST {base_url}/step`   -- body: `{"action": "<action string>"}`
///
/// Both return JSON matching [`ServerResponse`].
#[derive(Debug)]
pub struct AlfWorldEnv {
    /// Base URL of the ALFWorld server (e.g. `http://localhost:3000`).
    base_url: String,
    http: reqwest::Client,
    current_task_description: String,
    current_task_category: String,
    max_steps: usize,
    current_step: usize,
    done: bool,
}

/// The JSON shape returned by the ALFWorld server.
#[derive(Debug, Deserialize)]
struct ServerResponse {
    observation: String,
    #[serde(default)]
    available_actions: Option<Vec<String>>,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    reward: f64,
    #[serde(default)]
    task_description: Option<String>,
    #[serde(default)]
    task_category: Option<String>,
    #[serde(default)]
    info: Option<serde_json::Value>,
}

impl AlfWorldEnv {
    /// Create a new ALFWorld environment pointing at the given server.
    pub fn new(base_url: &str, max_steps: usize) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            http: reqwest::Client::new(),
            current_task_description: String::new(),
            current_task_category: String::new(),
            max_steps,
            current_step: 0,
            done: false,
        }
    }
}

impl Environment for AlfWorldEnv {
    async fn reset(&mut self, task_id: Option<&str>) -> Result<EnvObservation> {
        self.current_step = 0;
        self.done = false;

        let body = serde_json::json!({ "task_id": task_id });
        let resp: ServerResponse = self
            .http
            .post(format!("{}/reset", self.base_url))
            .json(&body)
            .send()
            .await
            .context("failed to reach ALFWorld server on reset")?
            .json()
            .await
            .context("failed to parse ALFWorld reset response")?;

        self.current_task_description = resp
            .task_description
            .unwrap_or_else(|| resp.observation.clone());
        self.current_task_category = resp.task_category.unwrap_or_default();

        tracing::debug!(
            task = %self.current_task_description,
            category = %self.current_task_category,
            "ALFWorld env reset"
        );

        Ok(EnvObservation {
            text: resp.observation,
            available_actions: resp.available_actions,
            done: false,
            reward: 0.0,
            info: resp.info.unwrap_or(serde_json::Value::Null),
        })
    }

    async fn step(&mut self, action: &str) -> Result<EnvObservation> {
        if self.done {
            anyhow::bail!("cannot step in a terminated episode");
        }

        self.current_step += 1;

        let body = serde_json::json!({ "action": action });
        let resp: ServerResponse = self
            .http
            .post(format!("{}/step", self.base_url))
            .json(&body)
            .send()
            .await
            .context("failed to reach ALFWorld server on step")?
            .json()
            .await
            .context("failed to parse ALFWorld step response")?;

        let truncated = self.current_step >= self.max_steps;
        let episode_done = resp.done || truncated;
        self.done = episode_done;

        Ok(EnvObservation {
            text: resp.observation,
            available_actions: resp.available_actions,
            done: episode_done,
            reward: resp.reward,
            info: resp.info.unwrap_or(serde_json::Value::Null),
        })
    }

    fn task_description(&self) -> &str {
        &self.current_task_description
    }

    fn task_category(&self) -> &str {
        &self.current_task_category
    }

    fn max_steps(&self) -> usize {
        self.max_steps
    }

    fn is_done(&self) -> bool {
        self.done
    }
}

// ---------------------------------------------------------------------------
// Mock ALFWorld environment for testing
// ---------------------------------------------------------------------------

/// A scripted mock of ALFWorld that replays predefined episodes.
///
/// Useful for deterministic tests of the trajectory collector, GRPO training
/// loop, and skill evolution pipeline without requiring a running ALFWorld
/// server.
#[derive(Debug, Clone)]
pub struct MockAlfWorldEnv {
    /// Pool of scripted episodes that the mock cycles through.
    episodes: Vec<MockEpisode>,
    /// Index of the current episode in `episodes`.
    episode_index: usize,
    /// Index of the next step within the current episode.
    step_index: usize,
    max_steps: usize,
    done: bool,
}

/// A single canned episode.
#[derive(Debug, Clone)]
struct MockEpisode {
    task_description: String,
    task_category: String,
    /// Each entry is (expected_observation, reward, done).
    /// The first entry is the initial observation (returned by `reset`).
    steps: Vec<MockStep>,
}

#[derive(Debug, Clone)]
struct MockStep {
    observation: String,
    reward: f64,
    done: bool,
}

impl MockAlfWorldEnv {
    /// Create a mock environment pre-loaded with realistic multi-step episodes
    /// covering the major ALFWorld task categories.
    pub fn new() -> Self {
        Self {
            episodes: Self::default_episodes(),
            episode_index: 0,
            step_index: 0,
            max_steps: 30,
            done: false,
        }
    }

    /// Create a mock with a custom step limit.
    pub fn with_max_steps(max_steps: usize) -> Self {
        Self {
            episodes: Self::default_episodes(),
            episode_index: 0,
            step_index: 0,
            max_steps,
            done: false,
        }
    }

    // -- built-in episodes --------------------------------------------------

    fn default_episodes() -> Vec<MockEpisode> {
        vec![
            // 1. Pick task (success)
            MockEpisode {
                task_description: "Put a clean apple in the fridge.".into(),
                task_category: "Pick".into(),
                steps: vec![
                    MockStep {
                        observation: "You are in the middle of a room. Looking quickly around you, you see a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 2, a countertop 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You arrive at countertop 1. On the countertop 1, you see an apple 1, a bread 1, and a knife 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You pick up the apple 1 from the countertop 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You arrive at sinkbasin 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You clean the apple 1 using the sinkbasin 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You arrive at fridge 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You open the fridge 1. The fridge 1 is open. In it, you see a lettuce 1 and a potato 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You put the apple 1 in/on the fridge 1.".into(),
                        reward: 1.0,
                        done: true,
                    },
                ],
            },
            // 2. Clean task (success)
            MockEpisode {
                task_description: "Clean a mug and put it on the shelf.".into(),
                task_category: "Clean".into(),
                steps: vec![
                    MockStep {
                        observation: "You are in the middle of a room. Looking quickly around you, you see a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a cabinet 2, a cabinet 1, a countertop 1, and a garbagecan 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You arrive at countertop 1. On the countertop 1, you see a mug 1 and a plate 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You pick up the mug 1 from the countertop 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You arrive at sinkbasin 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You clean the mug 1 using the sinkbasin 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You arrive at shelf 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You put the mug 1 in/on the shelf 1.".into(),
                        reward: 1.0,
                        done: true,
                    },
                ],
            },
            // 3. Heat task (success)
            MockEpisode {
                task_description: "Heat a potato and put it on the countertop.".into(),
                task_category: "Heat".into(),
                steps: vec![
                    MockStep {
                        observation: "You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 2, a countertop 1, a fridge 1, a microwave 1, a sinkbasin 1, a stoveburner 2, and a stoveburner 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You open the fridge 1. The fridge 1 is open. In it, you see a potato 1 and a tomato 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You pick up the potato 1 from the fridge 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You arrive at microwave 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You heat the potato 1 using the microwave 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You arrive at countertop 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You put the potato 1 in/on the countertop 1.".into(),
                        reward: 1.0,
                        done: true,
                    },
                ],
            },
            // 4. Cool task (success)
            MockEpisode {
                task_description: "Cool an apple and put it on the countertop.".into(),
                task_category: "Cool".into(),
                steps: vec![
                    MockStep {
                        observation: "You are in the middle of a room. Looking quickly around you, you see a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a fridge 1, a garbagecan 1, and a stoveburner 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You arrive at countertop 1. On the countertop 1, you see an apple 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You pick up the apple 1 from the countertop 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You arrive at fridge 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You cool the apple 1 with the fridge 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You arrive at countertop 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You put the apple 1 in/on the countertop 1.".into(),
                        reward: 1.0,
                        done: true,
                    },
                ],
            },
            // 5. Look task (failure -- agent runs out of useful steps)
            MockEpisode {
                task_description: "Examine the book under the desk lamp.".into(),
                task_category: "Look".into(),
                steps: vec![
                    MockStep {
                        observation: "You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a desklamp 1, a drawer 2, a drawer 1, a garbagecan 1, and a shelf 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You arrive at shelf 1. On the shelf 1, you see a pen 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You open the drawer 1. The drawer 1 is open. In it, you see a pencil 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You open the drawer 2. The drawer 2 is open. In it, you see a book 1 and a cd 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You pick up the book 1 from the drawer 2.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You arrive at desklamp 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You turn on the desklamp 1. You examine the book 1 under the desklamp 1.".into(),
                        reward: 1.0,
                        done: true,
                    },
                ],
            },
            // 6. PickTwo task (failure -- agent only picks one)
            MockEpisode {
                task_description: "Put two pencils on the desk.".into(),
                task_category: "PickTwo".into(),
                steps: vec![
                    MockStep {
                        observation: "You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, and a shelf 2, a shelf 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You open the drawer 1. The drawer 1 is open. In it, you see a pencil 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You pick up the pencil 1 from the drawer 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You arrive at desk 1. On the desk 1, you see a laptop 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "You put the pencil 1 in/on the desk 1.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "Nothing happens.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "Nothing happens.".into(),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "Nothing happens.".into(),
                        reward: 0.0,
                        done: true, // truncated
                    },
                ],
            },
        ]
    }

    fn current_episode(&self) -> &MockEpisode {
        &self.episodes[self.episode_index % self.episodes.len()]
    }
}

impl Default for MockAlfWorldEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for MockAlfWorldEnv {
    async fn reset(&mut self, _task_id: Option<&str>) -> Result<EnvObservation> {
        self.step_index = 0;
        self.done = false;

        let episode = self.current_episode();
        let first = &episode.steps[0];

        Ok(EnvObservation {
            text: first.observation.clone(),
            available_actions: None,
            done: false,
            reward: 0.0,
            info: serde_json::json!({
                "task_category": episode.task_category,
                "episode_index": self.episode_index,
            }),
        })
    }

    async fn step(&mut self, _action: &str) -> Result<EnvObservation> {
        if self.done {
            anyhow::bail!("cannot step in a terminated episode");
        }

        self.step_index += 1;
        let ep_idx = self.episode_index % self.episodes.len();
        let num_steps = self.episodes[ep_idx].steps.len();

        // If we've exhausted the scripted steps, return a terminal "Nothing
        // happens." observation.
        if self.step_index >= num_steps {
            self.done = true;
            return Ok(EnvObservation {
                text: "Nothing happens.".into(),
                available_actions: None,
                done: true,
                reward: 0.0,
                info: serde_json::Value::Null,
            });
        }

        // Extract fields we need before mutating self.
        let observation = self.episodes[ep_idx].steps[self.step_index].observation.clone();
        let reward = self.episodes[ep_idx].steps[self.step_index].reward;
        let step_done = self.episodes[ep_idx].steps[self.step_index].done;

        let truncated = self.step_index >= self.max_steps;
        let episode_done = step_done || truncated;
        self.done = episode_done;

        // When the episode ends, advance to the next canned episode for the
        // subsequent call to `reset`.
        if episode_done {
            self.episode_index += 1;
        }

        Ok(EnvObservation {
            text: observation,
            available_actions: None,
            done: episode_done,
            reward,
            info: serde_json::Value::Null,
        })
    }

    fn task_description(&self) -> &str {
        &self.current_episode().task_description
    }

    fn task_category(&self) -> &str {
        &self.current_episode().task_category
    }

    fn max_steps(&self) -> usize {
        self.max_steps
    }

    fn is_done(&self) -> bool {
        self.done
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_alfworld_runs_full_episode() {
        let mut env = MockAlfWorldEnv::new();
        let obs = env.reset(None).await.unwrap();
        assert!(!obs.done);
        assert!(obs.text.contains("room"));

        let mut total_reward = 0.0;
        let mut steps = 0;
        while !env.is_done() {
            let obs = env.step("go to countertop 1").await.unwrap();
            total_reward += obs.reward;
            steps += 1;
            if obs.done {
                break;
            }
        }
        assert!(steps > 0);
        // First episode is the "Pick" one which succeeds with reward 1.0.
        assert!((total_reward - 1.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn mock_alfworld_cycles_episodes() {
        let mut env = MockAlfWorldEnv::new();

        // Run through two full episodes.
        for expected_cat in &["Pick", "Clean"] {
            let _obs = env.reset(None).await.unwrap();
            assert_eq!(env.task_category(), *expected_cat);
            while !env.is_done() {
                let obs = env.step("noop").await.unwrap();
                if obs.done {
                    break;
                }
            }
        }
    }
}
