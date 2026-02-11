//! WebShop e-commerce browsing environment.
//!
//! WebShop presents the agent with product-search tasks on a simulated
//! e-commerce site. The action space is:
//!   - `search[query]`    -- issue a search
//!   - `click[element]`   -- click on a page element (product link, option, etc.)
//!   - `buy_now`          -- purchase the currently viewed item
//!
//! This module provides two implementations:
//! - [`WebShopEnv`]     connects to a running WebShop server via HTTP.
//! - [`MockWebShopEnv`] returns canned observation sequences for testing.

use anyhow::{Context, Result};
use serde::Deserialize;
use tracing;

use super::traits::{EnvObservation, Environment};

// ---------------------------------------------------------------------------
// HTTP-backed WebShop environment
// ---------------------------------------------------------------------------

/// A WebShop environment that communicates with an external server via HTTP.
///
/// Expected endpoints:
/// - `POST {base_url}/reset`  -- body: `{"task_id": <optional string>}`
/// - `POST {base_url}/step`   -- body: `{"action": "<action string>"}`
///
/// Both return JSON matching [`ServerResponse`].
#[derive(Debug)]
pub struct WebShopEnv {
    base_url: String,
    http: reqwest::Client,
    current_task_description: String,
    current_task_category: String,
    max_steps: usize,
    current_step: usize,
    done: bool,
}

/// JSON response from the WebShop server.
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

impl WebShopEnv {
    /// Create a new WebShop environment backed by the given server URL.
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

impl Environment for WebShopEnv {
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
            .context("failed to reach WebShop server on reset")?
            .json()
            .await
            .context("failed to parse WebShop reset response")?;

        self.current_task_description = resp
            .task_description
            .unwrap_or_else(|| resp.observation.clone());
        self.current_task_category = resp.task_category.unwrap_or_else(|| "search".into());

        tracing::debug!(
            task = %self.current_task_description,
            "WebShop env reset"
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
            .context("failed to reach WebShop server on step")?
            .json()
            .await
            .context("failed to parse WebShop step response")?;

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
// Mock WebShop environment for testing
// ---------------------------------------------------------------------------

/// A scripted mock of WebShop that replays canned episodes.
#[derive(Debug, Clone)]
pub struct MockWebShopEnv {
    episodes: Vec<MockEpisode>,
    episode_index: usize,
    step_index: usize,
    max_steps: usize,
    done: bool,
}

#[derive(Debug, Clone)]
struct MockEpisode {
    task_description: String,
    task_category: String,
    steps: Vec<MockStep>,
}

#[derive(Debug, Clone)]
struct MockStep {
    observation: String,
    available_actions: Option<Vec<String>>,
    reward: f64,
    done: bool,
}

impl MockWebShopEnv {
    /// Create a mock environment with realistic multi-step shopping episodes.
    pub fn new() -> Self {
        Self {
            episodes: Self::default_episodes(),
            episode_index: 0,
            step_index: 0,
            max_steps: 15,
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

    fn default_episodes() -> Vec<MockEpisode> {
        vec![
            // Episode 1: Successful product purchase
            MockEpisode {
                task_description: "I need a pair of noise-cancelling wireless headphones, black color, under $80.".into(),
                task_category: "search".into(),
                steps: vec![
                    // Initial observation (search page)
                    MockStep {
                        observation: "WebShop\n[Search bar] Search for a product\n[All Departments]".into(),
                        available_actions: Some(vec![
                            "search[noise cancelling wireless headphones black]".into(),
                        ]),
                        reward: 0.0,
                        done: false,
                    },
                    // After search
                    MockStep {
                        observation: "Search results for 'noise cancelling wireless headphones black':\n\n[Product 1] SoundMax Pro ANC Headphones - Black - $69.99\n★★★★☆ (2,341 ratings)\n\n[Product 2] AudioTech BT-500 Wireless Headphones - Black - $45.99\n★★★☆☆ (892 ratings)\n\n[Product 3] BassKing Noise Cancelling Over-Ear - Black/Silver - $89.99\n★★★★★ (567 ratings)".into(),
                        available_actions: Some(vec![
                            "click[Product 1]".into(),
                            "click[Product 2]".into(),
                            "click[Product 3]".into(),
                            "click[Next Page]".into(),
                        ]),
                        reward: 0.0,
                        done: false,
                    },
                    // After clicking Product 1
                    MockStep {
                        observation: "SoundMax Pro ANC Headphones\nPrice: $69.99\nColor: Black\nFeatures: Active Noise Cancellation, Bluetooth 5.2, 40-hour battery life, foldable design\nRating: ★★★★☆ (2,341 ratings)\n\n[Color: Black] [Color: White] [Color: Blue]\n[Buy Now] [Back to Search]".into(),
                        available_actions: Some(vec![
                            "click[Color: Black]".into(),
                            "click[Color: White]".into(),
                            "click[Color: Blue]".into(),
                            "buy_now".into(),
                            "click[Back to Search]".into(),
                        ]),
                        reward: 0.0,
                        done: false,
                    },
                    // After selecting Black color option
                    MockStep {
                        observation: "Selected: Color - Black\nSoundMax Pro ANC Headphones - Black\nPrice: $69.99\n\n[Buy Now] [Back to Search]".into(),
                        available_actions: Some(vec![
                            "buy_now".into(),
                            "click[Back to Search]".into(),
                        ]),
                        reward: 0.0,
                        done: false,
                    },
                    // After buy_now
                    MockStep {
                        observation: "Purchase successful! You bought: SoundMax Pro ANC Headphones - Black for $69.99.".into(),
                        available_actions: None,
                        reward: 1.0,
                        done: true,
                    },
                ],
            },
            // Episode 2: Partial match (wrong attribute selected)
            MockEpisode {
                task_description: "I want a stainless steel water bottle, 32oz, with a straw lid, in blue.".into(),
                task_category: "search".into(),
                steps: vec![
                    MockStep {
                        observation: "WebShop\n[Search bar] Search for a product\n[All Departments]".into(),
                        available_actions: Some(vec![
                            "search[stainless steel water bottle 32oz straw lid blue]".into(),
                        ]),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "Search results for 'stainless steel water bottle 32oz straw lid blue':\n\n[Product 1] HydroFlow 32oz Stainless Steel Bottle - Multiple Colors - $24.99\n★★★★★ (4,102 ratings)\n\n[Product 2] AquaPure Insulated Bottle 32oz - Blue - $19.99\n★★★★☆ (1,203 ratings)".into(),
                        available_actions: Some(vec![
                            "click[Product 1]".into(),
                            "click[Product 2]".into(),
                        ]),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "HydroFlow 32oz Stainless Steel Bottle\nPrice: $24.99\nMaterial: 18/8 Stainless Steel\nCapacity: 32oz\nLid options: Straw Lid, Flip Lid, Sport Cap\nColors: Blue, Red, Green, Black\nRating: ★★★★★ (4,102 ratings)\n\n[Color: Blue] [Color: Red] [Color: Green] [Color: Black]\n[Lid: Straw Lid] [Lid: Flip Lid] [Lid: Sport Cap]\n[Buy Now] [Back to Search]".into(),
                        available_actions: Some(vec![
                            "click[Color: Blue]".into(),
                            "click[Color: Red]".into(),
                            "click[Lid: Straw Lid]".into(),
                            "click[Lid: Flip Lid]".into(),
                            "buy_now".into(),
                        ]),
                        reward: 0.0,
                        done: false,
                    },
                    // Agent selects blue
                    MockStep {
                        observation: "Selected: Color - Blue\nHydroFlow 32oz Stainless Steel Bottle - Blue\nPrice: $24.99\n\n[Lid: Straw Lid] [Lid: Flip Lid] [Lid: Sport Cap]\n[Buy Now] [Back to Search]".into(),
                        available_actions: Some(vec![
                            "click[Lid: Straw Lid]".into(),
                            "click[Lid: Flip Lid]".into(),
                            "buy_now".into(),
                        ]),
                        reward: 0.0,
                        done: false,
                    },
                    // Agent buys without selecting straw lid
                    MockStep {
                        observation: "Purchase completed. You bought: HydroFlow 32oz Stainless Steel Bottle - Blue (Default Lid) for $24.99. Note: Straw lid was not selected.".into(),
                        available_actions: None,
                        reward: 0.5,
                        done: true,
                    },
                ],
            },
            // Episode 3: Failed purchase (too expensive)
            MockEpisode {
                task_description: "Find a lightweight running jacket, size medium, waterproof, under $40.".into(),
                task_category: "search".into(),
                steps: vec![
                    MockStep {
                        observation: "WebShop\n[Search bar] Search for a product\n[All Departments]".into(),
                        available_actions: Some(vec![
                            "search[lightweight running jacket medium waterproof]".into(),
                        ]),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "Search results for 'lightweight running jacket medium waterproof':\n\n[Product 1] RunElite Waterproof Jacket - $59.99\n★★★★☆ (723 ratings)\n\n[Product 2] SprintShield Rain Jacket - $49.99\n★★★☆☆ (412 ratings)\n\n[Product 3] BudgetRun Light Jacket - $34.99\n★★☆☆☆ (89 ratings)".into(),
                        available_actions: Some(vec![
                            "click[Product 1]".into(),
                            "click[Product 2]".into(),
                            "click[Product 3]".into(),
                        ]),
                        reward: 0.0,
                        done: false,
                    },
                    // Agent picks the expensive one
                    MockStep {
                        observation: "RunElite Waterproof Jacket\nPrice: $59.99\nSizes: S, M, L, XL\nFeatures: Waterproof, breathable, reflective strips\n\n[Size: S] [Size: M] [Size: L] [Size: XL]\n[Buy Now] [Back to Search]".into(),
                        available_actions: Some(vec![
                            "click[Size: M]".into(),
                            "buy_now".into(),
                            "click[Back to Search]".into(),
                        ]),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "Selected: Size - M\nRunElite Waterproof Jacket - Medium\nPrice: $59.99\n\n[Buy Now] [Back to Search]".into(),
                        available_actions: Some(vec![
                            "buy_now".into(),
                            "click[Back to Search]".into(),
                        ]),
                        reward: 0.0,
                        done: false,
                    },
                    MockStep {
                        observation: "Purchase completed. You bought: RunElite Waterproof Jacket - M for $59.99. Budget exceeded ($40 limit).".into(),
                        available_actions: None,
                        reward: 0.2,
                        done: true,
                    },
                ],
            },
        ]
    }

    fn current_episode(&self) -> &MockEpisode {
        &self.episodes[self.episode_index % self.episodes.len()]
    }
}

impl Default for MockWebShopEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for MockWebShopEnv {
    async fn reset(&mut self, _task_id: Option<&str>) -> Result<EnvObservation> {
        self.step_index = 0;
        self.done = false;

        let episode = self.current_episode();
        let first = &episode.steps[0];

        Ok(EnvObservation {
            text: first.observation.clone(),
            available_actions: first.available_actions.clone(),
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

        if self.step_index >= num_steps {
            self.done = true;
            return Ok(EnvObservation {
                text: "No more actions available.".into(),
                available_actions: None,
                done: true,
                reward: 0.0,
                info: serde_json::Value::Null,
            });
        }

        // Extract fields before mutating self.
        let observation = self.episodes[ep_idx].steps[self.step_index].observation.clone();
        let available_actions = self.episodes[ep_idx].steps[self.step_index].available_actions.clone();
        let reward = self.episodes[ep_idx].steps[self.step_index].reward;
        let step_done = self.episodes[ep_idx].steps[self.step_index].done;

        let truncated = self.step_index >= self.max_steps;
        let episode_done = step_done || truncated;
        self.done = episode_done;

        if episode_done {
            self.episode_index += 1;
        }

        Ok(EnvObservation {
            text: observation,
            available_actions,
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
    async fn mock_webshop_runs_full_episode() {
        let mut env = MockWebShopEnv::new();
        let obs = env.reset(None).await.unwrap();
        assert!(!obs.done);
        assert!(obs.text.contains("WebShop"));

        let mut total_reward = 0.0;
        let mut steps = 0;
        while !env.is_done() {
            let obs = env.step("search[headphones]").await.unwrap();
            total_reward += obs.reward;
            steps += 1;
            if obs.done {
                break;
            }
        }
        assert!(steps > 0);
        // First episode is a successful purchase with reward 1.0.
        assert!((total_reward - 1.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn mock_webshop_cycles_episodes() {
        let mut env = MockWebShopEnv::new();

        for _ in 0..3 {
            let _obs = env.reset(None).await.unwrap();
            while !env.is_done() {
                let obs = env.step("noop").await.unwrap();
                if obs.done {
                    break;
                }
            }
        }
        // Should have cycled through 3 episodes without error.
    }

    #[tokio::test]
    async fn mock_webshop_partial_reward() {
        let mut env = MockWebShopEnv::new();

        // Skip first episode.
        let _obs = env.reset(None).await.unwrap();
        while !env.is_done() {
            let obs = env.step("noop").await.unwrap();
            if obs.done {
                break;
            }
        }

        // Second episode: partial match (reward 0.5).
        let _obs = env.reset(None).await.unwrap();
        let mut total_reward = 0.0;
        while !env.is_done() {
            let obs = env.step("noop").await.unwrap();
            total_reward += obs.reward;
            if obs.done {
                break;
            }
        }
        assert!((total_reward - 0.5).abs() < f64::EPSILON);
    }
}
