//! Task environment abstractions and concrete implementations.
//!
//! Every environment implements the [`Environment`] trait so that the trajectory
//! collector can interact with it uniformly.
//!
//! Included environments:
//! - **ALFWorld** ([`alfworld`]) -- text-based household tasks (pick, clean,
//!   heat, cool, look, pick-two).
//! - **WebShop** ([`webshop`]) -- e-commerce product-search tasks (search,
//!   click, buy).
//!
//! Each environment module also exposes a `Mock*Env` variant that replays
//! canned episodes, making it possible to test the full SkillRL pipeline
//! without external dependencies.

pub mod alfworld;
pub mod traits;
pub mod webshop;

// Re-export the core trait and observation type at the module level.
pub use traits::{EnvConfig, EnvObservation, Environment};

// ---------------------------------------------------------------------------
// AnyEnv: enum dispatch wrapper for dynamic environment selection
// ---------------------------------------------------------------------------

/// An enum wrapper around all concrete environment types, enabling runtime
/// environment selection without `dyn` (which is incompatible with async trait
/// methods).
pub enum AnyEnv {
    AlfWorld(alfworld::AlfWorldEnv),
    MockAlfWorld(alfworld::MockAlfWorldEnv),
    WebShop(webshop::WebShopEnv),
    MockWebShop(webshop::MockWebShopEnv),
}

impl Environment for AnyEnv {
    async fn reset(&mut self, task_id: Option<&str>) -> anyhow::Result<EnvObservation> {
        match self {
            Self::AlfWorld(e) => e.reset(task_id).await,
            Self::MockAlfWorld(e) => e.reset(task_id).await,
            Self::WebShop(e) => e.reset(task_id).await,
            Self::MockWebShop(e) => e.reset(task_id).await,
        }
    }

    async fn step(&mut self, action: &str) -> anyhow::Result<EnvObservation> {
        match self {
            Self::AlfWorld(e) => e.step(action).await,
            Self::MockAlfWorld(e) => e.step(action).await,
            Self::WebShop(e) => e.step(action).await,
            Self::MockWebShop(e) => e.step(action).await,
        }
    }

    fn task_description(&self) -> &str {
        match self {
            Self::AlfWorld(e) => e.task_description(),
            Self::MockAlfWorld(e) => e.task_description(),
            Self::WebShop(e) => e.task_description(),
            Self::MockWebShop(e) => e.task_description(),
        }
    }

    fn task_category(&self) -> &str {
        match self {
            Self::AlfWorld(e) => e.task_category(),
            Self::MockAlfWorld(e) => e.task_category(),
            Self::WebShop(e) => e.task_category(),
            Self::MockWebShop(e) => e.task_category(),
        }
    }

    fn max_steps(&self) -> usize {
        match self {
            Self::AlfWorld(e) => e.max_steps(),
            Self::MockAlfWorld(e) => e.max_steps(),
            Self::WebShop(e) => e.max_steps(),
            Self::MockWebShop(e) => e.max_steps(),
        }
    }

    fn is_done(&self) -> bool {
        match self {
            Self::AlfWorld(e) => e.is_done(),
            Self::MockAlfWorld(e) => e.is_done(),
            Self::WebShop(e) => e.is_done(),
            Self::MockWebShop(e) => e.is_done(),
        }
    }
}
