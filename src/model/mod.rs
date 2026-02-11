//! Model client abstractions for interacting with LLM and embedding APIs.
//!
//! This module provides:
//! - [`api::LlmClient`] -- OpenAI-compatible chat completion client with
//!   log-probability support (needed for GRPO importance ratios).
//! - [`embedding::EmbeddingClient`] -- OpenAI-compatible embedding client
//!   for computing skill and task description vectors.
//! - [`prompt`] -- All prompt templates used throughout the SkillRL pipeline
//!   (skill-augmented action, distillation, evolution, cold-start).

pub mod api;
pub mod embedding;
pub mod prompt;

// Re-export the most commonly used types at the module level so that
// existing code that does `use crate::model::LlmClient` continues to work.
pub use api::{
    ChatMessage, ChatResponse, ChatResponseWithLogProbs, Choice, LlmClient, Usage,
};
pub use embedding::EmbeddingClient;
