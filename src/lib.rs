//! Krill: Recursive Skill-Augmented RL for LLM Agents
//!
//! Rust implementation of the SkillRL framework which bridges raw experience and policy
//! improvement through automatic skill discovery and recursive evolution.

pub mod agent;
pub mod config;
pub mod env;
pub mod model;
pub mod skill;
pub mod trajectory;
pub mod training;
