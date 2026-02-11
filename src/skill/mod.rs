//! Skill module for SkillRL.
//!
//! This module implements the full skill lifecycle:
//!
//! 1. **Types** ([`types`]) -- Core data structures for skills and categories.
//! 2. **Library** ([`library`]) -- The hierarchical [`SkillBank`] that stores general and
//!    task-specific skills with evolution history tracking.
//! 3. **Retrieval** ([`retrieval`]) -- Embedding-based top-K skill retrieval using cosine
//!    similarity, implementing `S_ret = TopK({s in S_k : sim(e_d, e_s) > delta}, K)`.
//! 4. **Distillation** ([`distillation`]) -- Teacher-model-driven extraction of strategic
//!    skills from successful (`s+ = M_T(tau+, d)`) and failed (`s- = M_T(tau-, d)`)
//!    trajectories.
//! 5. **Evolution** ([`evolution`]) -- Recursive skill evolution that analyses failure
//!    categories, samples representative trajectories, and generates new skills to fill
//!    gaps: `S_new = M_T(T_val^-, SkillBank)`.

pub mod distillation;
pub mod evolution;
pub mod library;
pub mod retrieval;
pub mod types;

// Re-export the most commonly used items at the module level.
pub use distillation::SkillDistiller;
pub use evolution::SkillEvolver;
pub use library::SkillBank;
pub use retrieval::{cosine_similarity, SkillRetriever};
pub use types::{Skill, SkillCategory};
