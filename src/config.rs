use serde::{Deserialize, Serialize};

/// Complete configuration for the SkillRL training pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillRLConfig {
    pub sft: SftConfig,
    pub rl: RlConfig,
    pub skill_retrieval: SkillRetrievalConfig,
    pub evolution: EvolutionConfig,
    pub model: ModelConfig,
}

/// Cold-start supervised fine-tuning configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftConfig {
    /// Learning rate for SFT (default: 1e-4).
    pub learning_rate: f64,
    /// Batch size for SFT (default: 16).
    pub batch_size: usize,
    /// Number of training epochs (default: 3).
    pub epochs: usize,
}

/// RL training configuration (GRPO).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlConfig {
    /// Learning rate for RL (default: 1e-6).
    pub learning_rate: f64,
    /// Batch size for RL (default: 64).
    pub batch_size: usize,
    /// Group size G for GRPO advantage estimation (default: 8).
    pub group_size: usize,
    /// KL divergence coefficient beta (default: 0.01).
    pub kl_coeff: f64,
    /// PPO clipping epsilon (default: 0.2).
    pub clip_epsilon: f64,
    /// Penalty for invalid actions (default: 0.1).
    pub invalid_action_penalty: f64,
    /// Maximum prompt length in tokens (default: 6000).
    pub max_prompt_length: usize,
    /// Maximum response length in tokens (default: 1024).
    pub max_response_length: usize,
    /// Total training epochs (default: 150).
    pub training_epochs: usize,
}

/// Skill retrieval configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillRetrievalConfig {
    /// Number of top skills to retrieve (default: 6).
    pub top_k: usize,
    /// Minimum similarity threshold (default: 0.4).
    pub similarity_threshold: f64,
}

/// Recursive evolution configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    /// Validation interval in training steps (default: 5).
    pub validation_interval: usize,
    /// Maximum new skills per evolution cycle (default: 3).
    pub max_new_skills: usize,
    /// Success rate threshold below which evolution triggers (default: 0.4).
    pub evolution_threshold: f64,
    /// Max trajectories to analyze when success << threshold.
    pub max_analysis_deep: usize,
    /// Max trajectories to analyze when success is near threshold.
    pub max_analysis_shallow: usize,
}

/// Model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Base URL for the policy model API.
    pub policy_api_base: String,
    /// Model identifier for the policy (e.g., "Qwen/Qwen2.5-7B-Instruct").
    pub policy_model_id: String,
    /// Base URL for the teacher model API.
    pub teacher_api_base: String,
    /// Model identifier for the teacher (e.g., "o3").
    pub teacher_model_id: String,
    /// API key for the policy model.
    pub policy_api_key: String,
    /// API key for the teacher model.
    pub teacher_api_key: String,
    /// Base URL for the embedding model API.
    pub embedding_api_base: String,
    /// Model identifier for embeddings.
    pub embedding_model_id: String,
    /// API key for the embedding model.
    pub embedding_api_key: String,
}

impl Default for SkillRLConfig {
    fn default() -> Self {
        Self {
            sft: SftConfig {
                learning_rate: 1e-4,
                batch_size: 16,
                epochs: 3,
            },
            rl: RlConfig {
                learning_rate: 1e-6,
                batch_size: 64,
                group_size: 8,
                kl_coeff: 0.01,
                clip_epsilon: 0.2,
                invalid_action_penalty: 0.1,
                max_prompt_length: 6000,
                max_response_length: 1024,
                training_epochs: 150,
            },
            skill_retrieval: SkillRetrievalConfig {
                top_k: 6,
                similarity_threshold: 0.4,
            },
            evolution: EvolutionConfig {
                validation_interval: 5,
                max_new_skills: 3,
                evolution_threshold: 0.4,
                max_analysis_deep: 10,
                max_analysis_shallow: 5,
            },
            model: ModelConfig {
                policy_api_base: "http://localhost:8000/v1".into(),
                policy_model_id: "Qwen/Qwen2.5-7B-Instruct".into(),
                teacher_api_base: "https://api.openai.com/v1".into(),
                teacher_model_id: "o3".into(),
                policy_api_key: String::new(),
                teacher_api_key: String::new(),
                embedding_api_base: "https://api.openai.com/v1".into(),
                embedding_model_id: "text-embedding-3-small".into(),
                embedding_api_key: String::new(),
            },
        }
    }
}
