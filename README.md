# SkillRL

Rust implementation of **[SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning](https://arxiv.org/abs/2602.08234)**.

SkillRL bridges raw experience and policy improvement through automatic skill discovery and recursive evolution. It trains LLM-based agents by:

1. Collecting interaction trajectories in task environments
2. Distilling reusable skills from successes and failures via a teacher model
3. Cold-start fine-tuning with skill-augmented demonstrations
4. RL training (GRPO) with recursive skill evolution

## Architecture

```
skillrl/
├── src/
│   ├── main.rs              # CLI entrypoint (train, collect, distill, sft, rl, inspect)
│   ├── lib.rs               # Library root
│   ├── config.rs            # All hyperparameters from the paper
│   │
│   ├── agent/               # Skill-augmented agent
│   │   └── agent.rs         #   SkillAgent: act(), act_with_logprob(), action parsing
│   │
│   ├── env/                 # Task environments
│   │   ├── traits.rs        #   Environment trait (reset, step, observations)
│   │   ├── alfworld.rs      #   ALFWorld: household tasks (live HTTP + mock)
│   │   └── webshop.rs       #   WebShop: e-commerce tasks (live HTTP + mock)
│   │
│   ├── model/               # LLM and embedding API clients
│   │   ├── api.rs           #   LlmClient: chat completions, logprobs
│   │   ├── embedding.rs     #   EmbeddingClient: embed, embed_batch
│   │   └── prompt.rs        #   All prompt templates (action, distillation, evolution, SFT)
│   │
│   ├── skill/               # Hierarchical skill system
│   │   ├── types.rs         #   Skill struct, SkillCategory (General / TaskSpecific)
│   │   ├── library.rs       #   SkillBank: two-tier storage, save/load, evolution history
│   │   ├── retrieval.rs     #   SkillRetriever: TopK cosine similarity retrieval
│   │   ├── distillation.rs  #   SkillDistiller: teacher-driven skill extraction
│   │   └── evolution.rs     #   SkillEvolver: recursive skill evolution
│   │
│   ├── trajectory/          # Trajectory collection and storage
│   │   ├── types.rs         #   Step, Trajectory, TrajectoryBatch, TrajectoryBuffer
│   │   └── collector.rs     #   TrajectoryCollector: episode rollout with skill augmentation
│   │
│   └── training/            # Training algorithms
│       ├── advantage.rs     #   GRPO group-relative advantage estimation
│       ├── grpo.rs          #   GrpoTrainer: clipped surrogate + KL objective
│       ├── sft.rs           #   SftTrainer: cold-start supervised fine-tuning
│       └── pipeline.rs      #   TrainingPipeline: full Algorithm 1 orchestration
│
├── Cargo.toml
└── README.md
```

~8,200 lines of Rust across 27 source files, with 91 unit tests.

## Building

```bash
cargo build --release
```

## Usage

### Full training pipeline

Runs all 4 phases end-to-end:

```bash
skillrl train --env alfworld --mock
```

### Individual phases

```bash
# Phase 1: Collect trajectories
skillrl collect --episodes 64 --output data/trajectories.json

# Phase 2: Distill skills from trajectories
skillrl distill --trajectories data/trajectories.json --output data/skill_bank.json

# Phase 3: Cold-start supervised fine-tuning
skillrl sft --skill-bank data/skill_bank.json --num-examples 7500

# Phase 4: RL training with recursive skill evolution
skillrl rl --skill-bank data/skill_bank.json
```

### Inspect a skill bank

```bash
skillrl inspect data/skill_bank.json
```

### Global options

| Flag | Default | Description |
|------|---------|-------------|
| `--config <path>` | *(none)* | Path to a JSON configuration file |
| `--env <alfworld\|webshop>` | `alfworld` | Which environment to use |
| `--mock` | `true` | Use mock environments (no live server needed) |

## Configuration

Pass a JSON file via `--config` to override defaults. All fields are optional — unspecified values use paper defaults.

```json
{
  "sft": {
    "learning_rate": 1e-4,
    "batch_size": 16,
    "epochs": 3
  },
  "rl": {
    "learning_rate": 1e-6,
    "batch_size": 64,
    "group_size": 8,
    "kl_coeff": 0.01,
    "clip_epsilon": 0.2,
    "invalid_action_penalty": 0.1,
    "max_prompt_length": 6000,
    "max_response_length": 1024,
    "training_epochs": 150
  },
  "skill_retrieval": {
    "top_k": 6,
    "similarity_threshold": 0.4
  },
  "evolution": {
    "validation_interval": 5,
    "max_new_skills": 3,
    "evolution_threshold": 0.4,
    "max_analysis_deep": 10,
    "max_analysis_shallow": 5
  },
  "model": {
    "policy_api_base": "http://localhost:8000/v1",
    "policy_model_id": "Qwen/Qwen2.5-7B-Instruct",
    "teacher_api_base": "https://api.openai.com/v1",
    "teacher_model_id": "o3",
    "policy_api_key": "",
    "teacher_api_key": "",
    "embedding_api_base": "https://api.openai.com/v1",
    "embedding_model_id": "text-embedding-3-small",
    "embedding_api_key": ""
  }
}
```

## Key algorithms

### GRPO (Group Relative Policy Optimization)

The RL objective (Equation 9 from the paper):

```
J(θ) = E[ 1/G Σ min(ρ_i · A_i, clip(ρ_i, 1-ε, 1+ε) · A_i) - β · D_KL(π_θ ‖ π_ref) ]
```

Where advantages are computed as group-relative z-scores: `A_i = (R_i - mean(R)) / std(R)`

### Skill retrieval

```
S_ret = TopK({ s ∈ S_k : sim(e_d, e_s) > δ }, K)
```

Cosine similarity over embeddings, with configurable threshold δ and top-K.

### Recursive evolution

When per-category accuracy drops below the evolution threshold:

```
S_new = M_T(T_val⁻, SkillBank)
SkillBank ← SkillBank ∪ S_new
```

The teacher model analyzes failure trajectories in context of the current skill bank and proposes new or refined skills.

## Environments

**ALFWorld** — Text-based household tasks: Pick, Look, Clean, Heat, Cool, PickTwo. The mock environment provides scripted episodes for all 6 task categories.

**WebShop** — E-commerce product search and purchase. The mock provides 3 episodes with varying reward levels (1.0, 0.5, 0.2).

For live environments, run the corresponding servers:
- ALFWorld: `http://localhost:3000`
- WebShop: `http://localhost:3001`

## Testing

```bash
cargo test
```

Runs 91 unit tests covering all modules — advantage estimation, GRPO loss, skill retrieval, environment simulation, trajectory collection, action parsing, prompt construction, and more.

## Paper reference

```bibtex
@article{skillrl2025,
  title={SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning},
  year={2025},
  url={https://arxiv.org/abs/2602.08234}
}
```

## License

[MIT](LICENSE)
