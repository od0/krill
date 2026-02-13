# Krill

Rust implementation of **[SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning](https://arxiv.org/abs/2602.08234)**.

Krill bridges raw experience and policy improvement through automatic skill discovery and recursive evolution. It trains LLM-based agents by:

1. Collecting interaction trajectories in task environments
2. Distilling reusable skills from successes and failures via a teacher model
3. Cold-start fine-tuning with skill-augmented demonstrations
4. RL training (GRPO) with recursive skill evolution

## Quick Start

Distill a real skill bank from ALFWorld trajectories using OpenAI:

```bash
# Build
cargo build --release

# Download pre-collected ALFWorld trajectories (3,119 episodes)
curl -L -o data/alfworld_raw.json \
  "https://huggingface.co/datasets/agent-eto/eto-sft-trajectory/resolve/main/data/alfworld_sft.json"

# Convert to SkillRL format (100 trajectories)
python3 scripts/convert_eto_trajectories.py \
  --input data/alfworld_raw.json \
  --output data/alfworld_trajectories.json \
  --max 100

# Distill skills using gpt-4o as teacher
export OPENAI_API_KEY="sk-..."
cargo run -- distill \
  --trajectories data/alfworld_trajectories.json \
  --output data/skill_bank.json \
  --config config.demo.json

# Inspect the result
cargo run -- inspect data/skill_bank.json
```

This produces a skill bank with ~470 skills (general + task-specific) in about 9 minutes. See [docs/demo-run.md](docs/demo-run.md) for full details and sample output.

## Architecture

```
krill/
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
├── scripts/
│   └── convert_eto_trajectories.py  # Convert HuggingFace ETO data to SkillRL format
│
├── docs/
│   ├── walkthrough.md       # Conceptual tutorial on the 4-phase pipeline
│   ├── demo-run.md          # Full worked example with real data
│   └── empathic-relevance.md
│
├── config.demo.json         # Demo config (OpenAI models)
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
krill train --env alfworld --mock
```

### Individual phases

```bash
# Phase 1: Collect trajectories
krill collect --episodes 64 --output data/trajectories.json

# Phase 2: Distill skills from trajectories
krill distill --trajectories data/trajectories.json --output data/skill_bank.json

# Phase 3: Cold-start supervised fine-tuning
krill sft --skill-bank data/skill_bank.json --num-examples 7500

# Phase 4: RL training with recursive skill evolution
krill rl --skill-bank data/skill_bank.json
```

### Inspect a skill bank

```bash
krill inspect data/skill_bank.json
```

### Global options

| Flag | Default | Description |
|------|---------|-------------|
| `--config <path>` | *(none)* | Path to a JSON configuration file |
| `--env <alfworld\|webshop>` | `alfworld` | Which environment to use |
| `--mock` | `true` | Use mock environments (no live server needed) |

## Configuration

Pass a JSON file via `--config` to override defaults. All fields are optional — unspecified values use paper defaults. API keys can be set in the config file or via environment variable:

```bash
export OPENAI_API_KEY="sk-..."  # fills in any empty *_api_key fields
```

The included `config.demo.json` uses OpenAI models throughout (this is what the Quick Start uses):

```json
{
  "model": {
    "policy_api_base": "https://api.openai.com/v1",
    "policy_model_id": "gpt-4o-mini",
    "teacher_api_base": "https://api.openai.com/v1",
    "teacher_model_id": "gpt-4o",
    "embedding_api_base": "https://api.openai.com/v1",
    "embedding_model_id": "text-embedding-3-small"
  }
}
```

The paper uses a local Qwen2.5-7B-Instruct as the policy model (served via vLLM at `localhost:8000`) and o3 as the teacher. Any OpenAI-compatible API will work — swap in your preferred models and endpoints.

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

## Documentation

- **[Walkthrough](docs/walkthrough.md)** — Conceptual tutorial explaining the 4-phase pipeline, how skills work, and the key insight
- **[Demo Run](docs/demo-run.md)** — Full worked example: downloading real ALFWorld data, distilling 474 skills with gpt-4o, and inspecting results

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
