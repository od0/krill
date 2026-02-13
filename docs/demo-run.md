# SkillRL Demo Run: End-to-End Skill Distillation

This documents a full run of the SkillRL pipeline on real ALFWorld trajectory data, from data acquisition through skill distillation and inspection.

## Data Source

**Dataset:** [ETO SFT Trajectory](https://huggingface.co/datasets/agent-eto/eto-sft-trajectory)
- Contains 3,119 successful ALFWorld trajectories collected via the ETO (Exploratory Training with Online feedback) method
- Format: conversation-style (human/gpt turns) with thought/action pairs
- File: `alfworld_sft.json` (18MB)

The original SkillRL paper (arXiv:2602.08234) collects its own trajectories using Qwen2.5-7B-Instruct interacting with ALFWorld. We use pre-collected SFT trajectories as a practical alternative that avoids installing the full ALFWorld environment.

### Download

```bash
curl -L -o data/alfworld_raw.json \
  "https://huggingface.co/datasets/agent-eto/eto-sft-trajectory/resolve/main/data/alfworld_sft.json"
```

## Step 1: Convert Trajectories

The raw ETO data is in conversation format. We convert it to SkillRL's trajectory format using a Python script.

```bash
python3 scripts/convert_eto_trajectories.py \
  --input data/alfworld_raw.json \
  --output data/alfworld_trajectories.json \
  --max 100
```

### Conversion Output

```
Converted 100 trajectories:
  Clean: 13
  Cool: 13
  Heat: 13
  Look: 10
  Pick: 32
  PickTwo: 19
  Total steps: 1,096
  Avg steps/trajectory: 11.0
```

The converter:
- Parses human/gpt conversation turns into observation/action steps
- Infers task category from the game file path (e.g., `pick_heat_then_place` → Heat)
- Marks all trajectories as successful (reward 1.0) since these are SFT demonstrations
- Extracts actions from `Action: <action>` patterns in GPT responses

### Trajectory Format

Each converted trajectory looks like:

```json
{
  "id": "067c055a-...",
  "task_description": "put a toiletpaper in toiletpaperhanger.",
  "task_category": "Pick",
  "steps": [
    {
      "observation": "You are in the middle of a room. Looking quickly around you, you see...",
      "action": "go to toilet 1",
      "reward": 0.0,
      "step_index": 0
    },
    ...
    {
      "observation": "You put the toiletpaper 1 in/on the toiletpaperhanger 1.",
      "action": "put toiletpaper 1 in/on toiletpaperhanger 1",
      "reward": 1.0,
      "step_index": 5
    }
  ],
  "total_reward": 1.0,
  "success": true,
  "metadata": {
    "environment": "alfworld",
    "num_steps": 6
  }
}
```

## Step 2: Distill Skills

This uses a teacher model (gpt-4o) to analyze trajectories and extract reusable skills, with text-embedding-3-small for embedding generation.

### Configuration

`config.demo.json`:
```json
{
  "sft": { "learning_rate": 1e-4, "batch_size": 16, "epochs": 3 },
  "rl": { "learning_rate": 1e-6, "batch_size": 64, "group_size": 8, "kl_coeff": 0.01, "clip_epsilon": 0.2, "invalid_action_penalty": 0.1, "max_prompt_length": 6000, "max_response_length": 1024, "training_epochs": 150 },
  "skill_retrieval": { "top_k": 6, "similarity_threshold": 0.4 },
  "evolution": { "validation_interval": 5, "max_new_skills": 3, "evolution_threshold": 0.4, "max_analysis_deep": 10, "max_analysis_shallow": 5 },
  "model": {
    "policy_api_base": "https://api.openai.com/v1",
    "policy_model_id": "gpt-4o-mini",
    "teacher_api_base": "https://api.openai.com/v1",
    "teacher_model_id": "gpt-4o",
    "embedding_api_base": "https://api.openai.com/v1",
    "embedding_model_id": "text-embedding-3-small",
    "policy_api_key": "",
    "teacher_api_key": "",
    "embedding_api_key": ""
  }
}
```

API keys are loaded from the `OPENAI_API_KEY` environment variable when left empty in the config.

### Run Distillation

```bash
export OPENAI_API_KEY="sk-proj-..."

cargo run -- distill \
  --trajectories data/alfworld_trajectories.json \
  --output data/demo_skill_bank.json \
  --config config.demo.json
```

### Distillation Output

The distillation took approximately 9 minutes and processed all 100 trajectories:

```
Distilling skills from data/alfworld_trajectories.json...
  Processing trajectory 1/100 (category: Pick)...
  Processing trajectory 2/100 (category: Pick)...
  ...
  Processing trajectory 100/100 (category: Cool)...
  Generating embeddings for 474 skills...
  Saved 474 skills to data/demo_skill_bank.json
```

**Result: 474 skills distilled from 100 trajectories.**

For each trajectory, the teacher model (gpt-4o) analyzes the observation/action sequence and extracts:
- A **skill name** (e.g., "Microwave Heating Sequence")
- A **principle** (what to do)
- A **when_to_apply** trigger (when this skill is relevant)
- A **category** (General or task-specific like Heat, Clean, etc.)

Each skill then gets a 1536-dimensional embedding from text-embedding-3-small for retrieval.

## Step 3: Inspect the Skill Bank

```bash
cargo run -- inspect data/demo_skill_bank.json
```

### Results

```
Skill Bank: data/demo_skill_bank.json
  Total skills: 474
  General skills: 333
  Evolution cycle: 0

Skills by category:
  Clean:    18
  Cool:     18
  Heat:     21
  Look:     18
  Pick:     42
  PickTwo:  24
  general: 333
```

### Example General Skills

| ID | Name | Principle |
|----|------|-----------|
| `bc77b218` | Identify Target Location | Quickly assess the environment to locate the target object or location needed for task completion. |
| `e84ee8a7` | Optimal Path Planning | Navigate directly to key locations to minimize unnecessary movement and time wastage. |
| `6ec57ad0` | Sequential Exploration | Systematically check each potential location for the target object until it is found. |
| `23d5d189` | Sequential Task Execution | Perform actions in a logical and correct sequence to achieve the desired outcome. |
| `5ea92cb5` | State Verification | Verify the state of an object (e.g., open/closed) before performing an action that depends on that state. |

### Example Task-Specific Skills

**Heat:**
```
[5d352009] Heat Management
  Principle: Safely handle hot objects and determine appropriate places for cooling or storage.
  Apply when: When dealing with tasks that involve hot items that need to be cooled down or stored safely.
```

**Clean:**
```
[88f57798] Clean object
  Principle: Use available resources to clean an object before further use.
  Apply when: When an object is required to be clean for the task or for hygiene purposes.
```

**Cool:**
```
[3d538b70] Utilize Cooling Mechanism
  Principle: Use available cooling appliances to change the temperature of an object.
  Apply when: When an object needs to be cooled as part of a task, particularly when a cooling appliance like a fridge is accessible.
```

**Look:**
```
[20aabb18] Interaction with Objects
  Principle: Interact with objects to uncover hidden items or information that could be crucial for task completion.
  Apply when: Use when encountering objects that can be interacted with, such as opening drawers or using light sources.
```

**Pick:**
```
[8396667d] Task Completion
  Principle: Utilize collected objects at the target location to complete the intended task.
  Apply when: Use when you have the required objects and are at the appropriate location to finalize the task.
```

**PickTwo:**
```
[a28b4db9] Task-Specific Object Placement
  Principle: Place objects at a specified location to complete a task.
  Apply when: Applicable when the task demands placing particular items in a designated area to achieve a goal.
```

## Summary

| Metric | Value |
|--------|-------|
| Raw trajectories available | 3,119 |
| Trajectories converted | 100 |
| Total steps across trajectories | 1,096 |
| Average steps per trajectory | 11.0 |
| Skills distilled | 474 |
| General skills | 333 |
| Task-specific skills | 141 |
| Task categories covered | 6 (Pick, PickTwo, Clean, Heat, Cool, Look) |
| Teacher model | gpt-4o |
| Embedding model | text-embedding-3-small |
| Embedding dimensions | 1,536 |
| Distillation time | ~9 minutes |

## Files Produced

| File | Description | Size |
|------|-------------|------|
| `data/alfworld_raw.json` | Raw ETO SFT trajectories from HuggingFace | 18MB |
| `data/alfworld_trajectories.json` | 100 converted trajectories in SkillRL format | ~250KB |
| `data/demo_skill_bank.json` | 474 distilled skills with embeddings | ~3MB |

## Next Steps

With the skill bank in hand, the remaining pipeline phases would be:

1. **Cold-start SFT** (`krill sft`): Fine-tune a policy model on skill-augmented demonstrations
2. **RL training** (`krill rl`): Train with GRPO, using recursive skill evolution to grow the skill bank as training progresses
3. **Evaluation**: Test the trained policy on held-out ALFWorld tasks, measuring success rate by category
