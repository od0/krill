# SkillRL Walkthrough

## The Problem

You have an LLM agent that needs to do tasks in an environment — like navigating a house to "put a clean apple in the fridge" (ALFWorld) or shopping for "a red cotton t-shirt under $30" (WebShop).

The naive approach: let the agent try stuff, save every trajectory it generates, and stuff those raw trajectories into its context window next time as "memory."

This sucks because:
- Raw trajectories are long and noisy (most steps are boring)
- They eat up your context window
- The agent has to re-derive the lessons every time

## The SkillRL Idea

Instead of memorizing raw experience, **distill it into skills** — concise, reusable principles.

Run the `inspect` command and look at what a skill actually is:

```
[5d352009] Heat Management
  Principle: Safely handle hot objects and determine appropriate places
             for cooling or storage.
  Apply when: When dealing with tasks that involve hot items that need to
              be cooled down or stored safely.

[6ec57ad0] Sequential Exploration
  Principle: Systematically check each potential location for the target
             object until it is found.
  Apply when: When the target object is not immediately visible, requiring
              a methodical search.
```

That's a **skill** — a name, a principle (what to do), and a trigger (when to apply it). Way more useful than a 50-step trajectory dump.

## The 4-Phase Pipeline

This is the core of the paper (Algorithm 1).

### Phase 1: Collect Trajectories

```bash
krill collect --episodes 64 --mock
```

The agent interacts with the environment. Each episode produces a `Trajectory` — a sequence of (observation, action, reward) steps. Some succeed, some fail. We save both.

In our code, this is `TrainingPipeline::collect_trajectories()` in `src/training/pipeline.rs`. It loops over episodes, calls the policy LLM for each action, and records everything.

You can also use pre-collected trajectories instead of running live collection. For example, the [ETO SFT trajectory dataset](https://huggingface.co/datasets/agent-eto/eto-sft-trajectory) has 3,119 successful ALFWorld episodes. The `scripts/convert_eto_trajectories.py` script converts these into SkillRL format. This is what we used for the [demo run](demo-run.md).

### Phase 2: Distill Skills

```bash
krill distill --trajectories data/trajectories.json
```

This is where the magic happens. A **teacher model** (a stronger LLM like o3 or gpt-4o) looks at the trajectories and extracts skills:

- From **successes**: "What strategic pattern made this work?" → General principles
- From **failures**: "What went wrong and how to avoid it?" → Lessons learned

The paper notation: `s+ = M_T(tau+, d)` and `s- = M_T(tau-, d)`

Skills get organized into a **SkillBank** with two tiers:
- **General skills** — apply everywhere (e.g., "Sequential Exploration", "Optimal Path Planning", "State Verification")
- **Task-specific skills** — apply to one category (e.g., "Heat Management" only for Heat tasks, "Clean object" only for Clean tasks)

You saw this hierarchy in the `inspect` output.

### Phase 3: Cold-Start SFT

```bash
krill sft --skill-bank data/skill_bank.json
```

Now we fine-tune the policy model. The teacher generates **skill-augmented demonstrations** — ideal trajectories that show the agent *using* the skills it just learned. The policy is trained on these via standard supervised fine-tuning (cross-entropy loss).

This gives the agent a warm start — it already knows what skills exist and roughly how to use them before RL begins.

### Phase 4: RL with Recursive Evolution

```bash
krill rl --skill-bank data/skill_bank.json
```

This is the main training loop. It uses **GRPO** (Group Relative Policy Optimization) — a variant of PPO designed for LLMs:

1. For each task, generate G completions (group_size=8)
2. Compute advantages as z-scores within each group: `A_i = (R_i - mean) / std`
3. Optimize the clipped surrogate objective with a KL penalty against a reference policy

The key innovation is **recursive evolution**: every `validation_interval` steps, the system checks per-category accuracy. If a category is failing (below 40% success rate):

```
S_new = M_T(failures, SkillBank)    // teacher analyzes failures
SkillBank <- SkillBank U S_new       // add new skills
```

The skill bank **grows** during training. In our [demo run](demo-run.md), the initial distillation produced 474 skills from 100 trajectories:

```
Total skills: 474
  General: 333
  Pick:     42
  PickTwo:  24
  Heat:     21
  Clean:    18
  Cool:     18
  Look:     18
```

During RL training, evolution would add more skills as the agent encounters failures — e.g., the agent keeps failing at Heat tasks -> the system notices -> teacher proposes new Heat-specific skills -> agent gets better -> cycle continues.

## How Skills Are Used at Inference

When the agent gets a task like "heat the potato", it:

1. **Embeds** the task description into a vector
2. **Retrieves** the top-K most relevant skills via cosine similarity (plus all general skills)
3. **Injects** them into the prompt as structured context
4. **Acts** with chain-of-thought reasoning grounded in the retrieved skills

The paper notation: `S_ret = TopK({s in S_k : sim(e_d, e_s) > delta}, K)`

This is in `src/agent/agent.rs` — the `act()` method, and `src/skill/retrieval.rs` for the retrieval logic.

## The Key Insight

The whole system is a loop:

```
experience -> skills -> better policy -> new experience -> better skills -> ...
```

Traditional approaches store raw trajectories (noisy, long). SkillRL compresses them into principles (10-20% fewer tokens) that are actually more useful for reasoning. And the skill bank co-evolves with the policy — it's not a static knowledge base.

## Try It

For a full worked example — downloading real ALFWorld data, distilling skills with gpt-4o, and inspecting the results — see [demo-run.md](demo-run.md).
