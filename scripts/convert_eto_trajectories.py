#!/usr/bin/env python3
"""Convert ETO ALFWorld SFT trajectories to SkillRL format.

Source: https://huggingface.co/datasets/agent-eto/eto-sft-trajectory
"""

import json
import re
import sys
import uuid

CATEGORY_MAP = {
    "pick_and_place": "Pick",
    "pick_clean_then_place": "Clean",
    "pick_heat_then_place": "Heat",
    "pick_cool_then_place": "Cool",
    "look_at_obj": "Look",
    "pick_two_obj": "PickTwo",
}

def infer_category(game_file: str, task_desc: str) -> str:
    for key, cat in CATEGORY_MAP.items():
        if key in game_file:
            return cat
    # Fallback to task description
    td = task_desc.lower()
    if "clean" in td:
        return "Clean"
    if "hot" in td or "heat" in td:
        return "Heat"
    if "cool" in td:
        return "Cool"
    if "examine" in td or "look" in td:
        return "Look"
    if "two" in td:
        return "PickTwo"
    return "Pick"

def convert(raw: list, max_trajectories: int = 0) -> list:
    trajectories = []
    for entry in raw:
        convos = entry["conversations"]
        game_file = entry.get("game_file", "")

        # Extract task description and observation/action pairs
        task_desc = ""
        steps = []
        step_idx = 0
        pending_obs = None

        for msg in convos:
            text = msg["value"].strip()
            if msg["from"] == "human":
                # Check if this contains the task description
                if "Your task is to:" in text:
                    task_desc = text.split("Your task is to:")[-1].strip()
                    # The room description is also the first observation
                    room_desc = text.split("\nYour task is to:")[0].strip()
                    # Remove the system prompt prefix if present
                    if "you see" in room_desc:
                        # Extract just the room observation
                        idx = room_desc.rfind("You are in the middle")
                        if idx >= 0:
                            room_desc = room_desc[idx:]
                    pending_obs = room_desc
                elif text.startswith("Observation:"):
                    pending_obs = text[len("Observation:"):].strip()
                elif text == "OK":
                    continue
                else:
                    # Some observations don't have the prefix
                    if pending_obs is None and "you see" in text.lower():
                        pending_obs = text
            elif msg["from"] == "gpt":
                if text == "OK":
                    continue
                # Extract action from "Thought: ...\nAction: ..."
                action_match = re.search(r'Action:\s*(.+?)(?:\n|$)', text)
                if action_match and pending_obs is not None:
                    action = action_match.group(1).strip()
                    steps.append({
                        "observation": pending_obs,
                        "action": action,
                        "reward": 0.0,
                        "step_index": step_idx,
                        "action_log_prob": None,
                        "ref_log_prob": None,
                    })
                    step_idx += 1
                    pending_obs = None

        if not steps or not task_desc:
            continue

        # These are SFT trajectories (successful demonstrations)
        # Mark the last step with reward 1.0
        steps[-1]["reward"] = 1.0

        category = infer_category(game_file, task_desc)

        trajectories.append({
            "id": str(uuid.uuid4()),
            "task_description": task_desc,
            "task_category": category,
            "steps": steps,
            "total_reward": 1.0,
            "success": True,
            "metadata": {
                "environment": "alfworld",
                "num_steps": len(steps),
                "total_tokens": 0,
                "skills_used": [],
            },
        })

        if max_trajectories and len(trajectories) >= max_trajectories:
            break

    return trajectories


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/alfworld_raw.json")
    parser.add_argument("--output", default="data/alfworld_trajectories.json")
    parser.add_argument("--max", type=int, default=100,
                        help="Max trajectories to convert (0 = all)")
    args = parser.parse_args()

    with open(args.input) as f:
        raw = json.load(f)

    trajectories = convert(raw, args.max)

    # Stats
    cats = {}
    for t in trajectories:
        cats[t["task_category"]] = cats.get(t["task_category"], 0) + 1

    print(f"Converted {len(trajectories)} trajectories:")
    for cat in sorted(cats):
        print(f"  {cat}: {cats[cat]}")
    total_steps = sum(t["metadata"]["num_steps"] for t in trajectories)
    print(f"  Total steps: {total_steps}")
    print(f"  Avg steps/trajectory: {total_steps / len(trajectories):.1f}")

    with open(args.output, "w") as f:
        json.dump(trajectories, f, indent=2)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
