#!/usr/bin/env python3
"""Generate ALFWorld trajectory data for SkillRL training.

This script generates realistic ALFWorld trajectories in the format expected by
the Rust SkillRL implementation. It covers all 6 ALFWorld task categories:
  - Pick:    pick up an object and place it in a receptacle
  - Look:    examine an object under a desklamp
  - Clean:   clean an object at sinkbasin and place it
  - Heat:    heat an object in microwave and place it
  - Cool:    cool an object in fridge and place it
  - PickTwo: pick up two instances of an object and place them

The trajectories are based on the ALFWorld environment's actual observation and
action formats, matching the TextWorld-based interface used in the original
ALFWorld paper (Shridhar et al., 2021).

Usage:
    # If alfworld is installed:
    python scripts/generate_trajectories.py --mode live

    # Using built-in trajectory templates (no alfworld needed):
    python scripts/generate_trajectories.py --mode synthetic

    # Download and parse ReAct prompts:
    python scripts/generate_trajectories.py --mode react
"""

import argparse
import json
import os
import sys
import uuid
from typing import Any

# ---------------------------------------------------------------------------
# ALFWorld action/observation vocabulary (matches the real environment)
# ---------------------------------------------------------------------------

ROOM_DESCRIPTIONS = [
    "You are in the middle of a room. Looking quickly around you, you see a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.",
    "You are in the middle of a room. Looking quickly around you, you see a armchair 1, a bed 1, a cabinet 1, a desk 2, a desk 1, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a laundryhamper 1, a safe 1, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, a shelf 1, and a sidetable 2, a sidetable 1.",
    "You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a handtowelholder 1, a shelf 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.",
    "You are in the middle of a room. Looking quickly around you, you see a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.",
    "You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a shelf 3, a shelf 2, a shelf 1, a sidetable 2, a sidetable 1, and a safe 1.",
    "You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 2, a countertop 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.",
    "You are in the middle of a room. Looking quickly around you, you see a armchair 2, a armchair 1, a coffeetable 1, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a shelf 2, a shelf 1, a sidetable 2, a sidetable 1, a sofa 1, and a tvstand 1.",
]

KITCHEN_OBJECTS = [
    "apple", "bread", "butterknife", "cup", "dishsponge", "egg", "fork",
    "knife", "ladle", "lettuce", "mug", "pan", "peppershaker", "plate",
    "pot", "potato", "saltshaker", "soapbottle", "spatula", "spoon",
    "tomato", "bowl",
]

BEDROOM_OBJECTS = [
    "alarmclock", "book", "cd", "cellphone", "creditcard", "desklamp",
    "keychain", "laptop", "pen", "pencil", "pillow", "remotecontrol",
    "vase", "watch",
]

BATHROOM_OBJECTS = [
    "candle", "cloth", "handtowel", "plunger", "scrubbrush", "soapbar",
    "soapbottle", "spraybottle", "tissuebox", "toiletpaper", "towel",
]

KITCHEN_RECEPTACLES = [
    "countertop", "cabinet", "shelf", "diningtable", "drawer", "fridge",
    "garbagecan", "microwave", "sinkbasin", "stoveburner",
]

BEDROOM_RECEPTACLES = [
    "desk", "dresser", "shelf", "sidetable", "drawer", "bed", "safe",
    "garbagecan",
]


def make_step(observation: str, action: str, reward: float, step_index: int) -> dict:
    return {
        "observation": observation,
        "action": action,
        "reward": reward,
        "step_index": step_index,
        "action_log_prob": None,
        "ref_log_prob": None,
    }


def make_trajectory(
    task_description: str,
    task_category: str,
    steps: list[dict],
    success: bool,
) -> dict:
    total_reward = sum(s["reward"] for s in steps)
    if success and total_reward == 0:
        # Ensure successful trajectories get reward 1.0 on last step
        steps[-1]["reward"] = 1.0
        total_reward = 1.0
    return {
        "id": str(uuid.uuid4()),
        "task_description": task_description,
        "task_category": task_category,
        "steps": steps,
        "total_reward": total_reward,
        "success": success,
        "metadata": {
            "environment": "alfworld",
            "num_steps": len(steps),
            "total_tokens": 0,
            "skills_used": [],
        },
    }


# ---------------------------------------------------------------------------
# Trajectory generators for each task category
# ---------------------------------------------------------------------------

def gen_pick_success(obj: str, obj_id: int, src: str, src_id: int,
                     dst: str, dst_id: int, room_desc: str) -> dict:
    """Pick task: find object, pick it up, put it in receptacle."""
    task = f"put a {obj} in/on {dst} {dst_id}."
    steps = [
        make_step(room_desc, f"go to {src} {src_id}", 0.0, 0),
        make_step(
            f"On the {src} {src_id}, you see a {obj} {obj_id}, a plate 1, and a knife 2.",
            f"take {obj} {obj_id} from {src} {src_id}",
            0.0, 1,
        ),
        make_step(
            f"You pick up the {obj} {obj_id} from the {src} {src_id}.",
            f"go to {dst} {dst_id}",
            0.0, 2,
        ),
        make_step(
            f"You arrive at {dst} {dst_id}." if dst != "fridge" else
            f"The {dst} {dst_id} is closed.",
            f"open {dst} {dst_id}" if dst == "fridge" else f"put {obj} {obj_id} in/on {dst} {dst_id}",
            0.0, 3,
        ),
    ]
    if dst == "fridge":
        steps.append(make_step(
            f"You open the {dst} {dst_id}. The {dst} {dst_id} is open. In it, you see a tomato 1.",
            f"put {obj} {obj_id} in/on {dst} {dst_id}",
            0.0, 4,
        ))
        steps.append(make_step(
            f"You put the {obj} {obj_id} in/on the {dst} {dst_id}.",
            "inventory", 1.0, 5,
        ))
    else:
        steps.append(make_step(
            f"You put the {obj} {obj_id} in/on the {dst} {dst_id}.",
            "inventory", 1.0, 4,
        ))
    return make_trajectory(task, "Pick", steps, True)


def gen_pick_fail(obj: str, obj_id: int, room_desc: str) -> dict:
    """Pick task: fail to find the object after searching."""
    task = f"put a {obj} in/on shelf 1."
    steps = [
        make_step(room_desc, "go to countertop 1", 0.0, 0),
        make_step(
            "On the countertop 1, you see a bread 1 and a saltshaker 1.",
            "go to countertop 2", 0.0, 1,
        ),
        make_step(
            "On the countertop 2, you see a peppershaker 1 and a spoon 1.",
            "go to cabinet 1", 0.0, 2,
        ),
        make_step(
            "The cabinet 1 is closed.",
            "open cabinet 1", 0.0, 3,
        ),
        make_step(
            "You open the cabinet 1. The cabinet 1 is open. In it, you see nothing.",
            "go to cabinet 2", 0.0, 4,
        ),
        make_step(
            "The cabinet 2 is closed.",
            "open cabinet 2", 0.0, 5,
        ),
        make_step(
            "You open the cabinet 2. The cabinet 2 is open. In it, you see a plate 2.",
            "go to cabinet 3", 0.0, 6,
        ),
        make_step(
            "The cabinet 3 is closed.",
            "open cabinet 3", 0.0, 7,
        ),
        make_step(
            "You open the cabinet 3. The cabinet 3 is open. In it, you see nothing.",
            "go to diningtable 1", 0.0, 8,
        ),
        make_step(
            "On the diningtable 1, you see a fork 1, a bread 2, and a cup 1.",
            f"take {obj} {obj_id} from diningtable 1", 0.0, 9,
        ),
        make_step(
            "Nothing happens.",
            "go to fridge 1", 0.0, 10,
        ),
        make_step(
            "The fridge 1 is closed.",
            "open fridge 1", 0.0, 11,
        ),
        make_step(
            "You open the fridge 1. The fridge 1 is open. In it, you see a egg 1 and a potato 1.",
            f"take {obj} {obj_id} from fridge 1", 0.0, 12,
        ),
        make_step(
            "Nothing happens.",
            "look", 0.0, 13,
        ),
    ]
    return make_trajectory(task, "Pick", steps, False)


def gen_look_success(obj: str, obj_id: int, src: str, src_id: int,
                     room_desc: str) -> dict:
    """Look task: find object, pick it up, examine under desklamp."""
    task = f"examine the {obj} with the desklamp."
    steps = [
        make_step(room_desc, f"go to {src} {src_id}", 0.0, 0),
        make_step(
            f"On the {src} {src_id}, you see a {obj} {obj_id}, a pencil 1, and a creditcard 1.",
            f"take {obj} {obj_id} from {src} {src_id}",
            0.0, 1,
        ),
        make_step(
            f"You pick up the {obj} {obj_id} from the {src} {src_id}.",
            "go to desklamp 1",
            0.0, 2,
        ),
        make_step(
            "You see a desklamp 1. The desklamp 1 is turned off.",
            "use desklamp 1",
            0.0, 3,
        ),
        make_step(
            f"You turn on the desklamp 1. You examine the {obj} {obj_id} under the desklamp 1.",
            "inventory",
            1.0, 4,
        ),
    ]
    return make_trajectory(task, "Look", steps, True)


def gen_look_fail(obj: str, obj_id: int, room_desc: str) -> dict:
    """Look task: find object but fail to find desklamp or use it properly."""
    task = f"examine the {obj} with the desklamp."
    steps = [
        make_step(room_desc, "go to desk 1", 0.0, 0),
        make_step(
            "On the desk 1, you see a laptop 1 and a pen 1.",
            "go to desk 2", 0.0, 1,
        ),
        make_step(
            f"On the desk 2, you see a {obj} {obj_id} and a book 2.",
            f"take {obj} {obj_id} from desk 2", 0.0, 2,
        ),
        make_step(
            f"You pick up the {obj} {obj_id} from the desk 2.",
            "go to sidetable 1", 0.0, 3,
        ),
        make_step(
            "On the sidetable 1, you see a alarmclock 1.",
            "go to sidetable 2", 0.0, 4,
        ),
        make_step(
            "On the sidetable 2, you see a cd 1.",
            "use sidetable 2", 0.0, 5,
        ),
        make_step(
            "Nothing happens.",
            "go to dresser 1", 0.0, 6,
        ),
        make_step(
            "On the dresser 1, you see a keychain 1 and a vase 1.",
            f"examine {obj} {obj_id}", 0.0, 7,
        ),
        make_step(
            f"This is a normal {obj} {obj_id}.",
            f"use {obj} {obj_id}", 0.0, 8,
        ),
        make_step(
            "Nothing happens.",
            "look", 0.0, 9,
        ),
    ]
    return make_trajectory(task, "Look", steps, False)


def gen_clean_success(obj: str, obj_id: int, src: str, src_id: int,
                      dst: str, dst_id: int, room_desc: str) -> dict:
    """Clean task: find, pick up, clean at sinkbasin, place in destination."""
    task = f"put a clean {obj} in/on {dst} {dst_id}."
    steps = [
        make_step(room_desc, f"go to {src} {src_id}", 0.0, 0),
        make_step(
            f"On the {src} {src_id}, you see a {obj} {obj_id} and a dishsponge 1.",
            f"take {obj} {obj_id} from {src} {src_id}",
            0.0, 1,
        ),
        make_step(
            f"You pick up the {obj} {obj_id} from the {src} {src_id}.",
            "go to sinkbasin 1",
            0.0, 2,
        ),
        make_step(
            "You arrive at sinkbasin 1.",
            f"clean {obj} {obj_id} with sinkbasin 1",
            0.0, 3,
        ),
        make_step(
            f"You clean the {obj} {obj_id} using the sinkbasin 1.",
            f"go to {dst} {dst_id}",
            0.0, 4,
        ),
        make_step(
            f"You arrive at {dst} {dst_id}.",
            f"put {obj} {obj_id} in/on {dst} {dst_id}",
            0.0, 5,
        ),
        make_step(
            f"You put the {obj} {obj_id} in/on the {dst} {dst_id}.",
            "inventory",
            1.0, 6,
        ),
    ]
    return make_trajectory(task, "Clean", steps, True)


def gen_clean_fail(obj: str, obj_id: int, room_desc: str) -> dict:
    """Clean task: find object, forget to clean, or clean wrong thing."""
    task = f"put a clean {obj} in/on countertop 1."
    steps = [
        make_step(room_desc, "go to countertop 1", 0.0, 0),
        make_step(
            f"On the countertop 1, you see a {obj} {obj_id}, a fork 1, and a soapbottle 1.",
            f"take {obj} {obj_id} from countertop 1",
            0.0, 1,
        ),
        make_step(
            f"You pick up the {obj} {obj_id} from the countertop 1.",
            "go to countertop 2",
            0.0, 2,
        ),
        make_step(
            "You arrive at countertop 2. On the countertop 2, you see a bread 1.",
            f"put {obj} {obj_id} in/on countertop 2",
            0.0, 3,
        ),
        make_step(
            f"You put the {obj} {obj_id} in/on the countertop 2.",
            "look",
            0.0, 4,
        ),
        # Agent forgot to clean first - task not achieved
    ]
    return make_trajectory(task, "Clean", steps, False)


def gen_clean_fail_wrong_order(obj: str, obj_id: int, room_desc: str) -> dict:
    """Clean task: go to sink first without having the object."""
    task = f"put a clean {obj} in/on shelf 1."
    steps = [
        make_step(room_desc, "go to sinkbasin 1", 0.0, 0),
        make_step(
            "You arrive at sinkbasin 1. On the sinkbasin 1, you see nothing.",
            f"clean {obj} 1 with sinkbasin 1",
            0.0, 1,
        ),
        make_step(
            "Nothing happens.",
            "go to countertop 1",
            0.0, 2,
        ),
        make_step(
            f"On the countertop 1, you see a {obj} {obj_id} and a plate 1.",
            f"take {obj} {obj_id} from countertop 1",
            0.0, 3,
        ),
        make_step(
            f"You pick up the {obj} {obj_id} from the countertop 1.",
            "go to shelf 1",
            0.0, 4,
        ),
        make_step(
            "On the shelf 1, you see a vase 1.",
            f"put {obj} {obj_id} in/on shelf 1",
            0.0, 5,
        ),
        make_step(
            f"You put the {obj} {obj_id} in/on the shelf 1.",
            "look",
            0.0, 6,
        ),
        # Forgot to clean - placed uncleaned object
    ]
    return make_trajectory(task, "Clean", steps, False)


def gen_heat_success(obj: str, obj_id: int, src: str, src_id: int,
                     dst: str, dst_id: int, room_desc: str) -> dict:
    """Heat task: find, pick up, heat in microwave, place in destination."""
    task = f"put a hot {obj} in/on {dst} {dst_id}."
    steps = [
        make_step(room_desc, f"go to {src} {src_id}", 0.0, 0),
    ]

    if src == "fridge":
        steps.append(make_step(
            f"The {src} {src_id} is closed.",
            f"open {src} {src_id}",
            0.0, 1,
        ))
        steps.append(make_step(
            f"You open the {src} {src_id}. The {src} {src_id} is open. In it, you see a {obj} {obj_id} and a lettuce 1.",
            f"take {obj} {obj_id} from {src} {src_id}",
            0.0, 2,
        ))
        next_idx = 3
    else:
        steps.append(make_step(
            f"On the {src} {src_id}, you see a {obj} {obj_id}, a pan 1, and a spatula 1.",
            f"take {obj} {obj_id} from {src} {src_id}",
            0.0, 1,
        ))
        next_idx = 2

    steps.append(make_step(
        f"You pick up the {obj} {obj_id} from the {src} {src_id}.",
        "go to microwave 1",
        0.0, next_idx,
    ))
    steps.append(make_step(
        "The microwave 1 is closed.",
        "heat {obj} {obj_id} with microwave 1".format(obj=obj, obj_id=obj_id),
        0.0, next_idx + 1,
    ))
    steps.append(make_step(
        f"You heat the {obj} {obj_id} using the microwave 1.",
        f"go to {dst} {dst_id}",
        0.0, next_idx + 2,
    ))
    steps.append(make_step(
        f"You arrive at {dst} {dst_id}.",
        f"put {obj} {obj_id} in/on {dst} {dst_id}",
        0.0, next_idx + 3,
    ))
    steps.append(make_step(
        f"You put the {obj} {obj_id} in/on the {dst} {dst_id}.",
        "inventory",
        1.0, next_idx + 4,
    ))
    return make_trajectory(task, "Heat", steps, True)


def gen_heat_fail(obj: str, obj_id: int, room_desc: str) -> dict:
    """Heat task: fail by placing without heating."""
    task = f"put a hot {obj} in/on countertop 1."
    steps = [
        make_step(room_desc, "go to fridge 1", 0.0, 0),
        make_step(
            "The fridge 1 is closed.",
            "open fridge 1",
            0.0, 1,
        ),
        make_step(
            f"You open the fridge 1. The fridge 1 is open. In it, you see a {obj} {obj_id} and a tomato 1.",
            f"take {obj} {obj_id} from fridge 1",
            0.0, 2,
        ),
        make_step(
            f"You pick up the {obj} {obj_id} from the fridge 1.",
            "go to countertop 1",
            0.0, 3,
        ),
        make_step(
            "You arrive at countertop 1. On the countertop 1, you see a bread 1 and a knife 1.",
            f"put {obj} {obj_id} in/on countertop 1",
            0.0, 4,
        ),
        make_step(
            f"You put the {obj} {obj_id} in/on the countertop 1.",
            "look",
            0.0, 5,
        ),
        # Forgot to heat - task not achieved
    ]
    return make_trajectory(task, "Heat", steps, False)


def gen_cool_success(obj: str, obj_id: int, src: str, src_id: int,
                     dst: str, dst_id: int, room_desc: str) -> dict:
    """Cool task: find, pick up, cool in fridge, place in destination."""
    task = f"put a cool {obj} in/on {dst} {dst_id}."
    steps = [
        make_step(room_desc, f"go to {src} {src_id}", 0.0, 0),
        make_step(
            f"On the {src} {src_id}, you see a {obj} {obj_id}, a mug 1, and a saltshaker 1.",
            f"take {obj} {obj_id} from {src} {src_id}",
            0.0, 1,
        ),
        make_step(
            f"You pick up the {obj} {obj_id} from the {src} {src_id}.",
            "go to fridge 1",
            0.0, 2,
        ),
        make_step(
            "The fridge 1 is closed.",
            f"cool {obj} {obj_id} with fridge 1",
            0.0, 3,
        ),
        make_step(
            f"You cool the {obj} {obj_id} using the fridge 1.",
            f"go to {dst} {dst_id}",
            0.0, 4,
        ),
        make_step(
            f"You arrive at {dst} {dst_id}.",
            f"put {obj} {obj_id} in/on {dst} {dst_id}",
            0.0, 5,
        ),
        make_step(
            f"You put the {obj} {obj_id} in/on the {dst} {dst_id}.",
            "inventory",
            1.0, 6,
        ),
    ]
    return make_trajectory(task, "Cool", steps, True)


def gen_cool_fail(obj: str, obj_id: int, room_desc: str) -> dict:
    """Cool task: searched extensively but couldn't find the object."""
    task = f"put a cool {obj} in/on countertop 1."
    steps = [
        make_step(room_desc, "go to countertop 1", 0.0, 0),
        make_step(
            "On the countertop 1, you see a peppershaker 1, a plate 2, and a soapbottle 1.",
            "go to countertop 2", 0.0, 1,
        ),
        make_step(
            "On the countertop 2, you see a bread 1 and a fork 1.",
            "go to countertop 3", 0.0, 2,
        ),
        make_step(
            "On the countertop 3, you see a knife 1 and a spoon 2.",
            "go to diningtable 1", 0.0, 3,
        ),
        make_step(
            "On the diningtable 1, you see a bowl 1, a cup 1, and a potato 1.",
            "go to cabinet 1", 0.0, 4,
        ),
        make_step(
            "The cabinet 1 is closed.",
            "open cabinet 1", 0.0, 5,
        ),
        make_step(
            "You open the cabinet 1. The cabinet 1 is open. In it, you see nothing.",
            "go to cabinet 2", 0.0, 6,
        ),
        make_step(
            "The cabinet 2 is closed.",
            "open cabinet 2", 0.0, 7,
        ),
        make_step(
            "You open the cabinet 2. The cabinet 2 is open. In it, you see a mug 2.",
            "go to cabinet 3", 0.0, 8,
        ),
        make_step(
            "The cabinet 3 is closed.",
            "open cabinet 3", 0.0, 9,
        ),
        make_step(
            "You open the cabinet 3. The cabinet 3 is open. In it, you see nothing.",
            "go to fridge 1", 0.0, 10,
        ),
        make_step(
            "The fridge 1 is closed.",
            "open fridge 1", 0.0, 11,
        ),
        make_step(
            "You open the fridge 1. The fridge 1 is open. In it, you see a egg 1 and a lettuce 1.",
            "look", 0.0, 12,
        ),
    ]
    return make_trajectory(task, "Cool", steps, False)


def gen_picktwo_success(obj: str, obj_id1: int, obj_id2: int,
                        src1: str, src1_id: int,
                        src2: str, src2_id: int,
                        dst: str, dst_id: int,
                        room_desc: str) -> dict:
    """PickTwo task: find two of same object, place both in receptacle."""
    task = f"find two {obj}s and put them in/on {dst} {dst_id}."
    steps = [
        make_step(room_desc, f"go to {src1} {src1_id}", 0.0, 0),
        make_step(
            f"On the {src1} {src1_id}, you see a {obj} {obj_id1} and a plate 1.",
            f"take {obj} {obj_id1} from {src1} {src1_id}",
            0.0, 1,
        ),
        make_step(
            f"You pick up the {obj} {obj_id1} from the {src1} {src1_id}.",
            f"go to {dst} {dst_id}",
            0.0, 2,
        ),
        make_step(
            f"You arrive at {dst} {dst_id}.",
            f"put {obj} {obj_id1} in/on {dst} {dst_id}",
            0.0, 3,
        ),
        make_step(
            f"You put the {obj} {obj_id1} in/on the {dst} {dst_id}.",
            f"go to {src2} {src2_id}",
            0.0, 4,
        ),
        make_step(
            f"On the {src2} {src2_id}, you see a {obj} {obj_id2}, a knife 1, and a bread 1.",
            f"take {obj} {obj_id2} from {src2} {src2_id}",
            0.0, 5,
        ),
        make_step(
            f"You pick up the {obj} {obj_id2} from the {src2} {src2_id}.",
            f"go to {dst} {dst_id}",
            0.0, 6,
        ),
        make_step(
            f"You arrive at {dst} {dst_id}. On the {dst} {dst_id}, you see a {obj} {obj_id1}.",
            f"put {obj} {obj_id2} in/on {dst} {dst_id}",
            0.0, 7,
        ),
        make_step(
            f"You put the {obj} {obj_id2} in/on the {dst} {dst_id}.",
            "inventory",
            1.0, 8,
        ),
    ]
    return make_trajectory(task, "PickTwo", steps, True)


def gen_picktwo_fail(obj: str, obj_id1: int, room_desc: str) -> dict:
    """PickTwo task: find only one of the two objects."""
    task = f"find two {obj}s and put them in/on shelf 1."
    steps = [
        make_step(room_desc, "go to countertop 1", 0.0, 0),
        make_step(
            f"On the countertop 1, you see a {obj} {obj_id1}, a bread 1, and a plate 1.",
            f"take {obj} {obj_id1} from countertop 1",
            0.0, 1,
        ),
        make_step(
            f"You pick up the {obj} {obj_id1} from the countertop 1.",
            "go to shelf 1",
            0.0, 2,
        ),
        make_step(
            "On the shelf 1, you see nothing.",
            f"put {obj} {obj_id1} in/on shelf 1",
            0.0, 3,
        ),
        make_step(
            f"You put the {obj} {obj_id1} in/on the shelf 1.",
            "go to countertop 2",
            0.0, 4,
        ),
        make_step(
            "On the countertop 2, you see a fork 1 and a knife 2.",
            "go to diningtable 1",
            0.0, 5,
        ),
        make_step(
            "On the diningtable 1, you see a bowl 1, a cup 2, and a spoon 1.",
            "go to cabinet 1",
            0.0, 6,
        ),
        make_step(
            "The cabinet 1 is closed.",
            "open cabinet 1",
            0.0, 7,
        ),
        make_step(
            "You open the cabinet 1. The cabinet 1 is open. In it, you see a saltshaker 1.",
            "go to cabinet 2",
            0.0, 8,
        ),
        make_step(
            "The cabinet 2 is closed.",
            "open cabinet 2",
            0.0, 9,
        ),
        make_step(
            "You open the cabinet 2. The cabinet 2 is open. In it, you see nothing.",
            "look",
            0.0, 10,
        ),
    ]
    return make_trajectory(task, "PickTwo", steps, False)


# ---------------------------------------------------------------------------
# Additional variant generators for diversity
# ---------------------------------------------------------------------------

def gen_pick_with_search(obj: str, obj_id: int, dst: str, dst_id: int,
                         room_desc: str) -> dict:
    """Pick with extended search phase - eventually succeeds."""
    task = f"put a {obj} in/on {dst} {dst_id}."
    steps = [
        make_step(room_desc, "go to countertop 1", 0.0, 0),
        make_step(
            "On the countertop 1, you see a bread 1, a peppershaker 1, and a soapbottle 1.",
            "go to countertop 2", 0.0, 1,
        ),
        make_step(
            "On the countertop 2, you see a knife 1 and a pan 1.",
            "go to diningtable 1", 0.0, 2,
        ),
        make_step(
            f"On the diningtable 1, you see a {obj} {obj_id}, a cup 1, a fork 1, and a plate 1.",
            f"take {obj} {obj_id} from diningtable 1",
            0.0, 3,
        ),
        make_step(
            f"You pick up the {obj} {obj_id} from the diningtable 1.",
            f"go to {dst} {dst_id}",
            0.0, 4,
        ),
        make_step(
            f"You arrive at {dst} {dst_id}.",
            f"put {obj} {obj_id} in/on {dst} {dst_id}",
            0.0, 5,
        ),
        make_step(
            f"You put the {obj} {obj_id} in/on the {dst} {dst_id}.",
            "inventory",
            1.0, 6,
        ),
    ]
    return make_trajectory(task, "Pick", steps, True)


def gen_clean_extended(obj: str, obj_id: int, room_desc: str) -> dict:
    """Clean with search for the object in multiple locations."""
    task = f"put a clean {obj} in/on shelf 2."
    steps = [
        make_step(room_desc, "go to countertop 1", 0.0, 0),
        make_step(
            "On the countertop 1, you see a saltshaker 1 and a butterknife 1.",
            "go to countertop 2", 0.0, 1,
        ),
        make_step(
            "On the countertop 2, you see a peppershaker 1.",
            "go to diningtable 1", 0.0, 2,
        ),
        make_step(
            f"On the diningtable 1, you see a {obj} {obj_id}, a bread 1, a cup 1, and a fork 2.",
            f"take {obj} {obj_id} from diningtable 1",
            0.0, 3,
        ),
        make_step(
            f"You pick up the {obj} {obj_id} from the diningtable 1.",
            "go to sinkbasin 1",
            0.0, 4,
        ),
        make_step(
            "You arrive at sinkbasin 1.",
            f"clean {obj} {obj_id} with sinkbasin 1",
            0.0, 5,
        ),
        make_step(
            f"You clean the {obj} {obj_id} using the sinkbasin 1.",
            "go to shelf 2",
            0.0, 6,
        ),
        make_step(
            "On the shelf 2, you see nothing.",
            f"put {obj} {obj_id} in/on shelf 2",
            0.0, 7,
        ),
        make_step(
            f"You put the {obj} {obj_id} in/on the shelf 2.",
            "inventory",
            1.0, 8,
        ),
    ]
    return make_trajectory(task, "Clean", steps, True)


def gen_heat_from_countertop(obj: str, obj_id: int, room_desc: str) -> dict:
    """Heat success: object found on countertop."""
    task = f"put a hot {obj} in/on diningtable 1."
    steps = [
        make_step(room_desc, "go to countertop 1", 0.0, 0),
        make_step(
            f"On the countertop 1, you see a {obj} {obj_id} and a knife 1.",
            f"take {obj} {obj_id} from countertop 1",
            0.0, 1,
        ),
        make_step(
            f"You pick up the {obj} {obj_id} from the countertop 1.",
            "go to microwave 1",
            0.0, 2,
        ),
        make_step(
            "The microwave 1 is closed.",
            f"heat {obj} {obj_id} with microwave 1",
            0.0, 3,
        ),
        make_step(
            f"You heat the {obj} {obj_id} using the microwave 1.",
            "go to diningtable 1",
            0.0, 4,
        ),
        make_step(
            "On the diningtable 1, you see a bowl 1 and a cup 1.",
            f"put {obj} {obj_id} in/on diningtable 1",
            0.0, 5,
        ),
        make_step(
            f"You put the {obj} {obj_id} in/on the diningtable 1.",
            "inventory",
            1.0, 6,
        ),
    ]
    return make_trajectory(task, "Heat", steps, True)


def gen_heat_wrong_appliance(obj: str, obj_id: int, room_desc: str) -> dict:
    """Heat fail: try to heat with stoveburner (wrong approach)."""
    task = f"put a hot {obj} in/on countertop 1."
    steps = [
        make_step(room_desc, "go to countertop 2", 0.0, 0),
        make_step(
            f"On the countertop 2, you see a {obj} {obj_id} and a spatula 1.",
            f"take {obj} {obj_id} from countertop 2",
            0.0, 1,
        ),
        make_step(
            f"You pick up the {obj} {obj_id} from the countertop 2.",
            "go to stoveburner 1",
            0.0, 2,
        ),
        make_step(
            "On the stoveburner 1, you see nothing.",
            f"heat {obj} {obj_id} with stoveburner 1",
            0.0, 3,
        ),
        make_step(
            "Nothing happens.",
            "go to stoveburner 2",
            0.0, 4,
        ),
        make_step(
            "On the stoveburner 2, you see a pot 1.",
            f"heat {obj} {obj_id} with stoveburner 2",
            0.0, 5,
        ),
        make_step(
            "Nothing happens.",
            "go to countertop 1",
            0.0, 6,
        ),
        make_step(
            "You arrive at countertop 1. On the countertop 1, you see a bread 1 and a fork 1.",
            f"put {obj} {obj_id} in/on countertop 1",
            0.0, 7,
        ),
        make_step(
            f"You put the {obj} {obj_id} in/on the countertop 1.",
            "look",
            0.0, 8,
        ),
        # Failed: tried stoveburner instead of microwave
    ]
    return make_trajectory(task, "Heat", steps, False)


def gen_cool_from_stoveburner(obj: str, obj_id: int, room_desc: str) -> dict:
    """Cool success: object found on stoveburner."""
    task = f"put a cool {obj} in/on shelf 1."
    steps = [
        make_step(room_desc, "go to stoveburner 1", 0.0, 0),
        make_step(
            f"On the stoveburner 1, you see a {obj} {obj_id}.",
            f"take {obj} {obj_id} from stoveburner 1",
            0.0, 1,
        ),
        make_step(
            f"You pick up the {obj} {obj_id} from the stoveburner 1.",
            "go to fridge 1",
            0.0, 2,
        ),
        make_step(
            "The fridge 1 is closed.",
            f"cool {obj} {obj_id} with fridge 1",
            0.0, 3,
        ),
        make_step(
            f"You cool the {obj} {obj_id} using the fridge 1.",
            "go to shelf 1",
            0.0, 4,
        ),
        make_step(
            "On the shelf 1, you see a peppershaker 2.",
            f"put {obj} {obj_id} in/on shelf 1",
            0.0, 5,
        ),
        make_step(
            f"You put the {obj} {obj_id} in/on the shelf 1.",
            "inventory",
            1.0, 6,
        ),
    ]
    return make_trajectory(task, "Cool", steps, True)


def gen_look_with_search(obj: str, obj_id: int, room_desc: str) -> dict:
    """Look success: search multiple locations before finding object."""
    task = f"examine the {obj} with the desklamp."
    steps = [
        make_step(room_desc, "go to sidetable 1", 0.0, 0),
        make_step(
            "On the sidetable 1, you see a alarmclock 1.",
            "go to sidetable 2",
            0.0, 1,
        ),
        make_step(
            "On the sidetable 2, you see a keychain 1.",
            "go to desk 1",
            0.0, 2,
        ),
        make_step(
            "On the desk 1, you see a laptop 1 and a pencil 2.",
            "go to dresser 1",
            0.0, 3,
        ),
        make_step(
            f"On the dresser 1, you see a {obj} {obj_id}, a book 1, and a vase 1.",
            f"take {obj} {obj_id} from dresser 1",
            0.0, 4,
        ),
        make_step(
            f"You pick up the {obj} {obj_id} from the dresser 1.",
            "go to desklamp 1",
            0.0, 5,
        ),
        make_step(
            "You see a desklamp 1. The desklamp 1 is turned off.",
            "use desklamp 1",
            0.0, 6,
        ),
        make_step(
            f"You turn on the desklamp 1. You examine the {obj} {obj_id} under the desklamp 1.",
            "inventory",
            1.0, 7,
        ),
    ]
    return make_trajectory(task, "Look", steps, True)


def gen_picktwo_extended(obj: str, room_desc: str) -> dict:
    """PickTwo with search for both objects across many locations."""
    task = f"find two {obj}s and put them in/on diningtable 1."
    steps = [
        make_step(room_desc, "go to countertop 1", 0.0, 0),
        make_step(
            "On the countertop 1, you see a bread 1, a fork 1, and a pan 1.",
            "go to countertop 2",
            0.0, 1,
        ),
        make_step(
            f"On the countertop 2, you see a {obj} 1 and a peppershaker 1.",
            f"take {obj} 1 from countertop 2",
            0.0, 2,
        ),
        make_step(
            f"You pick up the {obj} 1 from the countertop 2.",
            "go to diningtable 1",
            0.0, 3,
        ),
        make_step(
            "On the diningtable 1, you see a bowl 1 and a cup 1.",
            f"put {obj} 1 in/on diningtable 1",
            0.0, 4,
        ),
        make_step(
            f"You put the {obj} 1 in/on the diningtable 1.",
            "go to cabinet 1",
            0.0, 5,
        ),
        make_step(
            "The cabinet 1 is closed.",
            "open cabinet 1",
            0.0, 6,
        ),
        make_step(
            "You open the cabinet 1. The cabinet 1 is open. In it, you see nothing.",
            "go to cabinet 2",
            0.0, 7,
        ),
        make_step(
            "The cabinet 2 is closed.",
            "open cabinet 2",
            0.0, 8,
        ),
        make_step(
            f"You open the cabinet 2. The cabinet 2 is open. In it, you see a {obj} 2.",
            f"take {obj} 2 from cabinet 2",
            0.0, 9,
        ),
        make_step(
            f"You pick up the {obj} 2 from the cabinet 2.",
            "go to diningtable 1",
            0.0, 10,
        ),
        make_step(
            f"On the diningtable 1, you see a {obj} 1, a bowl 1, and a cup 1.",
            f"put {obj} 2 in/on diningtable 1",
            0.0, 11,
        ),
        make_step(
            f"You put the {obj} 2 in/on the diningtable 1.",
            "inventory",
            1.0, 12,
        ),
    ]
    return make_trajectory(task, "PickTwo", steps, True)


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_all_trajectories() -> list[dict]:
    """Generate a comprehensive set of ALFWorld trajectories."""
    trajectories: list[dict] = []

    # Room descriptions for kitchen and bedroom
    kr = ROOM_DESCRIPTIONS  # various room descriptions

    # ===== PICK trajectories (10) =====
    # Successful picks
    trajectories.append(gen_pick_success("apple", 1, "countertop", 1, "shelf", 1, kr[0]))
    trajectories.append(gen_pick_success("mug", 1, "countertop", 2, "cabinet", 3, kr[3]))
    trajectories.append(gen_pick_success("plate", 1, "diningtable", 1, "shelf", 2, kr[0]))
    trajectories.append(gen_pick_success("bowl", 1, "countertop", 1, "fridge", 1, kr[5]))
    trajectories.append(gen_pick_success("saltshaker", 1, "diningtable", 1, "drawer", 1, kr[3]))
    trajectories.append(gen_pick_with_search("cup", 1, "cabinet", 1, kr[0]))
    trajectories.append(gen_pick_with_search("egg", 1, "fridge", 1, kr[5]))
    trajectories.append(gen_pick_with_search("spoon", 1, "shelf", 1, kr[3]))
    # Failed picks
    trajectories.append(gen_pick_fail("tomato", 1, kr[0]))
    trajectories.append(gen_pick_fail("lettuce", 1, kr[5]))

    # ===== LOOK trajectories (10) =====
    # Successful looks
    trajectories.append(gen_look_success("cd", 1, "desk", 1, kr[1]))
    trajectories.append(gen_look_success("book", 1, "sidetable", 1, kr[1]))
    trajectories.append(gen_look_success("pen", 1, "desk", 2, kr[4]))
    trajectories.append(gen_look_success("cellphone", 1, "dresser", 1, kr[1]))
    trajectories.append(gen_look_success("alarmclock", 1, "sidetable", 2, kr[4]))
    trajectories.append(gen_look_with_search("vase", 1, kr[1]))
    trajectories.append(gen_look_with_search("creditcard", 1, kr[4]))
    trajectories.append(gen_look_with_search("keychain", 1, kr[1]))
    # Failed looks
    trajectories.append(gen_look_fail("watch", 1, kr[1]))
    trajectories.append(gen_look_fail("pencil", 2, kr[4]))

    # ===== CLEAN trajectories (10) =====
    # Successful cleans
    trajectories.append(gen_clean_success("apple", 1, "countertop", 1, "shelf", 1, kr[0]))
    trajectories.append(gen_clean_success("mug", 1, "diningtable", 1, "cabinet", 2, kr[3]))
    trajectories.append(gen_clean_success("plate", 1, "countertop", 2, "shelf", 2, kr[0]))
    trajectories.append(gen_clean_success("bowl", 1, "diningtable", 1, "countertop", 1, kr[5]))
    trajectories.append(gen_clean_success("cup", 1, "countertop", 1, "shelf", 3, kr[3]))
    trajectories.append(gen_clean_extended("lettuce", 1, kr[0]))
    trajectories.append(gen_clean_extended("tomato", 1, kr[5]))
    # Failed cleans
    trajectories.append(gen_clean_fail("potato", 1, kr[0]))
    trajectories.append(gen_clean_fail("egg", 1, kr[5]))
    trajectories.append(gen_clean_fail_wrong_order("mug", 2, kr[3]))

    # ===== HEAT trajectories (10) =====
    # Successful heats
    trajectories.append(gen_heat_success("potato", 1, "fridge", 1, "countertop", 1, kr[0]))
    trajectories.append(gen_heat_success("egg", 1, "fridge", 1, "diningtable", 1, kr[3]))
    trajectories.append(gen_heat_success("apple", 1, "countertop", 1, "shelf", 1, kr[5]))
    trajectories.append(gen_heat_success("tomato", 1, "fridge", 1, "countertop", 2, kr[0]))
    trajectories.append(gen_heat_success("bread", 1, "countertop", 2, "diningtable", 1, kr[3]))
    trajectories.append(gen_heat_from_countertop("potato", 2, kr[0]))
    trajectories.append(gen_heat_from_countertop("egg", 2, kr[5]))
    # Failed heats
    trajectories.append(gen_heat_fail("apple", 1, kr[0]))
    trajectories.append(gen_heat_fail("potato", 1, kr[3]))
    trajectories.append(gen_heat_wrong_appliance("tomato", 1, kr[5]))

    # ===== COOL trajectories (10) =====
    # Successful cools
    trajectories.append(gen_cool_success("apple", 1, "countertop", 1, "shelf", 1, kr[0]))
    trajectories.append(gen_cool_success("mug", 1, "diningtable", 1, "countertop", 2, kr[3]))
    trajectories.append(gen_cool_success("bowl", 1, "countertop", 2, "shelf", 2, kr[5]))
    trajectories.append(gen_cool_success("potato", 1, "countertop", 1, "diningtable", 1, kr[0]))
    trajectories.append(gen_cool_success("tomato", 1, "countertop", 1, "countertop", 2, kr[3]))
    trajectories.append(gen_cool_from_stoveburner("pan", 1, kr[0]))
    trajectories.append(gen_cool_from_stoveburner("pot", 1, kr[5]))
    # Failed cools
    trajectories.append(gen_cool_fail("lettuce", 1, kr[0]))
    trajectories.append(gen_cool_fail("egg", 1, kr[3]))
    trajectories.append(gen_cool_fail("bread", 1, kr[5]))

    # ===== PICKTWO trajectories (10) =====
    # Successful picktwos
    trajectories.append(gen_picktwo_success(
        "apple", 1, 2, "countertop", 1, "countertop", 2,
        "shelf", 1, kr[0],
    ))
    trajectories.append(gen_picktwo_success(
        "plate", 1, 2, "diningtable", 1, "countertop", 1,
        "cabinet", 2, kr[3],
    ))
    trajectories.append(gen_picktwo_success(
        "cup", 1, 2, "countertop", 1, "diningtable", 1,
        "shelf", 2, kr[5],
    ))
    trajectories.append(gen_picktwo_success(
        "mug", 1, 2, "countertop", 2, "shelf", 1,
        "diningtable", 1, kr[0],
    ))
    trajectories.append(gen_picktwo_success(
        "saltshaker", 1, 2, "diningtable", 1, "countertop", 2,
        "drawer", 1, kr[3],
    ))
    trajectories.append(gen_picktwo_extended("spoon", kr[0]))
    trajectories.append(gen_picktwo_extended("fork", kr[5]))
    # Failed picktwos
    trajectories.append(gen_picktwo_fail("potato", 1, kr[0]))
    trajectories.append(gen_picktwo_fail("egg", 1, kr[3]))
    trajectories.append(gen_picktwo_fail("tomato", 1, kr[5]))

    return trajectories


def try_live_generation() -> list[dict] | None:
    """Try to generate trajectories using actual ALFWorld environment."""
    try:
        import alfworld
        import alfworld.agents.environment as environment
        import yaml

        print("ALFWorld found! Generating live trajectories...")

        # Load config
        config_path = os.path.join(
            os.path.dirname(alfworld.__file__),
            "configs", "base_config.yaml"
        )
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Override for batch environment
        config["dataset"]["eval_ood_data_path"] = ""
        config["dataset"]["eval_id_data_path"] = ""

        env = environment.AlfredTWEnv(config, train_eval="train")
        env = env.init_env(batch_size=1)

        trajectories = []
        for episode in range(60):
            obs, info = env.reset()
            obs_text = obs[0]
            admissible = info.get("admissible_commands", [[]])[0]

            task_desc = info.get("extra.gamefile", ["unknown"])[0]
            # Infer category from task
            if "clean" in task_desc.lower():
                cat = "Clean"
            elif "heat" in task_desc.lower() or "hot" in task_desc.lower():
                cat = "Heat"
            elif "cool" in task_desc.lower():
                cat = "Cool"
            elif "examine" in task_desc.lower() or "look" in task_desc.lower():
                cat = "Look"
            elif "two" in task_desc.lower():
                cat = "PickTwo"
            else:
                cat = "Pick"

            steps_data = []
            done = False
            step_idx = 0
            max_steps = 30

            while not done and step_idx < max_steps:
                # Simple heuristic: pick random admissible action
                import random
                if admissible:
                    action = random.choice(admissible)
                else:
                    action = "look"

                steps_data.append(make_step(obs_text, action, 0.0, step_idx))

                obs, scores, dones, infos = env.step([action])
                obs_text = obs[0]
                done = dones[0]
                admissible = infos.get("admissible_commands", [[]])[0]
                step_idx += 1

            success = done and scores[0] > 0
            if success and steps_data:
                steps_data[-1]["reward"] = 1.0

            traj = make_trajectory(task_desc, cat, steps_data, success)
            trajectories.append(traj)
            print(f"  Episode {episode + 1}/60: {cat} - {'SUCCESS' if success else 'FAIL'} ({step_idx} steps)")

        env.close()
        return trajectories

    except ImportError:
        print("ALFWorld not installed, falling back to synthetic generation.")
        return None
    except Exception as e:
        print(f"ALFWorld error: {e}")
        print("Falling back to synthetic generation.")
        return None


def try_react_download() -> list[dict] | None:
    """Try to download and parse ReAct ALFWorld prompts."""
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/ysymyth/ReAct/master/prompts/alfworld_3prompts.json"
        print(f"Downloading ReAct prompts from {url}...")
        response = urllib.request.urlopen(url, timeout=15)
        data = json.loads(response.read().decode("utf-8"))
        print("  Downloaded successfully, parsing trajectories...")

        trajectories = []
        for key, prompt_text in data.items():
            # Parse the trajectories from the prompt text
            # ReAct prompts contain examples like:
            # > observation
            # action
            lines = prompt_text.strip().split("\n")
            # Extract task and steps
            task_desc = ""
            steps_data = []
            step_idx = 0
            current_obs = ""

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("Interact with a household"):
                    continue
                if line.startswith("Here are two examples"):
                    continue
                # Look for observation/action patterns
                if line.startswith(">"):
                    current_obs = line[1:].strip()
                    if not task_desc and "Your task is to:" in current_obs:
                        task_desc = current_obs.split("Your task is to:")[-1].strip()
                elif current_obs:
                    action = line
                    steps_data.append(make_step(current_obs, action, 0.0, step_idx))
                    step_idx += 1
                    current_obs = ""

            if steps_data and task_desc:
                # Infer category
                if "clean" in task_desc.lower():
                    cat = "Clean"
                elif "heat" in task_desc.lower() or "hot" in task_desc.lower():
                    cat = "Heat"
                elif "cool" in task_desc.lower():
                    cat = "Cool"
                elif "examine" in task_desc.lower():
                    cat = "Look"
                elif "two" in task_desc.lower():
                    cat = "PickTwo"
                else:
                    cat = "Pick"

                traj = make_trajectory(task_desc, cat, steps_data, True)
                trajectories.append(traj)

        if trajectories:
            print(f"  Parsed {len(trajectories)} trajectories from ReAct prompts")
            return trajectories
        return None

    except Exception as e:
        print(f"Failed to download ReAct prompts: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate ALFWorld trajectory data for SkillRL"
    )
    parser.add_argument(
        "--mode",
        choices=["live", "synthetic", "react", "auto"],
        default="auto",
        help="Generation mode: live (alfworld env), synthetic (templates), "
             "react (download ReAct prompts), auto (try all)",
    )
    parser.add_argument(
        "--output",
        default="data/alfworld_trajectories.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    trajectories = None

    if args.mode == "live":
        trajectories = try_live_generation()
    elif args.mode == "react":
        trajectories = try_react_download()
        if trajectories and len(trajectories) < 30:
            print("  Supplementing with synthetic trajectories...")
            trajectories.extend(generate_all_trajectories())
    elif args.mode == "synthetic":
        trajectories = generate_all_trajectories()
    else:  # auto
        trajectories = try_live_generation()
        if not trajectories:
            react_trajs = try_react_download()
            if react_trajs:
                trajectories = react_trajs
                if len(trajectories) < 30:
                    trajectories.extend(generate_all_trajectories())
            else:
                trajectories = generate_all_trajectories()

    if not trajectories:
        trajectories = generate_all_trajectories()

    # Summary statistics
    categories = {}
    successes = 0
    for t in trajectories:
        cat = t["task_category"]
        categories[cat] = categories.get(cat, 0) + 1
        if t["success"]:
            successes += 1

    print(f"\nGenerated {len(trajectories)} trajectories:")
    for cat in sorted(categories.keys()):
        print(f"  {cat}: {categories[cat]}")
    print(f"  Success rate: {successes}/{len(trajectories)} "
          f"({100 * successes / len(trajectories):.1f}%)")

    # Write output
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.output,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(trajectories, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
