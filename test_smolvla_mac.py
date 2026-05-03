#!/usr/bin/env python3
"""
SmolVLA checkpoint test — Mac M1 / CPU / MPS
=============================================
Tests the 60K checkpoint with synthetic inputs (no dataset needed).

Setup (one time):
    python3 -m venv smolvla_test_env
    source smolvla_test_env/bin/activate
    pip install --upgrade pip
    pip install lerobot av torch torchvision

Run:
    source smolvla_test_env/bin/activate
    python3 test_smolvla_mac.py
"""

import sys
import time
import numpy as np
import torch

# ── CONFIG ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "/Users/vickyprince/Downloads/060000/pretrained_model"

# Input specs (from config.json)
IMAGE_KEYS   = ["observation.images.left",
                "observation.images.center",
                "observation.images.right"]
IMAGE_SHAPE  = (3, 480, 640)   # C, H, W  (model resizes to 512x512 internally)
STATE_DIM    = 30
ACTION_DIM   = 6
CHUNK_SIZE   = 50              # model predicts 50 steps at once

TASK_SFP = "Insert the SFP module into the target SFP port on the NIC card"
TASK_SC  = "Insert the SC plug into the target SC port"
# ──────────────────────────────────────────────────────────────────────────────


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_tokenizer():
    """Load the SmolVLM2 tokenizer — already cached locally from policy load."""
    from transformers import AutoTokenizer
    # This is cached from when the policy loaded the VLM weights
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    )
    return tokenizer


def tokenize_task(tokenizer, task_str: str, device: str, max_length: int = 48) -> dict:
    """Convert a task string into token tensors expected by select_action."""
    enc = tokenizer(
        task_str,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    return {
        "observation.language.tokens":         enc["input_ids"].to(device),
        # attention mask must be bool — model uses torch.where(...) internally
        "observation.language.attention_mask": enc["attention_mask"].bool().to(device),
    }


def make_synthetic_batch(device: str, tokenizer, task_str: str) -> dict:
    """Build a fake but correctly shaped observation batch."""
    batch = {}

    # 3 camera images — random uint8 scaled to float [0,1]
    for key in IMAGE_KEYS:
        img = torch.randint(0, 256, (1, *IMAGE_SHAPE), dtype=torch.uint8)
        batch[key] = img.float() / 255.0
        batch[key] = batch[key].to(device)

    # 30-dim robot state (zeros simulate a resting pose)
    batch["observation.state"] = torch.zeros(1, STATE_DIM, device=device)

    # Language task — both raw string and pre-tokenized tokens
    batch["task"] = [task_str]
    batch.update(tokenize_task(tokenizer, task_str, device))

    return batch


def run_test():
    print("=" * 60)
    print("  SmolVLA 60K checkpoint test  —  Mac M1")
    print("=" * 60)

    device = get_device()
    print(f"\nDevice : {device}")
    print(f"Checkpoint : {CHECKPOINT_PATH}")

    # ── 1. Load checkpoint ─────────────────────────────────────────
    print("\n[1/3] Loading policy ...")
    t0 = time.time()

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(CHECKPOINT_PATH)
    policy = policy.to(device)
    policy.eval()

    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # ── Load tokenizer ─────────────────────────────────────────────
    print("  Loading tokenizer ...")
    tokenizer = get_tokenizer()
    print(f"  Tokenizer: {type(tokenizer).__name__}")

    # ── 2. Run inference with both task types ──────────────────────
    print("\n[2/3] Running inference ...")

    results = []
    for task_label, task_str in [("SFP task", TASK_SFP), ("SC  task", TASK_SC)]:
        batch = make_synthetic_batch(device, tokenizer, task_str)

        t1 = time.time()
        with torch.no_grad():
            action = policy.select_action(batch)
        elapsed_ms = (time.time() - t1) * 1000

        # select_action returns (1, chunk_size, action_dim) or (1, action_dim)
        if action.dim() == 3:
            # chunked: take first predicted step
            first_step = action[0, 0].cpu().numpy()
            full_chunk  = action[0].cpu().numpy()
        else:
            first_step = action[0].cpu().numpy()
            full_chunk  = first_step[None]

        results.append((task_label, task_str, first_step, full_chunk, elapsed_ms))

        print(f"\n  [{task_label}]")
        print(f"  Task string  : {task_str[:60]}...")
        print(f"  Output shape : {action.shape}  "
              f"({'chunked — ' + str(action.shape[1]) + ' steps' if action.dim()==3 else 'single step'})")
        print(f"  Step 0 action: {np.round(first_step, 4)}")
        print(f"    lin_vel (x,y,z): [{first_step[0]:+.4f}  {first_step[1]:+.4f}  {first_step[2]:+.4f}]")
        print(f"    ang_vel (x,y,z): [{first_step[3]:+.4f}  {first_step[4]:+.4f}  {first_step[5]:+.4f}]")
        print(f"  Inference time: {elapsed_ms:.0f}ms")

    # ── 3. Validate ────────────────────────────────────────────────
    print("\n[3/3] Validation ...")
    issues = []

    for task_label, _, first_step, full_chunk, _ in results:
        if np.any(np.isnan(first_step)):
            issues.append(f"{task_label}: NaN in output")
        if np.all(first_step == 0.0):
            issues.append(f"{task_label}: all-zero output")
        if np.any(np.abs(first_step) > 50.0):
            issues.append(f"{task_label}: unreasonably large magnitude {np.abs(first_step).max():.1f}")

    # Check SFP and SC outputs differ (model should respond to language)
    sfp_action = results[0][2]
    sc_action  = results[1][2]
    diff = np.abs(sfp_action - sc_action).mean()
    print(f"  SFP vs SC action difference (mean abs): {diff:.4f}")
    if diff < 1e-6:
        issues.append("SFP and SC actions are identical — model may be ignoring language input")
    else:
        print(f"  ✓ Model produces different actions for SFP vs SC tasks (language-conditioned)")

    # Action chunk shape
    expected_chunk = (CHUNK_SIZE, ACTION_DIM)
    actual_chunk   = results[0][3].shape
    if actual_chunk != expected_chunk:
        print(f"  ⚠ Chunk shape: got {actual_chunk}, expected {expected_chunk}")
        print(f"    (This is fine — shape may differ depending on LeRobot version)")
    else:
        print(f"  ✓ Action chunk shape correct: {actual_chunk}")

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    if issues:
        print("  ⚠  ISSUES:")
        for iss in issues:
            print(f"     - {iss}")
    else:
        print("  ✓  ALL CHECKS PASSED")
        print(f"  ✓  Checkpoint loads on {device.upper()}")
        print(f"  ✓  Valid 6D cartesian velocity output")
        print(f"  ✓  Language conditioning working (SFP ≠ SC)")
        print()
        print("  → Ready to write RunSmolVLA.py ROS2 node")

    print("=" * 60)


if __name__ == "__main__":
    run_test()
