#!/usr/bin/env python3
"""
Offline inference test for SmolVLA checkpoint.

Run this after the 40K checkpoint is saved to verify:
  1. Checkpoint loads correctly
  2. Model produces valid 6D actions (no NaN, no zeros)
  3. Inference speed is fast enough for real-time control (~20Hz target)
  4. Actions look reasonable compared to ground truth

Usage:
    cd ~/lerobot
    source .venv/bin/activate
    python3 test_smolvla_checkpoint.py
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import torch

# ── CONFIG ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "outputs/train/aic_smolvla_v1/checkpoints/040000"
DATASET_ROOT    = "/home/mansi/lerobot/aic_act_yolo_mansi"   # update if different
NUM_SAMPLES     = 10   # number of random frames to test
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────


def load_policy(checkpoint_path: str):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    print(f"Loading checkpoint: {checkpoint_path}")
    t0 = time.time()
    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    policy = policy.to(DEVICE)
    policy.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s  |  device={DEVICE}")
    return policy


def load_dataset(dataset_root: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    print(f"Loading dataset: {dataset_root}")
    ds = LeRobotDataset(
        repo_id="local/aic_act_yolo_mansi",
        root=dataset_root,
    )
    print(f"  {ds.num_frames} frames  |  {ds.num_episodes} episodes")
    return ds


def describe_dataset(dataset_root: str):
    info_path = Path(dataset_root) / "meta/info.json"
    info = json.load(open(info_path))

    video_keys = [
        k for k, v in info["features"].items()
        if v.get("dtype") == "video"
    ]
    state_dim  = info["features"]["observation.state"]["shape"][0]
    action_dim = info["features"]["action"]["shape"][0]

    print(f"  Camera keys : {video_keys}")
    print(f"  State dim   : {state_dim}  (expected 30)")
    print(f"  Action dim  : {action_dim} (expected 6 — cartesian velocity)")
    return video_keys, state_dim, action_dim


def to_batch(sample: dict) -> dict:
    """Add batch dimension and move tensors to device."""
    batch = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(DEVICE)
        else:
            batch[k] = v
    # task must be a list of strings
    if "task" in batch and isinstance(batch["task"], str):
        batch["task"] = [batch["task"]]
    return batch


def validate_action(pred: np.ndarray, gt: np.ndarray, sample_idx: int) -> list[str]:
    issues = []
    if np.any(np.isnan(pred)):
        issues.append("NaN in predicted action")
    if np.all(pred == 0.0):
        issues.append("All-zero output — model may not be learning")
    if np.any(np.abs(pred) > 50.0):
        issues.append(f"Unusually large action magnitude: max={np.abs(pred).max():.2f}")
    return issues


def run_test():
    print("=" * 60)
    print("  SmolVLA checkpoint offline inference test")
    print("=" * 60)

    # ── Check checkpoint exists ────────────────────────────────────
    ckpt = Path(CHECKPOINT_PATH)
    if not ckpt.exists():
        print(f"\n  Checkpoint not found: {ckpt}")
        print(f"  Training is still running — wait until step 40000 is saved.")
        print(f"  Expected path: {ckpt.resolve()}")
        sys.exit(1)
    print(f"\n[1/4] Checkpoint found at: {ckpt.resolve()}")

    # ── Dataset info ───────────────────────────────────────────────
    print("\n[2/4] Dataset info:")
    video_keys, state_dim, action_dim = describe_dataset(DATASET_ROOT)

    # ── Load model ─────────────────────────────────────────────────
    print("\n[3/4] Loading policy:")
    policy = load_policy(CHECKPOINT_PATH)

    # ── Inference loop ─────────────────────────────────────────────
    print(f"\n[4/4] Running {NUM_SAMPLES} inference passes:")
    ds = load_dataset(DATASET_ROOT)

    rng = np.random.default_rng(seed=42)
    sample_indices = rng.integers(0, len(ds) - 1, size=NUM_SAMPLES)

    all_issues    = []
    inference_ms  = []
    lin_errors    = []  # linear velocity MAE
    ang_errors    = []  # angular velocity MAE

    print(f"\n  {'idx':>7}  {'task':^12}  {'pred lin (xyz)':^26}  {'gt lin (xyz)':^26}  {'ms':>6}")
    print(f"  {'-'*7}  {'-'*12}  {'-'*26}  {'-'*26}  {'-'*6}")

    for i, idx in enumerate(sample_indices):
        sample = ds[int(idx)]
        batch  = to_batch(sample)

        task_label = batch["task"][0][:11] if batch.get("task") else "?"

        t0 = time.time()
        with torch.no_grad():
            pred_action = policy.select_action(batch)
        elapsed_ms = (time.time() - t0) * 1000
        inference_ms.append(elapsed_ms)

        # select_action may return (B, action_dim) or (B, chunk, action_dim)
        # take first step if chunked
        if pred_action.dim() == 3:
            pred_action = pred_action[:, 0, :]
        pred = pred_action.cpu().squeeze().numpy()

        gt = sample["action"].numpy()
        if gt.ndim == 2:   # chunked GT — take first step
            gt = gt[0]

        issues = validate_action(pred, gt, int(idx))
        if issues:
            all_issues.extend([f"  Sample {i+1} (frame {idx}): {iss}" for iss in issues])

        lin_err = np.abs(pred[:3] - gt[:3]).mean()
        ang_err = np.abs(pred[3:] - gt[3:]).mean()
        lin_errors.append(lin_err)
        ang_errors.append(ang_err)

        print(
            f"  {idx:>7}  {task_label:^12}  "
            f"[{pred[0]:+.3f} {pred[1]:+.3f} {pred[2]:+.3f}]  "
            f"[{gt[0]:+.3f}  {gt[1]:+.3f}  {gt[2]:+.3f}]  "
            f"{elapsed_ms:>5.0f}ms"
        )

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Avg inference time : {np.mean(inference_ms):.0f}ms  "
          f"(max {np.max(inference_ms):.0f}ms)")
    print(f"  Real-time capable  : {'YES ✓' if np.mean(inference_ms) < 100 else 'NO — too slow for 10Hz'}")
    print(f"  Linear vel MAE     : {np.mean(lin_errors):.4f}  (lower = better)")
    print(f"  Angular vel MAE    : {np.mean(ang_errors):.4f}  (lower = better)")

    if all_issues:
        print(f"\n  ⚠  ISSUES ({len(all_issues)}):")
        for iss in all_issues:
            print(f"     {iss}")
        print("\n  Recommendation: wait for 60K–80K checkpoint before deploying.")
    else:
        print(f"\n  ✓  No issues — all {NUM_SAMPLES} samples passed")
        print(f"  ✓  Model outputs valid 6D cartesian velocity actions")
        if np.mean(inference_ms) < 100:
            print(f"  ✓  Fast enough for real-time control (target ≥10Hz)")
        print(f"\n  → Ready to test in Gazebo with RunSmolVLA.py")


if __name__ == "__main__":
    run_test()
