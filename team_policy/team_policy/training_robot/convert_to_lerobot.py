"""
Convert collected HDF5 episodes to LeRobot dataset format for ACT training.

The action representation:
  - We compute delta TCP pose between consecutive frames (6D: dx,dy,dz + axis-angle drx,dry,drz)
  - This matches the 6D Cartesian twist action space that RunACT.py expects

Usage:
    cd ~/ros2_ws/src/aic
    pixi run python -m team_policy.training_robot.convert_to_lerobot \\
        --input /tmp/aic_dataset \\
        --output ./datasets/aic_pilot \\
        --success_only

Output structure (LeRobot HuggingFace format):
    datasets/aic_pilot/
        meta/
            info.json
            episodes.jsonl
            tasks.jsonl
            stats.json
        data/chunk-000/
            episode_000000.parquet
            ...
        videos/chunk-000/
            observation.images.left/episode_000000.mp4
            observation.images.center/episode_000000.mp4
            observation.images.right/episode_000000.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import List

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [qx,qy,qz,qw] to axis-angle [ax,ay,az]."""
    x, y, z, w = q
    sin_half = math.sqrt(x*x + y*y + z*z)
    if sin_half < 1e-9:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.atan2(sin_half, max(abs(w), 1e-12))
    if w < 0:
        angle = -angle
    return np.array([x/sin_half * angle, y/sin_half * angle, z/sin_half * angle], dtype=np.float32)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two [qx,qy,qz,qw] quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ], dtype=np.float32)


def quat_inverse(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([-x, -y, -z, w], dtype=np.float32)


def compute_delta_actions(tcp_poses: np.ndarray) -> np.ndarray:
    """
    tcp_poses: (T, 7) — each row is [x,y,z,qx,qy,qz,qw]
    Returns: (T-1, 6) delta actions [dx,dy,dz, drx,dry,drz]
    The last action is duplicated so shape stays (T, 6).
    """
    T = tcp_poses.shape[0]
    deltas = np.zeros((T, 6), dtype=np.float32)

    for i in range(T - 1):
        p_now  = tcp_poses[i, :3]
        p_next = tcp_poses[i+1, :3]
        q_now  = tcp_poses[i, 3:]
        q_next = tcp_poses[i+1, 3:]

        dp = p_next - p_now
        dq = quat_multiply(q_next, quat_inverse(q_now))
        dr = quat_to_axis_angle(dq)

        deltas[i] = np.concatenate([dp, dr])

    deltas[-1] = deltas[-2]  # duplicate last to keep T frames
    return deltas


def _decode_task_ids(raw_task_ids, fallback: str, count: int) -> List[str]:
    if raw_task_ids is None:
        return [fallback] * count
    decoded = []
    for item in raw_task_ids:
        if isinstance(item, bytes):
            decoded.append(item.decode("utf-8"))
        else:
            decoded.append(str(item))
    return decoded


# ---------------------------------------------------------------------------
# Dataset stats
# ---------------------------------------------------------------------------

def compute_stats(all_states: List[np.ndarray], all_actions: List[np.ndarray]):
    states  = np.concatenate(all_states,  axis=0)
    actions = np.concatenate(all_actions, axis=0)
    return {
        "observation.state": {
            "mean": states.mean(axis=0).tolist(),
            "std":  (states.std(axis=0) + 1e-6).tolist(),
            "min":  states.min(axis=0).tolist(),
            "max":  states.max(axis=0).tolist(),
        },
        "action": {
            "mean": actions.mean(axis=0).tolist(),
            "std":  (actions.std(axis=0) + 1e-6).tolist(),
            "min":  actions.min(axis=0).tolist(),
            "max":  actions.max(axis=0).tolist(),
        },
    }


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(input_dir: str, output_dir: str, success_only: bool) -> None:
    try:
        import h5py
        import pandas as pd
    except ImportError:
        raise RuntimeError("h5py and pandas are required: pixi add --pypi h5py pandas")

    episode_files = sorted(Path(input_dir).glob("episode_*.hdf5"))
    if not episode_files:
        raise FileNotFoundError(f"No HDF5 files found in {input_dir}")

    print(f"Found {len(episode_files)} episode files")

    out = Path(output_dir)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    data_dir  = out / "data"  / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)

    all_states:  List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    episode_meta = []
    lerobot_idx  = 0
    total_frames = 0

    for ep_file in episode_files:
        with h5py.File(ep_file, "r") as hf:
            meta    = hf["metadata"]
            success = bool(meta.attrs.get("success", 0))
            if success_only and not success:
                print(f"  skip {ep_file.name} (failed)")
                continue

            tcp_poses     = hf["observations/tcp_pose"][:]        # (T,7)
            tcp_vels      = hf["observations/tcp_velocity"][:]    # (T,6)
            tcp_errors    = hf["observations/tcp_error"][:]       # (T,6)
            joint_pos     = hf["observations/joint_positions"][:] # (T,7)
            wrist_force   = hf["observations/wrist_force"][:]     # (T,6)
            T = tcp_poses.shape[0]
            del wrist_force

            task_id = str(meta.attrs.get("task_id", "insert_cable"))
            task_ids = _decode_task_ids(
                hf["observations/task_id"][:] if "observations/task_id" in hf else None,
                fallback=task_id,
                count=T,
            )
            if "observations/relative_pose" in hf:
                relative_pose = hf["observations/relative_pose"][:]
                relative_valid = (
                    hf["observations/relative_pose_valid"][:].astype(bool)
                    if "observations/relative_pose_valid" in hf
                    else np.ones(T, dtype=bool)
                )
            else:
                relative_pose = np.zeros((T, 7), dtype=np.float32)
                relative_valid = np.zeros(T, dtype=bool)

            # Robot state: 7+6+6+7 = 26 dims (matches RunACT)
            state = np.concatenate([tcp_poses, tcp_vels, tcp_errors, joint_pos], axis=1)

            # Prefer expert labels recorded by the collector. Older files fall back to TCP deltas.
            if "actions/delta_pose" in hf:
                action = hf["actions/delta_pose"][:]
            else:
                action = compute_delta_actions(tcp_poses)
            velocity_action = (
                hf["actions/velocity"][:]
                if "actions/velocity" in hf
                else np.zeros_like(action)
            )

            all_states.append(state)
            all_actions.append(action)

            rows = []
            for t in range(T):
                rows.append({
                    "index":          total_frames + t,
                    "episode_index":  lerobot_idx,
                    "frame_index":    t,
                    "timestamp":      t * 0.05,  # 20 Hz → 50ms
                    "next.done":      int(t == T - 1),
                    "observation.state":  state[t].tolist(),
                    "observation.task_id": task_ids[t],
                    "observation.relative_pose": relative_pose[t].tolist(),
                    "observation.relative_pose.valid": bool(relative_valid[t]),
                    "action":             action[t].tolist(),
                    "action.velocity":    velocity_action[t].tolist(),
                    "task_index":         0,
                })

            df = pd.DataFrame(rows)
            df.to_parquet(data_dir / f"episode_{lerobot_idx:06d}.parquet", index=False)

            episode_meta.append({
                "episode_index": lerobot_idx,
                "tasks":         [f"insert {meta.attrs.get('port_type','cable')} cable"],
                "length":        T,
                "port_type":     str(meta.attrs.get("port_type", "")),
                "port_name":     str(meta.attrs.get("port_name", "")),
                "success":       success,
                "task_id":       task_id,
                "max_force":     float(meta.attrs.get("max_force", float("nan"))),
                "final_error":   float(meta.attrs.get("final_error", float("nan"))),
                "insertion_time": float(meta.attrs.get("insertion_time", float("nan"))),
                "contact_duration": float(meta.attrs.get("contact_duration", float("nan"))),
            })

            total_frames += T
            print(f"  converted {ep_file.name} → episode {lerobot_idx} ({T} frames)")
            lerobot_idx += 1

    if lerobot_idx == 0:
        print("No episodes passed the filter — nothing converted.")
        return

    # Write stats
    stats = compute_stats(all_states, all_actions)
    with open(out / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Write episode index
    with open(out / "meta" / "episodes.jsonl", "w") as f:
        for ep in episode_meta:
            f.write(json.dumps(ep) + "\n")

    # Write tasks
    with open(out / "meta" / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": "insert cable"}) + "\n")

    # Write dataset info
    info = {
        "codebase_version": "v2.1",
        "robot_type": "aic_ur5e",
        "total_episodes": lerobot_idx,
        "total_frames": total_frames,
        "fps": 20,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [26]},
            "observation.task_id": {"dtype": "string", "shape": [1]},
            "observation.relative_pose": {"dtype": "float32", "shape": [7]},
            "observation.relative_pose.valid": {"dtype": "bool", "shape": [1]},
            "action":            {"dtype": "float32", "shape": [6]},
            "action.velocity":   {"dtype": "float32", "shape": [6]},
        },
        "splits": {"train": f"0:{lerobot_idx}"},
    }
    with open(out / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nDone — {lerobot_idx} episodes, {total_frames} frames → {output_dir}")
    print("Next step: pixi run lerobot-train --dataset.repo_id=local/aic_pilot ...")


def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 episodes to LeRobot format")
    parser.add_argument("--input",        required=True, help="Directory with episode_*.hdf5 files")
    parser.add_argument("--output",       required=True, help="Output dataset directory")
    parser.add_argument("--success_only", action="store_true", default=True,
                        help="Only include successful episodes (default: True)")
    args = parser.parse_args()
    convert(args.input, args.output, args.success_only)


if __name__ == "__main__":
    main()
