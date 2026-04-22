"""
Convert collected HDF5 episodes to LeRobot dataset format for ACT training.

The action representation:
  - We compute delta TCP pose between consecutive frames (6D: dx,dy,dz + axis-angle drx,dry,drz)
  - This matches the 6D Cartesian twist action space used by the AIC LeRobot controller

Robot state (33D):
  - tcp_pose         7D  (x y z qx qy qz qw)
  - tcp_velocity     6D  (vx vy vz wx wy wz)  — Cartesian tool velocity
  - tcp_error        6D
  - joint_positions  7D
  - joint_velocity   7D  — per-joint velocity; tcp_velocity only tells you how the
                           tool moves in Cartesian space, not how each joint contributes

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
    Returns: (T, 6) delta actions [dx,dy,dz, drx,dry,drz]
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


def _decode_strings(raw_values) -> List[str]:
    decoded = []
    for item in raw_values:
        if isinstance(item, bytes):
            decoded.append(item.decode("utf-8"))
        else:
            decoded.append(str(item))
    return decoded


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def _write_video(frames: np.ndarray, path: Path, fps: int = 20) -> None:
    """Write (T, H, W, 3) uint8 RGB frames as an MP4 file."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python is required for video export")

    T, H, W, _ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (W, H))
    for t in range(T):
        writer.write(cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR))
    writer.release()


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

def convert(input_dir: str, output_dir: str, success_only: bool, max_final_error: float = 0.02) -> None:
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

    # Video output dirs (one sub-folder per camera)
    CAMERAS = ("left", "center", "right")
    video_dirs = {}
    for cam in CAMERAS:
        vd = out / "videos" / "chunk-000" / f"observation.images.{cam}"
        vd.mkdir(parents=True, exist_ok=True)
        video_dirs[cam] = vd

    all_states:  List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    episode_meta = []
    lerobot_idx  = 0
    total_frames = 0
    privileged_tf_frame_pairs: List[str] = []
    video_shape: dict = {}   # filled from first episode with images

    for ep_file in episode_files:
        with h5py.File(ep_file, "r") as hf:
            meta    = hf["metadata"]
            success = bool(meta.attrs.get("success", 0))
            final_error = float(meta.attrs.get("final_error", float("nan")))
            if success_only and not success:
                print(f"  skip {ep_file.name} (failed)")
                continue
            if not (final_error <= max_final_error):
                print(f"  skip {ep_file.name} (final_error={final_error:.3f}m > {max_final_error}m)")
                continue

            tcp_poses   = hf["observations/tcp_pose"][:]        # (T,7)
            tcp_vels    = hf["observations/tcp_velocity"][:]    # (T,6)
            tcp_errors  = hf["observations/tcp_error"][:]       # (T,6)
            joint_pos   = hf["observations/joint_positions"][:] # (T,7)
            joint_vel   = (                                      # (T,7) — per-joint velocity
                hf["observations/joint_velocity"][:]
                if "observations/joint_velocity" in hf
                else np.zeros_like(joint_pos)                   # back-compat for schema v3 files
            )
            T = tcp_poses.shape[0]

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

            if "observations/privileged_tf/transforms" in hf:
                privileged_tf = hf["observations/privileged_tf/transforms"][:]
                tf_count = privileged_tf.shape[1]
                privileged_tf_valid = (
                    hf["observations/privileged_tf/valid"][:].astype(bool)
                    if "observations/privileged_tf/valid" in hf
                    else np.ones((T, tf_count), dtype=bool)
                )
                ep_tf_frame_pairs = (
                    _decode_strings(hf["observations/privileged_tf/frame_pairs"][:])
                    if "observations/privileged_tf/frame_pairs" in hf
                    else [f"tf_{idx}" for idx in range(tf_count)]
                )
                if ep_tf_frame_pairs:
                    if not privileged_tf_frame_pairs:
                        privileged_tf_frame_pairs = list(ep_tf_frame_pairs)
                    elif ep_tf_frame_pairs != privileged_tf_frame_pairs:
                        print(
                            f"  warn {ep_file.name}: privileged TF frame pairs differ "
                            "from earlier episodes"
                        )
            else:
                privileged_tf = np.zeros((T, 0, 7), dtype=np.float32)
                privileged_tf_valid = np.zeros((T, 0), dtype=bool)
                ep_tf_frame_pairs = []

            # Robot state: 7+6+6+7+7 = 33 dims
            # joint_velocity added so the policy can see per-joint motion,
            # not just the aggregate Cartesian tcp_velocity.
            state = np.concatenate([tcp_poses, tcp_vels, tcp_errors, joint_pos, joint_vel], axis=1)

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

            # --- Write per-camera MP4 videos ---
            for cam in CAMERAS:
                img_key = f"observations/images/{cam}"
                if img_key in hf:
                    images = hf[img_key][:]   # (T, H, W, 3) uint8 RGB
                    if not video_shape:
                        _, H, W, _ = images.shape
                        video_shape = {"height": H, "width": W}
                    vid_path = video_dirs[cam] / f"episode_{lerobot_idx:06d}.mp4"
                    _write_video(images, vid_path)

            all_states.append(state)
            all_actions.append(action)

            rows = []
            for t in range(T):
                row = {
                    "index":          total_frames + t,
                    "episode_index":  lerobot_idx,
                    "frame_index":    t,
                    "timestamp":      t * 0.05,  # 20 Hz → 50 ms
                    "next.done":      int(t == T - 1),
                    "observation.state":  state[t].tolist(),
                    "observation.task_id": task_ids[t],
                    "observation.relative_pose": relative_pose[t].tolist(),
                    "observation.relative_pose.valid": bool(relative_valid[t]),
                    "observation.privileged_tf.transforms": privileged_tf[t].tolist(),
                    "observation.privileged_tf.valid": privileged_tf_valid[t].astype(bool).tolist(),
                    "action":             action[t].tolist(),
                    "action.velocity":    velocity_action[t].tolist(),
                    "task_index":         0,
                }
                rows.append(row)

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
                "privileged_tf_frame_pairs": ep_tf_frame_pairs,
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
    info: dict = {
        "codebase_version": "v2.1",
        "robot_type": "aic_ur5e",
        "total_episodes": lerobot_idx,
        "total_frames": total_frames,
        "fps": 20,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [33]},
            "observation.task_id": {"dtype": "string", "shape": [1]},
            "observation.relative_pose": {"dtype": "float32", "shape": [7]},
            "observation.relative_pose.valid": {"dtype": "bool", "shape": [1]},
            "action":            {"dtype": "float32", "shape": [6]},
            "action.velocity":   {"dtype": "float32", "shape": [6]},
        },
        "privileged_tf_frame_pairs": privileged_tf_frame_pairs,
        "splits": {"train": f"0:{lerobot_idx}"},
    }

    # Video features
    if video_shape:
        H = video_shape["height"]
        W = video_shape["width"]
        for cam in CAMERAS:
            info["features"][f"observation.images.{cam}"] = {
                "dtype": "video",
                "shape": [H, W, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": 20.0,
                    "video.codec": "mp4v",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }

    if privileged_tf_frame_pairs:
        info["features"]["observation.privileged_tf.transforms"] = {
            "dtype": "float32",
            "shape": [len(privileged_tf_frame_pairs), 7],
        }
        info["features"]["observation.privileged_tf.valid"] = {
            "dtype": "bool",
            "shape": [len(privileged_tf_frame_pairs)],
        }
    with open(out / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    has_video = bool(video_shape)
    print(f"\nDone — {lerobot_idx} episodes, {total_frames} frames → {output_dir}")
    print(f"  state: 33D (tcp_pose 7 + tcp_vel 6 + tcp_err 6 + joint_pos 7 + joint_vel 7)")
    print(f"  videos: {'written' if has_video else 'not found in HDF5'}")
    print("Next step: pixi run lerobot-train --dataset.repo_id=local/aic_pilot ...")


def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 episodes to LeRobot format")
    parser.add_argument("--input",        required=True, help="Directory with episode_*.hdf5 files")
    parser.add_argument("--output",       required=True, help="Output dataset directory")
    parser.add_argument("--success_only", action="store_true", default=True,
                        help="Only include successful episodes (default: True)")
    parser.add_argument("--max_final_error", type=float, default=0.02,
                        help="Max plug-to-port distance (m) at episode end (default: 0.02)")
    args = parser.parse_args()
    convert(args.input, args.output, args.success_only, args.max_final_error)


if __name__ == "__main__":
    main()
