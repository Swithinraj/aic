"""
Convert collected HDF5 episodes to LeRobot v3.0 dataset format for ACT training.

The action representation:
  - We compute delta TCP pose between consecutive frames (6D: dx,dy,dz + axis-angle drx,dry,drz)
  - This matches the 6D Cartesian twist action space used by the AIC LeRobot controller

Robot state (27D):
  - tcp_pose         7D  (x y z qx qy qz qw)
  - tcp_velocity     6D  (vx vy vz wx wy wz)  — Cartesian tool velocity
  - joint_positions  7D
  - joint_velocity   7D  — per-joint velocity; tcp_velocity only tells you how the
                           tool moves in Cartesian space, not how each joint contributes
  (tcp_error excluded — it is CheatCode-privileged at training time, near-zero at inference)

Usage:
    cd ~/ros2_ws/src/aic
    pixi run python -m team_policy.training_robot.convert_to_lerobot \\
        --input /tmp/aic_dataset \\
        --output ./datasets/aic_pilot \\
        --success_only

Output structure (LeRobot v3.0 format):
    datasets/aic_pilot/
        meta/
            info.json
            stats.json
            tasks.parquet
            episodes/
                chunk-000/
                    file-000.parquet
        data/
            chunk-000/
                file-000.parquet          (all episodes merged)
        videos/
            observation.images.left/
                chunk-000/
                    file-000.mp4          (all episodes concatenated)
            observation.images.center/...
            observation.images.right/...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

CODEBASE_VERSION = "v3.0"
DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
DEFAULT_EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_TASKS_PATH = "meta/tasks.parquet"


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

    deltas[-1] = deltas[-2]
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

def _write_video(frames: np.ndarray, path: Path, fps: int = 10,
                 target_hw: tuple[int, int] | None = None) -> None:
    """Write (T, H, W, 3) uint8 RGB frames as an MP4 file.

    target_hw: optional (height, width) to resize frames before writing.
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python is required for video export")

    T, H, W, _ = frames.shape
    out_h, out_w = (target_hw[0], target_hw[1]) if target_hw else (H, W)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (out_w, out_h))
    for t in range(T):
        bgr = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
        if target_hw:
            bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(bgr)
    writer.release()


def _concat_videos_ffmpeg(input_paths: List[Path], output_path: Path) -> bool:
    """Concatenate MP4 files using ffmpeg. Returns True on success."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        list_file = f.name
        for p in input_paths:
            f.write(f"file '{p.resolve()}'\n")

    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
             "-c", "copy", str(output_path)],
            capture_output=True,
            timeout=300,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    finally:
        os.unlink(list_file)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _find_port_xyz_in_base(ep_tf_frame_pairs: List[str],
                           privileged_tf: np.ndarray,
                           privileged_tf_valid: np.ndarray,
                           T: int) -> np.ndarray:
    """Return (T, 3) port position in base_link from privileged_tf.

    Searches frame_pairs for the entry 'base_link<-*port*' and extracts xyz.
    Falls back to zeros if not found (policy sees unknown port position).
    """
    port_idx = None
    for i, fp in enumerate(ep_tf_frame_pairs):
        if "base_link<-" in fp and "port" in fp.lower():
            port_idx = i
            break
    if port_idx is None or privileged_tf.shape[1] == 0:
        return np.zeros((T, 3), dtype=np.float32)
    xyz = privileged_tf[:, port_idx, :3].astype(np.float32)
    if privileged_tf_valid.shape[1] > port_idx:
        xyz[~privileged_tf_valid[:, port_idx]] = 0.0
    return xyz


def compute_stats(all_states: List[np.ndarray], all_actions: List[np.ndarray]) -> dict:
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

def convert(input_dir: str, output_dir: str, success_only: bool,
            max_final_error: float = 0.02,
            image_height: int = 480, image_width: int = 640,
            target_hz: float = 10.0) -> None:
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

    CAMERAS = ("left", "center", "right")
    VIDEO_KEYS = [f"observation.images.{cam}" for cam in CAMERAS]

    all_states:  List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    all_rows:    List[list] = []          # list of per-episode row lists
    episode_meta_rows = []                # for episodes parquet
    lerobot_idx  = 0
    total_frames = 0
    privileged_tf_frame_pairs: List[str] = []
    video_shape: dict = {}

    target_hw = (image_height, image_width) if (image_height and image_width) else None
    print(f"Image resize: {'%dx%d (HxW)' % (image_height, image_width) if target_hw else 'original size'}")

    # Per-camera temp video paths (written per episode, concatenated at the end)
    tmp_video_paths: dict[str, List[Path]] = {vk: [] for vk in VIDEO_KEYS}

    for ep_file in episode_files:
        with h5py.File(ep_file, "r") as hf:
            meta    = hf["metadata"]
            success = bool(meta.attrs.get("success", 0))
            if success_only and not success:
                print(f"  skip {ep_file.name} (failed)")
                continue
            # final_error can be NaN when relative_pose was never valid — treat as passing.
            final_error_raw = meta.attrs.get("final_error", None)
            if final_error_raw is not None:
                final_error = float(final_error_raw)
                if not math.isnan(final_error) and final_error > max_final_error:
                    print(f"  skip {ep_file.name} (final_error={final_error:.3f}m > {max_final_error}m)")
                    continue

            T_full = int(meta.attrs.get("num_frames", hf["observations/tcp_pose"].shape[0]))
            ts_full = hf["observations/timestamps"][:]

            if T_full > 1:
                median_dt = float(np.median(np.diff(ts_full)))
                actual_hz = 1.0 / median_dt if median_dt > 0 else target_hz
            else:
                actual_hz = target_hz

            step = max(1, round(actual_hz / target_hz))
            indices = np.arange(0, T_full, step)

            tcp_poses  = hf["observations/tcp_pose"][:][indices]
            tcp_vels   = hf["observations/tcp_velocity"][:][indices]
            joint_pos  = hf["observations/joint_positions"][:][indices]
            joint_vel  = (
                hf["observations/joint_velocity"][:][indices]
                if "observations/joint_velocity" in hf
                else np.zeros((len(indices), 7), dtype=np.float32)
            )
            T = tcp_poses.shape[0]

            task_id = str(meta.attrs.get("task_id", "insert_cable"))
            task_ids = _decode_task_ids(
                hf["observations/task_id"][:][indices] if "observations/task_id" in hf else None,
                fallback=task_id,
                count=T,
            )
            if "observations/relative_pose" in hf:
                relative_pose = hf["observations/relative_pose"][:][indices]
                relative_valid = (
                    hf["observations/relative_pose_valid"][:][indices].astype(bool)
                    if "observations/relative_pose_valid" in hf
                    else np.ones(T, dtype=bool)
                )
            else:
                relative_pose = np.zeros((T, 7), dtype=np.float32)
                relative_valid = np.zeros(T, dtype=bool)

            if "observations/privileged_tf/transforms" in hf:
                privileged_tf = hf["observations/privileged_tf/transforms"][:][indices]
                tf_count = privileged_tf.shape[1]
                privileged_tf_valid = (
                    hf["observations/privileged_tf/valid"][:][indices].astype(bool)
                    if "observations/privileged_tf/valid" in hf
                    else np.ones((T, tf_count), dtype=bool)
                )
                ep_tf_frame_pairs = (
                    _decode_strings(hf["observations/privileged_tf/frame_pairs"][:])
                    if "observations/privileged_tf/frame_pairs" in hf
                    else [f"tf_{idx}" for idx in range(tf_count)]
                )
                if ep_tf_frame_pairs and not privileged_tf_frame_pairs:
                    privileged_tf_frame_pairs = list(ep_tf_frame_pairs)
            else:
                privileged_tf = np.zeros((T, 0, 7), dtype=np.float32)
                privileged_tf_valid = np.zeros((T, 0), dtype=bool)
                ep_tf_frame_pairs = []

            # Port xyz: prefer real YOLO detections; fall back to GT+noise for old episodes
            if "observations/yolo_port_xyz" in hf:
                yolo_xyz   = hf["observations/yolo_port_xyz"][:][indices].astype(np.float32)
                yolo_valid = (
                    hf["observations/yolo_port_valid"][:][indices].astype(bool)
                    if "observations/yolo_port_valid" in hf
                    else np.ones(T, dtype=bool)
                )
                gt_xyz  = _find_port_xyz_in_base(ep_tf_frame_pairs, privileged_tf, privileged_tf_valid, T)
                # Where YOLO didn't detect, use GT + 2cm noise to simulate YOLO accuracy
                noise   = np.random.normal(0.0, 0.02, gt_xyz.shape).astype(np.float32)
                port_xyz = np.where(yolo_valid[:, None], yolo_xyz, gt_xyz + noise)
                yolo_coverage = float(yolo_valid.mean()) * 100
                print(f"    yolo_port_xyz: {yolo_coverage:.0f}% frames detected by YOLO")
            else:
                # Old episode — use GT + noise to simulate YOLO
                gt_xyz   = _find_port_xyz_in_base(ep_tf_frame_pairs, privileged_tf, privileged_tf_valid, T)
                noise    = np.random.normal(0.0, 0.02, gt_xyz.shape).astype(np.float32)
                port_xyz = (gt_xyz + noise).astype(np.float32)
                print(f"    yolo_port_xyz: no YOLO recorded — using GT+noise fallback")
            state = np.concatenate([tcp_poses, tcp_vels, joint_pos, joint_vel, port_xyz], axis=1)
            # state: tcp_pose(7)+tcp_vel(6)+jpos(7)+jvel(7)+port_xyz(3) = 30D

            if "actions/delta_pose" in hf:
                action = hf["actions/delta_pose"][:][indices]
            else:
                action = compute_delta_actions(tcp_poses)
            velocity_action = (
                hf["actions/velocity"][:][indices]
                if "actions/velocity" in hf
                else np.zeros_like(action)
            )

            # --- Write per-episode per-camera temp videos ---
            for cam in CAMERAS:
                img_key = f"observations/images/{cam}"
                vk = f"observation.images.{cam}"
                if img_key in hf:
                    images = hf[img_key][:][indices]
                    if not video_shape:
                        _, H, W, _ = images.shape
                        video_shape = {
                            "height": target_hw[0] if target_hw else H,
                            "width":  target_hw[1] if target_hw else W,
                        }
                    tmp_dir = out / "_tmp_videos" / vk
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    tmp_path = tmp_dir / f"episode_{lerobot_idx:06d}.mp4"
                    _write_video(images, tmp_path, target_hw=target_hw)
                    tmp_video_paths[vk].append(tmp_path)

            all_states.append(state)
            all_actions.append(action)

            ep_start = total_frames
            ep_end   = total_frames + T

            # Collect per-episode arrays (numpy, explicit dtypes for correct parquet schema).
            # Only include features needed for ACT training — string and privileged fields
            # cannot be tensorized by the lerobot normalizer and must be excluded.
            ep_data = {
                "index":             np.arange(ep_start, ep_end, dtype=np.int64),
                "episode_index":     np.full(T, lerobot_idx, dtype=np.int64),
                "frame_index":       np.arange(T, dtype=np.int64),
                "timestamp":         (np.arange(T, dtype=np.float32) * 0.10),
                "task_index":        np.zeros(T, dtype=np.int64),
                "observation.state": state.astype(np.float32),
                "action":            action.astype(np.float32),
            }
            all_rows.append(ep_data)

            episode_meta_rows.append({
                "episode_index":           lerobot_idx,
                "tasks":                   [f"insert {meta.attrs.get('port_type','cable')} cable"],
                "length":                  T,
                "data/chunk_index":        0,
                "data/file_index":         0,
                "dataset_from_index":      ep_start,
                "dataset_to_index":        ep_end,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index":  0,
                # video timestamps (all episodes in file-000, sequential timestamps)
                **{f"videos/{vk}/chunk_index": 0 for vk in VIDEO_KEYS},
                **{f"videos/{vk}/file_index": 0 for vk in VIDEO_KEYS},
                **{f"videos/{vk}/from_timestamp": ep_start * 0.10 for vk in VIDEO_KEYS},
                **{f"videos/{vk}/to_timestamp": ep_end * 0.10 for vk in VIDEO_KEYS},
            })

            total_frames += T
            print(f"  converted {ep_file.name} → episode {lerobot_idx} ({T} frames)")
            lerobot_idx += 1

    if lerobot_idx == 0:
        print("No episodes passed the filter — nothing converted.")
        return

    # --- Write merged data parquet (using datasets for correct dtypes) ---
    import datasets as hf_datasets
    from lerobot.datasets.feature_utils import get_hf_features_from_features
    from lerobot.datasets.utils import DEFAULT_FEATURES

    # Build features spec (ACT training features + standard frame metadata).
    # Exclude string and privileged fields — they cannot be tensorized by the normalizer.
    data_features_spec = {
        "observation.state": {"dtype": "float32", "shape": (30,)},
        "action":            {"dtype": "float32", "shape": (6,)},
    }
    all_features_spec = {**data_features_spec, **DEFAULT_FEATURES}
    hf_features = get_hf_features_from_features(all_features_spec)

    # Concatenate per-episode arrays
    merged: dict = {}
    for key in all_rows[0]:
        if key == "observation.task_id":
            merged[key] = [tid for ep in all_rows for tid in ep[key]]
        else:
            merged[key] = np.concatenate([ep[key] for ep in all_rows], axis=0)

    ds = hf_datasets.Dataset.from_dict(merged, features=hf_features)

    data_path = out / DEFAULT_DATA_PATH.format(chunk_index=0, file_index=0)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(data_path))

    # --- Concatenate videos ---
    has_video = bool(video_shape)
    if has_video:
        for vk in VIDEO_KEYS:
            if not tmp_video_paths[vk]:
                continue
            vid_out = out / DEFAULT_VIDEO_PATH.format(video_key=vk, chunk_index=0, file_index=0)
            vid_out.parent.mkdir(parents=True, exist_ok=True)
            if len(tmp_video_paths[vk]) == 1:
                import shutil
                shutil.copy(tmp_video_paths[vk][0], vid_out)
            else:
                ok = _concat_videos_ffmpeg(tmp_video_paths[vk], vid_out)
                if not ok:
                    print(f"  warn: ffmpeg concatenation failed for {vk}, copying last episode only")
                    import shutil
                    shutil.copy(tmp_video_paths[vk][-1], vid_out)

    # Clean up temp videos
    import shutil as _shutil
    tmp_dir = out / "_tmp_videos"
    if tmp_dir.exists():
        _shutil.rmtree(tmp_dir)

    # --- Write stats ---
    stats = compute_stats(all_states, all_actions)
    # Add placeholder image stats (ImageNet values) for video keys.
    # lerobot's factory.py overwrites these with IMAGENET_STATS when
    # use_imagenet_stats=True, but the key must exist first.
    if has_video:
        imagenet_mean = [[[0.485]], [[0.456]], [[0.406]]]
        imagenet_std  = [[[0.229]], [[0.224]], [[0.225]]]
        for vk in VIDEO_KEYS:
            stats[vk] = {
                "mean": imagenet_mean,
                "std":  imagenet_std,
                "min":  [[0.0], [0.0], [0.0]],
                "max":  [[1.0], [1.0], [1.0]],
            }
    with open(out / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # --- Write tasks.parquet ---
    tasks_path = out / DEFAULT_TASKS_PATH
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"task_index": [0], "task": ["insert cable"]}).to_parquet(tasks_path, index=False)

    # --- Write episodes parquet ---
    ep_parquet_path = out / DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)
    ep_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(episode_meta_rows).to_parquet(ep_parquet_path, index=False)

    # --- Write info.json (v3.0) ---
    # Only include features that are tensorizeable (no strings, no privileged debug data).
    # Standard lerobot frame metadata (DEFAULT_FEATURES) is merged in at load time.
    info_features: dict = {
        "observation.state": {"dtype": "float32", "shape": [30], "fps": 10},
        "action":            {"dtype": "float32", "shape": [6],  "fps": 10},
        "timestamp":         {"dtype": "float32", "shape": [1],  "fps": 10},
        "frame_index":       {"dtype": "int64",   "shape": [1],  "fps": 10},
        "episode_index":     {"dtype": "int64",   "shape": [1],  "fps": 10},
        "index":             {"dtype": "int64",   "shape": [1],  "fps": 10},
        "task_index":        {"dtype": "int64",   "shape": [1],  "fps": 10},
    }

    if has_video:
        H = video_shape["height"]
        W = video_shape["width"]
        for vk in VIDEO_KEYS:
            info_features[vk] = {
                "dtype": "video",
                "shape": [H, W, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": 10.0,
                    "video.codec": "mp4v",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
                "fps": 10,
            }

    info: dict = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": "aic_ur5e",
        "total_episodes": lerobot_idx,
        "total_frames": total_frames,
        "fps": 10,
        "data_path": DEFAULT_DATA_PATH,
        "video_path": DEFAULT_VIDEO_PATH if has_video else None,
        "features": info_features,
    }

    with open(out / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nDone — {lerobot_idx} episodes, {total_frames} frames → {output_dir}")
    print(f"  format: LeRobot {CODEBASE_VERSION}")
    print(f"  state: 30D (tcp_pose 7 + tcp_vel 6 + joint_pos 7 + joint_vel 7 + port_xyz 3)")
    print(f"  videos: {'written + concatenated' if has_video else 'not found in HDF5'}")
    print("Next step: pixi run lerobot-train --dataset.repo_id=local/aic_pilot ...")


def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 episodes to LeRobot v3.0 format")
    parser.add_argument("--input",        required=True, help="Directory with episode_*.hdf5 files")
    parser.add_argument("--output",       required=True, help="Output dataset directory")
    parser.add_argument("--success_only", action="store_true", default=True,
                        help="Only include successful episodes (default: True)")
    parser.add_argument("--max_final_error", type=float, default=0.02,
                        help="Max plug-to-port distance (m) at episode end (default: 0.02)")
    parser.add_argument("--image_height", type=int, default=480,
                        help="Resize images to this height (default: 480). Set 0 to keep original.")
    parser.add_argument("--image_width", type=int, default=640,
                        help="Resize images to this width (default: 640). Set 0 to keep original.")
    parser.add_argument("--target_hz", type=float, default=10.0,
                        help="Target Hz to downsample episodes to (default: 10.0)")
    args = parser.parse_args()
    convert(args.input, args.output, args.success_only, args.max_final_error,
            args.image_height, args.image_width, args.target_hz)


if __name__ == "__main__":
    main()
