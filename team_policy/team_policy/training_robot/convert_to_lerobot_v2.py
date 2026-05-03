"""
Convert schema v5/v6/v7/v8/v9 HDF5 episodes to LeRobot v3.0 dataset format — 77D state.

State layout (77 dimensions):
  [0:7]   tcp_pose        7D  x y z qx qy qz qw
  [7:13]  tcp_velocity    6D  vx vy vz wx wy wz
  [13:19] tcp_error       6D  controller tracking error (available at inference)
  [19:26] joint_positions 7D
  [26:33] joint_velocity  7D
  [33:36] port_xyz        3D  fused YOLO xyz in base_link (hold-last, zeros before first det)
  [36:37] yolo_valid      1D  fresh target detection flag, not hold-last existence
  [37:38] yolo_age        1D  seconds since last valid target detection
  [38:41] port_delta_tcp  3D  port_xyz - tcp position in base_link
  [41:47] tared_wrist_force_torque 6D  tare-subtracted fx fy fz tx ty tz
  [47:54] yolo_left       7D  conf cx cy w h valid age
  [54:61] yolo_center     7D
  [61:68] yolo_right      7D
  [68:70] plug_type_onehot 2D  [is_sfp, is_sc]
  [70:77] target_module_onehot 7D exact target module identity

Action (6D): delta TCP pose [dx dy dz drx dry drz] (current→commanded)

Backward compatibility:
  v5 episodes (no yolo_per_camera group):
    - tcp_error, wrist_force always present in HDF5 → used as-is
    - yolo_per_camera: filled with zeros [0,0,0,0,0,0,MAX_AGE]
    - port_xyz: YOLO hold-last from yolo_port_xyz/yolo_port_valid as usual
    - tared_wrist_force_torque: falls back to raw wrist_force
    - plug_type_onehot: inferred from metadata when possible
    - target_module_onehot: inferred from metadata when possible
    - yolo_valid/yolo_age: only schema v9 records true fused freshness; older schemas are legacy approximations

Usage:
    cd ~/ros2_ws/src/aic
    pixi run python -m team_policy.training_robot.convert_to_lerobot_v2 \\
        --input /media/$USER/seagate/aic_episodes/run_001 \\
        --output ./datasets/aic_v3_77d \\
        --success_only

Output: LeRobot v3.0 dataset with observation.state shape=[77], action shape=[6]
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

from team_policy.training_robot.episode_recorder_v2 import (
    TARGET_MODULE_NAMES,
    build_plug_type_onehot,
    build_target_module_onehot,
)

CODEBASE_VERSION = "v3.0"
DEFAULT_DATA_PATH     = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH    = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
DEFAULT_EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_TASKS_PATH    = "meta/tasks.parquet"

STATE_DIM  = 77
ACTION_DIM = 6
CAMERAS    = ("left", "center", "right")

_MAX_AGE_S = 10.0
_ZERO7 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _MAX_AGE_S], dtype=np.float32)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    sh = math.sqrt(x*x + y*y + z*z)
    if sh < 1e-9:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.atan2(sh, max(abs(w), 1e-12))
    if w < 0:
        angle = -angle
    return np.array([x/sh*angle, y/sh*angle, z/sh*angle], dtype=np.float32)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ], dtype=np.float32)


def _quat_inverse(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([-x, -y, -z, w], dtype=np.float32)


def _compute_delta_actions(tcp_poses: np.ndarray) -> np.ndarray:
    """Fallback: compute delta actions from consecutive tcp_poses when delta_pose not in HDF5."""
    T = tcp_poses.shape[0]
    deltas = np.zeros((T, 6), dtype=np.float32)
    for i in range(T - 1):
        dp = tcp_poses[i+1, :3] - tcp_poses[i, :3]
        dq = _quat_multiply(tcp_poses[i+1, 3:], _quat_inverse(tcp_poses[i, 3:]))
        dr = _quat_to_axis_angle(dq)
        deltas[i] = np.concatenate([dp, dr])
    deltas[-1] = deltas[-2]
    return deltas


def _decode_strings(raw) -> List[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in raw]


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def _write_video(frames: np.ndarray, path: Path, fps: int = 10,
                 target_hw: tuple | None = None) -> None:
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python required for video export")
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
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        list_file = f.name
        for p in input_paths:
            f.write(f"file '{p.resolve()}'\n")
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
             "-c", "copy", str(output_path)],
            capture_output=True, timeout=300,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    finally:
        os.unlink(list_file)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

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
# Per-episode 77D state builder
# ---------------------------------------------------------------------------

def _build_state_77d(
    tcp_poses:    np.ndarray,   # (T, 7)
    tcp_vels:     np.ndarray,   # (T, 6)
    tcp_errors:   np.ndarray,   # (T, 6)
    joint_pos:    np.ndarray,   # (T, 7)
    joint_vel:    np.ndarray,   # (T, 7)
    port_xyz:     np.ndarray,   # (T, 3)  YOLO hold-last
    yolo_valid:   np.ndarray,   # (T, 1)  fresh detection flag
    yolo_age:     np.ndarray,   # (T, 1)  staleness seconds
    port_delta_tcp: np.ndarray, # (T, 3)  port_xyz - tcp position
    tared_wrist_force: np.ndarray,   # (T, 6)
    yolo_left:    np.ndarray,   # (T, 7)
    yolo_center:  np.ndarray,   # (T, 7)
    yolo_right:   np.ndarray,   # (T, 7)
    plug_type_onehot: np.ndarray,  # (T, 2)
    target_module_onehot: np.ndarray,  # (T, 7)
) -> np.ndarray:
    return np.concatenate(
        [
            tcp_poses,
            tcp_vels,
            tcp_errors,
            joint_pos,
            joint_vel,
            port_xyz,
            yolo_valid,
            yolo_age,
            port_delta_tcp,
            tared_wrist_force,
            yolo_left,
            yolo_center,
            yolo_right,
            plug_type_onehot,
            target_module_onehot,
        ],
        axis=1,
    ).astype(np.float32)  # (T, 77)


def _yolo_hold_last(yolo_xyz_raw: np.ndarray, yolo_valid: np.ndarray) -> np.ndarray:
    """Apply inference-matching hold-last to fused port_xyz (T,3)."""
    T = yolo_xyz_raw.shape[0]
    port_xyz = np.zeros((T, 3), dtype=np.float32)
    last_xyz = np.zeros(3, dtype=np.float32)
    seen_first = False
    for t in range(T):
        if yolo_valid[t]:
            last_xyz = yolo_xyz_raw[t]
            seen_first = True
        if seen_first:
            port_xyz[t] = last_xyz
    return port_xyz


def _legacy_fused_freshness_from_held_xyz(
    yolo_xyz_raw: np.ndarray,
    seen_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate fresh-valid and age for legacy schemas with held-last semantics.

    Pre-v9 episodes stored held yolo_port_xyz and a latched yolo_port_valid. We
    conservatively treat the first seen frame and frames where held xyz changes
    as fresh updates, then age everything else until the next change.
    """
    T = yolo_xyz_raw.shape[0]
    fresh = np.zeros(T, dtype=bool)
    age = np.full(T, _MAX_AGE_S, dtype=np.float32)
    last_fresh_idx: int | None = None
    prev_xyz: np.ndarray | None = None
    eps = 1e-6
    for t in range(T):
        if not seen_mask[t]:
            prev_xyz = None
            continue
        xyz = yolo_xyz_raw[t]
        is_fresh = prev_xyz is None or float(np.linalg.norm(xyz - prev_xyz)) > eps
        if is_fresh:
            fresh[t] = True
            age[t] = 0.0
            last_fresh_idx = t
        elif last_fresh_idx is not None:
            age[t] = min(_MAX_AGE_S, 0.10 * float(t - last_fresh_idx))
        prev_xyz = xyz
    return fresh, age


def _infer_plug_type(meta) -> str:
    plug_type = str(meta.attrs.get("plug_type", "")).strip().lower()
    if plug_type in {"sfp", "sc"}:
        return plug_type

    port_type = str(meta.attrs.get("port_type", "")).strip().lower()
    if "sc" in port_type:
        return "sc"
    if "sfp" in port_type or "nic" in port_type:
        return "sfp"
    return ""


def _infer_target_module_name(meta) -> str:
    value = str(meta.attrs.get("target_module_name", "")).strip().lower()
    return value if value in TARGET_MODULE_NAMES else ""


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(
    input_dir: str,
    output_dir: str,
    success_only: bool,
    max_final_error: float = 0.02,
    image_height: int = 480,
    image_width: int = 640,
    target_hz: float = 10.0,
) -> None:
    try:
        import h5py
        import pandas as pd
    except ImportError:
        raise RuntimeError("h5py and pandas required: pixi add --pypi h5py pandas")

    episode_files = sorted(Path(input_dir).glob("episode_*.hdf5"))
    if not episode_files:
        raise FileNotFoundError(f"No HDF5 files found in {input_dir}")

    print(f"Found {len(episode_files)} episode files in {input_dir}")

    out = Path(output_dir)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    VIDEO_KEYS = [f"observation.images.{cam}" for cam in CAMERAS]
    target_hw  = (image_height, image_width) if (image_height and image_width) else None
    print(f"State: 77D  |  Images: {'%dx%d (HxW)' % (image_height, image_width) if target_hw else 'original'}")

    all_states:  List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    all_rows:    List[dict] = []
    episode_meta_rows = []
    tmp_video_paths: dict[str, List[Path]] = {vk: [] for vk in VIDEO_KEYS}
    lerobot_idx  = 0
    total_frames = 0
    video_shape:  dict = {}

    for ep_file in episode_files:
        with h5py.File(ep_file, "r") as hf:
            meta    = hf["metadata"]
            success = bool(meta.attrs.get("success", 0))
            if success_only and not success:
                print(f"  skip {ep_file.name} (failed)")
                continue
            final_error_raw = meta.attrs.get("final_error", None)
            if final_error_raw is not None:
                fe = float(final_error_raw)
                if not math.isnan(fe) and fe > max_final_error:
                    print(f"  skip {ep_file.name} (final_error={fe:.3f}m > {max_final_error}m)")
                    continue

            schema_ver = str(meta.attrs.get("schema_version", "5"))
            T_full = int(meta.attrs.get("num_frames", hf["observations/tcp_pose"].shape[0]))
            ts_full = hf["observations/timestamps"][:]

            # Downsample to target_hz
            if T_full > 1:
                median_dt = float(np.median(np.diff(ts_full)))
                actual_hz = 1.0 / median_dt if median_dt > 0 else target_hz
            else:
                actual_hz = target_hz
            step    = max(1, round(actual_hz / target_hz))
            indices = np.arange(0, T_full, step)

            tcp_poses = hf["observations/tcp_pose"][:][indices]
            tcp_vels  = hf["observations/tcp_velocity"][:][indices]
            joint_pos = hf["observations/joint_positions"][:][indices]
            joint_vel = (
                hf["observations/joint_velocity"][:][indices]
                if "observations/joint_velocity" in hf
                else np.zeros((len(indices), 7), dtype=np.float32)
            )
            wrist_force = (
                hf["observations/wrist_force"][:][indices].astype(np.float32)
                if "observations/wrist_force" in hf
                else np.zeros((len(indices), 6), dtype=np.float32)
            )
            tared_wrist_force = (
                hf["observations/tared_wrist_force_torque"][:][indices].astype(np.float32)
                if "observations/tared_wrist_force_torque" in hf
                else wrist_force.copy()
            )
            tcp_errors = (
                hf["observations/tcp_error"][:][indices].astype(np.float32)
                if "observations/tcp_error" in hf
                else np.zeros((len(indices), 6), dtype=np.float32)
            )
            T = tcp_poses.shape[0]

            task_id = str(meta.attrs.get("task_id", "insert_cable"))

            # --- port_xyz (fused YOLO hold-last) + fresh valid/age ---
            if "observations/yolo_port_xyz" in hf:
                yolo_xyz_raw = hf["observations/yolo_port_xyz"][:][indices].astype(np.float32)
                yolo_seen = (
                    hf["observations/yolo_port_valid"][:][indices].astype(bool)
                    if "observations/yolo_port_valid" in hf
                    else np.ones(T, dtype=bool)
                )
                if "observations/yolo_port_age" in hf:
                    yolo_valid = yolo_seen.copy()
                    yolo_age = hf["observations/yolo_port_age"][:][indices].astype(np.float32)
                else:
                    yolo_valid, yolo_age = _legacy_fused_freshness_from_held_xyz(
                        yolo_xyz_raw,
                        yolo_seen,
                    )
                    print(
                        "    fused_yolo_age: not found — "
                        "legacy schema, freshness/staleness reconstructed from held xyz changes"
                    )
                port_xyz = _yolo_hold_last(yolo_xyz_raw, yolo_seen)
                yolo_cov = float(yolo_valid.mean()) * 100
                pre_yolo = int(np.argmax(yolo_valid)) if yolo_valid.any() else T
                print(
                    f"    fused_yolo: {yolo_cov:.0f}% fresh | {pre_yolo} cold-start zeros"
                )
            else:
                port_xyz = np.zeros((T, 3), dtype=np.float32)
                yolo_valid = np.zeros(T, dtype=bool)
                yolo_age = np.full(T, _MAX_AGE_S, dtype=np.float32)
                print(f"    fused_yolo: not found — using zeros")
            port_delta_tcp = np.zeros((T, 3), dtype=np.float32)
            if yolo_valid.any():
                first_valid = int(np.argmax(yolo_valid))
                port_delta_tcp[first_valid:] = (
                    port_xyz[first_valid:] - tcp_poses[first_valid:, :3].astype(np.float32)
                )
            if "observations/tared_wrist_force_torque" not in hf:
                print("    tared_wrench: not found — falling back to raw wrist_force")

            # --- per-camera YOLO features (v6) or zeros fallback ---
            # These features are already normalized in HDF5, so video resizing during
            # conversion does not require any bbox recomputation here.
            per_cam: dict[str, np.ndarray] = {}
            has_per_cam = "observations/yolo_per_camera" in hf
            for cam in CAMERAS:
                key = f"observations/yolo_per_camera/{cam}/features"
                if has_per_cam and key in hf:
                    feat = hf[key][:][indices].astype(np.float32)
                    per_cam[cam] = feat
                else:
                    per_cam[cam] = np.tile(_ZERO7, (T, 1))
            if not has_per_cam:
                print(f"    per_camera_yolo: v5 episode — zero-filled 21D")

            if "observations/plug_type_onehot" in hf:
                plug_type_onehot = hf["observations/plug_type_onehot"][:][indices].astype(np.float32)
            else:
                inferred_plug_type = _infer_plug_type(meta)
                plug_type_onehot = np.tile(build_plug_type_onehot(inferred_plug_type), (T, 1))
                print(
                    "    plug_type_onehot: not found — "
                    f"inferred from metadata as '{inferred_plug_type or 'unknown'}'"
                )

            if "observations/target_module_onehot" in hf:
                target_module_onehot = (
                    hf["observations/target_module_onehot"][:][indices].astype(np.float32)
                )
            else:
                inferred_target_module = _infer_target_module_name(meta)
                target_module_onehot = np.tile(
                    build_target_module_onehot(inferred_target_module),
                    (T, 1),
                )
                print(
                    "    target_module_onehot: not found — "
                    f"inferred from metadata as '{inferred_target_module or 'unknown'}'"
                )

            state = _build_state_77d(
                tcp_poses, tcp_vels, tcp_errors, joint_pos, joint_vel, port_xyz,
                yolo_valid.astype(np.float32).reshape(T, 1),
                yolo_age.reshape(T, 1),
                port_delta_tcp, tared_wrist_force,
                per_cam["left"], per_cam["center"], per_cam["right"],
                plug_type_onehot, target_module_onehot,
            )
            assert state.shape == (T, STATE_DIM), (
                f"State shape mismatch: got {state.shape}, expected ({T}, {STATE_DIM})"
            )

            # --- Actions ---
            if "actions/delta_pose" in hf:
                action = hf["actions/delta_pose"][:][indices].astype(np.float32)
            else:
                action = _compute_delta_actions(tcp_poses)

            # --- Videos ---
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

            all_rows.append({
                "index":             np.arange(ep_start, ep_end, dtype=np.int64),
                "episode_index":     np.full(T, lerobot_idx, dtype=np.int64),
                "frame_index":       np.arange(T, dtype=np.int64),
                "timestamp":         np.arange(T, dtype=np.float32) * 0.10,
                "task_index":        np.zeros(T, dtype=np.int64),
                "observation.state": state.astype(np.float32),
                "action":            action.astype(np.float32),
            })
            episode_meta_rows.append({
                "episode_index":           lerobot_idx,
                "tasks":                   [
                    "insert "
                    f"{meta.attrs.get('plug_type', 'plug')} "
                    f"{meta.attrs.get('plug_name', 'plug')} "
                    f"into {meta.attrs.get('target_module_name', 'module')}/"
                    f"{meta.attrs.get('port_name', 'port')}"
                ],
                "task_id":                 str(meta.attrs.get("task_id", "")),
                "cable_name":              str(meta.attrs.get("cable_name", "")),
                "plug_type":               str(meta.attrs.get("plug_type", "")),
                "plug_name":               str(meta.attrs.get("plug_name", "")),
                "port_type":               str(meta.attrs.get("port_type", "")),
                "port_name":               str(meta.attrs.get("port_name", "")),
                "target_module_name":      str(meta.attrs.get("target_module_name", "")),
                "length":                  T,
                "data/chunk_index":        0,
                "data/file_index":         0,
                "dataset_from_index":      ep_start,
                "dataset_to_index":        ep_end,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index":  0,
                **{f"videos/{vk}/chunk_index": 0  for vk in VIDEO_KEYS},
                **{f"videos/{vk}/file_index":  0  for vk in VIDEO_KEYS},
                **{f"videos/{vk}/from_timestamp": ep_start * 0.10 for vk in VIDEO_KEYS},
                **{f"videos/{vk}/to_timestamp":   ep_end   * 0.10 for vk in VIDEO_KEYS},
            })

            total_frames += T
            print(f"  converted {ep_file.name} → episode {lerobot_idx} "
                  f"({T} frames, schema_v{schema_ver})")
            lerobot_idx += 1

    if lerobot_idx == 0:
        print("No episodes passed the filter — nothing converted.")
        return

    # --- Write merged data parquet ---
    import datasets as hf_datasets
    from lerobot.datasets.feature_utils import get_hf_features_from_features
    from lerobot.datasets.utils import DEFAULT_FEATURES

    data_features_spec = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,)},
        "action":            {"dtype": "float32", "shape": (ACTION_DIM,)},
    }
    all_features_spec = {**data_features_spec, **DEFAULT_FEATURES}
    hf_features = get_hf_features_from_features(all_features_spec)

    merged: dict = {}
    for key in all_rows[0]:
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
                    import shutil
                    print(f"  warn: ffmpeg concat failed for {vk}")
                    shutil.copy(tmp_video_paths[vk][-1], vid_out)

    import shutil as _shutil
    tmp_dir = out / "_tmp_videos"
    if tmp_dir.exists():
        _shutil.rmtree(tmp_dir)

    # --- Stats ---
    stats = compute_stats(all_states, all_actions)
    if has_video:
        imagenet_mean = [[[0.485]], [[0.456]], [[0.406]]]
        imagenet_std  = [[[0.229]], [[0.224]], [[0.225]]]
        for vk in VIDEO_KEYS:
            stats[vk] = {
                "mean": imagenet_mean, "std": imagenet_std,
                "min":  [[0.0], [0.0], [0.0]], "max": [[1.0], [1.0], [1.0]],
            }
    with open(out / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # --- tasks.parquet ---
    import pandas as pd
    tasks_path = out / DEFAULT_TASKS_PATH
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"task_index": [0], "task": ["insert cable"]}).to_parquet(tasks_path, index=False)

    # --- episodes parquet ---
    ep_parquet_path = out / DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)
    ep_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(episode_meta_rows).to_parquet(ep_parquet_path, index=False)

    # --- info.json ---
    info_features: dict = {
        "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "fps": 10},
        "action":            {"dtype": "float32", "shape": [ACTION_DIM],  "fps": 10},
        "timestamp":         {"dtype": "float32", "shape": [1],           "fps": 10},
        "frame_index":       {"dtype": "int64",   "shape": [1],           "fps": 10},
        "episode_index":     {"dtype": "int64",   "shape": [1],           "fps": 10},
        "index":             {"dtype": "int64",   "shape": [1],           "fps": 10},
        "task_index":        {"dtype": "int64",   "shape": [1],           "fps": 10},
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
        "robot_type":       "aic_ur5e",
        "total_episodes":   lerobot_idx,
        "total_frames":     total_frames,
        "fps":              10,
        "data_path":        DEFAULT_DATA_PATH,
        "video_path":       DEFAULT_VIDEO_PATH if has_video else None,
        "features":         info_features,
        "state_layout": {
            "0:7":   "tcp_pose (x y z qx qy qz qw)",
            "7:13":  "tcp_velocity (vx vy vz wx wy wz)",
            "13:19": "tcp_error (6D controller tracking error)",
            "19:26": "joint_positions (7D)",
            "26:33": "joint_velocity (7D)",
            "33:36": "yolo_port_xyz_fused (held target port xyz in base_link)",
            "36:37": "yolo_valid_fresh (1=fresh target detection, 0=stale/absent)",
            "37:38": "yolo_age_seconds (staleness of held target port xyz)",
            "38:41": "port_delta_tcp (held yolo_port_xyz - tcp position)",
            "41:47": "tared_wrist_force_torque (fx fy fz tx ty tz)",
            "47:54": "yolo_left (conf cx cy w h valid age)",
            "54:61": "yolo_center (conf cx cy w h valid age)",
            "61:68": "yolo_right (conf cx cy w h valid age)",
            "68:70": "plug_type_onehot ([is_sfp, is_sc])",
            "70:77": "target_module_onehot (" + ", ".join(TARGET_MODULE_NAMES) + ")",
        },
    }
    with open(out / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nDone — {lerobot_idx} episodes, {total_frames} frames → {output_dir}")
    print(f"  format : LeRobot {CODEBASE_VERSION}")
    print(
        f"  state  : {STATE_DIM}D  "
        "(30D base + 6D tcp_error + 3D held fused xyz + 1D fresh valid + 1D age + 3D port_delta_tcp + 6D tared F/T + 21D per-cam YOLO + 2D plug type + 7D target module)"
    )
    print(f"  action : {ACTION_DIM}D  delta TCP pose")
    print(f"  videos : {'written + concatenated' if has_video else 'not found'}")
    print("Next step: pixi run lerobot-train --dataset.repo_id=local/aic_v3_77d ...")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HDF5 schema v5/v6/v7/v8/v9 episodes to LeRobot v3.0 — 77D state"
    )
    parser.add_argument("--input",           required=True)
    parser.add_argument("--output",          required=True)
    parser.add_argument("--success_only",    action="store_true", default=True)
    parser.add_argument("--max_final_error", type=float, default=0.02)
    parser.add_argument("--image_height",    type=int,   default=480)
    parser.add_argument("--image_width",     type=int,   default=640)
    parser.add_argument("--target_hz",       type=float, default=10.0)
    args = parser.parse_args()
    convert(
        args.input, args.output, args.success_only,
        args.max_final_error, args.image_height, args.image_width, args.target_hz,
    )


if __name__ == "__main__":
    main()
