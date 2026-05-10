from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from lerobot.datasets.feature_utils import get_hf_features_from_features
from lerobot.datasets.utils import DEFAULT_FEATURES

OLD_STATE_DIM = 77
NEW_STATE_DIM = 63
ACTION_DIM = 6

KEEP_IDX = np.r_[
    0:7,
    7:13,
    19:26,
    26:33,
    41:47,
    47:54,
    54:61,
    61:68,
    68:70,
    70:77,
].astype(np.int64)

STATE_LAYOUT_63D = {
    "0:7": "tcp_pose (x y z qx qy qz qw)",
    "7:13": "tcp_velocity (vx vy vz wx wy wz)",
    "13:20": "joint_positions (7D)",
    "20:27": "joint_velocity (7D)",
    "27:33": "tared_wrist_force_torque (fx fy fz tx ty tz)",
    "33:40": "yolo_left (conf cx cy w h valid age)",
    "40:47": "yolo_center (conf cx cy w h valid age)",
    "47:54": "yolo_right (conf cx cy w h valid age)",
    "54:56": "plug_type_onehot ([is_sfp, is_sc])",
    "56:63": "target_module_onehot (7D exact target module identity)",
}

REMOVED_STATE_FIELDS = [
    "13:19 tcp_error",
    "33:36 fused_yolo_port_xyz",
    "36:37 fused_yolo_valid",
    "37:38 fused_yolo_age",
    "38:41 port_delta_tcp",
]


def _stack_feature(values, expected_dim: int, name: str) -> np.ndarray:
    rows = values.to_numpy() if hasattr(values, "to_numpy") else values
    out = np.empty((len(rows), expected_dim), dtype=np.float32)
    for i, value in enumerate(rows):
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape[0] != expected_dim:
            raise ValueError(f"{name} row {i} has dim {arr.shape[0]}, expected {expected_dim}")
        out[i] = arr
    return out


def _init_acc(dim: int) -> dict:
    return {
        "n": 0,
        "sum": np.zeros(dim, dtype=np.float64),
        "sumsq": np.zeros(dim, dtype=np.float64),
        "min": np.full(dim, np.inf, dtype=np.float64),
        "max": np.full(dim, -np.inf, dtype=np.float64),
    }


def _update_acc(acc: dict, values: np.ndarray) -> None:
    x = values.astype(np.float64)
    acc["n"] += int(x.shape[0])
    acc["sum"] += x.sum(axis=0)
    acc["sumsq"] += np.square(x).sum(axis=0)
    acc["min"] = np.minimum(acc["min"], x.min(axis=0))
    acc["max"] = np.maximum(acc["max"], x.max(axis=0))


def _finalize_acc(acc: dict) -> dict:
    n = max(int(acc["n"]), 1)
    mean = acc["sum"] / n
    var = np.maximum(acc["sumsq"] / n - np.square(mean), 0.0)
    std = np.sqrt(var) + 1e-6
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": acc["min"].tolist(),
        "max": acc["max"].tolist(),
    }


def _link_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _prepare_output(src_root: Path, dst_root: Path, force: bool, inplace: bool) -> None:
    if inplace:
        return
    if dst_root.exists():
        if not force:
            raise FileExistsError(f"Output exists: {dst_root}. Use --force or choose another output.")
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    for src in src_root.rglob("*"):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        if len(rel.parts) >= 1 and rel.parts[0] == "data" and src.suffix == ".parquet":
            continue
        if rel == Path("meta/info.json") or rel == Path("meta/stats.json"):
            continue
        _link_or_copy_file(src, dst)


def _write_dataset_parquet(df: pd.DataFrame, state63: np.ndarray, dst_path: Path) -> np.ndarray:
    payload = {}
    for col in df.columns:
        if col == "observation.state":
            payload[col] = state63
        elif col == "action":
            payload[col] = _stack_feature(df[col], ACTION_DIM, "action")
        elif col in {"index", "episode_index", "frame_index", "task_index"}:
            payload[col] = df[col].to_numpy(dtype=np.int64)
        elif col == "timestamp":
            payload[col] = df[col].to_numpy(dtype=np.float32)
        else:
            payload[col] = df[col].tolist()

    features_spec = {
        **{
            "observation.state": {"dtype": "float32", "shape": (NEW_STATE_DIM,)},
            "action": {"dtype": "float32", "shape": (ACTION_DIM,)},
        },
        **DEFAULT_FEATURES,
    }

    hf_features = get_hf_features_from_features(features_spec)
    ds = Dataset.from_dict(payload, features=hf_features)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(dst_path))
    return payload["action"]


def _strip_data_files(src_root: Path, dst_root: Path, inplace: bool) -> tuple[dict, dict, int]:
    state_acc = _init_acc(NEW_STATE_DIM)
    action_acc = _init_acc(ACTION_DIM)
    total_frames = 0

    data_files = sorted(src_root.glob("data/chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No LeRobot data parquet files found under {src_root / 'data'}")

    for src_path in data_files:
        rel = src_path.relative_to(src_root)
        dst_path = src_path if inplace else dst_root / rel
        df = pd.read_parquet(src_path)

        state77 = _stack_feature(df["observation.state"], OLD_STATE_DIM, "observation.state")
        state63 = state77[:, KEEP_IDX].astype(np.float32)

        write_path = dst_path.with_suffix(".tmp.parquet") if inplace else dst_path
        action = _write_dataset_parquet(df, state63, write_path)

        if inplace:
            os.replace(write_path, dst_path)

        _update_acc(state_acc, state63)
        _update_acc(action_acc, action)
        total_frames += int(state63.shape[0])

        print(f"rewrote {rel}: {OLD_STATE_DIM}D -> {NEW_STATE_DIM}D, frames={state63.shape[0]}")

    return _finalize_acc(state_acc), _finalize_acc(action_acc), total_frames


def _update_info_json(src_root: Path, dst_root: Path, total_frames: int, inplace: bool) -> None:
    src_info = src_root / "meta" / "info.json"
    dst_info = src_info if inplace else dst_root / "meta" / "info.json"

    if src_info.exists():
        with open(src_info, "r") as f:
            info = json.load(f)
    else:
        info = {}

    info["total_frames"] = total_frames
    info["features"] = info.get("features", {})
    info["features"]["observation.state"] = {
        "dtype": "float32",
        "shape": [NEW_STATE_DIM],
        "fps": int(info.get("fps", 10)),
    }
    info["state_layout"] = STATE_LAYOUT_63D
    info["source_state_dim"] = OLD_STATE_DIM
    info["state_dim"] = NEW_STATE_DIM
    info["removed_state_fields"] = REMOVED_STATE_FIELDS

    dst_info.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_info, "w") as f:
        json.dump(info, f, indent=2)


def _update_stats_json(src_root: Path, dst_root: Path, state_stats: dict, action_stats: dict, inplace: bool) -> None:
    src_stats = src_root / "meta" / "stats.json"
    dst_stats = src_stats if inplace else dst_root / "meta" / "stats.json"

    if src_stats.exists():
        with open(src_stats, "r") as f:
            stats = json.load(f)
    else:
        stats = {}

    stats["observation.state"] = state_stats
    stats["action"] = action_stats

    dst_stats.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_stats, "w") as f:
        json.dump(stats, f, indent=2)


def strip_dataset(input_dir: str, output_dir: str | None, force: bool, inplace: bool) -> None:
    src_root = Path(input_dir).expanduser().resolve()

    if not src_root.exists():
        raise FileNotFoundError(src_root)

    if inplace:
        dst_root = src_root
    else:
        if output_dir is None:
            raise ValueError("--output is required unless --inplace is used")
        dst_root = Path(output_dir).expanduser().resolve()

    _prepare_output(src_root, dst_root, force, inplace)

    state_stats, action_stats, total_frames = _strip_data_files(src_root, dst_root, inplace)

    _update_info_json(src_root, dst_root, total_frames, inplace)
    _update_stats_json(src_root, dst_root, state_stats, action_stats, inplace)

    print(f"done: {dst_root}")
    print(f"state: {OLD_STATE_DIM}D -> {NEW_STATE_DIM}D")
    print("removed: tcp_error, fused_yolo_port_xyz, fused_yolo_valid, fused_yolo_age, port_delta_tcp")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--inplace", action="store_true")
    args = parser.parse_args()

    strip_dataset(args.input, args.output, args.force, args.inplace)


if __name__ == "__main__":
    main()
