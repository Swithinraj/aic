"""
Validate HDF5 episode files for Schema v5, v6, v7, v8, and v9.

Extends the original validate_episode.py with v6/v7/v8/v9-specific checks:
  * observations/yolo_per_camera/{left,center,right}/features  shape (T, 7)
  * Feature range: confidence in [0,1], bbox in [0,1], valid in {0,1}, age in [0,MAX_AGE]
  * image_width / image_height attrs on observations/images group
  * wrist_force present (always was, now enforced)
  * schema v7 datasets for tared wrench, port delta, and plug type onehot
  * schema v8 dataset for target_module_onehot
  * schema v9 dataset for fused yolo_port_age with fresh-valid semantics

Usage:
    cd ~/ros2_ws/src/aic
    pixi run python -m team_policy.training_robot.validate_episode_v2 \\
        /media/$USER/seagate/aic_episodes/run_session_001/episode_00000.hdf5

    # Validate entire directory
    pixi run python -m team_policy.training_robot.validate_episode_v2 \\
        /media/$USER/seagate/aic_episodes/run_session_001/

    # Validate all runs
    pixi run python -m team_policy.training_robot.validate_episode_v2 \\
        /media/$USER/seagate/aic_episodes/run_session_*/

    # Full per-episode report (schema + insertion success + scores)
    pixi run python -m team_policy.training_robot.check_episodes
    pixi run python -m team_policy.training_robot.check_episodes \\
        /media/$USER/seagate/aic_episodes/run_session_001
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from team_policy.training_robot.episode_recorder_v2 import TARGET_MODULE_NAMES

CAMERAS   = ("left", "center", "right")
MAX_AGE_S = 10.0

# Severity codes
_OK   = "OK"
_WARN = "WARN"
_FAIL = "FAIL"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _schema_version(hf) -> int:
    try:
        return int(hf["metadata"].attrs.get("schema_version", 0))
    except Exception:
        return 0


def _report(issues: List[Tuple[str, str, str]], label: str, severity: str, msg: str):
    issues.append((label, severity, msg))


def _check_dataset(hf, key: str, expected_shape_suffix: tuple,
                    T: int, issues: list, warn_only=False) -> np.ndarray | None:
    """Check that key exists and has shape (T, *expected_shape_suffix). Return array or None."""
    sev = _WARN if warn_only else _FAIL
    if key not in hf:
        _report(issues, key, sev, f"missing dataset")
        return None
    arr = hf[key][:]
    if arr.shape[0] != T:
        _report(issues, key, _FAIL, f"length {arr.shape[0]} != T={T}")
    if expected_shape_suffix and arr.shape[1:] != expected_shape_suffix:
        _report(issues, key, _FAIL,
                f"shape suffix {arr.shape[1:]} != expected {expected_shape_suffix}")
    return arr


def _check_range(arr: np.ndarray, lo: float, hi: float,
                  key: str, issues: list, warn_only=False) -> None:
    sev = _WARN if warn_only else _FAIL
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        _report(issues, key, _FAIL, "contains NaN/Inf")
        return
    bad_lo = int(np.sum(arr < lo))
    bad_hi = int(np.sum(arr > hi))
    if bad_lo or bad_hi:
        _report(issues, key, sev,
                f"{bad_lo} values < {lo}, {bad_hi} values > {hi}")


def _check_quat_norms(arr: np.ndarray, key: str, issues: list) -> None:
    norms = np.linalg.norm(arr, axis=-1)
    bad = np.sum(np.abs(norms - 1.0) > 0.01)
    if bad:
        _report(issues, key, _WARN, f"{bad} quaternions with |norm-1| > 0.01")


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------

def validate_file(path: str | Path) -> Tuple[bool, List[Tuple[str, str, str]]]:
    """Return (passed, issues). issues = list of (label, severity, message)."""
    try:
        import h5py
    except ImportError:
        return False, [("import", _FAIL, "h5py not installed")]

    issues: List[Tuple[str, str, str]] = []

    try:
        hf = h5py.File(str(path), "r")
    except Exception as e:
        return False, [("file", _FAIL, f"cannot open: {e}")]

    with hf:
        schema = _schema_version(hf)
        if schema < 5:
            _report(issues, "schema_version", _WARN,
                    f"found version {schema}, expected 5, 6, 7, 8, or 9")

        meta = hf.get("metadata")
        if meta is None:
            _report(issues, "metadata", _FAIL, "group missing")
            return False, issues

        T = int(meta.attrs.get("num_frames", 0))
        if T < 10:
            _report(issues, "num_frames", _FAIL, f"only {T} frames (minimum 10)")

        # --- core observations ---
        for key, suffix in [
            ("observations/tcp_pose",        (7,)),
            ("observations/tcp_velocity",    (6,)),
            ("observations/tcp_error",       (6,)),
            ("observations/joint_positions", (7,)),
            ("observations/joint_velocity",  (7,)),
            ("observations/wrist_force",     (6,)),
            ("observations/timestamps",      ()),
        ]:
            arr = _check_dataset(hf, key, suffix if suffix else None, T, issues)
            if arr is not None and "nan" not in key:
                if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                    _report(issues, key, _FAIL, "contains NaN/Inf")

        # quaternion norms
        if "observations/tcp_pose" in hf:
            _check_quat_norms(hf["observations/tcp_pose"][:, 3:], "tcp_pose quaternion", issues)

        # --- images ---
        if "observations/images" in hf:
            img_grp = hf["observations/images"]
            # image dims attrs (v6)
            if schema >= 6:
                if "height" not in img_grp.attrs or "width" not in img_grp.attrs:
                    _report(issues, "observations/images", _WARN,
                            "missing height/width attrs (expected in schema v6)")
            for cam in CAMERAS:
                key = f"observations/images/{cam}"
                if key not in hf:
                    _report(issues, key, _FAIL, "image dataset missing")
                    continue
                shape = hf[key].shape
                if shape[0] != T:
                    _report(issues, key, _FAIL, f"frame count {shape[0]} != T={T}")
                if len(shape) != 4 or shape[3] != 3:
                    _report(issues, key, _FAIL, f"unexpected shape {shape} (expected T,H,W,3)")
        else:
            _report(issues, "observations/images", _FAIL, "group missing")

        # --- YOLO fused (v5+) ---
        _check_dataset(hf, "observations/yolo_port_xyz",   (3,), T, issues, warn_only=True)
        _check_dataset(hf, "observations/yolo_port_valid", (),   T, issues, warn_only=True)

        if schema >= 7:
            _check_dataset(hf, "observations/tared_wrist_force_torque", (6,), T, issues)
            _check_dataset(hf, "observations/port_delta_tcp", (3,), T, issues)
            plug = _check_dataset(hf, "observations/plug_type_onehot", (2,), T, issues)
            if plug is not None:
                plug = plug.astype(np.float32)
                _check_range(plug, 0.0, 1.0, "observations/plug_type_onehot", issues)
                not_binary = np.sum((plug != 0.0) & (plug != 1.0))
                if not_binary:
                    _report(
                        issues,
                        "observations/plug_type_onehot",
                        _WARN,
                        f"{not_binary} values are not exactly 0.0 or 1.0",
                    )
        if schema >= 8:
            target = _check_dataset(
                hf,
                "observations/target_module_onehot",
                (len(TARGET_MODULE_NAMES),),
                T,
                issues,
            )
            if target is not None:
                target = target.astype(np.float32)
                _check_range(target, 0.0, 1.0, "observations/target_module_onehot", issues)
                not_binary = np.sum((target != 0.0) & (target != 1.0))
                if not_binary:
                    _report(
                        issues,
                        "observations/target_module_onehot",
                        _WARN,
                        f"{not_binary} values are not exactly 0.0 or 1.0",
                    )
        if schema >= 9:
            age = _check_dataset(hf, "observations/yolo_port_age", (), T, issues)
            if age is not None:
                age = age.astype(np.float32)
                _check_range(age, 0.0, MAX_AGE_S + 0.01, "observations/yolo_port_age", issues)

        # --- per-camera YOLO (v6) ---
        if schema >= 6:
            if "observations/yolo_per_camera" not in hf:
                _report(issues, "observations/yolo_per_camera", _FAIL,
                        "group missing (required for schema v6)")
            else:
                for cam in CAMERAS:
                    key = f"observations/yolo_per_camera/{cam}/features"
                    feat = _check_dataset(hf, key, (7,), T, issues)
                    if feat is not None:
                        feat = feat.astype(np.float32)
                        _check_range(feat[:, 0], 0.0, 1.0, f"{key}[conf]",   issues)
                        _check_range(feat[:, 1], 0.0, 1.0, f"{key}[cx_norm]", issues, warn_only=True)
                        _check_range(feat[:, 2], 0.0, 1.0, f"{key}[cy_norm]", issues, warn_only=True)
                        _check_range(feat[:, 3], 0.0, 1.0, f"{key}[w_norm]",  issues, warn_only=True)
                        _check_range(feat[:, 4], 0.0, 1.0, f"{key}[h_norm]",  issues, warn_only=True)
                        _check_range(feat[:, 5], 0.0, 1.0, f"{key}[valid]",   issues)
                        _check_range(feat[:, 6], 0.0, MAX_AGE_S + 0.01, f"{key}[age]", issues)
                        # valid must be 0.0 or 1.0
                        not_binary = np.sum((feat[:, 5] != 0.0) & (feat[:, 5] != 1.0))
                        if not_binary:
                            _report(issues, f"{key}[valid]", _WARN,
                                    f"{not_binary} frames not exactly 0.0 or 1.0")

        # --- actions ---
        for key, suffix in [
            ("actions/commanded_pose", (7,)),
            ("actions/delta_pose",     (6,)),
            ("actions/velocity",       (6,)),
        ]:
            _check_dataset(hf, key, suffix, T, issues)

        # --- relative_pose / privileged_tf ---
        _check_dataset(hf, "observations/relative_pose",       (7,), T, issues, warn_only=True)
        _check_dataset(hf, "observations/relative_pose_valid", (),   T, issues, warn_only=True)

        # --- metadata attrs ---
        required_attrs = ["schema_version", "episode_id", "task_id", "port_type",
                          "port_name", "success", "num_frames"]
        if schema >= 7:
            required_attrs.extend([
                "cable_type",
                "cable_name",
                "plug_type",
                "plug_name",
                "target_module_name",
                "time_limit_s",
                "wrist_force_tare",
            ])
        if schema >= 8:
            required_attrs.append("target_module_onehot_encoding")
        if schema >= 9:
            required_attrs.append("yolo_fresh_valid_fraction")
        for attr in required_attrs:
            if attr not in meta.attrs:
                _report(issues, f"metadata.{attr}", _WARN, "attribute missing")
        if schema >= 8 and "target_module_onehot_encoding" in meta.attrs:
            encoding = str(meta.attrs["target_module_onehot_encoding"])
            expected = ",".join(TARGET_MODULE_NAMES)
            if encoding != expected:
                _report(
                    issues,
                    "metadata.target_module_onehot_encoding",
                    _WARN,
                    f"'{encoding}' != expected '{expected}'",
                )

        # --- force sanity ---
        if "observations/wrist_force" in hf:
            wf = hf["observations/wrist_force"][:].astype(np.float32)
            force_mag = np.linalg.norm(wf[:, :3], axis=1)
            if np.max(force_mag) > 500.0:
                _report(issues, "wrist_force", _WARN,
                        f"max force {np.max(force_mag):.1f}N looks unrealistic")
            if np.all(force_mag < 0.01):
                _report(issues, "wrist_force", _WARN,
                        "all force readings ~0 — sensor may not be publishing")
        if schema >= 7 and "observations/tared_wrist_force_torque" in hf:
            twf = hf["observations/tared_wrist_force_torque"][:].astype(np.float32)
            if np.any(np.isnan(twf)) or np.any(np.isinf(twf)):
                _report(issues, "tared_wrist_force_torque", _FAIL, "contains NaN/Inf")

        # --- timestamp monotonicity ---
        if "observations/timestamps" in hf:
            ts = hf["observations/timestamps"][:].astype(np.float64)
            if T > 1:
                diffs = np.diff(ts)
                if np.any(diffs <= 0):
                    _report(issues, "timestamps", _FAIL, "non-monotonic timestamps")
                median_dt = float(np.median(diffs))
                if median_dt > 0.15:
                    _report(issues, "timestamps", _WARN,
                            f"median dt={median_dt:.3f}s (expected ~0.10s)")

    n_fail = sum(1 for _, sev, _ in issues if sev == _FAIL)
    passed = (n_fail == 0)
    return passed, issues


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_issues(issues: List[Tuple[str, str, str]]) -> None:
    for label, sev, msg in issues:
        sym = {"OK": "✓", "WARN": "⚠", "FAIL": "✗"}.get(sev, "?")
        print(f"  [{sym} {sev:4s}] {label}: {msg}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate HDF5 episode files (schema v5-v9)"
    )
    parser.add_argument(
        "path", nargs="+",
        help="HDF5 file(s) or directory containing episode_*.hdf5 files"
    )
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Only show failures and warnings")
    args = parser.parse_args()

    files: List[Path] = []
    for p in args.path:
        pp = Path(p)
        if pp.is_dir():
            files.extend(sorted(pp.glob("episode_*.hdf5")))
        elif pp.is_file():
            files.append(pp)
        else:
            print(f"Path not found: {p}", file=sys.stderr)

    if not files:
        print("No HDF5 files found.", file=sys.stderr)
        sys.exit(1)

    n_pass = 0
    n_fail = 0
    for ep_file in files:
        passed, issues = validate_file(ep_file)
        status = "PASS" if passed else "FAIL"
        print(f"\n{'='*60}")
        print(f"{ep_file.name}  →  {status}")
        if issues:
            if args.quiet:
                issues = [(l, s, m) for l, s, m in issues if s != _OK]
            _print_issues(issues)
        elif not args.quiet:
            print("  All checks passed.")
        if passed:
            n_pass += 1
        else:
            n_fail += 1

    print(f"\n{'='*60}")
    print(f"Summary: {n_pass} passed, {n_fail} failed out of {len(files)} episodes")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
