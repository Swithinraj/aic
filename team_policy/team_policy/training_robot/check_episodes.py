"""
Full episode validation check — prints a summary for every episode on Seagate.

Checks per episode:
  - Schema version
  - Frame count
  - All required datasets (shapes, ranges, NaN/Inf)
  - Timestamps monotonic
  - Images present
  - insertion_success recorded (new field)
  - insertion_event_received / insertion_event_data metadata
  - YOLO per-camera features
  - Wrist force sanity

Usage:
    cd ~/ros2_ws/src/aic
    pixi run python -m team_policy.training_robot.check_episodes
    pixi run python -m team_policy.training_robot.check_episodes /media/$USER/seagate/aic_episodes/run_session_001
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np

from team_policy.training_robot.validate_episode_v2 import validate_file

_W = 72


def _bar(char="─"):
    return char * _W


def _header(title: str):
    print(f"\n{'━' * _W}")
    print(f"  {title}")
    print(f"{'━' * _W}")


def _check_insertion(path: Path) -> list[str]:
    lines = []
    try:
        import h5py
        with h5py.File(str(path), "r") as hf:
            meta = hf.get("metadata")
            ev_received = int(meta.attrs.get("insertion_event_received", -1)) if meta else -1
            ev_data     = str(meta.attrs.get("insertion_event_data", "")) if meta else ""
            schema      = int(meta.attrs.get("schema_version", 0)) if meta else 0
            success     = int(meta.attrs.get("success", -1)) if meta else -1
            n_frames    = int(meta.attrs.get("num_frames", 0)) if meta else 0
            port_name   = str(meta.attrs.get("port_name", "?")) if meta else "?"
            port_type   = str(meta.attrs.get("port_type", "?")) if meta else "?"
            target_mod  = str(meta.attrs.get("target_module_name", "?")) if meta else "?"
            plug_type   = str(meta.attrs.get("plug_type", "?")) if meta else "?"
            duration    = float(meta.attrs.get("duration_s", 0.0)) if meta else 0.0
            final_err   = meta.attrs.get("final_error", float("nan")) if meta else float("nan")

            lines.append(f"  schema        : v{schema}")
            lines.append(f"  frames        : {n_frames}")
            lines.append(f"  duration      : {duration:.1f}s")
            lines.append(f"  success       : {'YES' if success == 1 else 'NO'}")
            lines.append(f"  port          : {port_type}/{port_name}")
            lines.append(f"  plug_type     : {plug_type}")
            lines.append(f"  target_module : {target_mod}")
            lines.append(f"  final_error   : {float(final_err):.4f}m" if not np.isnan(float(final_err)) else "  final_error   : n/a")

            if "observations/insertion_success" in hf:
                ins = hf["observations/insertion_success"][:].astype(np.float32)
                n_ones = int((ins == 1.0).sum())
                n_zeros = int((ins == 0.0).sum())
                first_one = int(np.argmax(ins == 1.0)) if n_ones > 0 else -1
                lines.append(f"  insertion_success dataset :")
                lines.append(f"    frames=0 : {n_zeros}   frames=1 : {n_ones}   first_1_at_frame : {first_one if n_ones > 0 else 'never'}")
                lines.append(f"    insertion_event_received : {ev_received}   data : '{ev_data}'")
            else:
                lines.append("  insertion_success : MISSING (old episode, pre-patch)")

    except Exception as e:
        lines.append(f"  [ERROR reading extra fields]: {e}")
    return lines


def check_all(episodes_dir: Path):
    files = sorted(episodes_dir.rglob("episode_*.hdf5"))

    if not files:
        print(f"\n[ERROR] No episode_*.hdf5 files found under: {episodes_dir}")
        sys.exit(1)

    _header(f"Episode Validation — {len(files)} files in {episodes_dir}")

    n_pass = 0
    n_fail = 0
    n_warn = 0

    for ep_path in files:
        run_name = ep_path.parent.name
        ep_name  = ep_path.name
        passed, issues = validate_file(ep_path)

        fails = [(l, s, m) for l, s, m in issues if s == "FAIL"]
        warns = [(l, s, m) for l, s, m in issues if s == "WARN"]

        status = "PASS" if passed else "FAIL"
        sym    = "✓" if passed else "✗"
        print(f"\n{_bar()}")
        print(f"  {sym} [{status}]  {run_name}/{ep_name}")
        print(_bar("─"))

        extra = _check_insertion(ep_path)
        for line in extra:
            print(line)

        if fails or warns:
            print(f"  {'─'*30}")
        for label, sev, msg in fails:
            print(f"  [✗ FAIL] {label}: {msg}")
        for label, sev, msg in warns:
            print(f"  [⚠ WARN] {label}: {msg}")
        if not issues:
            print("  All schema checks passed.")

        if passed:
            n_pass += 1
        else:
            n_fail += 1
        n_warn += len(warns)

    print(f"\n{'━'*_W}")
    print(f"  SUMMARY")
    print(f"{'━'*_W}")
    print(f"  Total episodes : {len(files)}")
    print(f"  Passed         : {n_pass}")
    print(f"  Failed         : {n_fail}")
    print(f"  Warnings       : {n_warn}")
    print(f"{'━'*_W}\n")

    sys.exit(0 if n_fail == 0 else 1)


def main():
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    else:
        root = Path(os.environ.get("EPISODES_DIR",
                    f"/media/{os.environ.get('USER','ibrahim')}/seagate/aic_episodes"))

    if not root.exists():
        print(f"[ERROR] Path not found: {root}")
        sys.exit(1)

    check_all(root)


if __name__ == "__main__":
    main()
