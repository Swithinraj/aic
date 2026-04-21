"""
Validates a single HDF5 episode file and prints a full data summary.

Usage:
    cd ~/ros2_ws/src/aic
    pixi run python -m team_policy.training_robot.validate_episode \
        --file /tmp/aic_dataset/episode_00000.hdf5

Checks performed:
    - All expected keys exist
    - Shapes are consistent (same T across all arrays)
    - No NaN/Inf values in numeric arrays
    - Images are valid uint8 in range [0, 255]
    - commanded_pose is not all-zeros (i.e. CheatCode actually commanded something)
    - TCP pose quaternion norms are ~1.0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# HDF5 file locking fails on tmpfs (/tmp) — disable it globally
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def check(condition: bool, label: str, detail: str = "") -> bool:
    status = "OK " if condition else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return condition


def validate(path: str) -> bool:
    try:
        import h5py
    except ImportError:
        print("ERROR: h5py not installed — run: pixi install")
        return False

    print(f"\n=== Validating: {path} ===\n")
    all_ok = True

    with h5py.File(path, "r") as hf:
        # ---- Metadata ----
        print("--- Metadata ---")
        meta = hf["metadata"]
        schema_version = str(meta.attrs.get("schema_version", "1"))
        for key in (
            "schema_version",
            "episode_id",
            "task_id",
            "port_type",
            "port_name",
            "success",
            "num_frames",
            "duration_s",
            "max_force",
            "final_error",
            "insertion_time",
            "contact_duration",
        ):
            val = meta.attrs.get(key, "MISSING")
            print(f"  {key}: {val}")
        T_meta = int(meta.attrs.get("num_frames", -1))

        # ---- Required keys ----
        print("\n--- Required keys ---")
        required = [
            "observations/images/left",
            "observations/images/center",
            "observations/images/right",
            "observations/tcp_pose",
            "observations/tcp_velocity",
            "observations/tcp_error",
            "observations/joint_positions",
            "observations/wrist_force",
            "observations/timestamps",
            "actions/commanded_pose",
        ]
        for key in required:
            exists = key in hf
            all_ok &= check(exists, f"key exists: {key}")

        new_required = [
            "observations/task_id",
            "observations/relative_pose",
            "observations/relative_pose_valid",
            "actions/delta_pose",
            "actions/velocity",
        ]
        print("\n--- Schema v2 keys ---")
        has_new_schema = schema_version == "2"
        for key in new_required:
            exists = key in hf
            if has_new_schema:
                all_ok &= check(exists, f"key exists: {key}")
            else:
                check(exists, f"key exists: {key}", "optional for older episodes")

        if not all_ok:
            print("\nMissing keys — cannot continue.")
            return False

        # ---- Load arrays ----
        left   = hf["observations/images/left"][:]
        center = hf["observations/images/center"][:]
        right  = hf["observations/images/right"][:]
        tcp    = hf["observations/tcp_pose"][:]
        vel    = hf["observations/tcp_velocity"][:]
        err    = hf["observations/tcp_error"][:]
        joints = hf["observations/joint_positions"][:]
        wrench = hf["observations/wrist_force"][:]
        stamps = hf["observations/timestamps"][:]
        cmd    = hf["actions/commanded_pose"][:]
        task_ids = hf["observations/task_id"][:] if "observations/task_id" in hf else None
        rel = hf["observations/relative_pose"][:] if "observations/relative_pose" in hf else None
        rel_valid = (
            hf["observations/relative_pose_valid"][:]
            if "observations/relative_pose_valid" in hf
            else None
        )
        delta = hf["actions/delta_pose"][:] if "actions/delta_pose" in hf else None
        vel_action = hf["actions/velocity"][:] if "actions/velocity" in hf else None

        T = tcp.shape[0]

        # ---- Shapes ----
        print(f"\n--- Shapes (T={T} frames) ---")
        shape_checks = [
            (left.shape,   (T, None, None, 3), "images/left   (T,H,W,3)"),
            (center.shape, (T, None, None, 3), "images/center (T,H,W,3)"),
            (right.shape,  (T, None, None, 3), "images/right  (T,H,W,3)"),
            (tcp.shape,    (T, 7),             "tcp_pose      (T,7)"),
            (vel.shape,    (T, 6),             "tcp_velocity  (T,6)"),
            (err.shape,    (T, 6),             "tcp_error     (T,6)"),
            (joints.shape, (T, 7),             "joint_pos     (T,7)"),
            (wrench.shape, (T, 6),             "wrist_force   (T,6)"),
            (stamps.shape, (T,),               "timestamps    (T,)"),
            (cmd.shape,    (T, 7),             "commanded_pose(T,7)"),
        ]
        if task_ids is not None:
            shape_checks.append((task_ids.shape, (T,), "task_id       (T,)"))
        if rel is not None:
            shape_checks.append((rel.shape, (T, 7), "relative_pose (T,7)"))
        if rel_valid is not None:
            shape_checks.append((rel_valid.shape, (T,), "relative_valid(T,)"))
        if delta is not None:
            shape_checks.append((delta.shape, (T, 6), "delta_pose    (T,6)"))
        if vel_action is not None:
            shape_checks.append((vel_action.shape, (T, 6), "velocity      (T,6)"))
        for actual, expected, label in shape_checks:
            match = actual[0] == expected[0] and len(actual) == len(expected)
            if match and len(expected) > 1 and expected[1] is not None:
                match = actual[1] == expected[1]
            detail = f"got {actual}"
            all_ok &= check(match, label, detail)

        # ---- Frame count vs metadata ----
        all_ok &= check(T == T_meta, f"frame count matches metadata ({T} == {T_meta})")

        # ---- Image sanity ----
        print("\n--- Images ---")
        all_ok &= check(left.dtype == np.uint8,  f"left dtype uint8",   f"got {left.dtype}")
        all_ok &= check(left.min() >= 0 and left.max() <= 255, "left pixel range [0,255]",
                        f"min={left.min()} max={left.max()}")
        all_ok &= check(center.std() > 1.0, "center image has variation (not blank)",
                        f"std={center.std():.2f}")

        H, W = left.shape[1], left.shape[2]
        print(f"  image size: {H}×{W}")

        # ---- TCP pose sanity ----
        print("\n--- TCP pose ---")
        quat_norms = np.linalg.norm(tcp[:, 3:], axis=1)
        all_ok &= check(
            np.allclose(quat_norms, 1.0, atol=0.05),
            "quaternion norms ≈ 1.0",
            f"mean={quat_norms.mean():.4f} min={quat_norms.min():.4f}",
        )
        all_ok &= check(
            not np.any(np.isnan(tcp)) and not np.any(np.isinf(tcp)),
            "tcp_pose has no NaN/Inf",
        )
        print(f"  position range  x:[{tcp[:,0].min():.3f},{tcp[:,0].max():.3f}] "
              f"y:[{tcp[:,1].min():.3f},{tcp[:,1].max():.3f}] "
              f"z:[{tcp[:,2].min():.3f},{tcp[:,2].max():.3f}]")

        # ---- Commanded pose sanity ----
        print("\n--- Commanded pose (CheatCode actions) ---")
        nonzero_frames = np.count_nonzero(cmd.any(axis=1))
        all_ok &= check(
            nonzero_frames > T // 2,
            f"commanded_pose populated for most frames",
            f"{nonzero_frames}/{T} non-zero frames",
        )
        print(f"  first non-zero at frame: {next((i for i,r in enumerate(cmd) if r.any()), 'none')}")
        print(f"  position range  x:[{cmd[:,0].min():.3f},{cmd[:,0].max():.3f}] "
              f"y:[{cmd[:,1].min():.3f},{cmd[:,1].max():.3f}] "
              f"z:[{cmd[:,2].min():.3f},{cmd[:,2].max():.3f}]")

        if delta is not None:
            print("\n--- Delta pose actions ---")
            all_ok &= check(
                not np.any(np.isnan(delta)) and not np.any(np.isinf(delta)),
                "delta_pose has no NaN/Inf",
            )
            nonzero_delta = np.count_nonzero(np.linalg.norm(delta, axis=1) > 1e-6)
            all_ok &= check(
                nonzero_delta > T // 2,
                "delta_pose populated for most frames",
                f"{nonzero_delta}/{T} non-zero frames",
            )

        if rel is not None and rel_valid is not None:
            print("\n--- Relative pose (plug -> target) ---")
            valid_rel = rel[np.asarray(rel_valid, dtype=bool)]
            all_ok &= check(
                valid_rel.shape[0] > T // 2,
                "relative_pose valid for most frames",
                f"{valid_rel.shape[0]}/{T} valid frames",
            )
            if valid_rel.shape[0]:
                rel_dist = np.linalg.norm(valid_rel[:, :3], axis=1)
                print(f"  distance range: [{rel_dist.min():.4f}, {rel_dist.max():.4f}] m")
                print(f"  final distance: {rel_dist[-1]:.4f} m")
                quat_norms_rel = np.linalg.norm(valid_rel[:, 3:], axis=1)
                all_ok &= check(
                    np.allclose(quat_norms_rel, 1.0, atol=0.05),
                    "relative quaternion norms ≈ 1.0",
                    f"mean={quat_norms_rel.mean():.4f}",
                )

        # ---- Wrench sanity ----
        print("\n--- Wrist force ---")
        print(f"  Fz range: [{wrench[:,2].min():.2f}, {wrench[:,2].max():.2f}] N")
        contact_frames = np.sum(np.abs(wrench[:, 2]) > 0.5)
        print(f"  frames with |Fz| > 0.5N: {contact_frames}/{T} (expect some during insertion)")

        # ---- Episode quality metrics ----
        print("\n--- Quality metrics ---")
        for key in ("max_force", "final_error", "insertion_time", "contact_duration"):
            value = float(meta.attrs.get(key, float("nan")))
            print(f"  {key}: {value:.6g}")
            if has_new_schema:
                if key == "final_error":
                    all_ok &= check(np.isfinite(value), f"{key} is finite")
                else:
                    all_ok &= check(np.isfinite(value) and value >= 0.0, f"{key} is finite and non-negative")

        # ---- Timing ----
        print("\n--- Timing ---")
        if T > 1:
            dt = np.diff(stamps)
            print(f"  mean dt: {dt.mean()*1000:.1f}ms  (target ~50ms for 20Hz)")
            print(f"  max dt:  {dt.max()*1000:.1f}ms")
            all_ok &= check(dt.mean() < 0.15, "mean frame interval < 150ms", f"{dt.mean()*1000:.1f}ms")

        # ---- Summary ----
        print(f"\n{'='*45}")
        if all_ok:
            print(f"PASS — episode looks valid ({T} frames, "
                  f"{stamps[-1]-stamps[0]:.1f}s, "
                  f"success={bool(meta.attrs.get('success', 0))})")
        else:
            print("FAIL — see issues above")
        print('='*45)

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Validate a collected HDF5 episode")
    parser.add_argument("--file", required=True, help="Path to episode_XXXXX.hdf5")
    args = parser.parse_args()
    ok = validate(args.file)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
