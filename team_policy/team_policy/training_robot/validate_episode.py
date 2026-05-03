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


def _schema_version_number(schema_version: str) -> int:
    try:
        return int(str(schema_version).split(".", maxsplit=1)[0])
    except (TypeError, ValueError):
        return 1


def _decode_strings(values) -> list[str]:
    decoded = []
    for item in values:
        if isinstance(item, bytes):
            decoded.append(item.decode("utf-8"))
        else:
            decoded.append(str(item))
    return decoded


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
        schema_version_num = _schema_version_number(schema_version)
        has_v2_schema = schema_version_num >= 2
        has_v3_schema = schema_version_num >= 3
        has_v4_schema = schema_version_num >= 4
        has_v5_schema = schema_version_num >= 5
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
            "sustained_penalty_duration_s",
            "yolo_valid_fraction",
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
        for key in new_required:
            exists = key in hf
            if has_v2_schema:
                all_ok &= check(exists, f"key exists: {key}")
            else:
                check(
                    True,
                    f"optional key: {key}",
                    "present" if exists else "not present in older episode",
                )

        v3_required = [
            "observations/privileged_tf/transforms",
            "observations/privileged_tf/valid",
            "observations/privileged_tf/frame_pairs",
        ]
        print("\n--- Schema v3 TF keys ---")
        for key in v3_required:
            exists = key in hf
            if has_v3_schema:
                all_ok &= check(exists, f"key exists: {key}")
            else:
                check(
                    True,
                    f"optional key: {key}",
                    "present" if exists else "not present in older episode",
                )

        v4_required = ["observations/joint_velocity"]
        print("\n--- Schema v4 keys ---")
        for key in v4_required:
            exists = key in hf
            if has_v4_schema:
                all_ok &= check(exists, f"key exists: {key}")
            else:
                check(
                    True,
                    f"optional key: {key}",
                    "present" if exists else "not present in older episode",
                )
        if has_v4_schema and all(key in hf for key in v4_required):
            print("Schema v4 keys — joint_velocity present")

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
        joint_vel = (
            hf["observations/joint_velocity"][:]
            if "observations/joint_velocity" in hf
            else None
        )
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
        privileged_tf = (
            hf["observations/privileged_tf/transforms"][:]
            if "observations/privileged_tf/transforms" in hf
            else None
        )
        privileged_tf_valid = (
            hf["observations/privileged_tf/valid"][:]
            if "observations/privileged_tf/valid" in hf
            else None
        )
        privileged_tf_frame_pairs = (
            _decode_strings(hf["observations/privileged_tf/frame_pairs"][:])
            if "observations/privileged_tf/frame_pairs" in hf
            else []
        )

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
        if joint_vel is not None:
            shape_checks.append((joint_vel.shape, (T, 7), "joint_velocity(T,7)"))
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

        if privileged_tf is not None and privileged_tf_valid is not None:
            tf_count = privileged_tf.shape[1] if privileged_tf.ndim >= 2 else -1
            all_ok &= check(
                privileged_tf.ndim == 3 and privileged_tf.shape == (T, tf_count, 7),
                "privileged_tf transforms (T,N,7)",
                f"got {privileged_tf.shape}",
            )
            all_ok &= check(
                privileged_tf_valid.shape == (T, tf_count),
                "privileged_tf valid mask (T,N)",
                f"got {privileged_tf_valid.shape}",
            )
            all_ok &= check(
                len(privileged_tf_frame_pairs) == tf_count,
                "privileged_tf frame pair labels (N)",
                f"{len(privileged_tf_frame_pairs)} labels for {tf_count} transforms",
            )

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

        if joint_vel is not None:
            print("\n--- Joint velocity ---")
            all_ok &= check(
                not np.any(np.isnan(joint_vel)) and not np.any(np.isinf(joint_vel)),
                "joint_velocity has no NaN/Inf",
            )
            nonzero_joint_vel = int(np.count_nonzero(np.abs(joint_vel) > 1e-8))
            all_ok &= check(
                nonzero_joint_vel > 0,
                "joint_velocity is not all zeros",
                f"{nonzero_joint_vel}/{joint_vel.size} non-zero values",
            )
            print(f"  velocity range: [{joint_vel.min():.6f}, {joint_vel.max():.6f}] rad/s")

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

        if privileged_tf is not None and privileged_tf_valid is not None:
            print("\n--- Privileged TF snapshots ---")
            tf_valid_mask = np.asarray(privileged_tf_valid, dtype=bool)
            for idx, label in enumerate(privileged_tf_frame_pairs):
                valid_count = int(tf_valid_mask[:, idx].sum())
                print(f"  {idx}: {label} valid={valid_count}/{T}")

            valid_tf = privileged_tf[tf_valid_mask]
            all_ok &= check(
                not np.any(np.isnan(valid_tf)) and not np.any(np.isinf(valid_tf)),
                "valid privileged_tf transforms have no NaN/Inf",
            )
            if valid_tf.shape[0]:
                tf_quat_norms = np.linalg.norm(valid_tf[:, 3:], axis=1)
                all_ok &= check(
                    np.allclose(tf_quat_norms, 1.0, atol=0.05),
                    "privileged_tf quaternion norms ≈ 1.0",
                    f"mean={tf_quat_norms.mean():.4f}",
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
            if has_v2_schema:
                if key == "final_error":
                    all_ok &= check(np.isfinite(value), f"{key} is finite")
                else:
                    all_ok &= check(np.isfinite(value) and value >= 0.0, f"{key} is finite and non-negative")

        if has_v5_schema:
            print("\n--- Schema v5 quality metrics ---")
            sus = float(meta.attrs.get("sustained_penalty_duration_s", float("nan")))
            yolo_frac = float(meta.attrs.get("yolo_valid_fraction", float("nan")))
            force_thr = float(meta.attrs.get("force_penalty_threshold_n", 20.0))
            port_name_str = str(meta.attrs.get("port_name", ""))
            print(f"  sustained_penalty_duration_s: {sus:.3f}s  (above {force_thr}N)")
            print(f"  yolo_valid_fraction: {yolo_frac:.2%}")
            all_ok &= check(
                np.isfinite(sus) and sus >= 0.0,
                "sustained_penalty_duration_s is finite and non-negative",
                f"{sus:.3f}s",
            )
            all_ok &= check(
                sus < 1.0,
                "sustained_penalty_duration_s < 1.0s (competition penalty threshold)",
                f"{sus:.3f}s",
            )
            if "sfp" in port_name_str.lower():
                all_ok &= check(
                    yolo_frac >= 0.5,
                    "yolo_valid_fraction >= 50% for SFP port",
                    f"{yolo_frac:.2%}",
                )

        # ---- Timing ----
        print("\n--- Timing ---")
        if T > 1:
            dt = np.diff(stamps)
            print(f"  mean dt: {dt.mean()*1000:.1f}ms  (target ~100ms for 10Hz, or downsampled later)")
            print(f"  max dt:  {dt.max()*1000:.1f}ms")
            all_ok &= check(dt.mean() < 0.20, "mean frame interval < 200ms", f"{dt.mean()*1000:.1f}ms")

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
