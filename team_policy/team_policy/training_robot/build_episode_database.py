"""
Scan all episode sources from both drives and build a SQLite database.

Sources scanned:
  seagate:        /media/ibrahim/seagate/aic_episodes
  zaid:           /media/ibrahim/friend_seagate/aic_episodes/aic_episodes_zaid
  ujjwal:         /media/ibrahim/friend_seagate/aic_episodes/episodes_ujjwal
  swithin:        /media/ibrahim/friend_seagate/aic_episodes/intrinsic_swithin
  kbp:            /media/ibrahim/friend_seagate/aic_episodes/KBP_Intrinsic

Rules:
  - Skips any path that is a FILE (not a directory) when walking run_session_* dirs
  - Only processes files matching episode_*.hdf5
  - Reads only HDF5 metadata attrs — never loads image data
  - Never writes to raw episode files

Usage:
    cd ~/ros2_ws/src/aic
    pixi run python team_policy/team_policy/training_robot/build_episode_database.py --dry_run
    pixi run python team_policy/team_policy/training_robot/build_episode_database.py
"""

import argparse
import hashlib
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = Path("/media/ibrahim/seagate/aic_dataset_preprocessed/database/aic_episodes.db")
LOG_PATH = Path("/media/ibrahim/seagate/aic_dataset_preprocessed/logs")

SOURCES = [
    {
        "drive": "seagate",
        "owner": "ibrahim",
        "root": Path("/media/ibrahim/seagate/aic_episodes"),
    },
    {
        "drive": "friend_seagate",
        "owner": "zaid",
        "root": Path("/media/ibrahim/friend_seagate/aic_episodes/aic_episodes_zaid"),
    },
    {
        "drive": "friend_seagate",
        "owner": "ujjwal",
        "root": Path("/media/ibrahim/friend_seagate/aic_episodes/episodes_ujjwal"),
    },
    {
        "drive": "friend_seagate",
        "owner": "swithin",
        "root": Path("/media/ibrahim/friend_seagate/aic_episodes/intrinsic_swithin"),
    },
    {
        "drive": "friend_seagate",
        "owner": "swithin",
        "root": Path("/media/ibrahim/friend_seagate/aic_episodes/intrinsic_swithin/old"),
    },
    {
        "drive": "friend_seagate",
        "owner": "kbp",
        "root": Path("/media/ibrahim/friend_seagate/aic_episodes/KBP_Intrinsic"),
    },
]

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_uuid              TEXT PRIMARY KEY,
    source_drive              TEXT,
    source_owner              TEXT,
    source_path               TEXT UNIQUE,
    run_folder                TEXT,
    episode_filename          TEXT,
    episode_id_local          INTEGER,
    task_id                   TEXT,
    plug_type                 TEXT,
    cable_type                TEXT,
    target_module_name        TEXT,
    schema_version            TEXT,
    success                   INTEGER,
    num_frames                INTEGER,
    duration_s                REAL,
    fps_estimated             REAL,
    final_error_m             REAL,
    max_force_n               REAL,
    yolo_valid_fraction       REAL,
    yolo_fresh_valid_fraction REAL,
    force_baseline_n          REAL,
    file_size_bytes           INTEGER,
    fingerprint               TEXT,
    has_left_cam              INTEGER,
    has_center_cam            INTEGER,
    has_right_cam             INTEGER,
    has_tcp_error             INTEGER,
    has_tared_wrench          INTEGER,
    has_yolo_per_camera       INTEGER,
    has_yolo_age              INTEGER,
    has_target_module_onehot  INTEGER,
    validation_status         TEXT,
    validation_notes          TEXT,
    preprocessing_status      TEXT DEFAULT 'pending',
    split                     TEXT DEFAULT 'unassigned',
    symlink_name              TEXT,
    added_at                  TEXT
);
"""


def fingerprint(path: Path) -> str:
    size = path.stat().st_size
    # lightweight: size + first 4KB hash
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(4096))
    return f"{size}_{h.hexdigest()[:12]}"


def read_episode_metadata(path: Path) -> dict:
    try:
        import h5py
    except ImportError:
        raise RuntimeError("h5py required: run inside pixi environment")

    result = {
        "schema_version": None,
        "episode_id_local": None,
        "task_id": None,
        "plug_type": None,
        "cable_type": None,
        "target_module_name": None,
        "success": None,
        "num_frames": None,
        "duration_s": None,
        "final_error_m": None,
        "max_force_n": None,
        "yolo_valid_fraction": None,
        "yolo_fresh_valid_fraction": None,
        "force_baseline_n": None,
        "has_left_cam": 0,
        "has_center_cam": 0,
        "has_right_cam": 0,
        "has_tcp_error": 0,
        "has_tared_wrench": 0,
        "has_yolo_per_camera": 0,
        "has_yolo_age": 0,
        "has_target_module_onehot": 0,
        "validation_status": "valid",
        "validation_notes": "",
    }

    try:
        with h5py.File(str(path), "r") as hf:
            if "metadata" not in hf:
                result["validation_status"] = "missing_metadata"
                result["validation_notes"] = "no metadata group"
                return result

            attrs = dict(hf["metadata"].attrs)
            result["schema_version"] = str(attrs.get("schema_version", "unknown"))
            result["episode_id_local"] = int(attrs.get("episode_id", -1))
            result["task_id"] = str(attrs.get("task_id", ""))
            result["plug_type"] = str(attrs.get("plug_type", ""))
            result["cable_type"] = str(attrs.get("cable_type", ""))
            result["target_module_name"] = str(attrs.get("target_module_name", ""))

            raw_success = attrs.get("success", None)
            result["success"] = int(raw_success) if raw_success is not None else None

            result["num_frames"] = int(attrs.get("num_frames", 0))
            result["duration_s"] = float(attrs.get("duration_s", 0.0))
            result["final_error_m"] = float(attrs.get("final_error", 0.0)) if "final_error" in attrs else None
            result["max_force_n"] = float(attrs.get("max_force", 0.0)) if "max_force" in attrs else None
            result["yolo_valid_fraction"] = float(attrs.get("yolo_valid_fraction", 0.0)) if "yolo_valid_fraction" in attrs else None
            result["yolo_fresh_valid_fraction"] = float(attrs.get("yolo_fresh_valid_fraction", 0.0)) if "yolo_fresh_valid_fraction" in attrs else None
            result["force_baseline_n"] = float(attrs.get("force_baseline_n", 0.0)) if "force_baseline_n" in attrs else None

            obs = hf.get("observations", {})
            images = obs.get("images", {}) if obs else {}
            result["has_left_cam"] = int("left" in images)
            result["has_center_cam"] = int("center" in images)
            result["has_right_cam"] = int("right" in images)
            result["has_tcp_error"] = int("tcp_error" in obs)
            result["has_tared_wrench"] = int("tared_wrist_force_torque" in obs)
            result["has_yolo_per_camera"] = int("yolo_per_camera" in obs)
            result["has_yolo_age"] = int("yolo_port_age" in obs)
            result["has_target_module_onehot"] = int("target_module_onehot" in obs)

            # basic sanity
            notes = []
            if result["num_frames"] < 10:
                result["validation_status"] = "invalid"
                notes.append(f"too_few_frames({result['num_frames']})")
            if result["success"] == 0:
                result["validation_status"] = "invalid"
                notes.append("success=0")
            if result["schema_version"] not in {"5", "6", "7", "8", "9"}:
                result["validation_status"] = "schema_too_old"
                notes.append(f"schema={result['schema_version']}")
            result["validation_notes"] = "; ".join(notes)

    except Exception as e:
        result["validation_status"] = "corrupted"
        result["validation_notes"] = str(e)

    return result


def collect_episode_files(source: dict) -> list[Path]:
    root = source["root"]
    if not root.exists():
        print(f"  WARNING: {root} does not exist, skipping")
        return []

    files = []
    # Walk: for each item in root, if it's a dir, glob episode_*.hdf5 inside it
    # Also handles flat layout (if root itself has episode_*.hdf5 directly)
    direct = sorted(root.glob("episode_*.hdf5"))
    if direct:
        files.extend(direct)

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            # skip files at this level (e.g. intrinsic_swithin/run_session_139)
            if entry.suffix in {".hdf5", ".h5"}:
                print(f"  SKIP (file not dir): {entry}")
            continue
        eps = sorted(entry.glob("episode_*.hdf5"))
        files.extend(eps)

    return files


def assign_splits(rows: list[dict], seed: int = 42) -> list[dict]:
    import random
    rng = random.Random(seed)

    # stratify by plug_type
    by_type: dict[str, list] = {}
    for r in rows:
        if r["validation_status"] != "valid":
            r["split"] = "excluded"
            continue
        pt = r["plug_type"] or "unknown"
        by_type.setdefault(pt, []).append(r)

    for pt, group in by_type.items():
        rng.shuffle(group)
        n = len(group)
        n_val = max(1, int(n * 0.1))
        n_test = max(1, int(n * 0.1))
        for i, r in enumerate(group):
            if i < n_val:
                r["split"] = "val"
            elif i < n_val + n_test:
                r["split"] = "test"
            else:
                r["split"] = "train"

    return rows


def build_database(dry_run: bool = False, split_seed: int = 42) -> None:
    log_file = LOG_PATH / f"build_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    LOG_PATH.mkdir(parents=True, exist_ok=True)

    all_rows = []
    seen_fingerprints: dict[str, str] = {}
    global_idx = 0

    print("\n=== Scanning episode sources ===")
    for source in SOURCES:
        print(f"\n[{source['owner'].upper()}] {source['root']}")
        files = collect_episode_files(source)
        print(f"  Found {len(files)} episode files")

        for ep_path in files:
            meta = read_episode_metadata(ep_path)
            fp = fingerprint(ep_path)
            file_size = ep_path.stat().st_size

            num_frames = meta["num_frames"] or 0
            duration = meta["duration_s"] or 0.0
            fps_est = round(num_frames / duration, 2) if duration > 0 else None

            notes = meta["validation_notes"] or ""
            if fp in seen_fingerprints:
                meta["validation_status"] = "duplicate"
                notes = f"duplicate_of={seen_fingerprints[fp]}; {notes}"
                meta["validation_notes"] = notes.strip("; ")
            else:
                seen_fingerprints[fp] = ep_path.name

            row = {
                "episode_uuid": str(uuid.uuid4()),
                "source_drive": source["drive"],
                "source_owner": source["owner"],
                "source_path": str(ep_path),
                "run_folder": ep_path.parent.name,
                "episode_filename": ep_path.name,
                "episode_id_local": meta["episode_id_local"],
                "task_id": meta["task_id"],
                "plug_type": meta["plug_type"],
                "cable_type": meta["cable_type"],
                "target_module_name": meta["target_module_name"],
                "schema_version": meta["schema_version"],
                "success": meta["success"],
                "num_frames": num_frames,
                "duration_s": duration,
                "fps_estimated": fps_est,
                "final_error_m": meta["final_error_m"],
                "max_force_n": meta["max_force_n"],
                "yolo_valid_fraction": meta["yolo_valid_fraction"],
                "yolo_fresh_valid_fraction": meta["yolo_fresh_valid_fraction"],
                "force_baseline_n": meta["force_baseline_n"],
                "file_size_bytes": file_size,
                "fingerprint": fp,
                "has_left_cam": meta["has_left_cam"],
                "has_center_cam": meta["has_center_cam"],
                "has_right_cam": meta["has_right_cam"],
                "has_tcp_error": meta["has_tcp_error"],
                "has_tared_wrench": meta["has_tared_wrench"],
                "has_yolo_per_camera": meta["has_yolo_per_camera"],
                "has_yolo_age": meta["has_yolo_age"],
                "has_target_module_onehot": meta["has_target_module_onehot"],
                "validation_status": meta["validation_status"],
                "validation_notes": meta["validation_notes"],
                "preprocessing_status": "pending",
                "split": "unassigned",
                "symlink_name": None,
                "added_at": datetime.now().isoformat(),
            }
            all_rows.append(row)
            global_idx += 1

    # assign splits
    all_rows = assign_splits(all_rows, seed=split_seed)

    # assign symlink names to valid episodes only
    sym_idx = 0
    for r in all_rows:
        if r["validation_status"] == "valid":
            r["symlink_name"] = f"episode_{sym_idx:06d}.hdf5"
            sym_idx += 1

    # print summary
    print(f"\n=== Summary ===")
    total = len(all_rows)
    by_status: dict[str, int] = {}
    by_owner: dict[str, int] = {}
    by_split: dict[str, int] = {}
    by_plug: dict[str, int] = {}
    for r in all_rows:
        by_status[r["validation_status"]] = by_status.get(r["validation_status"], 0) + 1
        by_owner[r["source_owner"]] = by_owner.get(r["source_owner"], 0) + 1
        by_split[r["split"]] = by_split.get(r["split"], 0) + 1
        pt = r["plug_type"] or "unknown"
        by_plug[pt] = by_plug.get(pt, 0) + 1

    print(f"Total episodes scanned : {total}")
    print(f"By owner               : {by_owner}")
    print(f"By validation status   : {by_status}")
    print(f"By split               : {by_split}")
    print(f"By plug type           : {by_plug}")
    print(f"Symlinked (valid only) : {sym_idx}")

    if dry_run:
        print("\n[DRY RUN] Database NOT written. Remove --dry_run to write.")
        return

    # write DB
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(CREATE_TABLE)
    conn.commit()

    cols = [
        "episode_uuid", "source_drive", "source_owner", "source_path",
        "run_folder", "episode_filename", "episode_id_local", "task_id",
        "plug_type", "cable_type", "target_module_name", "schema_version",
        "success", "num_frames", "duration_s", "fps_estimated",
        "final_error_m", "max_force_n", "yolo_valid_fraction",
        "yolo_fresh_valid_fraction", "force_baseline_n",
        "file_size_bytes", "fingerprint",
        "has_left_cam", "has_center_cam", "has_right_cam",
        "has_tcp_error", "has_tared_wrench", "has_yolo_per_camera",
        "has_yolo_age", "has_target_module_onehot",
        "validation_status", "validation_notes",
        "preprocessing_status", "split", "symlink_name", "added_at",
    ]
    placeholders = ", ".join("?" * len(cols))
    sql = f"INSERT OR REPLACE INTO episodes ({', '.join(cols)}) VALUES ({placeholders})"
    conn.executemany(sql, [[r[c] for c in cols] for r in all_rows])
    conn.commit()
    conn.close()

    print(f"\nDatabase written to: {DB_PATH}")
    print(f"Log: {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Build AIC episode SQLite database")
    parser.add_argument("--dry_run", action="store_true", help="Scan only, do not write DB")
    parser.add_argument("--split_seed", type=int, default=42, help="Random seed for train/val/test split")
    args = parser.parse_args()
    build_database(dry_run=args.dry_run, split_seed=args.split_seed)


if __name__ == "__main__":
    main()
