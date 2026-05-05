"""
Create a flat symlink directory from the episode database.

Each valid episode in the DB gets a symlink:
    /media/ibrahim/seagate/aic_dataset_preprocessed/merged_symlinks/episode_NNNNNN.hdf5
    -> <absolute source path on drive>

Also exports CSV manifests:
    manifests/all_episodes.csv
    manifests/train_manifest.csv
    manifests/val_manifest.csv
    manifests/test_manifest.csv

Rules:
  - Only creates symlinks — never copies or modifies data
  - Only processes episodes with validation_status = 'valid'
  - Skips duplicates automatically (already marked in DB)
  - Safe to re-run: existing correct symlinks are kept, wrong ones are fixed

Usage:
    cd ~/ros2_ws/src/aic
    pixi run python team_policy/team_policy/training_robot/create_merged_symlinks.py --dry_run
    pixi run python team_policy/team_policy/training_robot/create_merged_symlinks.py
"""

import argparse
import csv
import os
import sqlite3
from pathlib import Path

DB_PATH = Path("/media/ibrahim/seagate/aic_dataset_preprocessed/database/aic_episodes.db")
SYMLINK_DIR = Path("/media/ibrahim/seagate/aic_dataset_preprocessed/merged_symlinks")
MANIFEST_DIR = Path("/media/ibrahim/seagate/aic_dataset_preprocessed/manifests")

MANIFEST_COLS = [
    "episode_uuid", "symlink_name", "source_owner", "source_path",
    "run_folder", "plug_type", "target_module_name", "schema_version",
    "num_frames", "duration_s", "fps_estimated", "final_error_m",
    "yolo_valid_fraction", "validation_status", "split",
]


def create_symlinks(dry_run: bool = False, split: str = "all", schema_version: str = None) -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}\nRun build_episode_database.py first.")

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Build filter
    filters = ["validation_status = 'valid'"]
    params = []
    if split != "all":
        filters.append("split = ?")
        params.append(split)
    if schema_version:
        filters.append("schema_version = ?")
        params.append(schema_version)
    where = " AND ".join(filters)

    valid_rows = conn.execute(
        f"SELECT * FROM episodes WHERE {where} ORDER BY source_owner, source_path",
        params,
    ).fetchall()

    all_rows = conn.execute("SELECT * FROM episodes ORDER BY symlink_name").fetchall()
    conn.close()

    # Re-number sequentially so symlinks are always episode_000000, 000001, ...
    rows = []
    for i, r in enumerate(valid_rows):
        rows.append((f"episode_{i:06d}.hdf5", r))

    print(f"\n=== Symlink creation ({'DRY RUN' if dry_run else 'REAL'}) ===")
    if schema_version:
        print(f"Schema filter: v{schema_version} only")
    print(f"Episodes to link: {len(rows)}")

    created = 0
    skipped = 0
    fixed = 0
    errors = 0

    if not dry_run:
        SYMLINK_DIR.mkdir(parents=True, exist_ok=True)

    for sym_name, row in rows:
        target = Path(row["source_path"])
        link = SYMLINK_DIR / sym_name

        if not target.exists():
            print(f"  ERROR: source missing: {target}")
            errors += 1
            continue

        if dry_run:
            print(f"  WOULD LINK: {sym_name} -> {target}")
            created += 1
            continue

        if link.exists() or link.is_symlink():
            if link.is_symlink() and os.readlink(str(link)) == str(target):
                skipped += 1
                continue
            else:
                link.unlink()
                fixed += 1

        link.symlink_to(target)
        created += 1

    print(f"\nSymlinks: created={created}, skipped_existing={skipped}, fixed={fixed}, errors={errors}")

    # write manifests
    if not dry_run:
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        _write_manifest(all_rows, MANIFEST_DIR / "all_episodes.csv")
        for s in ("train", "val", "test"):
            subset = [r for r in all_rows if r["split"] == s]
            _write_manifest(subset, MANIFEST_DIR / f"{s}_manifest.csv")
        print(f"\nManifests written to: {MANIFEST_DIR}")
        print(f"  all_episodes.csv      : {len(all_rows)} rows")
        for s in ("train", "val", "test"):
            n = sum(1 for r in all_rows if r["split"] == s)
            print(f"  {s}_manifest.csv : {n} rows")
    else:
        print("\n[DRY RUN] No symlinks or manifests written. Remove --dry_run to apply.")


def _write_manifest(rows, path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row[c] for c in MANIFEST_COLS if c in row.keys()})


def print_db_summary() -> None:
    if not DB_PATH.exists():
        print("No database found.")
        return
    conn = sqlite3.connect(str(DB_PATH))
    total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    valid = conn.execute("SELECT COUNT(*) FROM episodes WHERE validation_status='valid'").fetchone()[0]
    dups = conn.execute("SELECT COUNT(*) FROM episodes WHERE validation_status='duplicate'").fetchone()[0]
    invalid = conn.execute("SELECT COUNT(*) FROM episodes WHERE validation_status='invalid'").fetchone()[0]
    by_split = conn.execute(
        "SELECT split, COUNT(*) FROM episodes WHERE validation_status='valid' GROUP BY split"
    ).fetchall()
    by_owner = conn.execute(
        "SELECT source_owner, COUNT(*) FROM episodes WHERE validation_status='valid' GROUP BY source_owner"
    ).fetchall()
    by_plug = conn.execute(
        "SELECT plug_type, COUNT(*) FROM episodes WHERE validation_status='valid' GROUP BY plug_type"
    ).fetchall()
    conn.close()

    print(f"\n=== Database Summary: {DB_PATH} ===")
    print(f"Total episodes : {total}")
    print(f"  valid        : {valid}")
    print(f"  duplicate    : {dups}")
    print(f"  invalid      : {invalid}")
    print(f"  other        : {total - valid - dups - invalid}")
    print(f"By split       : {dict(by_split)}")
    print(f"By owner       : {dict(by_owner)}")
    print(f"By plug type   : {dict(by_plug)}")


def main():
    parser = argparse.ArgumentParser(description="Create flat symlink directory from episode DB")
    parser.add_argument("--dry_run", action="store_true", help="Print what would be done, do not create symlinks")
    parser.add_argument("--split", default="all", choices=["all", "train", "val", "test"],
                        help="Which split to symlink (default: all valid episodes)")
    parser.add_argument("--schema_version", default=None,
                        help="Only symlink episodes with this schema version, e.g. 9")
    parser.add_argument("--summary", action="store_true", help="Print DB summary and exit")
    args = parser.parse_args()

    if args.summary:
        print_db_summary()
        return

    create_symlinks(dry_run=args.dry_run, split=args.split, schema_version=args.schema_version)


if __name__ == "__main__":
    main()
