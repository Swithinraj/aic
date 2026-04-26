#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

RUN_PATTERN = re.compile(r"^run_(\d+)$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-dir", default="/home/swithin/official_aic/aic/team_policy/team_policy/training_robot/episodes", help="Path to episodes directory")
    parser.add_argument("--start", required=True, help="Start number, example: 043")
    parser.add_argument("--dry-run", action="store_true", help="Only show changes, do not rename")
    args = parser.parse_args()

    episodes_dir = Path(args.episodes_dir).resolve()

    if not episodes_dir.exists():
        raise FileNotFoundError(f"Directory not found: {episodes_dir}")

    start_num = int(args.start)
    width = len(args.start)

    run_dirs = []

    for item in episodes_dir.iterdir():
        if item.is_dir():
            match = RUN_PATTERN.match(item.name)
            if match:
                old_num = int(match.group(1))
                run_dirs.append((old_num, item))

    run_dirs.sort(key=lambda x: x[0])

    if not run_dirs:
        print("No run folders found.")
        return

    rename_plan = []

    for i, (_, old_path) in enumerate(run_dirs):
        new_num = start_num + i
        new_name = f"run_{new_num:0{width}d}"
        new_path = episodes_dir / new_name
        rename_plan.append((old_path, new_path))

    print("Rename plan:")
    for old_path, new_path in rename_plan:
        print(f"{old_path.name} -> {new_path.name}")

    if args.dry_run:
        print("\nDry run only. No folders were renamed.")
        return

    temp_plan = []

    for i, (old_path, _) in enumerate(rename_plan):
        temp_path = episodes_dir / f"__temp_run_rename_{i:06d}__"
        old_path.rename(temp_path)
        temp_plan.append((temp_path, rename_plan[i][1]))

    for temp_path, final_path in temp_plan:
        temp_path.rename(final_path)

    print("\nRenaming completed.")


if __name__ == "__main__":
    main()