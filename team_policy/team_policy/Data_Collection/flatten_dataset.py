#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dataset_dir_pos", nargs="?", help="Folder containing timestamp subfolders")
    parser.add_argument("output_dir_pos", nargs="?", help="Folder where flattened images will be saved")
    parser.add_argument("--source_dataset_dir", dest="source_dataset_dir_flag", help="Folder containing timestamp subfolders")
    parser.add_argument("--output_dir", dest="output_dir_flag", help="Folder where flattened images will be saved")
    args = parser.parse_args()

    source_dataset_dir = args.source_dataset_dir_flag or args.source_dataset_dir_pos
    output_dir_arg = args.output_dir_flag or args.output_dir_pos
    if source_dataset_dir is None or output_dir_arg is None:
        parser.error("source_dataset_dir and output_dir are required")

    source_dir = Path(source_dataset_dir).expanduser().resolve()
    output_dir = Path(output_dir_arg).expanduser().resolve()

    if not source_dir.exists():
        print(f"Source folder does not exist: {source_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    folder_list = sorted([p for p in source_dir.iterdir() if p.is_dir()])

    image_index = 1

    for folder in folder_list:
        ordered_files = [
            folder / "left.png",
            folder / "center.png",
            folder / "right.png",
        ]

        if not all(f.exists() for f in ordered_files):
            print(f"Skipping {folder.name} because one or more files are missing")
            continue

        for image_file in ordered_files:
            destination_file = output_dir / f"{image_index}.png"
            shutil.copy2(image_file, destination_file)
            print(f"Copied {image_file} -> {destination_file}")
            image_index += 1

    print(f"Done. Total images saved: {image_index - 1}")


if __name__ == "__main__":
    main()
