#!/usr/bin/env python3
"""
Correlate saved training episodes with their session YAMLs and explain YOLO drops.

Usage:
    cd ~/ros2_ws/src/aic
    pixi run python -m team_policy.training_robot.analyze_yolo_sessions

    pixi run python -m team_policy.training_robot.analyze_yolo_sessions \
        --episodes-dir team_policy/team_policy/training_robot/episodes \
        --sessions-dir team_policy/team_policy/training_robot/configs/sessions \
        --yolo-threshold 0.98
"""
from __future__ import annotations

import argparse
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

# HDF5 file locking fails on tmpfs (/tmp) — disable it globally.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


@dataclass(frozen=True)
class TrialConfig:
    session_file: str
    session_label: str
    trial_index: int
    port_type: str
    port_name: str
    target_module_name: str
    pose_x: float
    pose_y: float
    pose_yaw: float
    target_translation: float | None
    mount_signature: str


@dataclass(frozen=True)
class GapStats:
    total_invalid: int
    leading_invalid: int
    trailing_invalid: int
    longest_invalid: int
    invalid_segments: int
    first_valid_index: int | None
    last_valid_index: int | None
    median_dt_s: float

    @property
    def leading_invalid_s(self) -> float:
        return self.leading_invalid * self.median_dt_s


@dataclass(frozen=True)
class EpisodeSummary:
    run_name: str
    episode_name: str
    run_id: int
    episode_id: int
    port_type: str
    port_name: str
    frames: int
    yolo_valid_fraction: float
    final_error_mm: float
    max_force_n: float
    trial: TrialConfig | None
    gaps: GapStats


def _mount_signature(scene: dict) -> str:
    def _translation(block_name: str) -> str:
        block = scene.get(block_name, {})
        if not block.get("entity_present", False):
            return "off"
        pose = block.get("entity_pose", {})
        value = pose.get("translation", None)
        return "?" if value is None else f"{float(value):+.2f}"

    return (
        f"lc0={_translation('lc_mount_rail_0')} "
        f"sfp0={_translation('sfp_mount_rail_0')} "
        f"sc0={_translation('sc_mount_rail_0')} "
        f"lc1={_translation('lc_mount_rail_1')}"
    )


def _extract_session_label(text: str, fallback: str) -> str:
    match = re.search(r"^# Board pose:\s*([A-Z])\s*$", text, re.MULTILINE)
    return match.group(1) if match else fallback


def _parse_trial_config(session_path: Path) -> dict[int, TrialConfig]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise SystemExit("PyYAML is required. Run inside the pixi environment.") from exc

    text = session_path.read_text()
    data = yaml.safe_load(text)
    trials = data.get("trials", {})
    session_label = _extract_session_label(text, session_path.stem)
    parsed: dict[int, TrialConfig] = {}

    for trial_key in sorted(trials, key=lambda name: int(name.rsplit("_", maxsplit=1)[1])):
        trial_num = int(trial_key.rsplit("_", maxsplit=1)[1])
        trial = trials[trial_key]
        scene = trial["scene"]["task_board"]
        task = trial["tasks"]["task_1"]
        pose = scene["pose"]
        target_module_name = str(task["target_module_name"])
        target_translation: float | None = None

        if target_module_name.startswith("nic_card_mount_"):
            rail = int(target_module_name.rsplit("_", maxsplit=1)[1])
            rail_cfg = scene[f"nic_rail_{rail}"]
            target_translation = float(rail_cfg["entity_pose"]["translation"])
        elif target_module_name.startswith("sc_port_"):
            rail = int(target_module_name.rsplit("_", maxsplit=1)[1])
            rail_cfg = scene[f"sc_rail_{rail}"]
            target_translation = float(rail_cfg["entity_pose"]["translation"])

        parsed[trial_num - 1] = TrialConfig(
            session_file=session_path.name,
            session_label=session_label,
            trial_index=trial_num,
            port_type=str(task["port_type"]),
            port_name=str(task["port_name"]),
            target_module_name=target_module_name,
            pose_x=float(pose["x"]),
            pose_y=float(pose["y"]),
            pose_yaw=float(pose["yaw"]),
            target_translation=target_translation,
            mount_signature=_mount_signature(scene),
        )
    return parsed


def _session_path_for_run(run_id: int, sessions_dir: Path) -> Path | None:
    candidates = [
        sessions_dir / f"session_{run_id:02d}.yaml",
        sessions_dir / f"session_{run_id:03d}.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _median_dt_s(timestamps: np.ndarray) -> float:
    if timestamps.size < 2:
        return 0.0
    dt = np.diff(timestamps.astype(np.float64))
    if not dt.size:
        return 0.0
    return float(np.median(dt))


def _gap_stats(yolo_valid: np.ndarray, timestamps: np.ndarray) -> GapStats:
    total_invalid = int((~yolo_valid).sum())
    leading_invalid = 0
    while leading_invalid < yolo_valid.size and not bool(yolo_valid[leading_invalid]):
        leading_invalid += 1

    trailing_invalid = 0
    idx = yolo_valid.size - 1
    while idx >= 0 and not bool(yolo_valid[idx]):
        trailing_invalid += 1
        idx -= 1

    longest_invalid = 0
    invalid_segments = 0
    current = 0
    for valid in yolo_valid:
        if bool(valid):
            current = 0
            continue
        if current == 0:
            invalid_segments += 1
        current += 1
        longest_invalid = max(longest_invalid, current)

    first_valid_index = next((i for i, valid in enumerate(yolo_valid) if bool(valid)), None)
    last_valid_index = next(
        (i for i in range(yolo_valid.size - 1, -1, -1) if bool(yolo_valid[i])),
        None,
    )

    return GapStats(
        total_invalid=total_invalid,
        leading_invalid=leading_invalid,
        trailing_invalid=trailing_invalid,
        longest_invalid=longest_invalid,
        invalid_segments=invalid_segments,
        first_valid_index=first_valid_index,
        last_valid_index=last_valid_index,
        median_dt_s=_median_dt_s(timestamps),
    )


def _classify_gap(stats: GapStats) -> str:
    if stats.total_invalid == 0:
        return "no YOLO drop"
    if stats.invalid_segments == 1 and stats.total_invalid == stats.leading_invalid:
        return "leading-only drop; likely initial viewpoint or detector warm-up"
    if stats.invalid_segments == 1 and stats.total_invalid == stats.trailing_invalid:
        return "trailing-only drop; likely episode ended before next YOLO update"
    if stats.invalid_segments == 1:
        return "single internal drop; likely temporary occlusion"
    return "multi-gap drop; intermittent detection loss"


def _load_episode_summary(ep_path: Path, trial: TrialConfig | None) -> EpisodeSummary:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise SystemExit("h5py is required. Run inside the pixi environment.") from exc

    run_name = ep_path.parent.name
    run_id = int(run_name.rsplit("_", maxsplit=1)[1])
    episode_id = int(ep_path.stem.rsplit("_", maxsplit=1)[1])

    with h5py.File(ep_path, "r") as hf:
        meta = hf["metadata"].attrs
        yolo_valid = hf["observations/yolo_port_valid"][:].astype(bool)
        timestamps = hf["observations/timestamps"][:]
        gaps = _gap_stats(yolo_valid, timestamps)

        return EpisodeSummary(
            run_name=run_name,
            episode_name=ep_path.name,
            run_id=run_id,
            episode_id=episode_id,
            port_type=str(meta.get("port_type", "?")),
            port_name=str(meta.get("port_name", "?")),
            frames=int(hf["observations/tcp_pose"].shape[0]),
            yolo_valid_fraction=float(meta.get("yolo_valid_fraction", math.nan)),
            final_error_mm=float(meta.get("final_error", math.nan)) * 1000.0,
            max_force_n=float(meta.get("max_force", math.nan)),
            trial=trial,
            gaps=gaps,
        )


def _iter_episode_files(episodes_dir: Path, run_filter: set[int] | None) -> Iterable[Path]:
    for run_dir in sorted(episodes_dir.glob("run_*")):
        try:
            run_id = int(run_dir.name.rsplit("_", maxsplit=1)[1])
        except ValueError:
            continue
        if run_filter is not None and run_id not in run_filter:
            continue
        for ep_path in sorted(run_dir.glob("episode_*.hdf5")):
            yield ep_path


def _print_port_summary(episodes: list[EpisodeSummary]) -> None:
    grouped: dict[str, list[EpisodeSummary]] = defaultdict(list)
    for episode in episodes:
        grouped[episode.port_type].append(episode)

    print("\nBy port type:")
    for port_type in sorted(grouped):
        rows = grouped[port_type]
        yolo_values = [row.yolo_valid_fraction for row in rows]
        print(
            f"  {port_type:<3} episodes={len(rows):2d}  "
            f"avg_yolo={sum(yolo_values) / len(yolo_values):.1%}  "
            f"min_yolo={min(yolo_values):.1%}"
        )


def _print_flagged_episodes(
    episodes: list[EpisodeSummary],
    yolo_threshold: float,
    limit: int,
) -> list[EpisodeSummary]:
    flagged = [ep for ep in episodes if ep.yolo_valid_fraction < yolo_threshold]
    flagged.sort(key=lambda ep: ep.yolo_valid_fraction)

    print(f"\nFlagged episodes (yolo_valid_fraction < {yolo_threshold:.1%}):")
    if not flagged:
        print("  none")
        return []

    for ep in flagged[:limit]:
        trial = ep.trial
        pose = "pose=?"
        session_part = "session=?"
        target_part = "target=?"
        mount_part = "mounts=?"
        if trial is not None:
            pose = (
                f"pose={trial.session_label} "
                f"(x={trial.pose_x:.3f}, y={trial.pose_y:.3f}, yaw={trial.pose_yaw:.2f})"
            )
            session_part = f"{trial.session_file} trial_{trial.trial_index}"
            target_translation = "?" if trial.target_translation is None else f"{trial.target_translation:+.3f}"
            target_part = f"target={trial.target_module_name} t={target_translation}"
            mount_part = f"mounts={trial.mount_signature}"

        print(
            f"  {ep.run_name}/{ep.episode_name}  {session_part}  "
            f"{ep.port_type}/{ep.port_name}  yolo={ep.yolo_valid_fraction:.1%}"
        )
        print(
            f"    invalid={ep.gaps.total_invalid}/{ep.frames}  "
            f"leading={ep.gaps.leading_invalid} ({ep.gaps.leading_invalid_s:.2f}s)  "
            f"longest={ep.gaps.longest_invalid}  segments={ep.gaps.invalid_segments}"
        )
        print(
            f"    {pose}  {target_part}  {mount_part}"
        )
        print(
            f"    reason={_classify_gap(ep.gaps)}  "
            f"final_error={ep.final_error_mm:.1f}mm  max_force={ep.max_force_n:.2f}N"
        )
    return flagged


def _print_group_correlation(
    episodes: list[EpisodeSummary],
    group_name: str,
    group_key,
) -> None:
    grouped: dict[str, list[EpisodeSummary]] = defaultdict(list)
    for episode in episodes:
        if episode.port_type != "sfp" or episode.trial is None:
            continue
        grouped[group_key(episode)].append(episode)

    print(f"\nSFP correlation by {group_name}:")
    if not grouped:
        print("  no SFP episodes with session metadata")
        return

    ranked = sorted(
        grouped.items(),
        key=lambda item: (
            sum(ep.yolo_valid_fraction for ep in item[1]) / len(item[1]),
            min(ep.yolo_valid_fraction for ep in item[1]),
        ),
    )
    for label, rows in ranked:
        avg_yolo = sum(ep.yolo_valid_fraction for ep in rows) / len(rows)
        min_yolo = min(ep.yolo_valid_fraction for ep in rows)
        avg_leading = sum(ep.gaps.leading_invalid for ep in rows) / len(rows)
        print(
            f"  {label}: n={len(rows):2d}  avg_yolo={avg_yolo:.1%}  "
            f"min_yolo={min_yolo:.1%}  avg_leading_invalid={avg_leading:.1f} frames"
        )


def _print_conclusion(flagged: list[EpisodeSummary]) -> None:
    print("\nHeuristic conclusion:")
    if not flagged:
        print("  No episodes crossed the YOLO drop threshold.")
        return

    leading_only = [
        ep
        for ep in flagged
        if ep.gaps.invalid_segments == 1 and ep.gaps.total_invalid == ep.gaps.leading_invalid
    ]
    print(
        f"  {len(leading_only)}/{len(flagged)} flagged episodes are pure leading-only drops."
    )
    if leading_only:
        print(
            "  That points to session pose/clutter affecting the opening view, "
            "not a broken YAML or random mid-episode detector instability."
        )
    else:
        print(
            "  These drops are not purely front-loaded, so inspect mid-episode occlusions "
            "or detector output timing in addition to the session config."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes-dir",
        type=Path,
        default=Path(__file__).parent / "episodes",
        help="Directory containing run_*/episode_*.hdf5 (default: training_robot/episodes)",
    )
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=Path(__file__).parent / "configs" / "sessions",
        help="Directory containing session_XX.yaml files",
    )
    parser.add_argument(
        "--runs",
        type=int,
        nargs="*",
        default=None,
        help="Optional run ids to analyze, e.g. --runs 21 22 23 24",
    )
    parser.add_argument(
        "--yolo-threshold",
        type=float,
        default=0.98,
        help="Episodes below this yolo_valid_fraction are flagged (default: 0.98)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max number of flagged episodes to print (default: 10)",
    )
    args = parser.parse_args()

    episodes_dir = args.episodes_dir.expanduser().resolve()
    sessions_dir = args.sessions_dir.expanduser().resolve()
    run_filter = set(args.runs) if args.runs else None

    if not episodes_dir.exists():
        raise SystemExit(f"Episodes directory not found: {episodes_dir}")
    if not sessions_dir.exists():
        raise SystemExit(f"Sessions directory not found: {sessions_dir}")

    episode_paths = list(_iter_episode_files(episodes_dir, run_filter))
    if not episode_paths:
        raise SystemExit(f"No episodes found under: {episodes_dir}")

    session_cache: dict[int, dict[int, TrialConfig]] = {}
    summaries: list[EpisodeSummary] = []
    missing_sessions: set[int] = set()

    for ep_path in episode_paths:
        run_id = int(ep_path.parent.name.rsplit("_", maxsplit=1)[1])
        if run_id not in session_cache:
            session_path = _session_path_for_run(run_id, sessions_dir)
            if session_path is None:
                missing_sessions.add(run_id)
                session_cache[run_id] = {}
            else:
                session_cache[run_id] = _parse_trial_config(session_path)
        episode_id = int(ep_path.stem.rsplit("_", maxsplit=1)[1])
        trial = session_cache[run_id].get(episode_id)
        summaries.append(_load_episode_summary(ep_path, trial))

    run_count = len({summary.run_name for summary in summaries})
    print(f"Episodes: {len(summaries)} across {run_count} runs")
    print(f"Episodes dir: {episodes_dir}")
    print(f"Sessions dir: {sessions_dir}")

    if missing_sessions:
        missing_str = ", ".join(f"run_{run_id:03d}" for run_id in sorted(missing_sessions))
        print(f"Warning: no matching session YAML found for {missing_str}")

    _print_port_summary(summaries)
    flagged = _print_flagged_episodes(summaries, args.yolo_threshold, args.limit)
    _print_group_correlation(
        summaries,
        "pose",
        lambda ep: (
            f"{ep.trial.session_label} "
            f"(x={ep.trial.pose_x:.3f}, y={ep.trial.pose_y:.3f}, yaw={ep.trial.pose_yaw:.2f})"
        ),
    )
    _print_group_correlation(summaries, "mount pattern", lambda ep: ep.trial.mount_signature)
    _print_conclusion(flagged)


if __name__ == "__main__":
    main()
