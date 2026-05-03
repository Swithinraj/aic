#!/usr/bin/env python3
"""
Generate competition-format session YAMLs.

Each session = 3 trials sharing ONE fixed board pose.
  - Trial 1: NIC insertion (sfp)
  - Trial 2: NIC insertion (sfp, different rail)
  - Trial 3: SC  insertion

Board pose is FIXED within a session so the model sees a consistent scene.
Board pose varies BETWEEN sessions for training diversity.

Only the target entity is spawned per trial (lean spawn = lower RAM).
Background mounts cycle through 3 sets for visual variety.

Run:
    python generate_competition_sessions.py [--sessions 50] [--out-dir sessions]
    python generate_competition_sessions.py --sessions 50 --trials-per-session 1
    python generate_competition_sessions.py --layout nic_triplets --sessions 100 --out-dir sessions_nic_3trial

Output:
    sessions/session_01.yaml ... session_50.yaml
    or, with --trials-per-session 1:
    sessions/session_001.yaml ... session_150.yaml
    or, with --layout nic_triplets:
    sessions_nic_3trial/session_01.yaml ... session_100.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Board poses (10 distinct table poses)
#
# Keep the board on the table: z is fixed, roll/pitch are fixed at zero.
# Only x/y and yaw vary between sessions.
# ---------------------------------------------------------------------------

POSES = [
    dict(x=0.160, y=-0.180, z=1.14, roll= 0.00, pitch= 0.00, yaw=2.80),  # A
    dict(x=0.140, y=-0.220, z=1.14, roll= 0.00, pitch= 0.00, yaw=3.35),  # B
    dict(x=0.175, y=-0.150, z=1.14, roll= 0.00, pitch= 0.00, yaw=2.55),  # C
    dict(x=0.155, y=-0.250, z=1.14, roll= 0.00, pitch= 0.00, yaw=3.10),  # D
    dict(x=0.195, y=-0.100, z=1.14, roll= 0.00, pitch= 0.00, yaw=2.70),  # E
    dict(x=0.120, y=-0.175, z=1.14, roll= 0.00, pitch= 0.00, yaw=2.90),  # F
    dict(x=0.215, y=-0.155, z=1.14, roll= 0.00, pitch= 0.00, yaw=3.20),  # G
    dict(x=0.165, y=-0.060, z=1.14, roll= 0.00, pitch= 0.00, yaw=2.65),  # H
    dict(x=0.185, y=-0.195, z=1.14, roll= 0.00, pitch= 0.00, yaw=3.50),  # I
    dict(x=0.135, y=-0.125, z=1.14, roll= 0.00, pitch= 0.00, yaw=2.45),  # J
]

# ---------------------------------------------------------------------------
# Background mounts (3 visual-clutter sets — keep translation small and fixed)
# ---------------------------------------------------------------------------

MOUNT_SETS = [
    dict(sfp_t= 0.03, sc_t=-0.02, lc0_t= 0.02, lc1_t=-0.01),
    dict(sfp_t= 0.05, sc_t= 0.04, lc0_t= 0.04, lc1_t= 0.03),
    dict(sfp_t=-0.03, sc_t=-0.05, lc0_t=-0.01, lc1_t=-0.03),
]

# ---------------------------------------------------------------------------
# NIC and SC rail translations (2 options each — small, realistic)
# ---------------------------------------------------------------------------

NIC_TRANS_OPTIONS = [0.015, -0.015, 0.000, 0.020, -0.010]
SC_TRANS_OPTIONS  = [0.042, -0.042, 0.000, 0.030, -0.030]

# ---------------------------------------------------------------------------
# Trial sequences for each session (NIC rail 0-4, SC rail 0-1)
# Cycle through all rails evenly across 50 sessions.
# Pattern per session: [nic_railA, nic_railB, sc_railC]
# ---------------------------------------------------------------------------

SESSION_PATTERNS = [
    (0, 2, 0), (1, 3, 1), (4, 0, 0), (2, 1, 1), (3, 4, 0),
    (0, 3, 1), (1, 4, 0), (2, 0, 1), (3, 1, 0), (4, 2, 1),
    (0, 4, 0), (1, 2, 1), (3, 0, 0), (4, 1, 1), (2, 3, 0),
    (0, 1, 1), (2, 4, 0), (3, 2, 1), (1, 0, 0), (4, 3, 1),
    (0, 2, 1), (1, 3, 0), (4, 0, 1), (2, 1, 0), (3, 4, 1),
    (0, 3, 0), (1, 4, 1), (2, 0, 0), (3, 1, 1), (4, 2, 0),
    (0, 4, 1), (1, 2, 0), (3, 0, 1), (4, 1, 0), (2, 3, 1),
    (0, 1, 0), (2, 4, 1), (3, 2, 0), (1, 0, 1), (4, 3, 0),
    (0, 2, 0), (1, 3, 1), (4, 0, 0), (2, 1, 1), (3, 4, 0),
    (0, 3, 1), (1, 4, 0), (2, 0, 1), (3, 1, 0), (4, 2, 1),
]


# ---------------------------------------------------------------------------
# YAML block helpers
# ---------------------------------------------------------------------------

HEADER_TMPL = """\
# Competition-format session — 3 trials, FIXED board pose.
#
# Board does NOT move between trials within this session.
# This prevents score drops from pose discontinuity.
# Board stays on the table: z=1.14, roll=0.0, pitch=0.0.
# Only x/y and yaw vary between sessions.
#
# Session {session_num:02d} of {total}
# Board pose: {pose_label}
# Trials: {trial_sequence}
#
# Usage:
#   aic_engine_config_file:={yaml_path}

scoring:
  topics:
    - topic:
        name: "/joint_states"
        type: "sensor_msgs/msg/JointState"
    - topic:
        name: "/tf"
        type: "tf2_msgs/msg/TFMessage"
    - topic:
        name: "/tf_static"
        type: "tf2_msgs/msg/TFMessage"
        latched: true
    - topic:
        name: "/scoring/tf"
        type: "tf2_msgs/msg/TFMessage"
    - topic:
        name: "/aic/gazebo/contacts/off_limit"
        type: "ros_gz_interfaces/msg/Contacts"
    - topic:
        name: "/fts_broadcaster/wrench"
        type: "geometry_msgs/msg/WrenchStamped"
    - topic:
        name: "/aic_controller/joint_commands"
        type: "aic_control_interfaces/msg/JointMotionUpdate"
    - topic:
        name: "/aic_controller/pose_commands"
        type: "aic_control_interfaces/msg/MotionUpdate"
    - topic:
        name: "/scoring/insertion_event"
        type: "std_msgs/msg/String"
    - topic:
        name: "/aic_controller/controller_state"
        type: "aic_control_interfaces/msg/ControllerState"

task_board_limits:
  nic_rail:
    min_translation: -0.0215
    max_translation: 0.0234
  sc_rail:
    min_translation: -0.06
    max_translation: 0.055
  mount_rail:
    min_translation: -0.09425
    max_translation: 0.09425
trials:
"""

SINGLE_TRIAL_HEADER_TMPL = """\
# Single-trial training session.
#
# Use this when the eval container leaves the gripper open between trials.
# Relaunch Gazebo for every file so the robot, gripper, cable, and board start fresh.
# Board stays on the table: z=1.14, roll=0.0, pitch=0.0.
# Only x/y and yaw vary between source sessions.
#
# File {file_num:03d} of {total_files}
# Source competition session {source_session:02d}, trial {source_trial}
# Board pose: {pose_label}
# Task: {task_label}
#
# Usage:
#   aic_engine_config_file:={yaml_path}

scoring:
  topics:
    - topic:
        name: "/joint_states"
        type: "sensor_msgs/msg/JointState"
    - topic:
        name: "/tf"
        type: "tf2_msgs/msg/TFMessage"
    - topic:
        name: "/tf_static"
        type: "tf2_msgs/msg/TFMessage"
        latched: true
    - topic:
        name: "/scoring/tf"
        type: "tf2_msgs/msg/TFMessage"
    - topic:
        name: "/aic/gazebo/contacts/off_limit"
        type: "ros_gz_interfaces/msg/Contacts"
    - topic:
        name: "/fts_broadcaster/wrench"
        type: "geometry_msgs/msg/WrenchStamped"
    - topic:
        name: "/aic_controller/joint_commands"
        type: "aic_control_interfaces/msg/JointMotionUpdate"
    - topic:
        name: "/aic_controller/pose_commands"
        type: "aic_control_interfaces/msg/MotionUpdate"
    - topic:
        name: "/scoring/insertion_event"
        type: "std_msgs/msg/String"
    - topic:
        name: "/aic_controller/controller_state"
        type: "aic_control_interfaces/msg/ControllerState"

task_board_limits:
  nic_rail:
    min_translation: -0.0215
    max_translation: 0.0234
  sc_rail:
    min_translation: -0.06
    max_translation: 0.055
  mount_rail:
    min_translation: -0.09425
    max_translation: 0.09425
trials:
"""

FOOTER = """\
robot:
  home_joint_positions:
    shoulder_pan_joint: -0.1597
    shoulder_lift_joint: -1.3542
    elbow_joint: -1.6648
    wrist_1_joint: -1.6933
    wrist_2_joint: 1.5710
    wrist_3_joint: 1.4110
"""

NIC_TRIPLET_TARGETS = [(rail, trans) for rail in range(5) for trans in (0.015, -0.015)]


def _pose_block(pose: dict, indent: int = 10) -> str:
    pad = " " * indent
    return (
        f"{pad}pose:\n"
        f"{pad}  x: {pose['x']:.3f}\n"
        f"{pad}  y: {pose['y']:.3f}\n"
        f"{pad}  z: {pose['z']:.2f}\n"
        f"{pad}  roll: {pose['roll']:.2f}\n"
        f"{pad}  pitch: {pose['pitch']:.2f}\n"
        f"{pad}  yaw: {pose['yaw']:.2f}\n"
    )


def _nic_rails_block(target_rail: int, translation: float, indent: int = 10) -> str:
    pad = " " * indent
    lines = []
    for i in range(5):
        if i == target_rail:
            lines.append(
                f"{pad}nic_rail_{i}:\n"
                f"{pad}  entity_present: True\n"
                f"{pad}  entity_name: \"nic_card_{i}\"\n"
                f"{pad}  entity_pose:\n"
                f"{pad}    translation: {translation:.3f}\n"
                f"{pad}    roll: 0.0\n"
                f"{pad}    pitch: 0.0\n"
                f"{pad}    yaw: 0.0\n"
            )
        else:
            lines.append(f"{pad}nic_rail_{i}:\n{pad}  entity_present: False\n")
    return "".join(lines)


def _sc_rails_block(target_rail: int, translation: float, indent: int = 10) -> str:
    pad = " " * indent
    lines = []
    for i in range(2):
        if i == target_rail:
            lines.append(
                f"{pad}sc_rail_{i}:\n"
                f"{pad}  entity_present: True\n"
                f"{pad}  entity_name: \"sc_mount_{i}\"\n"
                f"{pad}  entity_pose:\n"
                f"{pad}    translation: {translation:.3f}\n"
                f"{pad}    roll: 0.0\n"
                f"{pad}    pitch: 0.0\n"
                f"{pad}    yaw: 0.0\n"
            )
        else:
            lines.append(f"{pad}sc_rail_{i}:\n{pad}  entity_present: False\n")
    return "".join(lines)


def _background_mounts(mounts: dict, indent: int = 10, is_sc_trial: bool = False) -> str:
    """Background visual mounts.

    For SC trials, sc_mount_rail_0 is disabled because it shares entity_name
    'sc_mount_0' with sc_rail_0 (the functional port). Gazebo cannot have two
    entities with the same name — the conflict causes the functional port to
    spawn at the wrong position and CheatCode misses the insertion.
    """
    pad = " " * indent
    if is_sc_trial:
        sc_mount_block = f"{pad}sc_mount_rail_0:\n{pad}  entity_present: False\n"
    else:
        sc_mount_block = (
            f"{pad}sc_mount_rail_0:\n"
            f"{pad}  entity_present: True\n"
            f"{pad}  entity_name: \"sc_mount_0\"\n"
            f"{pad}  entity_pose:\n"
            f"{pad}    translation: {mounts['sc_t']:.2f}\n"
            f"{pad}    roll: 0.0\n"
            f"{pad}    pitch: 0.0\n"
            f"{pad}    yaw: 0.0\n"
        )
    return (
        f"{pad}lc_mount_rail_0:\n"
        f"{pad}  entity_present: True\n"
        f"{pad}  entity_name: \"lc_mount_0\"\n"
        f"{pad}  entity_pose:\n"
        f"{pad}    translation: {mounts['lc0_t']:.2f}\n"
        f"{pad}    roll: 0.0\n"
        f"{pad}    pitch: 0.0\n"
        f"{pad}    yaw: 0.0\n"
        f"{pad}sfp_mount_rail_0:\n"
        f"{pad}  entity_present: True\n"
        f"{pad}  entity_name: \"sfp_mount_0\"\n"
        f"{pad}  entity_pose:\n"
        f"{pad}    translation: {mounts['sfp_t']:.2f}\n"
        f"{pad}    roll: 0.0\n"
        f"{pad}    pitch: 0.0\n"
        f"{pad}    yaw: 0.0\n"
        + sc_mount_block +
        f"{pad}lc_mount_rail_1:\n"
        f"{pad}  entity_present: True\n"
        f"{pad}  entity_name: \"lc_mount_1\"\n"
        f"{pad}  entity_pose:\n"
        f"{pad}    translation: {mounts['lc1_t']:.2f}\n"
        f"{pad}    roll: 0.0\n"
        f"{pad}    pitch: 0.0\n"
        f"{pad}    yaw: 0.0\n"
        f"{pad}sfp_mount_rail_1:\n"
        f"{pad}  entity_present: False\n"
        f"{pad}sc_mount_rail_1:\n"
        f"{pad}  entity_present: False\n"
    )


def _sfp_cable(indent: int = 8) -> str:
    pad = " " * indent
    return (
        f"{pad}cables:\n"
        f"{pad}  cable_0:\n"
        f"{pad}    pose:\n"
        f"{pad}      gripper_offset:\n"
        f"{pad}        x: 0.0\n"
        f"{pad}        y: 0.015385\n"
        f"{pad}        z: 0.04245\n"
        f"{pad}      roll: 0.4432\n"
        f"{pad}      pitch: -0.4838\n"
        f"{pad}      yaw: 1.3303\n"
        f"{pad}    attach_cable_to_gripper: True\n"
        f"{pad}    cable_type: \"sfp_sc_cable\"\n"
    )


def _sc_cable(indent: int = 8) -> str:
    pad = " " * indent
    return (
        f"{pad}cables:\n"
        f"{pad}  cable_1:\n"
        f"{pad}    pose:\n"
        f"{pad}      gripper_offset:\n"
        f"{pad}        x: 0.0\n"
        f"{pad}        y: 0.015385\n"
        f"{pad}        z: 0.04045\n"
        f"{pad}      roll: 0.4432\n"
        f"{pad}      pitch: -0.4838\n"
        f"{pad}      yaw: 1.3303\n"
        f"{pad}    attach_cable_to_gripper: True\n"
        f"{pad}    cable_type: \"sfp_sc_cable_reversed\"\n"
    )


def _sfp_task(nic_rail: int, indent: int = 4) -> str:
    pad = " " * indent
    return (
        f"{pad}tasks:\n"
        f"{pad}  task_1:\n"
        f"{pad}    cable_type: \"sfp_sc\"\n"
        f"{pad}    cable_name: \"cable_0\"\n"
        f"{pad}    plug_type: \"sfp\"\n"
        f"{pad}    plug_name: \"sfp_tip\"\n"
        f"{pad}    port_type: \"sfp\"\n"
        f"{pad}    port_name: \"sfp_port_0\"\n"
        f"{pad}    target_module_name: \"nic_card_mount_{nic_rail}\"\n"
        f"{pad}    time_limit: 180\n"
    )


def _sc_task(sc_rail: int, indent: int = 4) -> str:
    pad = " " * indent
    return (
        f"{pad}tasks:\n"
        f"{pad}  task_1:\n"
        f"{pad}    cable_type: \"sfp_sc\"\n"
        f"{pad}    cable_name: \"cable_1\"\n"
        f"{pad}    plug_type: \"sc\"\n"
        f"{pad}    plug_name: \"sc_tip\"\n"
        f"{pad}    port_type: \"sc\"\n"
        f"{pad}    port_name: \"sc_port_base\"\n"
        f"{pad}    target_module_name: \"sc_port_{sc_rail}\"\n"
        f"{pad}    time_limit: 180\n"
    )


# ---------------------------------------------------------------------------
# Trial builder
# ---------------------------------------------------------------------------

def build_nic_trial(trial_num: int, pose: dict, nic_rail: int,
                    nic_trans: float, mounts: dict) -> str:
    lines = [f"  trial_{trial_num}:"]
    lines.append("    scene:")
    lines.append("        task_board:")
    lines.append(_pose_block(pose, indent=10).rstrip())
    lines.append(_nic_rails_block(nic_rail, nic_trans, indent=10).rstrip())
    lines.append(_sc_rails_block(-1, 0.0, indent=10).rstrip())  # all SC absent
    lines.append(_background_mounts(mounts, indent=10).rstrip())
    lines.append(_sfp_cable(indent=8).rstrip())
    lines.append(_sfp_task(nic_rail, indent=4).rstrip())
    return "\n".join(lines) + "\n"


def build_sc_trial(trial_num: int, pose: dict, sc_rail: int,
                   sc_trans: float, mounts: dict) -> str:
    lines = [f"  trial_{trial_num}:"]
    lines.append("    scene:")
    lines.append("        task_board:")
    lines.append(_pose_block(pose, indent=10).rstrip())
    lines.append(_nic_rails_block(-1, 0.0, indent=10).rstrip())  # all NIC absent
    lines.append(_sc_rails_block(sc_rail, sc_trans, indent=10).rstrip())
    lines.append(_background_mounts(mounts, indent=10, is_sc_trial=True).rstrip())
    lines.append(_sc_cable(indent=8).rstrip())
    lines.append(_sc_task(sc_rail, indent=4).rstrip())
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Session builder
# ---------------------------------------------------------------------------

def build_session(session_num: int, total: int, out_dir: Path) -> Path:
    pattern_idx = (session_num - 1) % len(SESSION_PATTERNS)
    pose_idx     = (session_num - 1) % len(POSES)
    mount_idx    = (session_num - 1) % len(MOUNT_SETS)

    nic_a, nic_b, sc_c = SESSION_PATTERNS[pattern_idx]
    pose   = POSES[pose_idx]
    mounts = MOUNT_SETS[mount_idx]

    nic_trans_a = NIC_TRANS_OPTIONS[nic_a % len(NIC_TRANS_OPTIONS)]
    nic_trans_b = NIC_TRANS_OPTIONS[nic_b % len(NIC_TRANS_OPTIONS)]
    sc_trans_c  = SC_TRANS_OPTIONS[sc_c  % len(SC_TRANS_OPTIONS)]

    pose_label = "ABCDEFGHIJ"[pose_idx]
    yaml_path  = out_dir / f"session_{session_num:02d}.yaml"

    if 11 <= session_num <= 20:
        trial_str = f"SC{sc_c} → NIC{nic_a} → NIC{nic_b}"
    else:
        trial_str = f"NIC{nic_a} → NIC{nic_b} → SC{sc_c}"

    header = HEADER_TMPL.format(
        session_num=session_num,
        total=total,
        pose_label=pose_label,
        trial_sequence=trial_str,
        yaml_path=str(yaml_path),
    )

    if 11 <= session_num <= 20:
        # SC port first, then two NIC insertions
        content = header
        content += build_sc_trial( 1, pose, sc_c,  sc_trans_c,  mounts)
        content += build_nic_trial(2, pose, nic_a, nic_trans_a, mounts)
        content += build_nic_trial(3, pose, nic_b, nic_trans_b, mounts)
    else:
        content = header
        content += build_nic_trial(1, pose, nic_a, nic_trans_a, mounts)
        content += build_nic_trial(2, pose, nic_b, nic_trans_b, mounts)
        content += build_sc_trial( 3, pose, sc_c,  sc_trans_c,  mounts)
    content += FOOTER

    yaml_path.write_text(content)
    return yaml_path


def build_nic_triplet_session(session_num: int, total: int, out_dir: Path) -> Path:
    _, pose, mounts, pose_label = _session_spec(session_num)
    yaml_path = out_dir / f"session_{session_num:02d}.yaml"

    round_idx = (session_num - 1) // len(POSES)
    base = round_idx % len(NIC_TRIPLET_TARGETS)
    targets = [NIC_TRIPLET_TARGETS[(base + offset) % len(NIC_TRIPLET_TARGETS)] for offset in (0, 3, 6)]
    trial_str = " → ".join(f"NIC{rail}({trans:+.3f})" for rail, trans in targets)

    header = HEADER_TMPL.format(
        session_num=session_num,
        total=total,
        pose_label=pose_label,
        trial_sequence=trial_str,
        yaml_path=str(yaml_path),
    )
    header = header.replace("# Competition-format session — 3 trials, FIXED board pose.",
                            "# NIC-only training session — 3 trials.")
    header = header.replace("# Board does NOT move between trials within this session.",
                            "# All three trials share one fixed board pose (consistent scene for the model).")
    header = header.replace("# This prevents score drops from pose discontinuity.",
                            "# Only the NIC rail target and translation change between trials.")

    content = header
    for trial_num, (nic_rail, nic_trans) in enumerate(targets, start=1):
        content += build_nic_trial(trial_num, pose, nic_rail, nic_trans, mounts)
    content += FOOTER

    yaml_path.write_text(content)
    return yaml_path


def _session_spec(session_num: int) -> tuple[tuple[int, int, int], dict, dict, str]:
    pattern_idx = (session_num - 1) % len(SESSION_PATTERNS)
    pose_idx = (session_num - 1) % len(POSES)
    mount_idx = (session_num - 1) % len(MOUNT_SETS)
    return (
        SESSION_PATTERNS[pattern_idx],
        POSES[pose_idx],
        MOUNT_SETS[mount_idx],
        "ABCDEFGHIJ"[pose_idx],
    )


def build_single_trial_session(
    file_num: int,
    total_files: int,
    source_session: int,
    source_trial: int,
    out_dir: Path,
) -> Path:
    (nic_a, nic_b, sc_c), pose, mounts, pose_label = _session_spec(source_session)

    yaml_path = out_dir / f"session_{file_num:03d}.yaml"
    if 11 <= source_session <= 20:
        if source_trial == 1:
            task_label = f"SC{sc_c}"
            body = build_sc_trial(1, pose, sc_c, SC_TRANS_OPTIONS[sc_c % len(SC_TRANS_OPTIONS)], mounts)
        elif source_trial == 2:
            task_label = f"NIC{nic_a}"
            body = build_nic_trial(1, pose, nic_a, NIC_TRANS_OPTIONS[nic_a % len(NIC_TRANS_OPTIONS)], mounts)
        elif source_trial == 3:
            task_label = f"NIC{nic_b}"
            body = build_nic_trial(1, pose, nic_b, NIC_TRANS_OPTIONS[nic_b % len(NIC_TRANS_OPTIONS)], mounts)
        else:
            raise ValueError(f"source_trial must be 1, 2, or 3, got {source_trial}")
    elif source_trial == 1:
        task_label = f"NIC{nic_a}"
        body = build_nic_trial(
            1, pose, nic_a, NIC_TRANS_OPTIONS[nic_a % len(NIC_TRANS_OPTIONS)], mounts
        )
    elif source_trial == 2:
        task_label = f"NIC{nic_b}"
        body = build_nic_trial(
            1, pose, nic_b, NIC_TRANS_OPTIONS[nic_b % len(NIC_TRANS_OPTIONS)], mounts
        )
    elif source_trial == 3:
        task_label = f"SC{sc_c}"
        body = build_sc_trial(
            1, pose, sc_c, SC_TRANS_OPTIONS[sc_c % len(SC_TRANS_OPTIONS)], mounts
        )
    else:
        raise ValueError(f"source_trial must be 1, 2, or 3, got {source_trial}")

    content = SINGLE_TRIAL_HEADER_TMPL.format(
        file_num=file_num,
        total_files=total_files,
        source_session=source_session,
        source_trial=source_trial,
        pose_label=pose_label,
        task_label=task_label,
        yaml_path=str(yaml_path),
    )
    content += body
    content += FOOTER
    yaml_path.write_text(content)
    return yaml_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=50,
                        help="Number of session YAMLs to generate (default: 50)")
    parser.add_argument("--trials-per-session", type=int, choices=(1, 3), default=3,
                        help="Use 3 for competition-format files, or 1 to relaunch Gazebo for every trial.")
    parser.add_argument("--layout", choices=("mixed", "nic_triplets"), default="mixed",
                        help="Session layout: 'mixed' for the original mixed-task generator, or 'nic_triplets' for NIC-only 3-trial sessions.")
    parser.add_argument("--out-dir", default="sessions",
                        help="Output directory (default: sessions/)")
    args = parser.parse_args()

    out_dir = Path(__file__).parent / args.out_dir
    out_dir.mkdir(exist_ok=True)

    if args.layout == "nic_triplets":
        if args.trials_per_session != 3:
            raise ValueError("layout 'nic_triplets' requires --trials-per-session 3")
        for i in range(1, args.sessions + 1):
            build_nic_triplet_session(i, args.sessions, out_dir)
            _, _, _, pose_label = _session_spec(i)
            round_idx = (i - 1) // len(POSES)
            base = round_idx % len(NIC_TRIPLET_TARGETS)
            targets = [NIC_TRIPLET_TARGETS[(base + offset) % len(NIC_TRIPLET_TARGETS)] for offset in (0, 3, 6)]
            trial_str = "→".join(f"NIC{rail}({trans:+.3f})" for rail, trans in targets)
            print(f"  session_{i:02d}.yaml  pose={pose_label}  {trial_str}")

        print(f"\nGenerated {args.sessions} NIC-only 3-trial session YAMLs in: {out_dir}/")
        print("Each session = 3 NIC trials, FIXED table pose, lean entity spawn.")
        print("Board z stays 1.14 and roll/pitch stay 0.0; only x/y/yaw vary between sessions.")
        print(f"\nLaunch one session at a time:")
        print(f"  aic_engine_config_file:={out_dir}/session_01.yaml")
        print("Collector can use the default num_episodes=3.")
        return

    if args.trials_per_session == 3:
        for i in range(1, args.sessions + 1):
            build_session(i, args.sessions, out_dir)
            nic_a, nic_b, sc_c = SESSION_PATTERNS[(i - 1) % len(SESSION_PATTERNS)]
            pose_label = "ABCDEFGHIJ"[(i - 1) % len(POSES)]
            if 11 <= i <= 20:
                trial_str = f"SC{sc_c}→NIC{nic_a}→NIC{nic_b}"
            else:
                trial_str = f"NIC{nic_a}→NIC{nic_b}→SC{sc_c}"
            print(f"  session_{i:02d}.yaml  pose={pose_label}  {trial_str}")

        print(f"\nGenerated {args.sessions} session YAMLs in: {out_dir}/")
        print(f"Each session = 3 trials, FIXED table pose, lean entity spawn.")
        print(f"Board z stays 1.14 and roll/pitch stay 0.0; only x/y/yaw vary between sessions.")
        print(f"\nLaunch one session at a time:")
        print(f"  aic_engine_config_file:={out_dir}/session_01.yaml")
        print(f"After it finishes, relaunch with session_02.yaml, etc.")
        print(f"\nFor 100 sessions: python generate_competition_sessions.py --sessions 100")
        return

    total_files = args.sessions * 3
    file_num = 1
    for source_session in range(1, args.sessions + 1):
        for source_trial in (1, 2, 3):
            build_single_trial_session(
                file_num, total_files, source_session, source_trial, out_dir
            )
            _, _, _, pose_label = _session_spec(source_session)
            print(
                f"  session_{file_num:03d}.yaml  source={source_session:02d}.{source_trial}  pose={pose_label}"
            )
            file_num += 1

    print(f"\nGenerated {total_files} single-trial YAMLs in: {out_dir}/")
    print("Each file = 1 trial. Relaunch Gazebo for every file so the gripper starts closed.")
    print("Board z stays 1.14 and roll/pitch stay 0.0; only x/y/yaw vary between source sessions.")
    print(f"\nLaunch one file at a time:")
    print(f"  aic_engine_config_file:={out_dir}/session_001.yaml")
    print("Use num_episodes:=1 and RUN_ID=run_001, then session_002.yaml with RUN_ID=run_002.")
    print("\nFor 300 single-trial files: python generate_competition_sessions.py --sessions 100 --trials-per-session 1")


if __name__ == "__main__":
    main()
