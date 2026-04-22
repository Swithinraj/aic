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

Output:
    sessions/session_01.yaml ... session_50.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Board poses (10 distinct positions/orientations)
# ---------------------------------------------------------------------------

POSES = [
    dict(x=0.160, y=-0.180, z=1.14, roll= 0.00, pitch= 0.00, yaw=2.80),  # A
    dict(x=0.140, y=-0.220, z=1.14, roll= 0.04, pitch=-0.03, yaw=3.35),  # B
    dict(x=0.175, y=-0.150, z=1.14, roll= 0.00, pitch= 0.00, yaw=2.55),  # C
    dict(x=0.155, y=-0.250, z=1.14, roll=-0.02, pitch= 0.02, yaw=3.10),  # D
    dict(x=0.195, y=-0.100, z=1.14, roll= 0.03, pitch=-0.02, yaw=2.70),  # E
    dict(x=0.120, y=-0.175, z=1.14, roll= 0.00, pitch= 0.00, yaw=2.90),  # F
    dict(x=0.215, y=-0.155, z=1.14, roll=-0.03, pitch= 0.03, yaw=3.20),  # G
    dict(x=0.165, y=-0.060, z=1.14, roll= 0.02, pitch=-0.01, yaw=2.65),  # H
    dict(x=0.185, y=-0.195, z=1.14, roll=-0.01, pitch= 0.01, yaw=3.50),  # I
    dict(x=0.135, y=-0.125, z=1.14, roll= 0.04, pitch=-0.04, yaw=2.45),  # J
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
#
# Session {session_num:02d} of {total}
# Board pose: {pose_label}
# Trials: NIC{nic_a} → NIC{nic_b} → SC{sc_c}
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


def _background_mounts(mounts: dict, indent: int = 10) -> str:
    pad = " " * indent
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
        f"{pad}sc_mount_rail_0:\n"
        f"{pad}  entity_present: True\n"
        f"{pad}  entity_name: \"sc_mount_0\"\n"
        f"{pad}  entity_pose:\n"
        f"{pad}    translation: {mounts['sc_t']:.2f}\n"
        f"{pad}    roll: 0.0\n"
        f"{pad}    pitch: 0.0\n"
        f"{pad}    yaw: 0.0\n"
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
    lines.append(_background_mounts(mounts, indent=10).rstrip())
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

    header = HEADER_TMPL.format(
        session_num=session_num,
        total=total,
        pose_label=pose_label,
        nic_a=nic_a, nic_b=nic_b, sc_c=sc_c,
        yaml_path=str(yaml_path),
    )

    content = header
    content += build_nic_trial(1, pose, nic_a, nic_trans_a, mounts)
    content += build_nic_trial(2, pose, nic_b, nic_trans_b, mounts)
    content += build_sc_trial( 3, pose, sc_c,  sc_trans_c,  mounts)
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
    parser.add_argument("--out-dir", default="sessions",
                        help="Output directory (default: sessions/)")
    args = parser.parse_args()

    out_dir = Path(__file__).parent / args.out_dir
    out_dir.mkdir(exist_ok=True)

    for i in range(1, args.sessions + 1):
        path = build_session(i, args.sessions, out_dir)
        nic_a, nic_b, sc_c = SESSION_PATTERNS[(i - 1) % len(SESSION_PATTERNS)]
        pose_label = "ABCDEFGHIJ"[(i - 1) % len(POSES)]
        print(f"  session_{i:02d}.yaml  pose={pose_label}  NIC{nic_a}→NIC{nic_b}→SC{sc_c}")

    print(f"\nGenerated {args.sessions} session YAMLs in: {out_dir}/")
    print(f"Each session = 3 trials, FIXED board pose, lean entity spawn.")
    print(f"\nLaunch one session at a time:")
    print(f"  aic_engine_config_file:={out_dir}/session_01.yaml")
    print(f"After it finishes, relaunch with session_02.yaml, etc.")
    print(f"\nFor 100 sessions: python generate_competition_sessions.py --sessions 100")


if __name__ == "__main__":
    main()
