#!/usr/bin/env python3
"""
Generate orientation_sweep_50_trials.yaml.

Every trial spawns a FULLY POPULATED board:
  - All 5 NIC rails occupied (nic_card_0 … nic_card_4)
  - Both SC rails occupied (sc_mount_0 and sc_mount_1)

The task still targets one specific port per trial so the robot learns
to navigate to the correct port among many visible connectors.

Run from repo root:
    python team_policy/team_policy/training_robot/configs/generate_50_trials.py
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Board pose diversity (10 positions)
# ---------------------------------------------------------------------------

POSITIONS = {
    "A": dict(x=0.160, y=-0.180, roll= 0.00, pitch= 0.00, yaw=2.80),
    "B": dict(x=0.140, y=-0.220, roll= 0.04, pitch=-0.03, yaw=3.35),
    "C": dict(x=0.175, y=-0.150, roll= 0.00, pitch= 0.00, yaw=2.55),
    "D": dict(x=0.155, y=-0.250, roll=-0.02, pitch= 0.02, yaw=3.10),
    "E": dict(x=0.195, y=-0.100, roll= 0.03, pitch=-0.02, yaw=2.70),
    "F": dict(x=0.120, y=-0.175, roll= 0.00, pitch= 0.00, yaw=2.90),
    "G": dict(x=0.215, y=-0.155, roll=-0.03, pitch= 0.03, yaw=3.20),
    "H": dict(x=0.165, y=-0.060, roll= 0.02, pitch=-0.01, yaw=2.65),
    "I": dict(x=0.185, y=-0.195, roll=-0.01, pitch= 0.01, yaw=3.50),
    "J": dict(x=0.135, y=-0.125, roll= 0.04, pitch=-0.04, yaw=2.45),
}

# ---------------------------------------------------------------------------
# Background mount sets (visual clutter variety)
# ---------------------------------------------------------------------------

MOUNT_SETS = {
    1: dict(sfp_t= 0.03, sc_t=-0.02, lc0_t= 0.02, lc1_t=-0.01),
    2: dict(sfp_t= 0.05, sc_t= 0.04, lc0_t= 0.04, lc1_t= 0.03),
    3: dict(sfp_t=-0.03, sc_t=-0.05, lc0_t=-0.01, lc1_t=-0.03),
}

# ---------------------------------------------------------------------------
# Per-rail translations for ALL NIC cards and SC mounts when fully populated.
# Each mount_set gets a different layout so the board looks different each time.
#   Index: [rail_0, rail_1, rail_2, rail_3, rail_4]
# ---------------------------------------------------------------------------

NIC_TRANSLATIONS = {
    # mount_set: translations for nic_rail_0 … nic_rail_4
    1: [ 0.010, -0.015,  0.020, -0.010,  0.005],
    2: [-0.010,  0.020, -0.015,  0.015, -0.005],
    3: [ 0.015, -0.005,  0.010, -0.020,  0.010],
}

SC_TRANSLATIONS = {
    # mount_set: [sc_rail_0_t, sc_rail_1_t]
    1: [-0.030,  0.030],
    2: [ 0.042, -0.042],
    3: [-0.050,  0.050],
}

# ---------------------------------------------------------------------------
# 50 trial specs: (position_key, rail_spec, target_translation, mount_set)
#
# rail_spec "NIC0"–"NIC4": SFP insertion on that NIC card's sfp_port_0
# rail_spec "SC0"/"SC1":   SC insertion on that SC port
#
# target_translation overrides the corresponding rail's entry in NIC_TRANSLATIONS
# or SC_TRANSLATIONS so the target port is at a known varied position.
# ---------------------------------------------------------------------------

TRIALS = [
    # --- 40 SFP/NIC trials (8 per rail, cycling through all 10 board poses) ---
    ("A", "NIC0",  0.015, 1), ("B", "NIC1",  0.020, 2), ("C", "NIC2", -0.010, 3),
    ("D", "NIC3",  0.000, 1), ("E", "NIC4", -0.015, 2), ("F", "NIC0", -0.015, 3),
    ("G", "NIC1",  0.015, 1), ("H", "NIC2",  0.000, 2), ("I", "NIC3", -0.020, 3),
    ("J", "NIC4",  0.020, 1), ("C", "NIC0",  0.000, 2), ("D", "NIC1", -0.010, 3),
    ("E", "NIC2",  0.015, 1), ("F", "NIC3",  0.020, 2), ("G", "NIC4", -0.010, 3),
    ("H", "NIC0",  0.020, 1), ("I", "NIC1",  0.000, 2), ("J", "NIC2", -0.015, 3),
    ("A", "NIC3",  0.015, 1), ("B", "NIC4", -0.020, 2), ("E", "NIC0", -0.010, 3),
    ("F", "NIC1",  0.020, 1), ("G", "NIC2",  0.000, 2), ("H", "NIC3", -0.015, 3),
    ("I", "NIC4",  0.010, 1), ("J", "NIC0",  0.015, 2), ("A", "NIC1", -0.015, 3),
    ("B", "NIC2",  0.020, 1), ("C", "NIC3",  0.010, 2), ("D", "NIC4",  0.000, 3),
    ("G", "NIC0", -0.020, 1), ("H", "NIC1",  0.010, 2), ("I", "NIC2", -0.020, 3),
    ("J", "NIC3",  0.000, 1), ("E", "NIC4",  0.015, 2), ("D", "NIC0",  0.010, 3),
    ("C", "NIC1", -0.020, 1), ("B", "NIC2",  0.010, 2), ("A", "NIC3", -0.010, 3),
    ("F", "NIC4",  0.020, 1),
    # --- 10 SC trials (5 per rail) ---
    ("C", "SC0", -0.042, 2), ("H", "SC1", -0.055, 3), ("B", "SC0",  0.042, 1),
    ("I", "SC1",  0.050, 2), ("E", "SC0",  0.000, 3), ("J", "SC1", -0.030, 1),
    ("A", "SC0", -0.025, 2), ("F", "SC1",  0.030, 3), ("G", "SC0",  0.025, 1),
    ("D", "SC1",  0.000, 2),
]

assert len(TRIALS) == 50, f"Expected 50 trials, got {len(TRIALS)}"


# ---------------------------------------------------------------------------
# YAML block builders
# ---------------------------------------------------------------------------

def _pose_block(pos: dict, indent: int = 10) -> str:
    pad = " " * indent
    return (
        f"{pad}pose:\n"
        f"{pad}  x: {pos['x']:.3f}\n"
        f"{pad}  y: {pos['y']:.3f}\n"
        f"{pad}  z: 1.14\n"
        f"{pad}  roll: {pos['roll']:.2f}\n"
        f"{pad}  pitch: {pos['pitch']:.2f}\n"
        f"{pad}  yaw: {pos['yaw']:.2f}\n"
    )


def _nic_rail_block(rail_idx: int, translation: float, indent: int = 10) -> str:
    """Always present — every trial has all 5 NIC cards."""
    pad = " " * indent
    return "\n".join([
        f"{pad}nic_rail_{rail_idx}:",
        f"{pad}  entity_present: True",
        f"{pad}  entity_name: \"nic_card_{rail_idx}\"",
        f"{pad}  entity_pose:",
        f"{pad}    translation: {translation:.3f}",
        f"{pad}    roll: 0.0",
        f"{pad}    pitch: 0.0",
        f"{pad}    yaw: 0.0",
    ]) + "\n"


def _sc_rail_block(rail_idx: int, translation: float, indent: int = 10) -> str:
    """Always present — every trial has both SC ports."""
    pad = " " * indent
    return "\n".join([
        f"{pad}sc_rail_{rail_idx}:",
        f"{pad}  entity_present: True",
        f"{pad}  entity_name: \"sc_mount_{rail_idx}\"",
        f"{pad}  entity_pose:",
        f"{pad}    translation: {translation:.3f}",
        f"{pad}    roll: 0.0",
        f"{pad}    pitch: 0.0",
        f"{pad}    yaw: 0.0",
    ]) + "\n"


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


def _sfp_cable_block(indent: int = 8) -> str:
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


def _sc_cable_block(indent: int = 8) -> str:
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


def _sfp_task_block(nic_rail_idx: int, indent: int = 4) -> str:
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
        f"{pad}    target_module_name: \"nic_card_mount_{nic_rail_idx}\"\n"
        f"{pad}    time_limit: 180\n"
    )


def _sc_task_block(sc_rail_idx: int, indent: int = 4) -> str:
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
        f"{pad}    target_module_name: \"sc_port_{sc_rail_idx}\"\n"
        f"{pad}    time_limit: 180\n"
    )


# ---------------------------------------------------------------------------
# Trial builder
# ---------------------------------------------------------------------------

def build_trial(trial_num: int, pos_key: str, rail_spec: str,
                target_translation: float, mount_set: int) -> str:
    pos    = POSITIONS[pos_key]
    mounts = MOUNT_SETS[mount_set]
    is_sc  = rail_spec.startswith("SC")
    prefix_len = 2 if is_sc else 3
    rail_idx   = int(rail_spec[prefix_len:])

    # Build per-rail translation tables; override the target rail with the
    # trial-specific translation for targeted diversity.
    nic_trans = list(NIC_TRANSLATIONS[mount_set])
    nic_trans[rail_idx if not is_sc else 0] = (
        nic_trans[rail_idx if not is_sc else 0] if is_sc else target_translation
    )

    sc_trans = list(SC_TRANSLATIONS[mount_set])
    if is_sc:
        sc_trans[rail_idx] = target_translation

    lines = [f"  trial_{trial_num}:"]
    lines.append("    scene:")
    lines.append("        task_board:")
    lines.append(_pose_block(pos, indent=10).rstrip())

    # All 5 NIC rails — always present
    for i in range(5):
        lines.append(_nic_rail_block(i, nic_trans[i], indent=10).rstrip())

    # Both SC rails — always present
    for i in range(2):
        lines.append(_sc_rail_block(i, sc_trans[i], indent=10).rstrip())

    lines.append(_background_mounts(mounts, indent=10).rstrip())
    lines.append(
        _sfp_cable_block(indent=8).rstrip()
        if not is_sc else
        _sc_cable_block(indent=8).rstrip()
    )
    lines.append(
        _sfp_task_block(rail_idx, indent=4).rstrip()
        if not is_sc else
        _sc_task_block(rail_idx, indent=4).rstrip()
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# File header / footer
# ---------------------------------------------------------------------------

HEADER = """\
# Training data collection config — 50 trials, fully populated board.
#
# Every trial spawns ALL components:
#   - 5 NIC cards (nic_rail_0 … nic_rail_4), each at a varied translation
#   - 2 SC ports  (sc_rail_0 and sc_rail_1),  each at a varied translation
#
# The task still targets one specific port per trial:
#   - 40 SFP trials: insert into sfp_port_0 on the target NIC card
#   - 10 SC  trials: insert into sc_port_base on the target SC mount
#
# Pass with:
#   aic_engine_config_file:=/path/to/orientation_sweep_50_trials.yaml
#
# Regenerate by running:
#   python team_policy/team_policy/training_robot/configs/generate_50_trials.py

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


def main() -> None:
    out_path = Path(__file__).parent / "orientation_sweep_50_trials.yaml"

    content = HEADER
    for i, (pos_key, rail_spec, translation, mount_set) in enumerate(TRIALS, start=1):
        content += build_trial(i, pos_key, rail_spec, translation, mount_set)
    content += FOOTER

    out_path.write_text(content)
    print(f"Written {len(TRIALS)} trials → {out_path}")
    print("Every trial: 5 NIC cards + 2 SC ports, all present simultaneously.")


if __name__ == "__main__":
    main()
