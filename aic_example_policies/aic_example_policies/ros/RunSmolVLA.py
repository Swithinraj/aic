#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
RunSmolVLA — SmolVLA inference node for AIC cable insertion.

Mirrors RunACT.py structure exactly. Differences vs ACT:
  - SmolVLAPolicy with language conditioning (SFP vs SC task strings)
  - 30D state vector: tcp_pose(7) + tcp_vel(6) + joint_pos(7) + joint_vel(7) + port_xyz(3)
  - IDENTITY image normalization (images passed as float [0,1], no mean/std)
  - MEAN_STD state normalization
  - Action chunk of 50 steps handled internally by select_action (returns 1 step per call)

Checkpoint layout expected at CHECKPOINT_PATH:
  pretrained_model/
    config.json
    model.safetensors
    policy_preprocessor_step_5_normalizer_processor.safetensors
    policy_postprocessor_step_0_unnormalizer_processor.safetensors
"""

import time
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Dict

from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3, Wrench

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from aic_control_interfaces.msg import (
    MotionUpdate,
    TrajectoryGenerationMode,
)

# ── LeRobot SmolVLA import workaround (bypass GROOT dataclass error) ───────────
# lerobot/policies/__init__.py imports ALL policies including groot, which has
# a broken @dataclass definition. We register stub modules so the __init__ is
# never executed, then import SmolVLA directly.
import lerobot as _lerobot_pkg
import types
import sys as _sys

_lerobot_root = Path(_lerobot_pkg.__file__).resolve().parent
_policies_dir = _lerobot_root / "policies"
_smolvla_dir  = _policies_dir / "smolvla"

_policies_pkg = types.ModuleType("lerobot.policies")
_policies_pkg.__path__ = [str(_policies_dir)]
_sys.modules["lerobot.policies"] = _policies_pkg

_smolvla_pkg = types.ModuleType("lerobot.policies.smolvla")
_smolvla_pkg.__path__ = [str(_smolvla_dir)]
_sys.modules["lerobot.policies.smolvla"] = _smolvla_pkg
# ──────────────────────────────────────────────────────────────────────────────

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from safetensors.torch import load_file
from transformers import AutoTokenizer


# ── Required checkpoint files ──────────────────────────────────────────────────
_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)

# Language task strings (must match training)
TASK_STRINGS = {
    "sfp": "Insert the SFP module into the target SFP port on the NIC card",
    "sc":  "Insert the SC plug into the target SC port",
}

# Image scaling — must match data collection (AICRobotAICControllerConfig default)
IMAGE_SCALE = 0.25

# Control rate
CONTROL_HZ = 10          # steps per second
EPISODE_DURATION_S = 30  # seconds per attempt

# Port position in base frame — mean from training data.
# Used as fallback when YOLO port detection is not available.
# Normalizes to ~0 which is safe. Replace with live YOLO detection for best results.
PORT_XYZ_MEAN = np.array([-0.44123670, 0.30071041, 0.14065616], dtype=np.float32)


class RunSmolVLA(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"RunSmolVLA init — device: {self.device}")

        # ── 0. Resolve checkpoint path from ROS parameter ──────────────────────
        raw_path = str(parent_node.declare_parameter("checkpoint_path", "").value)
        checkpoint_path = self._resolve_checkpoint_path(raw_path)
        self.get_logger().info(f"Checkpoint: {checkpoint_path}")

        # ── 1. Load Policy ─────────────────────────────────────────────────────
        self.get_logger().info(f"Loading SmolVLA from {checkpoint_path} ...")
        t0 = time.time()
        self.policy = SmolVLAPolicy.from_pretrained(str(checkpoint_path))
        self.policy.to(self.device)
        self.policy.eval()
        self.get_logger().info(f"SmolVLA loaded in {time.time()-t0:.1f}s")

        # ── 2. Load tokenizer ──────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        )
        self.get_logger().info("Tokenizer loaded.")

        # ── 3. Load normalization stats ────────────────────────────────────────
        # Preprocessor stats: state MEAN_STD normalization
        pre_stats = load_file(
            checkpoint_path / "policy_preprocessor_step_5_normalizer_processor.safetensors"
        )
        # Postprocessor stats: action MEAN_STD unnormalization
        post_stats = load_file(
            checkpoint_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        )

        def _t(stats, key):
            return stats[key].float().to(self.device)

        self.state_mean = _t(pre_stats,  "observation.state.mean")   # (30,)
        self.state_std  = _t(pre_stats,  "observation.state.std")    # (30,)
        self.action_mean = _t(post_stats, "action.mean")             # (6,)
        self.action_std  = _t(post_stats, "action.std")              # (6,)

        self.get_logger().info("Normalization stats loaded.")
        self.get_logger().info(f"  action_mean: {self.action_mean.cpu().numpy()}")
        self.get_logger().info(f"  action_std:  {self.action_std.cpu().numpy()}")

        # ── 4. Warm up the model (first call is slow on MPS/CUDA JIT) ─────────
        self._warmup()

    # ── Checkpoint resolution ─────────────────────────────────────────────────

    @staticmethod
    def _resolve_checkpoint_path(raw_path: str) -> Path:
        """
        Accept either:
          - direct path to pretrained_model/ directory, or
          - parent checkpoint dir (e.g. 060000/) that contains pretrained_model/
        Raises ValueError if path is empty, FileNotFoundError if files are missing.
        """
        raw = raw_path.strip()
        if not raw:
            raise ValueError(
                "RunSmolVLA requires -p checkpoint_path:=/path/to/pretrained_model "
                "(or the parent dir containing pretrained_model/)"
            )
        path = Path(raw).expanduser()
        candidates = [path, path / "pretrained_model"]
        for candidate in candidates:
            if all((candidate / f).exists() for f in _REQUIRED_FILES):
                return candidate
        missing = [f for f in _REQUIRED_FILES if not (path / f).exists()]
        raise FileNotFoundError(
            f"Not a valid pretrained_model directory: {path}. Missing: {missing}"
        )

    def _warmup(self):
        """Run one dummy forward pass so the first real step is fast."""
        self.get_logger().info("Warming up model ...")
        dummy = self._make_dummy_batch("sfp")
        with torch.inference_mode():
            self.policy.select_action(dummy)
        self.get_logger().info("Warmup done.")

    # ── Tokenization ──────────────────────────────────────────────────────────

    def _tokenize(self, task_str: str) -> dict:
        enc = self.tokenizer(
            task_str,
            return_tensors="pt",
            padding="max_length",
            max_length=48,
            truncation=True,
        )
        return {
            "observation.language.tokens":
                enc["input_ids"].to(self.device),
            "observation.language.attention_mask":
                enc["attention_mask"].bool().to(self.device),
        }

    # ── Image processing ──────────────────────────────────────────────────────

    @staticmethod
    def _ros_img_to_tensor(raw_img, device: torch.device, scale: float) -> torch.Tensor:
        """
        ROS Image → float32 [0,1] tensor of shape (1, C, H, W).
        SmolVLA uses IDENTITY image normalization — no mean/std subtraction.
        """
        img_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )
        if scale != 1.0:
            img_np = cv2.resize(img_np, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)
        return (
            torch.from_numpy(img_np.copy())
            .permute(2, 0, 1)   # HWC → CHW
            .float()
            .div(255.0)
            .unsqueeze(0)       # → (1, C, H, W)
            .to(device)
        )

    # ── State assembly ────────────────────────────────────────────────────────

    def _build_state(self, obs_msg: Observation) -> torch.Tensor:
        """
        Assemble 30D state vector and normalize with MEAN_STD.

        Ordering (matches data collection pipeline):
          [0:7]   tcp_pose      — position (3) + quaternion (4)
          [7:13]  tcp_velocity  — linear (3) + angular (3)
          [13:20] joint_positions — 7 joints
          [20:27] joint_velocity  — 7 joints (from joint_states.velocity)
          [27:30] port_xyz_in_base — YOLO-detected port; uses dataset mean as fallback
        """
        cs  = obs_msg.controller_state
        js  = obs_msg.joint_states

        tcp_pose = cs.tcp_pose
        tcp_vel  = cs.tcp_velocity

        state_np = np.array([
            # TCP pose (7)
            tcp_pose.position.x,
            tcp_pose.position.y,
            tcp_pose.position.z,
            tcp_pose.orientation.x,
            tcp_pose.orientation.y,
            tcp_pose.orientation.z,
            tcp_pose.orientation.w,
            # TCP velocity (6)
            tcp_vel.linear.x,
            tcp_vel.linear.y,
            tcp_vel.linear.z,
            tcp_vel.angular.x,
            tcp_vel.angular.y,
            tcp_vel.angular.z,
            # Joint positions (7)
            *list(js.position[:7]),
            # Joint velocities (7)
            *list(js.velocity[:7]),
            # Port XYZ in base (3) — dataset mean used as fallback
            PORT_XYZ_MEAN[0],
            PORT_XYZ_MEAN[1],
            PORT_XYZ_MEAN[2],
        ], dtype=np.float32)

        raw = torch.from_numpy(state_np).unsqueeze(0).to(self.device)  # (1, 30)
        normalized = (raw - self.state_mean) / (self.state_std + 1e-8)
        return normalized

    # ── Full observation dict ─────────────────────────────────────────────────

    def _prepare_observations(
        self, obs_msg: Observation, task_key: str
    ) -> Dict[str, torch.Tensor]:
        task_str = TASK_STRINGS[task_key]
        batch = {
            "observation.images.left":   self._ros_img_to_tensor(
                obs_msg.left_image,   self.device, IMAGE_SCALE),
            "observation.images.center": self._ros_img_to_tensor(
                obs_msg.center_image, self.device, IMAGE_SCALE),
            "observation.images.right":  self._ros_img_to_tensor(
                obs_msg.right_image,  self.device, IMAGE_SCALE),
            "observation.state": self._build_state(obs_msg),
            "task": [task_str],
        }
        batch.update(self._tokenize(task_str))
        return batch

    # ── Dummy batch for warmup ────────────────────────────────────────────────

    def _make_dummy_batch(self, task_key: str) -> dict:
        task_str = TASK_STRINGS[task_key]
        # Images: (1, 3, 288, 256) — 1152×1024 × 0.25
        dummy_img = torch.zeros(1, 3, 288, 256, device=self.device)
        batch = {
            "observation.images.left":   dummy_img,
            "observation.images.center": dummy_img,
            "observation.images.right":  dummy_img,
            "observation.state": torch.zeros(1, 30, device=self.device),
            "task": [task_str],
        }
        batch.update(self._tokenize(task_str))
        return batch

    # ── Motion update helper ──────────────────────────────────────────────────

    def _make_motion_update(self, action: np.ndarray) -> MotionUpdate:
        """Pack 6D cartesian velocity into a MotionUpdate message."""
        twist = Twist(
            linear=Vector3(
                x=float(action[0]),
                y=float(action[1]),
                z=float(action[2]),
            ),
            angular=Vector3(
                x=float(action[3]),
                y=float(action[4]),
                z=float(action[5]),
            ),
        )
        msg = MotionUpdate()
        msg.header.frame_id = "base_link"
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.velocity        = twist

        # Impedance parameters (tuned for cable insertion — same as RunACT)
        msg.target_stiffness = np.diag(
            [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
        ).flatten()
        msg.target_damping = np.diag(
            [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
        ).flatten()
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        return msg

    # ── Main inference loop ───────────────────────────────────────────────────

    def _run_episode(
        self,
        task_key: str,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.policy.reset()
        task_str = TASK_STRINGS[task_key]
        self.get_logger().info(f"RunSmolVLA starting — task: '{task_str}'")

        step_dt   = 1.0 / CONTROL_HZ
        start     = time.time()
        step_count = 0

        while time.time() - start < EPISODE_DURATION_S:
            t_loop = time.time()

            # 1. Observe
            obs_msg = get_observation()
            if obs_msg is None:
                self.get_logger().warn("No observation received — skipping step.")
                time.sleep(0.05)
                continue

            # 2. Build observation dict
            obs = self._prepare_observations(obs_msg, task_key)

            # 3. Inference
            t_infer = time.time()
            with torch.inference_mode():
                norm_action = self.policy.select_action(obs)
            infer_ms = (time.time() - t_infer) * 1000

            # 4. Unnormalize action  →  real cartesian velocity [m/s, rad/s]
            real_action = (norm_action * self.action_std) + self.action_mean
            action_np   = real_action[0].cpu().numpy()  # (6,)

            self.get_logger().info(
                f"step={step_count:4d}  "
                f"lin=[{action_np[0]:+.4f} {action_np[1]:+.4f} {action_np[2]:+.4f}]  "
                f"ang=[{action_np[3]:+.4f} {action_np[4]:+.4f} {action_np[5]:+.4f}]  "
                f"infer={infer_ms:.0f}ms"
            )

            # 5. Command robot
            motion_update = self._make_motion_update(action_np)
            move_robot(motion_update=motion_update)
            send_feedback(f"SmolVLA step {step_count} in progress")

            step_count += 1

            # 6. Hold control rate
            elapsed = time.time() - t_loop
            time.sleep(max(0.0, step_dt - elapsed))

        self.get_logger().info(
            f"RunSmolVLA episode done — {step_count} steps in {time.time()-start:.1f}s"
        )
        return True

    # ── AIC task entry points ─────────────────────────────────────────────────

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ):
        """
        AIC task handler — dispatches to SFP or SC based on task description.
        Called by the AIC framework when a cable insertion task is assigned.
        """
        task_desc = task.description.lower() if hasattr(task, "description") else ""

        if "sc" in task_desc or "sc_cable" in task_desc:
            task_key = "sc"
        else:
            task_key = "sfp"   # default to SFP

        self.get_logger().info(
            f"insert_cable() — task='{task_desc}' → key='{task_key}'"
        )
        return self._run_episode(task_key, get_observation, move_robot, send_feedback)
