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
RunSmolVLA_63d — SmolVLA inference node for AIC cable insertion (63D state).

Drop-in replacement for RunSmolVLA.py targeting the cluster-trained 63D checkpoint
(act_yolo_63d dataset, ≥40 K steps).

Key differences from the 30D variant:
  ─────────────────────────────────────────────────────────────────────────────
  State (63D)
    [0:3]   tcp_pos       — position xyz in base frame          (from controller_state)
    [3:7]   tcp_quat      — quaternion [x, y, z, w]             (from controller_state)
    [7:13]  tcp_vel       — linear (3) + angular (3)            (from controller_state)
    [13:20] joint_pos     — 7 joint positions (rad)             (from joint_states)
    [20:27] joint_vel     — 7 joint velocities (rad/s)          (from joint_states)
    [27:33] wrist_ft      — force (3) + torque (3) Nm, tared    (from wrist_wrench)
    [33:40] yolo_left     — [conf, cx, cy, w, h, valid, age]   (stale zeros at deploy)
    [40:47] yolo_center   — same                               (stale zeros at deploy)
    [47:54] yolo_right    — same                               (stale zeros at deploy)
    [54:56] plug_type     — [1,0]=SFP  [0,1]=SC               (from task description)
    [56:63] target_module — 7-D one-hot over module names      (from task description)
  ─────────────────────────────────────────────────────────────────────────────
  Normalization (MEAN_STD loaded from checkpoint safetensors, applied manually)
  Action (6D Cartesian velocity, MODE_VELOCITY, 10 Hz)
  Episode duration: 70 s  (training episodes top out at ~61 s)
  Chunk size: 50 steps = 5 s between VLM re-queries

Run:
    ros2 run aic_example_policies RunSmolVLA_63d \
        -p checkpoint_path:=/path/to/040000

Or pass the pretrained_model/ sub-dir directly:
    -p checkpoint_path:=/path/to/040000/pretrained_model
"""

import math
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

# ── LeRobot SmolVLA import workaround (bypass GROOT dataclass import error) ────
# lerobot/policies/__init__.py tries to import ALL policies including groot which
# has a broken @dataclass.  Registering stub modules skips the __init__ completely.
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


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

STATE_DIM  = 63
ACTION_DIM = 6

# Required checkpoint files
_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)

# Language task strings — must match training verbatim
TASK_STRINGS = {
    "sfp": "Insert the SFP module into the target SFP port on the NIC card",
    "sc":  "Insert the SC plug into the target SC port",
}

# Target module one-hot order (must match training dataset TASK_ONE_HOT_ORDER)
MODULE_NAMES = [
    "nic_card_mount_0",
    "nic_card_mount_1",
    "nic_card_mount_2",
    "nic_card_mount_3",
    "nic_card_mount_4",
    "sc_port_0",
    "sc_port_1",
]
_MODULE_INDEX = {name: i for i, name in enumerate(MODULE_NAMES)}

# Plug-type one-hot: index 0 = SFP, index 1 = SC
PLUG_TYPE_SFP = np.array([1.0, 0.0], dtype=np.float32)
PLUG_TYPE_SC  = np.array([0.0, 1.0], dtype=np.float32)

# Stale YOLO placeholder — used for all three cameras at deployment time.
# Training data had valid YOLO; these values normalise near the tail of the
# distribution. age=100 >> any training age so the model learns to ignore stale dims.
_YOLO_STALE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0], dtype=np.float32)

# Control parameters
CONTROL_HZ         = 10    # Hz
EPISODE_DURATION_S = 70    # seconds — training max is ~61 s

# Image downscale factor — must match data collection pipeline
IMAGE_SCALE = 0.25

# Velocity safety clamps — applied AFTER unnormalization to prevent runaway actions
# Training action_std: lin~0.005-0.048 m/s, ang~0.003-0.014 rad/s
# Clamp at ~3× max training std to allow normal variation but catch pathological outputs
_MAX_VEL_LIN_M_S   = 0.15   # m/s per axis — 15 cm/s
_MAX_VEL_ANG_RAD_S = 0.20   # rad/s per axis — ~11 deg/s

# F/T safety thresholds (same philosophy as the scoring team's script)
_CONTACT_FORCE_N   = 20.0   # N above baseline before hard stop
_TORQUE_LIMIT_NM   =  5.0   # Nm absolute before hard stop


# ─────────────────────────────────────────────────────────────────────────────
#  Policy class
# ─────────────────────────────────────────────────────────────────────────────

class RunSmolVLA_63d(Policy):
    """
    SmolVLA inference policy with 63D state for the AIC cable insertion challenge.
    """

    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"RunSmolVLA_63d init — device: {self.device}")

        # ── 0. Resolve checkpoint ──────────────────────────────────────────────
        raw_path = str(
            parent_node.declare_parameter("checkpoint_path", "").value
        )
        checkpoint_path = self._resolve_checkpoint_path(raw_path)
        self.get_logger().info(f"Checkpoint: {checkpoint_path}")

        # ── 1. Load SmolVLA policy ─────────────────────────────────────────────
        self.get_logger().info("Loading SmolVLA ...")
        t0 = time.time()
        self.policy = SmolVLAPolicy.from_pretrained(str(checkpoint_path))
        self.policy.to(self.device)
        self.policy.eval()
        # fp16 on CUDA halves VRAM (~2.7 GB → ~1.4 GB) with negligible accuracy loss
        # at inference time. Keep fp32 on CPU / MPS where fp16 is slower.
        if self.device.type == "cuda":
            self.get_logger().info("  Using fp16 on CUDA (halved VRAM)")
            # self.policy.half()
            # self.get_logger().info("  Using fp16 on CUDA (halved VRAM)")
        self.get_logger().info(f"SmolVLA loaded in {time.time() - t0:.1f}s")

        # ── 2. Load tokenizer ──────────────────────────────────────────────────
        # SmolVLA expects pre-tokenized language tokens in the batch —
        # select_action() does NOT tokenize internally.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.policy.config.vlm_model_name,
            padding_side="right",
        )
        self.get_logger().info(
            f"Tokenizer loaded ({self.policy.config.vlm_model_name})"
        )

        # ── 3. Load normalization stats manually ───────────────────────────────
        # from_pretrained loads model.safetensors; normalizer safetensors are
        # separate files that must be loaded explicitly.
        pre_stats  = load_file(
            str(checkpoint_path /
                "policy_preprocessor_step_5_normalizer_processor.safetensors")
        )
        post_stats = load_file(
            str(checkpoint_path /
                "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
        )

        def _t(stats: dict, key: str) -> torch.Tensor:
            return stats[key].float().to(self.device)

        self.state_mean  = _t(pre_stats,  "observation.state.mean")   # (63,)
        self.state_std   = _t(pre_stats,  "observation.state.std")    # (63,)
        self.action_mean = _t(post_stats, "action.mean")              # (6,)
        self.action_std  = _t(post_stats, "action.std")               # (6,)

        self.get_logger().info(
            f"State normalizer loaded — mean dim={self.state_mean.shape[0]}, "
            f"std dim={self.state_std.shape[0]}"
        )
        self.get_logger().info(
            f"Action unnormalizer — mean={self.action_mean.cpu().numpy()}"
        )
        self.get_logger().info(
            f"Action unnormalizer — std= {self.action_std.cpu().numpy()}"
        )

        # ── 3. Warm-up (amortise JIT / CUDA kernel compile on first call) ─────
        self._warmup()

    # ── Checkpoint resolution ─────────────────────────────────────────────────

    @staticmethod
    def _resolve_checkpoint_path(raw_path: str) -> Path:
        """
        Accept either:
          /path/to/040000                (contains pretrained_model/ sub-dir)
          /path/to/040000/pretrained_model   (contains files directly)
        Raises ValueError for empty path, FileNotFoundError if files are absent.
        """
        raw = raw_path.strip()
        if not raw:
            raise ValueError(
                "RunSmolVLA_63d requires:\n"
                "  ros2 run aic_example_policies RunSmolVLA_63d "
                "-p checkpoint_path:=/path/to/040000"
            )
        path = Path(raw).expanduser()
        for candidate in [path, path / "pretrained_model"]:
            if all((candidate / f).exists() for f in _REQUIRED_FILES):
                return candidate
        missing = [f for f in _REQUIRED_FILES if not (path / f).exists()]
        raise FileNotFoundError(
            f"Not a valid checkpoint directory: {path}\n"
            f"Missing files: {missing}"
        )

    # ── Warm-up ───────────────────────────────────────────────────────────────

    def _warmup(self):
        """One dummy forward pass so the first real inference step is fast."""
        self.get_logger().info("Warming up SmolVLA (one dummy forward pass) ...")
        dummy = self._make_dummy_batch("sfp")
        self.policy.reset()
        with torch.inference_mode():
            self.policy.select_action(dummy)
        self.get_logger().info("Warmup done.")

    # ── Language tokenization ─────────────────────────────────────────────────

    def _tokenize_task(self, task_str: str) -> dict:
        """Tokenize task instruction → observation.language.* tensors for batch."""
        enc = self.tokenizer(
            [task_str],
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

    # ── Task parsing ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_task(task: Task) -> tuple[str, np.ndarray, np.ndarray]:
        """
        Extract task_key, plug_type_vec, target_module_vec from AIC Task message.

        Reads task.port_type and task.target_module_name directly (same fields
        used by the scoring team's script). Falls back to parsing description.

        Returns:
            task_key          — "sfp" or "sc"
            plug_type_vec     — shape (2,)  one-hot
            target_module_vec — shape (7,)  one-hot (uniform 1/7 if module unknown)
        """
        # ── Plug type: prefer task.port_type field ────────────────────────────
        port_type = ""
        if hasattr(task, "port_type") and task.port_type:
            port_type = task.port_type.lower().strip()

        desc = (task.description.lower()
                if hasattr(task, "description") and task.description else "")

        if port_type == "sc" or "sc" in desc:
            task_key      = "sc"
            plug_type_vec = PLUG_TYPE_SC.copy()
        else:
            task_key      = "sfp"
            plug_type_vec = PLUG_TYPE_SFP.copy()

        # ── Target module: prefer task.target_module_name field ───────────────
        target_module_vec = np.zeros(7, dtype=np.float32)
        module_name = None

        if hasattr(task, "target_module_name") and task.target_module_name:
            module_name = task.target_module_name.lower().strip()

        # Fallback: scan description for known module name strings
        if module_name is None or module_name not in _MODULE_INDEX:
            for name in MODULE_NAMES:
                if name in desc:
                    module_name = name
                    break

        if module_name and module_name in _MODULE_INDEX:
            target_module_vec[_MODULE_INDEX[module_name]] = 1.0
        else:
            # Unknown module — uniform distribution stays in-distribution
            # (all-zeros would be a 7σ outlier in the normalized state)
            target_module_vec[:] = 1.0 / 7.0

        return task_key, plug_type_vec, target_module_vec

    # ── Image processing ──────────────────────────────────────────────────────

    def _ros_img_to_tensor(
        self, raw_img, scale: float = IMAGE_SCALE
    ) -> torch.Tensor:
        """
        Convert ROS Image → float [0,1] tensor  (1, C, H, W).

        Uses fp16 on CUDA to match the policy's precision and halve image memory.
        SmolVLA uses IDENTITY normalization — the VLM's SigLIP processor handles
        per-channel stats internally.
        """
        img_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )
        if scale != 1.0:
            img_np = cv2.resize(
                img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        t = (
            torch.from_numpy(img_np.copy())
            .permute(2, 0, 1)   # HWC → CHW
            .float()
            .div(255.0)
            .unsqueeze(0)       # (1, C, H, W)
            .to(self.device)
        )
        # if self.device.type == "cuda":
            # t = t.half()
        return t

    # ── State assembly ────────────────────────────────────────────────────────

    def _build_state(
        self,
        obs_msg: Observation,
        plug_type_vec: np.ndarray,
        target_module_vec: np.ndarray,
    ) -> torch.Tensor:
        """
        Assemble raw 63D state vector, normalise with MEAN_STD, return (1, 63) tensor.

        State layout (must match training pipeline exactly):
          [0:3]   tcp_pos      position xyz
          [3:7]   tcp_quat     quaternion [x, y, z, w]   ← NOT [w, x, y, z]
          [7:13]  tcp_vel      linear (3) + angular (3)
          [13:20] joint_pos    7 joint positions (rad)
          [20:27] joint_vel    7 joint velocities (rad/s)
          [27:33] wrist_ft     Fx,Fy,Fz,Tx,Ty,Tz  (already tared by AIC system)
          [33:40] yolo_left    [conf, cx, cy, w, h, valid, age]  ← stale at deploy
          [40:47] yolo_center  same
          [47:54] yolo_right   same
          [54:56] plug_type    one-hot [SFP, SC]
          [56:63] target_module 7D one-hot over MODULE_NAMES
        """
        cs = obs_msg.controller_state
        js = obs_msg.joint_states
        ww = obs_msg.wrist_wrench  # WrenchStamped — already tared

        tcp_pos  = cs.tcp_pose.position
        tcp_ori  = cs.tcp_pose.orientation
        tcp_vel  = cs.tcp_velocity
        ft       = ww.wrench

        state_np = np.array([
            # TCP position (3)
            tcp_pos.x,
            tcp_pos.y,
            tcp_pos.z,
            # TCP quaternion [x, y, z, w]  (NOT w-first) (4)
            tcp_ori.x,
            tcp_ori.y,
            tcp_ori.z,
            tcp_ori.w,
            # TCP velocity: linear (3) + angular (3)
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
            # Wrist F/T: force (3) + torque (3)
            ft.force.x,
            ft.force.y,
            ft.force.z,
            ft.torque.x,
            ft.torque.y,
            ft.torque.z,
            # YOLO per camera (7 × 3 = 21) — stale placeholder
            *_YOLO_STALE,   # left
            *_YOLO_STALE,   # center
            *_YOLO_STALE,   # right
            # Plug type (2)
            *plug_type_vec,
            # Target module (7)
            *target_module_vec,
        ], dtype=np.float32)

        assert len(state_np) == STATE_DIM, (
            f"State dim mismatch: built {len(state_np)}, expected {STATE_DIM}"
        )

        raw = torch.from_numpy(state_np).unsqueeze(0).to(self.device)  # (1, 63)
        normalized = (raw - self.state_mean) / (self.state_std + 1e-8)
        # if self.device.type == "cuda":
            # normalized = normalized.half()
        return normalized

    # ── Observation dict ──────────────────────────────────────────────────────

    def _prepare_batch(
        self,
        obs_msg: Observation,
        task_key: str,
        plug_type_vec: np.ndarray,
        target_module_vec: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        task_str = TASK_STRINGS[task_key]
        batch = {
            "observation.images.left":   self._ros_img_to_tensor(obs_msg.left_image),
            "observation.images.center": self._ros_img_to_tensor(obs_msg.center_image),
            "observation.images.right":  self._ros_img_to_tensor(obs_msg.right_image),
            "observation.state": self._build_state(
                obs_msg, plug_type_vec, target_module_vec),
            "task": [task_str],
        }
        batch.update(self._tokenize_task(task_str))
        return batch

    # ── Dummy batch for warm-up ───────────────────────────────────────────────

    def _make_dummy_batch(self, task_key: str) -> dict:
        task_str = TASK_STRINGS[task_key]
        # Images at 0.25 scale of 1152×1024 → 288×256 (HW rounded down)
        # Use the same dtype as the policy (fp16 on CUDA, fp32 elsewhere)
        # dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        dtype = torch.float32
        dummy_img   = torch.zeros(1, 3, 288, 256, device=self.device, dtype=dtype)
        dummy_state = torch.zeros(1, STATE_DIM,   device=self.device, dtype=dtype)
        batch = {
            "observation.images.left":   dummy_img,
            "observation.images.center": dummy_img,
            "observation.images.right":  dummy_img,
            "observation.state": dummy_state,
            "task": [task_str],
        }
        batch.update(self._tokenize_task(task_str))
        return batch

    # ── Motion update ─────────────────────────────────────────────────────────

    def _make_motion_update(self, action_np: np.ndarray) -> MotionUpdate:
        """
        Pack 6D Cartesian velocity [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]
        into a MotionUpdate with impedance parameters tuned for cable insertion.
        """
        twist = Twist(
            linear=Vector3(
                x=float(action_np[0]),
                y=float(action_np[1]),
                z=float(action_np[2]),
            ),
            angular=Vector3(
                x=float(action_np[3]),
                y=float(action_np[4]),
                z=float(action_np[5]),
            ),
        )
        msg = MotionUpdate()
        msg.header.frame_id = "base_link"
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.velocity        = twist

        # Impedance tuning — same as RunACT; good for compliant insertion
        msg.target_stiffness = np.diag(
            [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
        ).flatten().tolist()
        msg.target_damping = np.diag(
            [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
        ).flatten().tolist()
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
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        """
        Run one 70-second episode:
          step 0          → full VLM forward pass (~500 ms on GPU)
          steps 1–49      → fast buffer pop (~1 ms each)
          step 50         → re-query VLM (next 5-second chunk)
          ...
        """
        task_key, plug_type_vec, target_module_vec = self._parse_task(task)
        task_str = TASK_STRINGS[task_key]

        module_str = (MODULE_NAMES[int(np.argmax(target_module_vec))]
                      if target_module_vec.max() == 1.0 else "unknown")
        self.get_logger().info(
            f"RunSmolVLA_63d start — task_key={task_key!r}  "
            f"module={module_str}  task='{task_str}'"
        )

        self.policy.reset()

        # Free fragmented GPU memory from previous episode / warmup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            self.get_logger().info(f"  GPU free before episode: {free_gb:.2f} GB")

        step_dt         = 1.0 / CONTROL_HZ
        start           = time.time()
        step_count      = 0
        force_baseline  = None   # measured on first valid observation

        while time.time() - start < EPISODE_DURATION_S:
            t_loop = time.time()

            # ── Observe ────────────────────────────────────────────────────────
            obs_msg = get_observation()
            if obs_msg is None:
                self.get_logger().warn("No observation available — sleeping 50ms.")
                time.sleep(0.05)
                continue

            # ── F/T safety check ───────────────────────────────────────────────
            ww = obs_msg.wrist_wrench.wrench
            force_n   = math.sqrt(ww.force.x**2  + ww.force.y**2  + ww.force.z**2)
            torque_nm = math.sqrt(ww.torque.x**2 + ww.torque.y**2 + ww.torque.z**2)
            if force_baseline is None:
                force_baseline = force_n
                self.get_logger().info(f"  F/T baseline = {force_baseline:.2f} N")
            contact_force = force_n - force_baseline
            if contact_force > _CONTACT_FORCE_N or torque_nm > _TORQUE_LIMIT_NM:
                self.get_logger().warn(
                    f"F/T LIMIT — contact={contact_force:.1f}N "
                    f"(raw={force_n:.1f}N base={force_baseline:.1f}N) "
                    f"torque={torque_nm:.2f}Nm — STOPPING episode"
                )
                break

            # ── Build batch ────────────────────────────────────────────────────
            batch = self._prepare_batch(
                obs_msg, task_key, plug_type_vec, target_module_vec
            )

            # ── Inference ──────────────────────────────────────────────────────
            t_infer = time.time()
            with torch.inference_mode():
                norm_action = self.policy.select_action(batch)
            infer_ms = (time.time() - t_infer) * 1000
            is_new_chunk = infer_ms > 200  # full VLM forward pass > 200ms

            # ── Unnormalise → real Cartesian velocity ──────────────────────────
            real_action = (norm_action * self.action_std) + self.action_mean
            a = real_action[0].float().cpu().numpy()   # (6,) always fp32 for numpy

            # ── Velocity safety clamp ──────────────────────────────────────────
            a[:3] = np.clip(a[:3], -_MAX_VEL_LIN_M_S,   _MAX_VEL_LIN_M_S)
            a[3:] = np.clip(a[3:], -_MAX_VEL_ANG_RAD_S, _MAX_VEL_ANG_RAD_S)

            # ── Log ────────────────────────────────────────────────────────────
            chunk_marker = " ← NEW CHUNK" if is_new_chunk else ""
            self.get_logger().info(
                f"step={step_count:4d}  "
                f"lin=[{a[0]:+.4f} {a[1]:+.4f} {a[2]:+.4f}]  "
                f"ang=[{a[3]:+.4f} {a[4]:+.4f} {a[5]:+.4f}]  "
                f"Fz={ww.force.z:+.1f}N  infer={infer_ms:.0f}ms{chunk_marker}"
            )

            # ── Send to robot ──────────────────────────────────────────────────
            move_robot(motion_update=self._make_motion_update(a))
            send_feedback(
                f"SmolVLA_63d step={step_count}  "
                f"lin_z={a[2]:+.4f}m/s  contact={contact_force:+.1f}N  infer={infer_ms:.0f}ms"
            )

            step_count += 1

            # ── Rate control ───────────────────────────────────────────────────
            elapsed = time.time() - t_loop
            if elapsed > step_dt * 1.5:
                self.get_logger().warn(
                    f"[TIMING] step={step_count} took {elapsed*1000:.0f}ms "
                    f"(target={step_dt*1000:.0f}ms) — may drop frames"
                )
            time.sleep(max(0.0, step_dt - elapsed))

        elapsed_total = time.time() - start
        self.get_logger().info(
            f"RunSmolVLA_63d done — {step_count} steps in {elapsed_total:.1f}s"
        )
        return True

    # ── AIC entry point ───────────────────────────────────────────────────────

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ):
        """
        AIC framework entry point.  Called by AicModel when a cable insertion
        task is dispatched to this policy.
        """
        return self._run_episode(task, get_observation, move_robot, send_feedback)
