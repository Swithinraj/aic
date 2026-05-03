"""
Deploy a locally-trained SmolVLA policy for the AIC cable-insertion task.

ROS params (required):
    checkpoint_path  — absolute path to the pretrained_model/ folder, e.g.
        /home/ibrahim/ros2_ws/src/aic/outputs/train/aic_act_mansi_040000/040000/pretrained_model

ROS params (optional):
    task_instruction  — language prompt sent to the VLM (default: "insert cable")
                        Use "insert nic cable" for NIC port, "insert sc cable" for SC port.

State (30D, must match training):
    tcp_pose(7) + tcp_velocity(6) + joint_positions(7) + joint_velocity(7) + port_xyz(3)
    port_xyz comes from /fused_yolo/detections_json  (YOLO+depth planner must be running)

Action (6D delta TCP at 10 Hz):
    [dx, dy, dz, drx, dry, drz]  — position delta (m) + axis-angle rotation delta (rad)

Usage (3 terminals):
    # Terminal 1 — Gazebo:
    cd $AIC_ROOT && pixi run ros2 launch ...

    # Terminal 2 — YOLO+depth planner:
    cd $AIC_ROOT && pixi run ros2 run team_policy combined_yolo_depth_pose_planner

    # Terminal 3 — SmolVLA policy:
    cd $AIC_ROOT && pixi run ros2 run aic_model aic_model --ros-args \\
        -p use_sim_time:=true \\
        -p policy:=team_policy.run_smolvla \\
        -p checkpoint_path:=/home/ibrahim/ros2_ws/src/aic/outputs/train/aic_act_mansi_040000/040000/pretrained_model \\
        -p task_instruction:="insert nic cable"
"""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import sys
import types

import numpy as np
import torch
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
from rclpy.node import Node
from std_msgs.msg import Header, String
from transformers import AutoTokenizer

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task


def _patch_lerobot_groot_import_bug():
    """Stub out the broken GR00T policy before lerobot.policies.__init__ tries to import it."""
    if "lerobot.policies.groot.configuration_groot" in sys.modules:
        return
    groot_pkg = types.ModuleType("lerobot.policies.groot")
    groot_pkg.__path__ = []
    cfg_mod = types.ModuleType("lerobot.policies.groot.configuration_groot")
    modeling_mod = types.ModuleType("lerobot.policies.groot.modeling_groot")

    class GrootConfig:
        pass

    class GrootPolicy:
        pass

    cfg_mod.GrootConfig = GrootConfig
    modeling_mod.GrootPolicy = GrootPolicy
    sys.modules["lerobot.policies.groot"] = groot_pkg
    sys.modules["lerobot.policies.groot.configuration_groot"] = cfg_mod
    sys.modules["lerobot.policies.groot.modeling_groot"] = modeling_mod


_patch_lerobot_groot_import_bug()

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from safetensors.torch import load_file

# Safety clamps per 10 Hz step (100 ms) — same as run_act.py
_MAX_DELTA_POS_M   = 0.15   # 15 cm max position delta
_MAX_DELTA_ROT_RAD = 0.20   # ~11 deg max rotation delta


# ---------------------------------------------------------------------------
# Quaternion helpers (identical to run_act.py)
# ---------------------------------------------------------------------------

def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ], dtype=np.float64)


def _axis_angle_to_quat(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = rotvec / angle
    s = math.sin(angle / 2.0)
    c = math.cos(angle / 2.0)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, c])


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class RunSmolVLA(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)

        checkpoint_path = str(
            parent_node.declare_parameter("checkpoint_path", "").value
        )
        if not checkpoint_path:
            raise ValueError(
                "RunSmolVLA requires the 'checkpoint_path' ROS parameter.\n"
                "  -p checkpoint_path:=/absolute/path/to/pretrained_model"
            )

        # Task instruction must match training data labels exactly.
        # Training used: "insert cable" / "insert nic cable" / "insert sc cable"
        self._task_instruction = str(
            parent_node.declare_parameter("task_instruction", "insert cable").value
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        path = Path(checkpoint_path)

        # --- Load SmolVLA policy in fp16 to halve GPU memory (~2.7GB → ~1.4GB) ---
        self.policy = SmolVLAPolicy.from_pretrained(str(path))
        self.policy.eval()
        self.policy.to(self.device)
        if self.device.type == "cuda":
            self.policy.half()

        # --- Load tokenizer for language instruction ---
        # SmolVLA expects observation.language.tokens and observation.language.attention_mask
        # The tokenizer must match what was used during training.
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.policy.config.vlm_model_name,
            padding_side="right",
        )

        # --- Load normalisation stats ---
        # State is normalized with MEAN_STD before passing to the policy.
        # Action output is normalized; we denormalize it here.
        pre_path  = path / "policy_preprocessor_step_5_normalizer_processor.safetensors"
        post_path = path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"

        pre_stats  = load_file(str(pre_path))  if pre_path.exists()  else {}
        post_stats = load_file(str(post_path)) if post_path.exists() else {}

        def _get(d: dict, key: str, dim: int, default: float) -> torch.Tensor:
            if key in d:
                return d[key].to(self.device).float()
            # Try suffix match (lerobot sometimes prefixes with feature path)
            for k, v in d.items():
                if k.endswith("." + key) or k == key:
                    return v.to(self.device).float()
            self.get_logger().warning(f"Stat key '{key}' not found — using default {default}")
            return torch.full((dim,), default, device=self.device)

        STATE_DIM  = 30
        ACTION_DIM = 6

        self.state_mean  = _get(pre_stats,  "observation.state.mean", STATE_DIM,  0.0).view(1, -1)
        self.state_std   = _get(pre_stats,  "observation.state.std",  STATE_DIM,  1.0).view(1, -1)
        self.action_mean = _get(post_stats, "action.mean",            ACTION_DIM, 0.0).view(1, -1)
        self.action_std  = _get(post_stats, "action.std",             ACTION_DIM, 1.0).view(1, -1)

        # --- Subscribe to YOLO planner for port_xyz ---
        # The planner publishes /fused_yolo/detections_json with pose_base_link positions.
        # port_xyz is the 3rd component of the 30D state vector.
        self._port_xyz       = np.zeros(3, dtype=np.float32)
        self._port_lock      = threading.Lock()
        self._port_stamp     = 0.0  # time.time() of last valid YOLO detection
        # Per-episode task type set dynamically in insert_cable() — drives YOLO filtering
        self._current_task_type        = "all"
        self._current_task_instruction = self._task_instruction
        self._yolo_sub  = parent_node.create_subscription(
            String,
            "/fused_yolo/detections_json",
            self._yolo_cb,
            10,
        )

        self.get_logger().info(
            f"RunSmolVLA loaded:\n"
            f"  path        = {path}\n"
            f"  device      = {self.device}\n"
            f"  task        = '{self._task_instruction}'\n"
            f"  state_dim   = {STATE_DIM}D (tcp_pose+vel+jpos+jvel+port_xyz)\n"
            f"  action_dim  = {ACTION_DIM}D delta-TCP\n"
            f"  pre_stats   = {'found' if pre_stats  else 'MISSING (using defaults)'}\n"
            f"  post_stats  = {'found' if post_stats else 'MISSING (using defaults)'}\n"
            f"  state_mean  = {self.state_mean[0].cpu().numpy().round(4).tolist()}\n"
            f"  state_std   = {self.state_std[0].cpu().numpy().round(4).tolist()}\n"
            f"  action_mean = {self.action_mean[0].cpu().numpy().round(6).tolist()}\n"
            f"  action_std  = {self.action_std[0].cpu().numpy().round(6).tolist()}"
        )
        self._debug_step = 0  # global counter for throttled debug prints

    # ----------------------------------------------------------------
    # YOLO subscriber — extracts port_xyz from fused detections JSON
    # ----------------------------------------------------------------

    def _yolo_cb(self, msg: String) -> None:
        try:
            detections = json.loads(msg.data)
        except Exception:
            return

        # Select target family based on per-episode task type (set dynamically in insert_cable)
        task_type = self._current_task_type
        if task_type == "nic":
            target_classes = {"nic", "nic_card", "nic_port", "nic_card_0", "nic_card_1",
                              "nic_card_2", "nic_card_3", "sfp", "sfp_port", "sfp_module"}
            target_families = {"nic", "sfp"}
        elif task_type == "sc":
            target_classes = {"sc", "sc_port", "sc_port_0", "sc_port_1", "sc_port_base",
                              "sc_port_0", "sfp_sc"}
            target_families = {"sc"}
        else:
            target_classes = {"nic", "nic_card", "nic_port", "sfp", "sfp_port",
                              "sc", "sc_port", "sfp_sc"}
            target_families = {"nic", "sc", "sfp"}

        best = None
        best_conf = 0.0
        for det in detections:
            cn = str(det.get("class_name", "")).lower().replace(" ", "_")
            family = cn.split("_")[0]
            if cn not in target_classes and family not in target_families:
                continue
            conf = float(det.get("confidence", 0.0))
            if conf > best_conf:
                best = det
                best_conf = conf

        if best is None:
            self.get_logger().warning(
                f"[YOLO] No matching detection — classes seen: "
                f"{[d.get('class_name','?') for d in detections]}"
            )
            return

        pos = best.get("pose_base_link", {}).get("position", {})
        new_xyz = np.array(
            [float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))],
            dtype=np.float32,
        )
        with self._port_lock:
            old_xyz = self._port_xyz.copy()
            self._port_xyz   = new_xyz
            self._port_stamp = time.time()

        if np.linalg.norm(new_xyz - old_xyz) > 0.005:  # log only when it moves >5mm
            self.get_logger().info(
                f"[YOLO] port_xyz updated: {new_xyz.round(4).tolist()} "
                f"(class='{best.get('class_name','?')}' conf={best.get('confidence',0):.2f})"
            )

    # ----------------------------------------------------------------
    # Observation → model input
    # ----------------------------------------------------------------

    def _img_to_tensor(self, img_msg) -> torch.Tensor:
        """Convert ROS Image msg to float [0,1] tensor (1,C,H,W) on device."""
        arr = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
            img_msg.height, img_msg.width, 3
        )
        t = (
            torch.from_numpy(arr.copy())
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device)
        )
        if self.device.type == "cuda":
            t = t.half()
        return t

    def _build_state(self, obs_msg) -> torch.Tensor:
        """30D state: tcp_pose(7)+tcp_vel(6)+jpos(7)+jvel(7)+port_xyz(3), normalized."""
        cs  = obs_msg.controller_state
        tcp = cs.tcp_pose
        vel = cs.tcp_velocity
        js  = obs_msg.joint_states

        with self._port_lock:
            port = self._port_xyz.copy()

        raw = np.array([
            tcp.position.x,    tcp.position.y,    tcp.position.z,
            tcp.orientation.x, tcp.orientation.y, tcp.orientation.z, tcp.orientation.w,
            vel.linear.x,  vel.linear.y,  vel.linear.z,
            vel.angular.x, vel.angular.y, vel.angular.z,
            *list(js.position[:7]),
            *list(js.velocity[:7]),
            port[0], port[1], port[2],
        ], dtype=np.float32)

        if self._debug_step % 10 == 0:
            self.get_logger().info(
                f"[STATE step={self._debug_step}]\n"
                f"  tcp_pos    = [{raw[0]:.4f}, {raw[1]:.4f}, {raw[2]:.4f}]\n"
                f"  tcp_quat   = [{raw[3]:.4f}, {raw[4]:.4f}, {raw[5]:.4f}, {raw[6]:.4f}]\n"
                f"  tcp_vel    = [{raw[7]:.4f}, {raw[8]:.4f}, {raw[9]:.4f}]\n"
                f"  jpos[:4]   = {raw[13:17].round(4).tolist()}\n"
                f"  port_xyz   = [{port[0]:.4f}, {port[1]:.4f}, {port[2]:.4f}]"
                + ("  <-- ZEROS! YOLO not publishing?" if np.all(port == 0) else "")
            )

        t = torch.from_numpy(raw).unsqueeze(0).to(self.device)
        norm = (t - self.state_mean) / torch.clamp(self.state_std, min=1e-8)
        if self.device.type == "cuda":
            norm = norm.half()
        return norm

    def _tokenize_task(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize the task instruction. SmolVLA expects a trailing newline."""
        task_text = self._current_task_instruction
        if not task_text.endswith("\n"):
            task_text = task_text + "\n"

        encoded = self._tokenizer(
            [task_text],
            padding="longest",
            truncation=True,
            max_length=self.policy.config.tokenizer_max_length,
            return_tensors="pt",
        )
        input_ids      = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(dtype=torch.bool, device=self.device)
        return input_ids, attention_mask

    def _to_batch(self, obs_msg) -> dict:
        lang_tokens, lang_mask = self._tokenize_task()
        return {
            "observation.images.left":              self._img_to_tensor(obs_msg.left_image),
            "observation.images.center":            self._img_to_tensor(obs_msg.center_image),
            "observation.images.right":             self._img_to_tensor(obs_msg.right_image),
            "observation.state":                    self._build_state(obs_msg),
            "observation.language.tokens":          lang_tokens,
            "observation.language.attention_mask":  lang_mask,
        }

    # ----------------------------------------------------------------
    # Action → motion command (identical to run_act.py)
    # ----------------------------------------------------------------

    def _delta_to_motion(self, obs_msg, action_6d: np.ndarray) -> MotionUpdate:
        """Apply 6D delta to current TCP pose → absolute MODE_POSITION command."""
        cs  = obs_msg.controller_state
        tcp = cs.tcp_pose

        cur_pos  = np.array([tcp.position.x,    tcp.position.y,    tcp.position.z],   dtype=np.float64)
        cur_quat = np.array([tcp.orientation.x, tcp.orientation.y,
                              tcp.orientation.z, tcp.orientation.w], dtype=np.float64)

        dp = np.clip(action_6d[:3].astype(np.float64), -_MAX_DELTA_POS_M, _MAX_DELTA_POS_M)
        dr = action_6d[3:6].astype(np.float64)
        dr_norm = np.linalg.norm(dr)
        if dr_norm > _MAX_DELTA_ROT_RAD:
            dr = dr / dr_norm * _MAX_DELTA_ROT_RAD

        new_pos  = cur_pos + dp
        dq       = _axis_angle_to_quat(dr)
        new_quat = _quat_multiply(dq, cur_quat)
        nrm = np.linalg.norm(new_quat)
        if nrm > 1e-9:
            new_quat /= nrm

        target = Pose(
            position=Point(x=float(new_pos[0]), y=float(new_pos[1]), z=float(new_pos[2])),
            orientation=Quaternion(
                x=float(new_quat[0]), y=float(new_quat[1]),
                z=float(new_quat[2]), w=float(new_quat[3]),
            ),
        )

        mu = MotionUpdate()
        mu.header = Header(
            frame_id="base_link",
            stamp=self._parent_node.get_clock().now().to_msg(),
        )
        mu.pose = target
        mu.trajectory_generation_mode = TrajectoryGenerationMode(
            mode=TrajectoryGenerationMode.MODE_POSITION,
        )
        # Match stiffness/damping used during training (CheatCode collector)
        mu.target_stiffness = np.diag([90.0, 90.0, 90.0, 50.0, 50.0, 50.0]).flatten().tolist()
        mu.target_damping   = np.diag([50.0, 50.0, 50.0, 20.0, 20.0, 20.0]).flatten().tolist()
        mu.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        mu.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        return mu

    # ----------------------------------------------------------------
    # Entry point
    # ----------------------------------------------------------------

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ) -> bool:
        # --- Determine task type and instruction from the task object ---
        module = task.target_module_name.lower()
        port_t = task.port_type.lower()
        if "nic" in module or port_t == "sfp":
            self._current_task_type        = "nic"
            self._current_task_instruction = "insert nic cable"
        elif port_t == "sc" or "sc" in module:
            self._current_task_type        = "sc"
            self._current_task_instruction = "insert sc cable"
        else:
            self._current_task_type        = "all"
            self._current_task_instruction = self._task_instruction

        # Reset port_xyz at episode start, then wait for YOLO before running policy.
        # Do NOT start inference with port_xyz=[0,0,0] — training mean is ~[-0.44, 0.30, 0.14] so
        # zeros produce a +6σ outlier in the normalized state and the policy behaves randomly.
        with self._port_lock:
            self._port_xyz = np.zeros(3, dtype=np.float32)

        self.policy.reset()

        # --- Free fragmented CUDA memory before inference ---
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.get_logger().info(
            f"RunSmolVLA.insert_cable() start — task: {task}\n"
            f"  task_type   = {self._current_task_type}\n"
            f"  instruction = '{self._current_task_instruction}'\n"
            f"  GPU free    = {torch.cuda.mem_get_info()[0] / 1024**3:.2f}GB / {torch.cuda.mem_get_info()[1] / 1024**3:.2f}GB"
            if torch.cuda.is_available() else
            f"RunSmolVLA.insert_cable() start — task: {task}"
        )

        TIME_LIMIT_S      = 60.0
        STEP_S            = 1.0 / 10.0   # 10 Hz — matches training control frequency
        CONTACT_FORCE_N   = 20.0  # contact force above baseline that triggers stop
        TORQUE_LIMIT_NM   = 5.0
        step_count        = 0
        force_baseline_n  = None  # measured on first observation
        start             = time.time()

        # --- Wait for YOLO to provide a valid port_xyz before starting inference ---
        YOLO_WAIT_S = 10.0
        yolo_t0 = time.time()
        while time.time() - yolo_t0 < YOLO_WAIT_S:
            with self._port_lock:
                port_valid = np.any(self._port_xyz != 0.0)
            if port_valid:
                with self._port_lock:
                    pv = self._port_xyz.copy()
                self.get_logger().info(
                    f"[YOLO] Valid port_xyz after {time.time()-yolo_t0:.1f}s: {pv.round(4).tolist()}"
                )
                break
            time.sleep(0.1)
        else:
            with self._port_lock:
                pv = self._port_xyz.copy()
            self.get_logger().warning(
                f"[YOLO] No valid detection after {YOLO_WAIT_S:.0f}s — port_xyz still "
                f"{pv.tolist()}. Policy will run with zeros (expect degraded insertion)."
            )

        chunk_idx = 0  # tracks which action in the 50-step chunk we're on

        while time.time() - start < TIME_LIMIT_S:
            t0 = time.time()

            obs_msg = get_observation()
            if obs_msg is None:
                self.get_logger().warning("No observation — skipping step")
                time.sleep(STEP_S)
                continue

            # Force/torque safety check — baseline-subtracted to match training
            wrench      = obs_msg.wrist_wrench.wrench
            force_norm  = math.sqrt(wrench.force.x**2  + wrench.force.y**2  + wrench.force.z**2)
            torque_norm = math.sqrt(wrench.torque.x**2 + wrench.torque.y**2 + wrench.torque.z**2)
            if force_baseline_n is None:
                force_baseline_n = force_norm
                self.get_logger().info(f"[F/T] baseline = {force_baseline_n:.2f}N")
            contact_force = force_norm - force_baseline_n
            if contact_force > CONTACT_FORCE_N or torque_norm > TORQUE_LIMIT_NM:
                self.get_logger().warning(
                    f"RunSmolVLA force limit — contact={contact_force:.1f}N "
                    f"(raw={force_norm:.1f}N base={force_baseline_n:.1f}N) "
                    f"torque={torque_norm:.2f}Nm — stopping"
                )
                break

            # Log force/torque every 10 steps
            if step_count % 10 == 0:
                self.get_logger().info(
                    f"[F/T step={step_count}] contact={contact_force:.2f}N "
                    f"raw={force_norm:.2f}N torque={torque_norm:.3f}Nm"
                )

            # Warn if YOLO hasn't updated in >2s (port_xyz may be stale/zeroed)
            with self._port_lock:
                yolo_age = time.time() - self._port_stamp if self._port_stamp > 0 else float("inf")
            if step_count % 10 == 0 and yolo_age > 2.0:
                self.get_logger().warning(
                    f"[YOLO] No detection for {yolo_age:.1f}s at step={step_count} "
                    f"— port_xyz may be stale. Check combined_yolo_depth_pose_planner."
                )

            batch = self._to_batch(obs_msg)

            infer_t0 = time.time()
            with torch.inference_mode():
                # select_action returns one action from the 50-step chunk.
                # Chunk is regenerated automatically when exhausted (every ~5s at 10Hz).
                norm_action = self.policy.select_action(batch)  # (1, 6) normalised
            infer_ms = (time.time() - infer_t0) * 1000

            # Detect when a new chunk is fetched (inference takes >200ms = real forward pass)
            is_new_chunk = infer_ms > 200
            if is_new_chunk:
                chunk_idx = 0
                with self._port_lock:
                    port_now = self._port_xyz.copy()
                self.get_logger().info(
                    f"[CHUNK step={step_count}] new 50-action chunk generated "
                    f"(infer={infer_ms:.0f}ms) port_xyz={port_now.round(4).tolist()}"
                )
            else:
                chunk_idx += 1

            # Denormalize: model outputs normalised actions, we restore original scale
            raw_action = norm_action * self.action_std + self.action_mean
            action_np  = raw_action[0].cpu().numpy().astype(np.float64)

            # Log every step for first 10, then every 5 steps
            if step_count < 10 or step_count % 5 == 0:
                self.get_logger().info(
                    f"[ACT step={step_count} chunk_idx={chunk_idx}] "
                    f"norm={norm_action[0].cpu().numpy().round(3).tolist()} "
                    f"raw(m/rad)={action_np.round(4).tolist()} "
                    f"infer={infer_ms:.0f}ms"
                )

            self._debug_step += 1
            mu = self._delta_to_motion(obs_msg, action_np)
            move_robot(motion_update=mu)

            step_count += 1
            if step_count % 10 == 0:
                with self._port_lock:
                    port = self._port_xyz.copy()
                send_feedback(
                    f"RunSmolVLA step={step_count} elapsed={time.time()-start:.1f}s "
                    f"force={force_norm:.1f}N port_xyz={port.round(3).tolist()}"
                )

            elapsed = time.time() - t0
            if elapsed > STEP_S * 1.5:
                self.get_logger().warning(f"[TIMING] step={step_count} took {elapsed*1000:.0f}ms (target={STEP_S*1000:.0f}ms)")
            time.sleep(max(0.0, STEP_S - elapsed))

        self.get_logger().info(f"RunSmolVLA finished — {step_count} steps")
        return True


# aic_model resolves the class by the last component of the module path.
# -p policy:=team_policy.run_smolvla  →  looks for `run_smolvla` attribute
run_smolvla = RunSmolVLA
