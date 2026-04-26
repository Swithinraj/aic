from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, Wrench
from rclpy.node import Node

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import GetObservationCallback, MoveRobotCallback, Policy, SendFeedbackCallback
from aic_task_interfaces.msg import Task

from safetensors.torch import load_file
import sys
import types


def _patch_lerobot_groot_import_bug():
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

import sys
import types


def _patch_lerobot_groot_import_bug():
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

import sys
import types


def _patch_lerobot_groot_import_bug():
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

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def _axis_angle_to_quat(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    axis = rotvec / angle
    s = math.sin(angle * 0.5)
    c = math.cos(angle * 0.5)

    return np.array(
        [
            float(axis[0] * s),
            float(axis[1] * s),
            float(axis[2] * s),
            float(c),
        ],
        dtype=np.float64,
    )


class LeRobotACTPolicy(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)

        self.node = parent_node

        self.checkpoint_path = self._resolve_checkpoint_path()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.act_hz = float(parent_node.declare_parameter("act_hz", 10.0).value)
        self.action_frame = str(parent_node.declare_parameter("action_frame", "base_link").value)
        self.action_scale = float(parent_node.declare_parameter("action_scale", 0.35).value)
        self.max_linear_velocity_mps = float(parent_node.declare_parameter("max_linear_velocity_mps", 0.04).value)
        self.max_angular_velocity_rps = float(parent_node.declare_parameter("max_angular_velocity_rps", 0.15).value)
        self.min_tcp_z = float(parent_node.declare_parameter("min_tcp_z", 0.10).value)
        self.max_total_drop_m = float(parent_node.declare_parameter("max_total_drop_m", 0.08).value)
        self.duration_sec = float(parent_node.declare_parameter("act_duration_sec", 0.0).value)
        self._start_tcp_z = None

        self.policy = self._load_policy(self.checkpoint_path)

        pre_stats = self._load_safetensor_if_exists(
            self.checkpoint_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        )
        post_stats = self._load_safetensor_if_exists(
            self.checkpoint_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        )

        self.state_mean = self._load_stat(pre_stats, "observation.state.mean", 33, 0.0).view(1, -1)
        self.state_std = self._load_stat(pre_stats, "observation.state.std", 33, 1.0).view(1, -1)

        self.action_mean = self._load_stat(post_stats, "action.mean", 6, 0.0).view(1, -1)
        self.action_std = self._load_stat(post_stats, "action.std", 6, 1.0).view(1, -1)

        self.state_dim = int(self.state_mean.shape[-1])
        self.action_dim = int(self.action_mean.shape[-1])

        if self.state_dim != 33:
            raise ValueError(f"Expected observation.state dimension 33, got {self.state_dim}")

        if self.action_dim != 6:
            raise ValueError(f"Expected ACT action dimension 6 for Cartesian delta TCP control, got {self.action_dim}")

        self.img_mean = _IMAGENET_MEAN.to(self.device)
        self.img_std = _IMAGENET_STD.to(self.device)

        self.get_logger().info(
            "LeRobot ACT policy loaded "
            f"path={self.checkpoint_path} "
            f"device={self.device} "
            f"state_dim={self.state_dim} "
            f"action_dim={self.action_dim} "
            f"act_hz={self.act_hz}"
        )

    def _resolve_checkpoint_path(self) -> Path:
        raw = str(self.node.declare_parameter("checkpoint_path", "").value).strip()

        if raw:
            path = Path(raw).expanduser().resolve()
        else:
            package_root = Path(__file__).resolve().parents[1]
            path = package_root / "model"

        if path.name == "pretrained_model":
            ckpt = path
        elif (path / "pretrained_model").is_dir():
            ckpt = path / "pretrained_model"
        else:
            numeric_dirs = []
            if path.is_dir():
                for item in path.iterdir():
                    if item.is_dir() and item.name.isdigit() and (item / "pretrained_model").is_dir():
                        numeric_dirs.append(item)

            if not numeric_dirs:
                raise FileNotFoundError(
                    "Could not find pretrained_model. Pass for example: "
                    "-p checkpoint_path:=/home/swithin/official_aic/aic/team_policy/model/040000/pretrained_model"
                )

            ckpt = sorted(numeric_dirs, key=lambda p: int(p.name))[-1] / "pretrained_model"

        required = [
            "config.json",
            "model.safetensors",
            "policy_preprocessor_step_3_normalizer_processor.safetensors",
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        ]

        missing = [name for name in required if not (ckpt / name).exists()]
        if missing:
            raise FileNotFoundError(f"Checkpoint folder is missing files: {missing}. Path: {ckpt}")

        return ckpt

    def _load_policy(self, path: Path) -> ACTPolicy:
        import draccus

        with open(path / "config.json", "r") as f:
            cfg_dict = json.load(f)

        cfg_dict.pop("type", None)

        config = draccus.decode(ACTConfig, cfg_dict)

        policy = ACTPolicy(config)
        weights = load_file(str(path / "model.safetensors"))
        policy.load_state_dict(weights)
        policy.eval()
        policy.to(self.device)

        return policy

    def _load_safetensor_if_exists(self, path: Path) -> dict:
        if not path.exists():
            return {}
        return load_file(str(path))

    def _load_stat(self, stats: dict, key: str, dim: int, default: float) -> torch.Tensor:
        if key in stats:
            return stats[key].float().to(self.device)

        suffix = "." + key
        for k, v in stats.items():
            if k.endswith(suffix):
                return v.float().to(self.device)

        self.get_logger().warning(f"Missing stat {key}; using default={default}")
        return torch.full((dim,), float(default), dtype=torch.float32, device=self.device)

    def _ros_image_to_rgb(self, img_msg) -> np.ndarray:
        height = int(img_msg.height)
        width = int(img_msg.width)
        step = int(img_msg.step)
        encoding = str(img_msg.encoding).lower()

        channels = 3
        if encoding in ("rgba8", "bgra8"):
            channels = 4
        elif encoding in ("mono8", "8uc1"):
            channels = 1

        raw = np.frombuffer(img_msg.data, dtype=np.uint8)
        arr = raw.reshape(height, step)[:, : width * channels]
        arr = arr.reshape(height, width, channels)

        if channels == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif encoding == "bgr8":
            arr = arr[:, :, ::-1]
        elif encoding == "bgra8":
            arr = arr[:, :, [2, 1, 0]]
        elif encoding == "rgba8":
            arr = arr[:, :, :3]
        else:
            arr = arr[:, :, :3]

        return np.ascontiguousarray(arr)

    def _image_to_tensor(self, img_msg) -> torch.Tensor:
        arr = self._ros_image_to_rgb(img_msg)
        t = torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        return (t - self.img_mean) / self.img_std

    def _build_state(self, obs_msg) -> torch.Tensor:
        cs = obs_msg.controller_state
        tcp = cs.tcp_pose
        vel = cs.tcp_velocity
        js = obs_msg.joint_states

        position = list(js.position[:7])
        velocity = list(js.velocity[:7])

        while len(position) < 7:
            position.append(0.0)

        while len(velocity) < 7:
            velocity.append(0.0)

        tcp_error = list(cs.tcp_error)
        while len(tcp_error) < 6:
            tcp_error.append(0.0)

        raw = np.array(
            [
                tcp.position.x,
                tcp.position.y,
                tcp.position.z,
                tcp.orientation.x,
                tcp.orientation.y,
                tcp.orientation.z,
                tcp.orientation.w,
                vel.linear.x,
                vel.linear.y,
                vel.linear.z,
                vel.angular.x,
                vel.angular.y,
                vel.angular.z,
                *tcp_error[:6],
                *position[:7],
                *velocity[:7],
            ],
            dtype=np.float32,
        )

        t = torch.from_numpy(raw).unsqueeze(0).to(self.device)
        return (t - self.state_mean) / torch.clamp(self.state_std, min=1e-6)

    def _build_batch(self, obs_msg) -> dict:
        return {
            "observation.images.left": self._image_to_tensor(obs_msg.left_image),
            "observation.images.center": self._image_to_tensor(obs_msg.center_image),
            "observation.images.right": self._image_to_tensor(obs_msg.right_image),
            "observation.state": self._build_state(obs_msg),
        }

    def _extract_action(self, model_output) -> np.ndarray:
        if isinstance(model_output, dict):
            if "action" in model_output:
                model_output = model_output["action"]
            else:
                first_key = next(iter(model_output.keys()))
                model_output = model_output[first_key]

        if isinstance(model_output, np.ndarray):
            action = torch.from_numpy(model_output).to(self.device)
        else:
            action = model_output

        if action.ndim == 1:
            action = action.view(1, -1)

        if action.ndim == 3:
            action = action[:, 0, :]

        action = action.float()

        if action.shape[-1] != self.action_dim:
            raise ValueError(f"Model returned action dim {action.shape[-1]}, expected {self.action_dim}")

        action = action * self.action_std + self.action_mean
        return action[0].detach().cpu().numpy().astype(np.float64)

    def _action_to_motion_update(self, obs_msg, action_6d: np.ndarray) -> MotionUpdate:
        action_6d = np.nan_to_num(action_6d.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

        linear = action_6d[:3] * self.action_scale
        angular = action_6d[3:6] * self.action_scale

        linear = np.clip(
            linear,
            -self.max_linear_velocity_mps,
            self.max_linear_velocity_mps,
        )

        angular = np.clip(
            angular,
            -self.max_angular_velocity_rps,
            self.max_angular_velocity_rps,
        )

        tcp_z = float(obs_msg.controller_state.tcp_pose.position.z)

        if self._start_tcp_z is None:
            self._start_tcp_z = tcp_z

        too_low = tcp_z <= self.min_tcp_z
        dropped_too_much = tcp_z <= self._start_tcp_z - self.max_total_drop_m

        if too_low or dropped_too_much:
            if self.action_frame == "base_link":
                linear[2] = max(0.0, linear[2])
            else:
                linear[:] = 0.0

        msg = MotionUpdate()
        msg.header.frame_id = self.action_frame
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.velocity = Twist(
            linear=Vector3(
                x=float(linear[0]),
                y=float(linear[1]),
                z=float(linear[2]),
            ),
            angular=Vector3(
                x=float(angular[0]),
                y=float(angular[1]),
                z=float(angular[2]),
            ),
        )

        msg.target_stiffness = [
            90.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 90.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 90.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 5.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 5.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 5.0,
        ]

        msg.target_damping = [
            50.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 50.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 50.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 5.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 5.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 5.0,
        ]

        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )

        msg.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY

        return msg

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ) -> bool:
        del kwargs

        try:
            self.policy.reset()
        except Exception:
            pass

        step_dt = 1.0 / max(self.act_hz, 1e-6)

        if self.duration_sec > 0.0:
            run_sec = self.duration_sec
        else:
            task_limit = float(getattr(task, "time_limit", 60.0))
            run_sec = max(1.0, task_limit - 1.0)

        start = time.monotonic()
        step = 0

        send_feedback(
            f"lerobot_act/start task={task.id} checkpoint={self.checkpoint_path} run_sec={run_sec:.1f}"
        )

        while time.monotonic() - start < run_sec:
            t0 = time.monotonic()

            obs_msg = get_observation()
            if obs_msg is None:
                send_feedback("lerobot_act/waiting_for_observation")
                time.sleep(step_dt)
                continue

            batch = self._build_batch(obs_msg)

            with torch.inference_mode():
                model_output = self.policy.select_action(batch)

            action = self._extract_action(model_output)
            motion_update = self._action_to_motion_update(obs_msg, action)

            move_robot(motion_update=motion_update)

            step += 1
            if step % int(max(1, round(self.act_hz))) == 0:
                elapsed = time.monotonic() - start
                tcp_z = float(obs_msg.controller_state.tcp_pose.position.z)
                send_feedback(
                    "lerobot_act/step "
                    f"step={step} elapsed={elapsed:.1f} tcp_z={tcp_z:.3f} "
                    f"action_vel=({action[0]:.4f},{action[1]:.4f},{action[2]:.4f},"
                    f"{action[3]:.4f},{action[4]:.4f},{action[5]:.4f})"
                )

            sleep_time = step_dt - (time.monotonic() - t0)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

        send_feedback(f"lerobot_act/timeout_or_finished steps={step} elapsed={time.monotonic() - start:.1f}")
        return False


mypolicy_lerobot = LeRobotACTPolicy
mypolicy = LeRobotACTPolicy