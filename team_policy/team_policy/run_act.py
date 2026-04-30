"""
Deploy a locally-trained ACT policy for the AIC cable-insertion task.

HYBRID APPROACH:
  Phase 1 — ACT model with 10-step closed-loop replanning (coarse approach)
  Phase 2 — Force-guided insertion (slow push along insertion axis)

The model gets the cable CLOSE to the port; the force-guided phase
pushes it the final few mm into the port with compliant impedance.
"""
from __future__ import annotations

import json
import math
import sys
import time
import types
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
from rclpy.node import Node
from std_msgs.msg import Header

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task

# ---------------------------------------------------------------------------
# LeRobot ACT import workaround (bypass GROOT dataclass error)
# ---------------------------------------------------------------------------
import lerobot as _lerobot_pkg

_lerobot_root = Path(_lerobot_pkg.__file__).resolve().parent
_policies_dir = _lerobot_root / "policies"
_act_dir = _policies_dir / "act"

_policies_pkg = types.ModuleType("lerobot.policies")
_policies_pkg.__path__ = [str(_policies_dir)]
sys.modules["lerobot.policies"] = _policies_pkg

_act_pkg = types.ModuleType("lerobot.policies.act")
_act_pkg.__path__ = [str(_act_dir)]
sys.modules["lerobot.policies.act"] = _act_pkg

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from safetensors.torch import load_file

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_IMG_H, _IMG_W = 480, 640

# Force/torque safety
_FORCE_HARD_N    = 80.0    # Emergency stop
_TORQUE_HARD_NM  = 12.0

# Timing
_TIME_LIMIT_S = 150.0      # Total time budget (session allows 180s)
_STEP_HZ      = 10.0
_STEP_S        = 1.0 / _STEP_HZ

# ACT phase parameters
_REPLAN_EVERY  = 10        # Re-query model every 10 steps (1s) for closed-loop
_EMA_ALPHA     = 0.7       # Smoothing at chunk boundaries

# Impedance — approach phase (matches training)
_STIFFNESS_APPROACH = np.diag([90.0, 90.0, 90.0, 50.0, 50.0, 50.0]).flatten().tolist()
_DAMPING_APPROACH   = np.diag([50.0, 50.0, 50.0, 20.0, 20.0, 20.0]).flatten().tolist()

# Impedance — insertion phase (stiff enough to push, but with wrench compliance)
_STIFFNESS_INSERT = np.diag([90.0, 90.0, 90.0, 50.0, 50.0, 50.0]).flatten().tolist()
_DAMPING_INSERT   = np.diag([50.0, 50.0, 50.0, 20.0, 20.0, 20.0]).flatten().tolist()

# Insertion phase parameters
_INSERT_STEP_M       = 0.001    # 1mm per step along insertion axis
_INSERT_DEPTH_M      = 0.040    # Push 40mm to ensure full engagement
_INSERT_FORCE_THRESH = 5.0      # Force (N) indicating contact with port
_STALL_WINDOW        = 30       # Steps to average for stall detection
_STALL_VEL_THRESH    = 0.0003   # m/step — below this = stalled
_MIN_ACT_STEPS       = 100      # Minimum ACT steps before allowing insertion trigger
_ACT_TIMEOUT_S       = 80.0     # Max time in ACT phase before forcing insertion


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product  a ⊗ b  with (x,y,z,w) convention."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ], dtype=np.float64)


def _axis_angle_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """Axis-angle (3D) → unit quaternion (x,y,z,w)."""
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = rotvec / angle
    s = math.sin(angle / 2.0)
    c = math.cos(angle / 2.0)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, c])


def _quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) → 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class RunACT(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)

        checkpoint_path = str(
            parent_node.declare_parameter("checkpoint_path", "").value
        )
        if not checkpoint_path:
            raise ValueError(
                "RunACT requires the 'checkpoint_path' ROS parameter.\n"
                "  -p checkpoint_path:=/absolute/path/to/pretrained_model"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        path = Path(checkpoint_path)

        # --- Load config + weights ---
        with open(path / "config.json") as f:
            cfg_dict = json.load(f)
        cfg_dict.pop("type", None)

        import draccus
        config = draccus.decode(ACTConfig, cfg_dict)
        self.policy = ACTPolicy(config)
        self.policy.load_state_dict(load_file(path / "model.safetensors"))
        self.policy.eval()
        self.policy.to(self.device)

        # --- Load normalisation stats ---
        pre_path  = path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        post_path = path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"

        pre_stats  = load_file(str(pre_path))  if pre_path.exists()  else {}
        post_stats = load_file(str(post_path)) if post_path.exists() else {}

        def _get(d, key, shape, default):
            if key in d:
                return d[key].to(self.device).float()
            self.get_logger().warning(f"Stat key '{key}' not found — using default {default}")
            return torch.full(shape, default, device=self.device)

        STATE_DIM, ACTION_DIM = 33, 6
        self.state_mean  = _get(pre_stats,  "observation.state.mean", (STATE_DIM,),  0.0).view(1, -1)
        self.state_std   = _get(pre_stats,  "observation.state.std",  (STATE_DIM,),  1.0).view(1, -1)
        self.action_mean = _get(post_stats, "action.mean",            (ACTION_DIM,), 0.0).view(1, -1)
        self.action_std  = _get(post_stats, "action.std",             (ACTION_DIM,), 1.0).view(1, -1)

        self._img_mean = _IMAGENET_MEAN.to(self.device)
        self._img_std  = _IMAGENET_STD.to(self.device)

        # EMA state
        self._prev_action: np.ndarray | None = None

        self.get_logger().info(
            f"RunACT loaded (HYBRID mode):\n"
            f"  path        = {path}\n"
            f"  device      = {self.device}\n"
            f"  replan      = every {_REPLAN_EVERY} steps\n"
            f"  time_limit  = {_TIME_LIMIT_S}s\n"
            f"  insert_step = {_INSERT_STEP_M*1000:.1f}mm\n"
            f"  pre_stats   = {'found' if pre_stats else 'MISSING'}\n"
            f"  post_stats  = {'found' if post_stats else 'MISSING'}"
        )

    # ----------------------------------------------------------------
    # Observation → model input
    # ----------------------------------------------------------------

    def _img_to_tensor(self, img_msg) -> torch.Tensor:
        arr = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
            img_msg.height, img_msg.width, 3
        )
        t = (
            torch.from_numpy(arr.copy())
            .permute(2, 0, 1).float().div(255.0)
            .unsqueeze(0).to(self.device)
        )
        if t.shape[2] != _IMG_H or t.shape[3] != _IMG_W:
            t = F.interpolate(t, size=(_IMG_H, _IMG_W), mode="bilinear", align_corners=False)
        return (t - self._img_mean) / self._img_std

    def _build_state(self, obs_msg) -> torch.Tensor:
        cs  = obs_msg.controller_state
        tcp = cs.tcp_pose
        vel = cs.tcp_velocity
        js  = obs_msg.joint_states

        raw = np.array([
            tcp.position.x,    tcp.position.y,    tcp.position.z,
            tcp.orientation.x, tcp.orientation.y, tcp.orientation.z, tcp.orientation.w,
            vel.linear.x,  vel.linear.y,  vel.linear.z,
            vel.angular.x, vel.angular.y, vel.angular.z,
            *list(cs.tcp_error),
            *list(js.position[:7]),
            *list(js.velocity[:7]),
        ], dtype=np.float32)

        t = torch.from_numpy(raw).unsqueeze(0).to(self.device)
        return (t - self.state_mean) / self.state_std

    def _to_batch(self, obs_msg) -> dict:
        return {
            "observation.images.left":   self._img_to_tensor(obs_msg.left_image),
            "observation.images.center": self._img_to_tensor(obs_msg.center_image),
            "observation.images.right":  self._img_to_tensor(obs_msg.right_image),
            "observation.state":         self._build_state(obs_msg),
        }

    # ----------------------------------------------------------------
    # Action helpers
    # ----------------------------------------------------------------

    def _smooth_action(self, action_np: np.ndarray) -> np.ndarray:
        if self._prev_action is None:
            self._prev_action = action_np.copy()
            return action_np
        smoothed = _EMA_ALPHA * action_np + (1.0 - _EMA_ALPHA) * self._prev_action
        self._prev_action = smoothed.copy()
        return smoothed

    def _make_motion(self, obs_msg, target_pose: Pose, stiffness, damping,
                     wrench_gains=None) -> MotionUpdate:
        mu = MotionUpdate()
        mu.header = Header(
            frame_id="base_link",
            stamp=self._parent_node.get_clock().now().to_msg(),
        )
        mu.pose = target_pose
        mu.trajectory_generation_mode = TrajectoryGenerationMode(
            mode=TrajectoryGenerationMode.MODE_POSITION,
        )
        mu.target_stiffness = stiffness
        mu.target_damping   = damping
        mu.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        mu.wrench_feedback_gains_at_tip = wrench_gains or [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        return mu

    def _delta_to_pose(self, obs_msg, action_6d: np.ndarray) -> Pose:
        tcp = obs_msg.controller_state.tcp_pose
        cur_pos  = np.array([tcp.position.x, tcp.position.y, tcp.position.z], dtype=np.float64)
        cur_quat = np.array([tcp.orientation.x, tcp.orientation.y,
                              tcp.orientation.z, tcp.orientation.w], dtype=np.float64)

        new_pos = cur_pos + action_6d[:3].astype(np.float64)
        dr = action_6d[3:6].astype(np.float64)
        dq = _axis_angle_to_quat(dr)
        new_quat = _quat_multiply(dq, cur_quat)
        nrm = np.linalg.norm(new_quat)
        if nrm > 1e-9:
            new_quat /= nrm

        return Pose(
            position=Point(x=float(new_pos[0]), y=float(new_pos[1]), z=float(new_pos[2])),
            orientation=Quaternion(
                x=float(new_quat[0]), y=float(new_quat[1]),
                z=float(new_quat[2]), w=float(new_quat[3]),
            ),
        )

    def _get_tcp_state(self, obs_msg):
        """Extract TCP position, quaternion, force/torque norms."""
        tcp = obs_msg.controller_state.tcp_pose
        pos = np.array([tcp.position.x, tcp.position.y, tcp.position.z])
        quat = np.array([tcp.orientation.x, tcp.orientation.y,
                         tcp.orientation.z, tcp.orientation.w])
        wrench = obs_msg.wrist_wrench.wrench
        force_n = math.sqrt(wrench.force.x**2 + wrench.force.y**2 + wrench.force.z**2)
        torque_n = math.sqrt(wrench.torque.x**2 + wrench.torque.y**2 + wrench.torque.z**2)
        return pos, quat, force_n, torque_n

    # ----------------------------------------------------------------
    # Phase 1: ACT approach with closed-loop replanning
    # ----------------------------------------------------------------

    def _run_act_phase(self, get_observation, move_robot, send_feedback, start_time):
        """Run ACT model with 10-step replanning. Returns (obs, pos, quat, recent_actions, step_count)."""
        self.policy.reset()
        self._prev_action = None
        step_count = 0
        start_pos = None
        pos_history = deque(maxlen=_STALL_WINDOW)
        recent_actions = deque(maxlen=30)  # Track recent actions for insertion direction

        self.get_logger().info("=== PHASE 1: ACT approach (closed-loop) ===")

        while time.time() - start_time < _ACT_TIMEOUT_S:
            if time.time() - start_time > _TIME_LIMIT_S - 30:
                # Reserve 30s for insertion phase
                self.get_logger().info("ACT phase: time budget exhausted, switching to insertion")
                break

            t0 = time.time()
            obs_msg = get_observation()
            if obs_msg is None:
                time.sleep(_STEP_S)
                continue

            pos, quat, force_n, torque_n = self._get_tcp_state(obs_msg)

            # Hard stop check
            if force_n > _FORCE_HARD_N or torque_n > _TORQUE_HARD_NM:
                self.get_logger().error(f"HARD STOP — F={force_n:.1f}N T={torque_n:.2f}Nm")
                return None, None, None, None, step_count

            if start_pos is None:
                start_pos = pos.copy()
            pos_history.append(pos.copy())

            # Closed-loop: replan every N steps
            if step_count > 0 and step_count % _REPLAN_EVERY == 0:
                self.policy._action_queue.clear()

            # Model inference
            batch = self._to_batch(obs_msg)
            with torch.inference_mode():
                norm_action = self.policy.select_action(batch)

            raw_action = (norm_action * self.action_std + self.action_mean)
            action_np  = raw_action[0].cpu().numpy().astype(np.float64)

            # Smooth
            action_np = self._smooth_action(action_np)
            recent_actions.append(action_np.copy())

            # Send command
            target_pose = self._delta_to_pose(obs_msg, action_np)
            mu = self._make_motion(obs_msg, target_pose,
                                   _STIFFNESS_APPROACH, _DAMPING_APPROACH)
            move_robot(motion_update=mu)

            # Stall detection
            if len(pos_history) >= _STALL_WINDOW and step_count >= _MIN_ACT_STEPS:
                positions = np.array(list(pos_history))
                per_step_vel = np.linalg.norm(np.diff(positions, axis=0), axis=1).mean()
                if per_step_vel < _STALL_VEL_THRESH:
                    self.get_logger().info(
                        f"Stall detected at step {step_count} "
                        f"(vel={per_step_vel*1000:.3f}mm/step) — switching to insertion"
                    )
                    break

            # Logging
            if step_count < 5 or step_count % 20 == 0:
                dist = np.linalg.norm(pos - start_pos)
                self.get_logger().info(
                    f"ACT step={step_count:4d} | "
                    f"tcp=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}) | "
                    f"Δ=({action_np[0]:+.5f},{action_np[1]:+.5f},{action_np[2]:+.5f},"
                    f"{action_np[3]:+.5f},{action_np[4]:+.5f},{action_np[5]:+.5f}) | "
                    f"F={force_n:.1f}N | travel={dist:.3f}m"
                )

            step_count += 1
            if step_count % 50 == 0:
                elapsed = time.time() - start_time
                send_feedback(
                    f"ACT step={step_count} t={elapsed:.0f}s F={force_n:.1f}N "
                    f"tcp=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})"
                )

            time.sleep(max(0.0, _STEP_S - (time.time() - t0)))

        # Return last observation state for insertion phase
        obs_msg = get_observation()
        if obs_msg is None:
            return None, None, None, recent_actions, step_count
        pos, quat, _, _ = self._get_tcp_state(obs_msg)
        return obs_msg, pos, quat, recent_actions, step_count

    # ----------------------------------------------------------------
    # Phase 2: Force-guided insertion
    # ----------------------------------------------------------------

    def _run_insertion_phase(self, get_observation, move_robot, send_feedback,
                             start_time, init_pos, init_quat, recent_actions):
        """Slowly push along insertion axis with compliant impedance."""
        self.get_logger().info("=== PHASE 2: Force-guided insertion ===")

        # Compute insertion direction from recent ACT actions
        if recent_actions and len(recent_actions) >= 5:
            actions_arr = np.array(list(recent_actions))
            avg_delta_pos = actions_arr[:, :3].mean(axis=0)
            dir_norm = np.linalg.norm(avg_delta_pos)
            if dir_norm > 1e-6:
                insert_dir = avg_delta_pos / dir_norm
            else:
                # Fallback: use gripper Z-axis (local forward)
                R = _quat_to_rotation_matrix(init_quat)
                insert_dir = R[:, 2]  # Z-axis of gripper frame
        else:
            R = _quat_to_rotation_matrix(init_quat)
            insert_dir = R[:, 2]

        self.get_logger().info(
            f"Insertion direction: ({insert_dir[0]:+.3f},{insert_dir[1]:+.3f},{insert_dir[2]:+.3f})"
        )

        # Also compute average rotation delta for gradual roll correction
        avg_rot_delta = np.zeros(3)
        if recent_actions and len(recent_actions) >= 5:
            actions_arr = np.array(list(recent_actions))
            avg_rot_delta = actions_arr[:, 3:6].mean(axis=0)
            # Scale down rotation — apply gently during insertion
            avg_rot_delta *= 0.3

        # CRITICAL: accumulate target position independently of actual TCP
        # The actual TCP may not move immediately due to impedance, but
        # the target must keep advancing to create the driving force.
        target_pos = init_pos.copy()
        cur_quat = init_quat.copy()
        total_push = 0.0
        insert_steps = 0
        contact_detected = False

        while time.time() - start_time < _TIME_LIMIT_S:
            t0 = time.time()

            obs_msg = get_observation()
            if obs_msg is None:
                time.sleep(_STEP_S)
                continue

            pos, quat, force_n, torque_n = self._get_tcp_state(obs_msg)

            # Hard stop
            if force_n > _FORCE_HARD_N or torque_n > _TORQUE_HARD_NM:
                self.get_logger().error(
                    f"INSERT HARD STOP — F={force_n:.1f}N T={torque_n:.2f}Nm"
                )
                break

            # Update orientation from actual (for logging), but DON'T reset target_pos
            cur_quat = quat.copy()

            # Detect contact
            if force_n > _INSERT_FORCE_THRESH and not contact_detected:
                contact_detected = True
                self.get_logger().info(
                    f"Contact detected at F={force_n:.1f}N — continuing insertion"
                )

            # Check if we've pushed enough
            if total_push >= _INSERT_DEPTH_M:
                self.get_logger().info(
                    f"Insertion depth reached: {total_push*1000:.1f}mm"
                )
                break

            # If force is very high (>40N), slow down the step size
            step_size = _INSERT_STEP_M
            if force_n > 40.0:
                step_size = _INSERT_STEP_M * 0.3
            elif force_n > 25.0:
                step_size = _INSERT_STEP_M * 0.6

            # Advance TARGET position along insertion direction (accumulate!)
            target_pos = target_pos + insert_dir * step_size
            new_pos = target_pos

            # Apply gentle rotation correction
            rot_scale = 0.1 if force_n > 15.0 else 0.3
            dr = avg_rot_delta * rot_scale
            dq = _axis_angle_to_quat(dr)
            new_quat = _quat_multiply(dq, cur_quat)
            nrm = np.linalg.norm(new_quat)
            if nrm > 1e-9:
                new_quat /= nrm

            target_pose = Pose(
                position=Point(x=float(new_pos[0]), y=float(new_pos[1]), z=float(new_pos[2])),
                orientation=Quaternion(
                    x=float(new_quat[0]), y=float(new_quat[1]),
                    z=float(new_quat[2]), w=float(new_quat[3]),
                ),
            )

            # Use full stiffness to actually push the cable in
            mu = self._make_motion(
                obs_msg, target_pose,
                _STIFFNESS_INSERT, _DAMPING_INSERT,
                wrench_gains=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            )
            move_robot(motion_update=mu)

            total_push += step_size
            insert_steps += 1

            if insert_steps % 10 == 0 or insert_steps < 3:
                tcp_travel = np.linalg.norm(pos - init_pos)
                self.get_logger().info(
                    f"INSERT step={insert_steps:3d} | "
                    f"tcp=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}) | "
                    f"tgt=({target_pos[0]:+.4f},{target_pos[1]:+.4f},{target_pos[2]:+.4f}) | "
                    f"F={force_n:.1f}N | push={total_push*1000:.1f}mm | moved={tcp_travel*1000:.1f}mm"
                )

            if insert_steps % 20 == 0:
                send_feedback(
                    f"INSERT step={insert_steps} F={force_n:.1f}N push={total_push*1000:.1f}mm"
                )

            time.sleep(max(0.0, _STEP_S - (time.time() - t0)))

        # Hold position for 3 seconds to let connector stabilize
        self.get_logger().info("Holding position for connector stabilization (3s)...")
        hold_start = time.time()
        while time.time() - hold_start < 3.0 and time.time() - start_time < _TIME_LIMIT_S:
            obs_msg = get_observation()
            if obs_msg is not None:
                pos, quat, force_n, _ = self._get_tcp_state(obs_msg)
                # Just re-send current position to hold
                hold_pose = Pose(
                    position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                    orientation=Quaternion(
                        x=float(quat[0]), y=float(quat[1]),
                        z=float(quat[2]), w=float(quat[3]),
                    ),
                )
                mu = self._make_motion(obs_msg, hold_pose,
                                       _STIFFNESS_INSERT, _DAMPING_INSERT)
                move_robot(motion_update=mu)
            time.sleep(_STEP_S)

        return insert_steps

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
        self.get_logger().info(f"RunACT.insert_cable() HYBRID start — task: {task}")
        start = time.time()

<<<<<<< Updated upstream
        TIME_LIMIT_S    = 120.0          # doubled: give model more time for final approach
        STEP_S          = 1.0 / 10.0   # 10 Hz matches training
        FORCE_LIMIT_N   = 80.0         # raised: scoring only penalises >20N for >1s
        TORQUE_LIMIT_NM = 15.0         # raised: allow insertion torques
        EMERGENCY_FORCE = 150.0        # hard abort only at dangerous levels
        step_count      = 0
        force_exceeded_count = 0
        start           = time.time()
=======
        # Phase 1: ACT approach
        obs_msg, pos, quat, recent_actions, act_steps = self._run_act_phase(
            get_observation, move_robot, send_feedback, start
        )
>>>>>>> Stashed changes

        self.get_logger().info(
            f"ACT phase complete: {act_steps} steps in {time.time()-start:.1f}s"
        )

        # Phase 2: Force-guided insertion (if we have valid state)
        insert_steps = 0
        if pos is not None and time.time() - start < _TIME_LIMIT_S - 5:
            insert_steps = self._run_insertion_phase(
                get_observation, move_robot, send_feedback,
                start, pos, quat, recent_actions
            )

<<<<<<< Updated upstream
            # Force/torque safety check — soft pause instead of hard stop
            wrench = obs_msg.wrist_wrench.wrench
            force_norm  = math.sqrt(wrench.force.x**2  + wrench.force.y**2  + wrench.force.z**2)
            torque_norm = math.sqrt(wrench.torque.x**2 + wrench.torque.y**2 + wrench.torque.z**2)

            # Emergency hard stop at extreme force
            if force_norm > EMERGENCY_FORCE:
                self.get_logger().warning(
                    f"RunACT EMERGENCY force — {force_norm:.1f}N — aborting"
                )
                break

            # Soft limit: skip action but keep looping (robot holds position)
            if force_norm > FORCE_LIMIT_N or torque_norm > TORQUE_LIMIT_NM:
                force_exceeded_count += 1
                if force_exceeded_count % 10 == 1:
                    self.get_logger().warning(
                        f"RunACT force soft-limit — force={force_norm:.1f}N torque={torque_norm:.2f}Nm — pausing motion"
                    )
                time.sleep(STEP_S)
                step_count += 1
                continue

            batch = self._to_batch(obs_msg)

            with torch.inference_mode():
                norm_action = self.policy.select_action(batch)   # (1, 6) normalised

            raw_action = (norm_action * self.action_std + self.action_mean)
            action_np  = raw_action[0].cpu().numpy().astype(np.float64)

            if step_count < 5:
                self.get_logger().info(
                    f"step={step_count} norm={norm_action[0].cpu().numpy().round(3).tolist()} "
                    f"raw={action_np.round(4).tolist()}"
                )

            mu = self._delta_to_motion(obs_msg, action_np)
            move_robot(motion_update=mu)

            step_count += 1
            if step_count % 10 == 0:
                send_feedback(f"RunACT step={step_count} elapsed={time.time()-start:.1f}s force={force_norm:.1f}N")

            time.sleep(max(0.0, STEP_S - (time.time() - t0)))

        self.get_logger().info(f"RunACT finished — {step_count} steps")
=======
        elapsed = time.time() - start
        self.get_logger().info(
            f"RunACT HYBRID finished — ACT:{act_steps} + INSERT:{insert_steps} "
            f"steps in {elapsed:.1f}s"
        )
>>>>>>> Stashed changes
        return True


# aic_model resolves the class by the last component of the module path.
run_act = RunACT
