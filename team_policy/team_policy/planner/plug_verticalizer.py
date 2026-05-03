from __future__ import annotations

from dataclasses import dataclass
import math
from geometry_msgs.msg import Pose, Quaternion


@dataclass
class PlugVerticalizerConfig:
    desired_signed_angle_deg: float = 0.0
    angle_tolerance_deg: float = 3.0
    control_gain: float = 0.6
    max_step_deg: float = 8.0
    correction_axis: str = "tool_z"
    ground_axis_in_base: tuple[float, float, float] = (0.0, 0.0, 1.0)
    gripper_axis_for_ground: str = "tool_z"
    max_iterations: int = 10
    settle_time_sec: float = 0.35


PSEUDO_ALGORITHM = """
Upgrade path for true 3D plug-to-ground perpendicularity

1. Detect plug keypoints in at least two wrist cameras.
2. Use CameraInfo intrinsics and TF camera extrinsics.
3. Recover the 3D plug axis by triangulation or PnP.
4. Express the plug axis in base_link.
5. Rotate the gripper until the plug axis aligns with base_link +Z.
6. Keep image-plane verticalization only as the fine correction loop.
"""


class PlugVerticalizer:
    def __init__(self, config: PlugVerticalizerConfig | None = None):
        self.config = config or PlugVerticalizerConfig()

    def is_aligned(self, signed_angle_deg: float) -> bool:
        return abs(self.compute_error_deg(signed_angle_deg)) <= self.config.angle_tolerance_deg

    def compute_error_deg(self, signed_angle_deg: float) -> float:
        return signed_angle_deg - self.config.desired_signed_angle_deg

    def build_correction_target(self, current_pose: Pose, signed_angle_deg: float) -> Pose:
        error_deg = self.compute_error_deg(signed_angle_deg)
        commanded_step_deg = -self.config.control_gain * error_deg
        commanded_step_deg = max(-self.config.max_step_deg, min(self.config.max_step_deg, commanded_step_deg))

        axis = self._axis_vector(self.config.correction_axis)
        q_current = self._quat_to_list(current_pose.orientation)
        q_step_local = self._quat_from_axis_angle(axis, math.radians(commanded_step_deg))
        q_target = self._quat_multiply(q_current, q_step_local)
        q_target = self._quat_normalize(q_target)

        target = Pose()
        target.position.x = current_pose.position.x
        target.position.y = current_pose.position.y
        target.position.z = current_pose.position.z
        target.orientation = Quaternion(x=q_target[0], y=q_target[1], z=q_target[2], w=q_target[3])
        return target

    def compute_gripper_ground_angle_deg(self, current_pose: Pose) -> float:
        q_current = self._quat_to_list(current_pose.orientation)
        gripper_axis_local = self._axis_vector(self.config.gripper_axis_for_ground)
        gripper_axis_base = self._rotate_vector(q_current, gripper_axis_local)
        ground_axis = self.config.ground_axis_in_base
        return self._angle_between_vectors_deg(gripper_axis_base, ground_axis)

    def _axis_vector(self, name: str) -> tuple[float, float, float]:
        if name == "tool_x":
            return (1.0, 0.0, 0.0)
        if name == "tool_y":
            return (0.0, 1.0, 0.0)
        return (0.0, 0.0, 1.0)

    def _quat_to_list(self, q: Quaternion) -> list[float]:
        return [float(q.x), float(q.y), float(q.z), float(q.w)]

    def _quat_from_axis_angle(self, axis: tuple[float, float, float], angle_rad: float) -> list[float]:
        ax, ay, az = axis
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm <= 1e-12:
            return [0.0, 0.0, 0.0, 1.0]
        ax /= norm
        ay /= norm
        az /= norm
        s = math.sin(angle_rad * 0.5)
        c = math.cos(angle_rad * 0.5)
        return [ax * s, ay * s, az * s, c]

    def _quat_multiply(self, q1: list[float], q2: list[float]) -> list[float]:
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]

    def _quat_conjugate(self, q: list[float]) -> list[float]:
        return [-q[0], -q[1], -q[2], q[3]]

    def _rotate_vector(self, q: list[float], v: tuple[float, float, float]) -> tuple[float, float, float]:
        vq = [v[0], v[1], v[2], 0.0]
        qq = self._quat_normalize(q)
        rotated = self._quat_multiply(self._quat_multiply(qq, vq), self._quat_conjugate(qq))
        return (rotated[0], rotated[1], rotated[2])

    def _angle_between_vectors_deg(self, a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        ax, ay, az = a
        bx, by, bz = b
        an = math.sqrt(ax * ax + ay * ay + az * az)
        bn = math.sqrt(bx * bx + by * by + bz * bz)
        if an <= 1e-12 or bn <= 1e-12:
            return 0.0
        dot = (ax * bx + ay * by + az * bz) / (an * bn)
        dot = max(-1.0, min(1.0, dot))
        return math.degrees(math.acos(dot))

    def _quat_normalize(self, q: list[float]) -> list[float]:
        n = math.sqrt(sum(v * v for v in q))
        if n <= 1e-12:
            return [0.0, 0.0, 0.0, 1.0]
        return [v / n for v in q]
