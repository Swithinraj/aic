from copy import deepcopy
import math
from typing import Dict, List, Tuple

from geometry_msgs.msg import Pose


Bounds = Dict[str, Tuple[float, float]]


def run_search(
    start_pose: Pose,
    goal_pose: Pose,
    clearance_z: float,
    workspace_bounds: Bounds,
    max_segment_length: float = 0.05,
) -> List[Pose]:
    coarse_waypoints = build_clearance_waypoints(
        start_pose=start_pose,
        goal_pose=goal_pose,
        clearance_z=clearance_z,
        workspace_bounds=workspace_bounds,
    )
    return densify_waypoints(coarse_waypoints, max_segment_length=max_segment_length)


def run_direct_path(
    start_pose: Pose,
    goal_pose: Pose,
    workspace_bounds: Bounds,
    max_segment_length: float = 0.05,
) -> List[Pose]:
    clamped_start = clamp_pose(start_pose, workspace_bounds)
    clamped_goal = clamp_pose(goal_pose, workspace_bounds)
    return densify_waypoints([clamped_start, clamped_goal], max_segment_length)


def build_clearance_waypoints(
    start_pose: Pose,
    goal_pose: Pose,
    clearance_z: float,
    workspace_bounds: Bounds,
) -> List[Pose]:
    clamped_start = clamp_pose(start_pose, workspace_bounds)
    clamped_goal = clamp_pose(goal_pose, workspace_bounds)
    safe_z = clamp_value(
        max(clamped_start.position.z, clamped_goal.position.z, clearance_z),
        *workspace_bounds["z"],
    )

    # Waypoint 1: lift straight up from start
    waypoint_1 = copy_pose(clamped_start)
    waypoint_1.position.z = safe_z

    # Waypoint 2: translate horizontally above goal at safe height
    waypoint_2 = copy_pose(clamped_goal)
    waypoint_2.position.z = safe_z

    # Waypoint 3: descend to goal
    waypoint_3 = copy_pose(clamped_goal)

    return [waypoint_1, waypoint_2, waypoint_3]


def densify_waypoints(waypoints: List[Pose], max_segment_length: float) -> List[Pose]:
    if not waypoints:
        return []

    dense: List[Pose] = [copy_pose(waypoints[0])]
    for idx in range(len(waypoints) - 1):
        start_pose = dense[-1]
        end_pose = waypoints[idx + 1]
        distance = position_distance(start_pose, end_pose)
        if distance <= max_segment_length:
            dense.append(copy_pose(end_pose))
            continue

        num_steps = int(distance / max_segment_length)
        for step in range(1, num_steps + 1):
            alpha = step / (num_steps + 1)
            dense.append(interpolate_pose(start_pose, end_pose, alpha))
        dense.append(copy_pose(end_pose))
    return dense


def quaternion_slerp(q0: Pose, q1: Pose, fraction: float) -> Tuple[float, float, float, float]:
    """Spherical linear interpolation between the orientations of two Poses."""
    v0 = [q0.orientation.x, q0.orientation.y, q0.orientation.z, q0.orientation.w]
    v1 = [q1.orientation.x, q1.orientation.y, q1.orientation.z, q1.orientation.w]

    dot = sum(a * b for a, b in zip(v0, v1))
    # Ensure shortest path by flipping if needed
    if dot < 0.0:
        v1 = [-x for x in v1]
        dot = -dot

    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        res = [a + fraction * (b - a) for a, b in zip(v0, v1)]
        length = sum(x * x for x in res) ** 0.5
        if length < 1e-6:
            return tuple(v0)
        return tuple(x / length for x in res)

    theta_0 = math.acos(min(1.0, dot))
    theta = theta_0 * fraction
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return tuple(s0 * a + s1 * b for a, b in zip(v0, v1))


def clamp_pose(pose: Pose, workspace_bounds: Bounds) -> Pose:
    clamped = copy_pose(pose)
    clamped.position.x = clamp_value(clamped.position.x, *workspace_bounds["x"])
    clamped.position.y = clamp_value(clamped.position.y, *workspace_bounds["y"])
    clamped.position.z = clamp_value(clamped.position.z, *workspace_bounds["z"])
    return clamped


def clamp_value(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def copy_pose(pose: Pose) -> Pose:
    return deepcopy(pose)


def position_distance(pose_a: Pose, pose_b: Pose) -> float:
    dx = pose_a.position.x - pose_b.position.x
    dy = pose_a.position.y - pose_b.position.y
    dz = pose_a.position.z - pose_b.position.z
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def interpolate_pose(start_pose: Pose, end_pose: Pose, alpha: float) -> Pose:
    pose = copy_pose(start_pose)
    pose.position.x = start_pose.position.x + alpha * (end_pose.position.x - start_pose.position.x)
    pose.position.y = start_pose.position.y + alpha * (end_pose.position.y - start_pose.position.y)
    pose.position.z = start_pose.position.z + alpha * (end_pose.position.z - start_pose.position.z)

    qx, qy, qz, qw = quaternion_slerp(start_pose, end_pose, alpha)
    pose.orientation.x = float(qx)
    pose.orientation.y = float(qy)
    pose.orientation.z = float(qz)
    pose.orientation.w = float(qw)

    return pose
