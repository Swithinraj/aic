from copy import deepcopy
from typing import Dict, List, Tuple

from geometry_msgs.msg import Pose


Bounds = Dict[str, Tuple[float, float]]


def run_search(
    start_pose: Pose,
    goal_pose: Pose,
    clearance_z: float,
    workspace_bounds: Bounds,
) -> List[Pose]:
    return build_clearance_waypoints(
        start_pose=start_pose,
        goal_pose=goal_pose,
        clearance_z=clearance_z,
        workspace_bounds=workspace_bounds,
    )


def build_clearance_waypoints(
    start_pose: Pose,
    goal_pose: Pose,
    clearance_z: float,
    workspace_bounds: Bounds,
) -> List[Pose]:
    clamped_goal = clamp_pose(goal_pose, workspace_bounds)
    safe_z = clamp_value(
        max(start_pose.position.z, clamped_goal.position.z, clearance_z),
        *workspace_bounds["z"],
    )

    waypoint_1 = copy_pose(start_pose)
    waypoint_1.position.z = safe_z

    waypoint_2 = copy_pose(clamped_goal)
    waypoint_2.position.z = safe_z

    waypoint_3 = copy_pose(clamped_goal)

    return [waypoint_1, waypoint_2, waypoint_3]


PSEUDO_ALGORITHM = """
Suggested replacement for run_search(...): Weighted A* in Cartesian space

1. Convert start pose and goal pose into a 3D grid or lattice in base_link.
2. Define neighbor actions, for example:
   - +/- x step
   - +/- y step
   - +/- z step
   - optional diagonal moves
3. Reject states outside workspace bounds.
4. Reject states inside forbidden regions or known obstacle volumes.
5. Cost g(n):
   - translation distance
   - optional penalty for z motion
   - optional penalty near obstacles
6. Heuristic h(n):
   - Euclidean distance to goal
7. Priority:
   - f(n) = g(n) + w * h(n), where w is usually 1.2 to 2.0
8. Stop when goal cell is reached.
9. Reconstruct the path from parent links.
10. Convert path cells back into Pose waypoints.
11. Smooth or sparsify waypoints before execution.
"""


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
