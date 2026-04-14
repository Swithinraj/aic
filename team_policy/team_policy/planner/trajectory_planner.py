import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
# from geometry_msgs.msg import Pose

import rclpy
from rclpy.node import Node

from urdf_parser_py.urdf import URDF

class Pose:
    class Position:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    class Orientation:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
            self.w = 1.0

    def __init__(self):
        self.position = Pose.Position()
        self.orientation = Pose.Orientation()

def pose_to_arrays(pose):
    p = np.array([pose.position.x,
                  pose.position.y,
                  pose.position.z])

    q = np.array([pose.orientation.x,
                  pose.orientation.y,
                  pose.orientation.z,
                  pose.orientation.w])

    return p, q


def arrays_to_pose(p, q):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = p
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
    return pose


def quaternion_angle(q1, q2):
    """Compute angular distance between two quaternions."""
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    return 2 * np.arccos(abs(dot))


def interpolate_poses_task_space(start_pose,
                                     goal_pose,
                                     pos_resolution=0.5,   # meters
                                     rot_resolution=0.5):  # radians
    # Extract
    p_start, q_start = pose_to_arrays(start_pose)
    p_goal,  q_goal  = pose_to_arrays(goal_pose)

    # Normalize quaternions
    q_start = q_start / np.linalg.norm(q_start)
    q_goal  = q_goal  / np.linalg.norm(q_goal)

    # --- Compute required steps ---
    # Position distance
    pos_dist = np.linalg.norm(p_goal - p_start)

    # Orientation distance
    rot_dist = quaternion_angle(q_start, q_goal)

    # Steps needed for each
    n_pos = int(np.ceil(pos_dist / pos_resolution)) if pos_resolution > 0 else 1
    n_rot = int(np.ceil(rot_dist / rot_resolution)) if rot_resolution > 0 else 1

    # Take the maximum to satisfy both constraints
    steps = max(n_pos, n_rot, 1)

    # --- Setup SLERP ---
    key_times = [0, 1]
    key_rots = R.from_quat([q_start, q_goal])
    slerp = Slerp(key_times, key_rots)

    trajectory = []

    for t in np.linspace(0, 1, steps + 1):
        # Interpolate position
        p_t = (1 - t) * p_start + t * p_goal

        # Interpolate orientation
        q_t = slerp([t])[0].as_quat()

        trajectory.append(arrays_to_pose(p_t, q_t))

    return trajectory
