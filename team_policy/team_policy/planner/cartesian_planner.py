from dataclasses import dataclass
from typing import List

from geometry_msgs.msg import Pose

from team_policy.planner.search_backend import run_search


@dataclass
class PlannerConfig:
    clearance_z: float = 0.40 #adding z-cearence for moving between two poses
    workspace_min_x: float = -0.70
    workspace_max_x: float = 0.10
    workspace_min_y: float = -0.40
    workspace_max_y: float = 0.70
    workspace_min_z: float = 0.10
    workspace_max_z: float = 0.65
    max_segment_length: float = 0.03


class CartesianPlanner:
    def __init__(self, config: PlannerConfig | None = None):
        self.config = config or PlannerConfig()

    def plan_from_current_pose(self, current_pose: Pose, target_pose: Pose) -> List[Pose]:
        return run_search(
            start_pose=current_pose,
            goal_pose=target_pose,
            clearance_z=self.config.clearance_z,
            workspace_bounds={
                "x": (self.config.workspace_min_x, self.config.workspace_max_x),
                "y": (self.config.workspace_min_y, self.config.workspace_max_y),
                "z": (self.config.workspace_min_z, self.config.workspace_max_z),
            },
            max_segment_length=self.config.max_segment_length,
        )