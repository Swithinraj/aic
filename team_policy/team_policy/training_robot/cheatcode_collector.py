"""
Autonomous data collection policy.

Run this as aic_model — it wraps CheatCode so the aic_engine orchestrates
trials normally.  For every successful trial it saves an HDF5 episode.

Usage (Terminal 1 — eval container WITH ground truth):
    distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=true start_aic_engine:=true

Usage (Terminal 2 — run collector):
    cd ~/ros2_ws/src/aic
    pixi reinstall ros-kilted-team-policy
    pixi run ros2 run aic_model aic_model --ros-args \\
        -p use_sim_time:=true \\
        -p policy:=team_policy.training_robot.cheatcode_collector \\
        -p output_dir:=/tmp/aic_dataset \\
        -p num_episodes:=50 \\
        -p success_only:=true
"""
from __future__ import annotations

import pathlib
import threading
import time

import numpy as np

_DEFAULT_DATASET_DIR = str(pathlib.Path(__file__).parent / "dataset")
from aic_example_policies.ros.CheatCode import CheatCode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from rclpy.time import Time
from tf2_ros import TransformException

from team_policy.training_robot.episode_recorder import EpisodeRecorder  # type: ignore[import-unresolved]


class DataCollectionPolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)

        self._output_dir   = str(parent_node.declare_parameter("output_dir",   "/tmp/aic_dataset").value)
        self._num_episodes = int(parent_node.declare_parameter("num_episodes", 50).value)
        self._success_only = bool(parent_node.declare_parameter("success_only", True).value)

        self._recorder    = EpisodeRecorder(self._output_dir)
        self._cheatcode   = CheatCode(parent_node)
        self._episode_idx = 0
        self._saved_count = 0

        self.get_logger().info(
            f"DataCollectionPolicy ready — target={self._num_episodes} episodes, "
            f"output={self._output_dir}, success_only={self._success_only}"
        )

    @staticmethod
    def _task_frames(task: Task) -> tuple[str, str, str]:
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        plug_frame = f"{task.cable_name}/{task.plug_name}_link"
        task_board_frame = "task_board/task_board_base_link"
        return port_frame, plug_frame, task_board_frame

    @staticmethod
    def _format_tf_pair(target_frame: str, source_frame: str) -> str:
        return f"{target_frame}<-{source_frame}"

    def _lookup_pose_array(self, target_frame: str, source_frame: str) -> np.ndarray | None:
        """Return source-frame pose in target-frame coordinates as [x,y,z,qx,qy,qz,qw]."""
        try:
            tf_msg = self._parent_node._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
            ).transform
        except TransformException:
            return None

        return np.array(
            [
                tf_msg.translation.x,
                tf_msg.translation.y,
                tf_msg.translation.z,
                tf_msg.rotation.x,
                tf_msg.rotation.y,
                tf_msg.rotation.z,
                tf_msg.rotation.w,
            ],
            dtype=np.float32,
        )

    def _relative_pose_plug_to_target(self, task: Task) -> np.ndarray | None:
        """Return target port pose in the plug-tip frame as [x,y,z,qx,qy,qz,qw]."""
        port_frame, plug_frame, _ = self._task_frames(task)
        return self._lookup_pose_array(plug_frame, port_frame)

    def _privileged_tf_pairs(self, task: Task) -> list[tuple[str, str]]:
        """Selected TFs saved for offline debugging/training analysis."""
        port_frame, plug_frame, task_board_frame = self._task_frames(task)
        return [
            ("base_link", task_board_frame),
            ("base_link", port_frame),
            ("base_link", plug_frame),
            ("base_link", "gripper/tcp"),
            (plug_frame, port_frame),
        ]

    def _privileged_tf_snapshot(
        self,
        frame_pairs: list[tuple[str, str]],
    ) -> tuple[np.ndarray, np.ndarray]:
        transforms = np.zeros((len(frame_pairs), 7), dtype=np.float32)
        valid = np.zeros(len(frame_pairs), dtype=np.bool_)
        for idx, (target_frame, source_frame) in enumerate(frame_pairs):
            pose = self._lookup_pose_array(target_frame, source_frame)
            if pose is None:
                continue
            transforms[idx] = pose
            valid[idx] = True
        return transforms, valid

    # ------------------------------------------------------------------
    # Policy entry point — called once per trial by aic_engine
    # ------------------------------------------------------------------

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:

        if self._episode_idx >= self._num_episodes:
            self.get_logger().info(
                f"Reached target of {self._num_episodes} episodes. "
                "Returning True to let aic_engine finish gracefully."
            )
            return True

        episode_id = self._episode_idx
        self._episode_idx += 1

        port_type = str(getattr(task, "port_type", "unknown"))
        port_name = str(getattr(task, "port_name", "unknown"))
        task_id   = str(getattr(task, "id", str(episode_id)))

        send_feedback(f"collector/episode={episode_id} port={port_type}/{port_name}")
        self._recorder.start_episode(episode_id, task_id, port_type, port_name)
        tf_pairs = self._privileged_tf_pairs(task)
        self._recorder.set_privileged_tf_frame_pairs(
            [self._format_tf_pair(target, source) for target, source in tf_pairs]
        )

        # Wrap move_robot to capture CheatCode's commanded target pose.
        # CheatCode uses set_pose_target() which puts the goal in motion_update.pose
        # (MODE_POSITION).  We read that field directly — NOT the current TCP pose.
        def recording_move_robot(motion_update=None, joint_motion_update=None):
            if motion_update is not None:
                p = getattr(motion_update, "pose", None)
                if p is not None:
                    pos = p.position
                    ori = p.orientation
                    commanded = np.array(
                        [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w],
                        dtype=np.float32,
                    )
                    self._recorder.update_commanded_pose(commanded)
            if motion_update is not None:
                move_robot(motion_update=motion_update)
            elif joint_motion_update is not None:
                move_robot(joint_motion_update=joint_motion_update)

        # Background thread records observations at ~10 Hz while CheatCode runs.
        recording_active = threading.Event()
        recording_active.set()

        def obs_loop():
            while recording_active.is_set():
                obs = get_observation()
                if obs is not None:
                    relative_pose = self._relative_pose_plug_to_target(task)
                    privileged_tf, privileged_tf_valid = self._privileged_tf_snapshot(tf_pairs)
                    self._recorder.record_frame(
                        obs,
                        relative_pose=relative_pose,
                        privileged_tf=privileged_tf,
                        privileged_tf_valid=privileged_tf_valid,
                    )
                time.sleep(0.10)

        obs_thread = threading.Thread(target=obs_loop, daemon=True)
        obs_thread.start()

        try:
            success = self._cheatcode.insert_cable(
                task=task,
                get_observation=get_observation,
                move_robot=recording_move_robot,
                send_feedback=send_feedback,
            )
        except Exception as exc:
            self.get_logger().error(f"CheatCode raised: {exc}")
            success = False
        finally:
            recording_active.clear()
            obs_thread.join(timeout=2.0)

        if self._success_only and not success:
            self._recorder.end_episode(success=False)
            send_feedback(f"collector/episode={episode_id} FAILED — discarded")
        else:
            path = self._recorder.end_episode(success=success)
            if path:
                self._saved_count += 1
                self.get_logger().info(
                    f"[{self._saved_count}/{self._num_episodes}] Saved {path}"
                )
                send_feedback(
                    f"collector/saved episode={episode_id} success={success} "
                    f"total_saved={self._saved_count} path={path}"
                )
            else:
                send_feedback(f"collector/episode={episode_id} too short — discarded")

        return success


# aic_model expects the class name to match the last part of the module path.
# Usage: -p policy:=team_policy.training_robot.cheatcode_collector
cheatcode_collector = DataCollectionPolicy


def main():
    raise SystemExit(
        "DataCollectionPolicy is not a standalone node.\n"
        "Run it via aic_model:\n"
        "  pixi run ros2 run aic_model aic_model --ros-args \\\n"
        "    -p use_sim_time:=true \\\n"
        "    -p policy:=team_policy.training_robot.cheatcode_collector"
    )
