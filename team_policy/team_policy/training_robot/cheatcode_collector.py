"""
Autonomous data collection policy.

Wraps CheatCode so aic_engine orchestrates trials normally.
Every successful trial saves one HDF5 episode with YOLO port detections.

See with_yolo_training.md for the full 3-terminal collection workflow.
Each Gazebo session runs 3 trials automatically then exits.

Quick usage (after entering pixi shell):
    # Terminal 1 — sim with ground truth
    distrobox enter -r aic_eval -- /entrypoint.sh \\
        ground_truth:=true start_aic_engine:=true gazebo_gui:=false

    # Terminal 2 — collector + embedded YOLO planner (change run_NNN each session)
    ros2 run aic_model aic_model --ros-args \\
        -p use_sim_time:=true \\
        -p policy:=team_policy.training_robot.cheatcode_collector \\
        -p output_dir:=$TRAIN_ROOT/episodes/run_001
"""
from __future__ import annotations

import json
import pathlib
import threading
import time

import numpy as np
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

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
from std_msgs.msg import String
from tf2_ros import TransformException

from team_policy.training_robot.episode_recorder import (  # type: ignore[import-unresolved]
    DEFAULT_MAX_FINAL_ERROR_M,
    DEFAULT_MAX_SUSTAINED_FORCE_DURATION_S,
    EpisodeRecorder,
)


class DataCollectionPolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)

        self._output_dir   = str(parent_node.declare_parameter("output_dir",   "/tmp/aic_dataset").value)
        self._num_episodes = int(parent_node.declare_parameter("num_episodes", 3).value)
        self._success_only = bool(parent_node.declare_parameter("success_only", True).value)
        self._start_yolo_planner = bool(parent_node.declare_parameter("start_yolo_planner", True).value)
        # Quality gates: maximum final insertion error per port-name prefix.
        # Evaluated after CheatCode reports success. Set to None to disable a gate.
        self._max_final_error_by_port: dict = DEFAULT_MAX_FINAL_ERROR_M
        self._max_sustained_force_s: float = DEFAULT_MAX_SUSTAINED_FORCE_DURATION_S

        self._recorder    = EpisodeRecorder(self._output_dir)
        self._cheatcode   = CheatCode(parent_node)
        self._embedded_yolo_node = None
        self._embedded_yolo_executor = None
        self._embedded_yolo_thread = None

        # Resume from existing files so relaunching never overwrites saved episodes.
        _existing = sorted(pathlib.Path(self._output_dir).glob("episode_*.hdf5"))
        self._episode_idx = len(_existing)
        self._saved_count = len(_existing)

        # YOLO port detection — updated from /fused_yolo/detections_json
        self._yolo_lock          = threading.Lock()
        self._yolo_port_xyz      = np.zeros(3, dtype=np.float32)
        self._yolo_port_valid    = False
        self._current_port_name  = ""
        self._current_port_type  = ""
        self._current_module_name = ""

        parent_node.create_subscription(
            String, "/fused_yolo/detections_json",
            self._cb_fused_yolo, 10,
        )

        if self._start_yolo_planner:
            self._start_embedded_yolo_planner()

        self.get_logger().info(
            f"DataCollectionPolicy ready — target={self._num_episodes} episodes, "
            f"output={self._output_dir}, success_only={self._success_only}"
        )

    def _start_embedded_yolo_planner(self) -> None:
        from team_policy.planner.combined_yolo_depth_pose_planner import (
            CombinedYoloDepthPosePlanner,
        )

        self._embedded_yolo_node = CombinedYoloDepthPosePlanner()
        self._embedded_yolo_executor = MultiThreadedExecutor(num_threads=2)
        self._embedded_yolo_executor.add_node(self._embedded_yolo_node)

        def spin_yolo() -> None:
            try:
                self._embedded_yolo_executor.spin()
            except ExternalShutdownException:
                pass
            except Exception as exc:
                self.get_logger().error(f"Embedded YOLO planner stopped: {exc}")

        self._embedded_yolo_thread = threading.Thread(
            target=spin_yolo,
            name="embedded_yolo_planner",
            daemon=True,
        )
        self._embedded_yolo_thread.start()
        self.get_logger().info("Embedded YOLO planner started")

    def shutdown(self) -> None:
        if self._embedded_yolo_executor is not None:
            self._embedded_yolo_executor.shutdown()
        if self._embedded_yolo_thread is not None:
            self._embedded_yolo_thread.join(timeout=2.0)
        if self._embedded_yolo_node is not None:
            self._embedded_yolo_node.destroy_node()
        self._embedded_yolo_node = None
        self._embedded_yolo_executor = None
        self._embedded_yolo_thread = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

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

    def _cb_fused_yolo(self, msg: String) -> None:
        """Store the latest YOLO-detected port xyz in base_link for the current task."""
        try:
            dets = json.loads(msg.data)
        except Exception:
            return
        with self._yolo_lock:
            target_port = self._current_port_name
            target_type = self._current_port_type
            target_module = self._current_module_name
        best_rank = None
        best_conf = float("-inf")
        best_xyz = None
        for det in dets:
            rank = self._target_match_rank(det, target_type, target_port, target_module)
            if rank is None:
                continue
            pos = det.get("pose_base_link", {}).get("position", {})
            if not pos:
                continue
            xyz = np.array([
                float(pos.get("x", 0.0)),
                float(pos.get("y", 0.0)),
                float(pos.get("z", 0.0)),
            ], dtype=np.float32)
            conf = float(det.get("confidence", 0.0))
            if best_rank is None or rank < best_rank or (rank == best_rank and conf > best_conf):
                best_rank = rank
                best_conf = conf
                best_xyz = xyz
        if best_xyz is not None:
            with self._yolo_lock:
                self._yolo_port_xyz   = best_xyz
                self._yolo_port_valid = True

    @staticmethod
    def _norm_name(value: object) -> str:
        return str(value).strip().lower()

    def _target_match_rank(
        self,
        det: dict,
        target_type: str,
        target_port: str,
        target_module: str,
    ) -> int | None:
        names = {
            self._norm_name(det.get("instance_name", "")),
            self._norm_name(det.get("class_name", "")),
        }
        names.discard("")
        if not names:
            return None

        exact_aliases = {target_port} if target_port else set()
        if target_type == "sc" and target_module:
            exact_aliases.add(target_module)
        if any(name in exact_aliases for name in names):
            return 0

        if target_type == "sc":
            # SC tasks identify the target by module (`sc_port_0` / `sc_port_1`)
            # while the detector may temporarily publish only the generic family name.
            if any(name == "sc_port" or name.startswith("sc_port_") for name in names):
                return 1

        if target_port and any(target_port in name or name in target_port for name in names):
            return 2
        if target_type == "sc" and target_module and any(target_module in name or name in target_module for name in names):
            return 3
        return None

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
        target_module_name = str(getattr(task, "target_module_name", "unknown"))
        task_id   = str(getattr(task, "id", str(episode_id)))

        send_feedback(f"collector/episode={episode_id} port={port_type}/{port_name}")
        with self._yolo_lock:
            self._current_port_name   = port_name.strip().lower()
            self._current_port_type   = port_type.strip().lower()
            self._current_module_name = target_module_name.strip().lower()
            self._yolo_port_xyz       = np.zeros(3, dtype=np.float32)
            self._yolo_port_valid     = False

        self._recorder.start_episode(episode_id, task_id, port_type, port_name)
        tf_pairs = self._privileged_tf_pairs(task)
        self._recorder.set_privileged_tf_frame_pairs(
            [self._format_tf_pair(target, source) for target, source in tf_pairs]
        )

        # Wrap move_robot to:
        #   1. Capture CheatCode's commanded target pose for the action log.
        #   2. Inject wrench_feedback_gains so the impedance controller provides
        #      passive lateral compliance during insertion.  When the plug contacts
        #      a port wall the controller auto-deflects sideways instead of pushing
        #      harder, which keeps forces below the 20N competition penalty line.
        #      Gains [0.5, 0.5, 0.5, 0, 0, 0] match the official CheatCode reference.
        _WRENCH_COMPLIANCE = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]

        def recording_move_robot(motion_update=None, joint_motion_update=None):
            if motion_update is not None:
                # Inject compliance gains — overrides whatever CheatCode left in the message.
                motion_update.wrench_feedback_gains_at_tip = _WRENCH_COMPLIANCE
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

        # Measure resting force baseline before CheatCode moves anything.
        # The F/T sensor reads ~20N at rest (gripper + cable weight in sim).
        # We subtract this so the quality gate checks actual contact force only.
        force_baseline_n = 0.0
        for _ in range(10):
            obs0 = get_observation()
            if obs0 is not None:
                w = obs0.wrist_wrench.wrench.force
                force_baseline_n = float(
                    np.sqrt(w.x * w.x + w.y * w.y + w.z * w.z)
                )
                break
            time.sleep(0.05)
        self.get_logger().info(
            f"collector/episode={episode_id} force_baseline={force_baseline_n:.1f}N"
        )

        # Background thread records observations at ~10 Hz while CheatCode runs.
        recording_active = threading.Event()
        recording_active.set()

        def obs_loop():
            _step = 0.10  # 10 Hz
            while recording_active.is_set():
                t0 = time.time()
                obs = get_observation()
                if obs is not None:
                    relative_pose = self._relative_pose_plug_to_target(task)
                    privileged_tf, privileged_tf_valid = self._privileged_tf_snapshot(tf_pairs)
                    with self._yolo_lock:
                        yolo_xyz   = self._yolo_port_xyz.copy()
                        yolo_valid = self._yolo_port_valid
                    self._recorder.record_frame(
                        obs,
                        relative_pose=relative_pose,
                        privileged_tf=privileged_tf,
                        privileged_tf_valid=privileged_tf_valid,
                        yolo_port_xyz=yolo_xyz if yolo_valid else None,
                    )
                time.sleep(max(0.0, _step - (time.time() - t0)))

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
            self._recorder.end_episode(success=False, force_baseline_n=force_baseline_n)
            send_feedback(f"collector/episode={episode_id} FAILED — discarded")
        else:
            # Resolve per-port final-error threshold: match on port_name prefix.
            max_err = self._max_final_error_by_port.get("default")
            for key, threshold in self._max_final_error_by_port.items():
                if key != "default" and port_name.startswith(key):
                    max_err = threshold
                    break

            path = self._recorder.end_episode(
                success=success,
                max_final_error_m=max_err,
                max_sustained_force_duration_s=self._max_sustained_force_s,
                force_baseline_n=force_baseline_n,
            )
            if path:
                self._saved_count += 1
                self.get_logger().info(
                    f"[{self._saved_count}/{self._num_episodes}] Saved {path}"
                )
                send_feedback(
                    f"collector/saved episode={episode_id} success={success} "
                    f"total_saved={self._saved_count} path={path}"
                )
            elif len(list(pathlib.Path(self._output_dir).glob("episode_*.hdf5"))) == self._saved_count:
                send_feedback(
                    f"collector/episode={episode_id} rejected by quality gate — discarded "
                    f"(port={port_name} max_err={max_err}m max_force_s={self._max_sustained_force_s}s)"
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
