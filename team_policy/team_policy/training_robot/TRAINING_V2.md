# AIC Cable-Insertion Training Pipeline — V2

**Current repo status: Schema v6 episodes, 63D ACT state, per-camera YOLO + force/torque**

---

## Overview

This document describes the **current V2 pipeline as implemented in this repo**
for collecting, validating, converting, training, and deploying the AIC
cable-insertion policy.

This guide is intentionally code-accurate. A few inline comments and log strings
inside the Python files still mention **57D**, but the actual implemented V2
state in the current repo is **63D**.

### What changed from V1

| Feature | V1 (`run_act.py` / `convert_to_lerobot.py`) | V2 (`run_act_v2.py` / `convert_to_lerobot_v2.py`) |
|---|---|---|
| TCP pose | 7D | 7D |
| TCP velocity | 6D | 6D |
| TCP error | not in state | 6D |
| Joint positions | 7D | 7D |
| Joint velocity | 7D | 7D |
| Fused YOLO `port_xyz` | 3D | 3D |
| Wrist force/torque | not in state | 6D |
| YOLO left camera | not in state | 7D |
| YOLO center camera | not in state | 7D |
| YOLO right camera | not in state | 7D |
| Total | 30D | 63D |

### Why V2 exists

V1 gave the policy:

- robot proprioception
- 3 camera images
- one fused `port_xyz` hint in `base_link`

V2 adds two missing signal families:

- **Force/torque** from `observations/wrist_force`
- **Per-camera YOLO 2D features** from the three camera-specific detection topics

That makes the policy less dependent on a single fused depth estimate near the
insertion region, where the fused `z` estimate is often the least reliable.

### Important implementation note about fused `port_xyz`

The fused `port_xyz` comes from `/fused_yolo/detections_json`, published by
[combined_yolo_depth_pose_planner.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/planner/combined_yolo_depth_pose_planner.py:288).
That output is **not a true multi-camera 3D fusion**. The planner collects
observations from all cameras, then chooses **one best observation per instance**
(preferring the configured camera, usually `center`, otherwise highest confidence)
before transforming that pose into `base_link`.

So in practice, the fused `port_xyz` is best thought of as:

- a single selected camera's 3D estimate
- transformed into `base_link`
- held last when detections disappear

It is still useful as a coarse approach hint, but V2 adds per-camera 2D features
so the model is not forced to trust that fused 3D estimate alone.

---

## Current V2 state layout (63D)

The implemented V2 state is:

```text
[0:7]    tcp_pose
[7:13]   tcp_velocity
[13:19]  tcp_error
[19:26]  joint_positions
[26:33]  joint_velocity
[33:36]  port_xyz_fused
[36:42]  wrist_force
[42:49]  yolo_left
[49:56]  yolo_center
[56:63]  yolo_right
```

### Detailed layout

| Indices | Dims | Field | Notes |
|---|---:|---|---|
| `0:7` | 7 | `tcp_pose` | `x y z qx qy qz qw` |
| `7:13` | 6 | `tcp_velocity` | `vx vy vz wx wy wz` |
| `13:19` | 6 | `tcp_error` | from `controller_state.tcp_error` |
| `19:26` | 7 | `joint_positions` | first 7 joints |
| `26:33` | 7 | `joint_velocity` | first 7 joint velocities |
| `33:36` | 3 | `port_xyz_fused` | fused YOLO hold-last in `base_link` |
| `36:42` | 6 | `wrist_force` | `fx fy fz tx ty tz` |
| `42:49` | 7 | `yolo_left` | per-camera YOLO vector |
| `49:56` | 7 | `yolo_center` | per-camera YOLO vector |
| `56:63` | 7 | `yolo_right` | per-camera YOLO vector |

### Why `tcp_error` is included in V2

Unlike V1, the current V2 converter and deployment runner both include
`tcp_error` in the state:

- [convert_to_lerobot_v2.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/convert_to_lerobot_v2.py:171)
- [run_act_v2.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/run_act_v2.py:164)

So the current V2 checkpoint format is **not** "30D base + F/T + YOLO"; it is
"30D base + `tcp_error` + F/T + per-camera YOLO" = **63D**.

---

## Per-camera YOLO feature vector (7D per camera)

Each camera contributes one 7D feature vector:

```text
[confidence, bbox_cx_norm, bbox_cy_norm, bbox_w_norm, bbox_h_norm, valid_float, age_seconds]
```

| Slot | Name | Description |
|---|---|---|
| `0` | `confidence` | YOLO confidence score |
| `1` | `bbox_cx_norm` | bounding-box center x, normalized |
| `2` | `bbox_cy_norm` | bounding-box center y, normalized |
| `3` | `bbox_w_norm` | bounding-box width, normalized |
| `4` | `bbox_h_norm` | bounding-box height, normalized |
| `5` | `valid_float` | `1.0` if detection age `< 0.15s`, else `0.0` |
| `6` | `age_seconds` | seconds since last detection, clamped to `10.0` |

The builder lives in
[episode_recorder_v2.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/episode_recorder_v2.py:108).

### No-detection representation

If there has been no recent detection, the vector becomes:

```text
[0, 0, 0, 0, 0, 0, 10]
```

That means:

- zero confidence
- zero box geometry
- invalid flag
- maxed-out age

### Per-camera YOLO topic source

The collector listens to:

- `/left_camera/yolo/detections_json`
- `/center_camera/yolo/detections_json`
- `/right_camera/yolo/detections_json`

Those come from the planner's inherited multi-camera YOLO detector path, not
from the fused `base_link` output.

### Current normalization behavior

At collection time, bounding boxes are normalized using the recorded image size
from the observation stream, which is usually the original camera resolution
(`1024x1152`) in
[cheatcode_collector_v2.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/cheatcode_collector_v2.py:355).

At deployment time, `run_act_v2.py` currently normalizes live per-camera
bounding boxes using `480x640`, matching the resized training videos but **not**
the collector's original-image normalization:

- [run_act_v2.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/run_act_v2.py:76)
- [run_act_v2.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/run_act_v2.py:144)

This README does not change that behavior, but it is important to know when
interpreting results.

---

## Action representation (6D)

The action is:

```text
[dx, dy, dz, drx, dry, drz]
```

This is the 6D delta pose from **current TCP pose to commanded TCP pose**.

For V2, the converter prefers `actions/delta_pose` from HDF5 and only falls
back to frame-to-frame TCP differencing if that key is missing:

- [convert_to_lerobot_v2.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/convert_to_lerobot_v2.py:288)

---

## File inventory

| File | Role |
|---|---|
| [training_robot/cheatcode_collector_v2.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/cheatcode_collector_v2.py:1) | V2 data collection policy |
| [training_robot/episode_recorder_v2.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/episode_recorder_v2.py:1) | Schema v6 HDF5 recorder |
| [training_robot/validate_episode_v2.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/validate_episode_v2.py:1) | Schema v5/v6 validator with v6 checks |
| [training_robot/convert_to_lerobot_v2.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/convert_to_lerobot_v2.py:1) | Schema v5/v6 HDF5 to LeRobot converter |
| [run_act_v2.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/run_act_v2.py:1) | V2 deployment runner for 63D checkpoints |
| [training_robot/aic_collect_v2.sh](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/aic_collect_v2.sh:1) | Helper collection script |

### Legacy V1 files still present

These are still valid for the older 30D pipeline:

- [training_robot/cheatcode_collector.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/cheatcode_collector.py:1)
- [training_robot/episode_recorder.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/episode_recorder.py:1)
- [training_robot/validate_episode.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/validate_episode.py:1)
- [training_robot/convert_to_lerobot.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/convert_to_lerobot.py:1)
- [run_act.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/run_act.py:1)

---

## End-to-end V2 data flow

```text
Gazebo sim (ground_truth:=true during collection)
    -> joint states, wrist wrench, 3 RGB cameras, 3 stereo depth streams, TF

CombinedYoloDepthPosePlanner
    -> /left_camera/yolo/detections_json
    -> /center_camera/yolo/detections_json
    -> /right_camera/yolo/detections_json
    -> /fused_yolo/detections_json

cheatcode_collector_v2.py
    -> wraps CheatCode expert
    -> records images, proprioception, tcp_error, wrist_force
    -> records fused yolo_port_xyz
    -> records per-camera YOLO 7D features
    -> writes Schema v6 HDF5 via EpisodeRecorderV2

validate_episode_v2.py
    -> checks shapes, ranges, timestamps, images, v6 YOLO groups

convert_to_lerobot_v2.py
    -> reads Schema v5/v6 HDF5
    -> applies hold-last to fused yolo_port_xyz
    -> builds 63D observation.state
    -> writes LeRobot v3.0 parquet + videos

lerobot-train --policy=act
    -> trains ACT on observation.state + images -> action

run_act_v2.py
    -> rebuilds same 63D state at inference
    -> runs inherited ACT approach + YOLO lateral guard + force-guided insertion
```

---

## HDF5 Schema v6

Schema v6 is written by
[EpisodeRecorderV2](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/episode_recorder_v2.py:182).

```text
episode_NNNNN.hdf5
├── observations/
│   ├── images/
│   │   ├── left                     (T, H, W, 3) uint8
│   │   ├── center                   (T, H, W, 3) uint8
│   │   ├── right                    (T, H, W, 3) uint8
│   │   └── attrs: height, width
│   ├── tcp_pose                     (T, 7) float32
│   ├── tcp_velocity                 (T, 6) float32
│   ├── tcp_error                    (T, 6) float32
│   ├── joint_positions              (T, 7) float32
│   ├── joint_velocity               (T, 7) float32
│   ├── wrist_force                  (T, 6) float32
│   ├── timestamps                   (T,) float64
│   ├── task_id                      (T,) string
│   ├── relative_pose                (T, 7) float32
│   ├── relative_pose_valid          (T,) bool
│   ├── yolo_port_xyz                (T, 3) float32
│   ├── yolo_port_valid              (T,) bool
│   ├── privileged_tf/
│   │   ├── transforms               (T, N, 7) float32
│   │   ├── valid                    (T, N) bool
│   │   └── frame_pairs              (N,) string
│   └── yolo_per_camera/
│       ├── left/features            (T, 7) float32
│       ├── center/features          (T, 7) float32
│       └── right/features           (T, 7) float32
├── actions/
│   ├── commanded_pose               (T, 7) float32
│   ├── delta_pose                   (T, 6) float32
│   └── velocity                     (T, 6) float32
└── metadata/
    └── attrs: schema_version="6", episode_id, task_id, port_type, port_name,
               success, num_frames, duration_s, max_force, final_error,
               insertion_time, contact_duration, sustained_penalty_duration_s,
               force_baseline_n, yolo_valid_fraction, image_height, image_width
```

### Backward compatibility

Schema v6 preserves the legacy fused YOLO keys:

- `observations/yolo_port_xyz`
- `observations/yolo_port_valid`

So the V1 converter and validator can still open these episodes as a fallback.

---

## Quick-start paths

```bash
export AIC_ROOT=~/ros2_ws/src/aic
export TRAIN_ROOT=$AIC_ROOT/team_policy/team_policy/training_robot
export SEAGATE=/media/$USER/seagate/aic_episodes
export DATASET_DIR=$TRAIN_ROOT/lerobot_datasets/aic_v2_63d
```

---

## Step-by-step workflow

## 1. Launch simulation for collection

Use ground-truth TF during data collection so CheatCode and privileged labels work.

```bash
distrobox enter -r aic_eval -- /entrypoint.sh \
    ground_truth:=true \
    start_aic_engine:=true \
    gazebo_gui:=false
```

## 2. Collect episodes

### Minimal manual collection

```bash
cd $AIC_ROOT
pixi shell

export RUN=run_001

ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=team_policy.training_robot.cheatcode_collector_v2 \
    -p output_dir:=$SEAGATE/$RUN \
    -p num_episodes:=3 \
    -p success_only:=true
```

### Helper script

You can also use the helper script:

```bash
cd $TRAIN_ROOT
bash aic_collect_v2.sh
```

### What the collector currently does

The V2 collector:

- starts the embedded combined YOLO planner by default
- subscribes to `/fused_yolo/detections_json`
- subscribes to all three per-camera YOLO JSON topics
- records a background observation loop at about `10 Hz`
- measures a resting force baseline once at episode start
- injects impedance compliance gains `[0.5, 0.5, 0.5, 0.0, 0.0, 0.0]`
- stores per-camera 7D YOLO features into `observations/yolo_per_camera`
- stores fused held YOLO `port_xyz` for compatibility
- applies final-error and sustained-force quality gates before saving

### Example run directories

```bash
$SEAGATE/
    run_001/
    run_sc_001/
    run_sc_002/
    run_nic_001/
```

### SC-focused collection example

```bash
ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=team_policy.training_robot.cheatcode_collector_v2 \
    -p output_dir:=$SEAGATE/run_sc_001 \
    -p num_episodes:=3
```

### NIC-focused collection example

```bash
ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=team_policy.training_robot.cheatcode_collector_v2 \
    -p output_dir:=$SEAGATE/run_nic_001 \
    -p num_episodes:=3
```

## 3. Validate episodes

Validate every run before conversion.

### Single file

```bash
pixi run python -m team_policy.training_robot.validate_episode_v2 \
    $SEAGATE/run_001/episode_00000.hdf5
```

### Whole run directory

```bash
pixi run python -m team_policy.training_robot.validate_episode_v2 \
    $SEAGATE/run_001/
```

### Multiple run directories

```bash
pixi run python -m team_policy.training_robot.validate_episode_v2 \
    $SEAGATE/run_*/
```

### What `validate_episode_v2.py` checks

- schema version is compatible
- required observation datasets exist
- image datasets exist and match episode length
- quaternion norms are reasonable
- timestamps are monotonic
- `wrist_force` is present and not obviously broken
- fused YOLO datasets exist when expected
- `yolo_per_camera/{left,center,right}/features` exists for schema v6
- confidence, normalized bbox values, validity, and age are within expected ranges

## 4. Convert to LeRobot format

### Convert one run

```bash
pixi run python -m team_policy.training_robot.convert_to_lerobot_v2 \
    --input  $SEAGATE/run_001 \
    --output $DATASET_DIR \
    --success_only \
    --max_final_error 0.02 \
    --image_height 480 \
    --image_width 640 \
    --target_hz 10.0
```

### Merge multiple runs first, then convert

```bash
export MERGED=$SEAGATE/merged_v2
mkdir -p $MERGED
cp $SEAGATE/run_*/episode_*.hdf5 $MERGED/
```

To renumber the merged files sequentially:

```bash
cd $MERGED
ls episode_*.hdf5 | sort | awk 'BEGIN{n=0}{printf "mv %s episode_%05d.hdf5\n", $0, n++}' | bash
```

Then convert:

```bash
pixi run python -m team_policy.training_robot.convert_to_lerobot_v2 \
    --input  $MERGED \
    --output $DATASET_DIR \
    --success_only
```

### What the converter currently does

The V2 converter:

- accepts schema v5 and v6 episodes together
- filters failed episodes if `--success_only`
- filters by `final_error`
- downsamples to `target_hz`
- applies hold-last to fused `yolo_port_xyz`
- zero-fills per-camera YOLO features for v5 episodes
- builds one 63D `observation.state` tensor
- writes LeRobot v3.0 parquet and concatenated MP4s

### LeRobot output structure

```text
$DATASET_DIR/
├── data/
│   └── chunk-000/
│       └── file-000.parquet
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet
└── videos/
    ├── observation.images.left/
    ├── observation.images.center/
    └── observation.images.right/
```

### `info.json` / `stats.json`

The converter writes:

- `meta/info.json` with `observation.state.shape = [63]`
- `meta/stats.json` with per-dimension mean/std/min/max

### V5 compatibility behavior

If an input episode is schema v5:

- `tcp_error` and `wrist_force` are still read if present
- `yolo_per_camera` is missing, so V2 zero-fills it as:
  `[0,0,0,0,0,0,10]`

That allows mixed v5/v6 conversion, but the richer per-camera signal will only
come from v6 episodes.

## 5. Train with ACT

### Minimal training command

```bash
pixi run lerobot-train \
    --dataset.repo_id=local/aic_v2_63d \
    --dataset.root=$DATASET_DIR \
    --policy=act \
    --policy.chunk_size=100 \
    --policy.n_action_steps=10 \
    --training.num_workers=4 \
    --training.batch_size=8 \
    --training.num_steps=200000 \
    --output_dir=./outputs/aic_v2_63d_run1
```

### Practical notes

- The dataset feature name is still `observation.state`; V2 packs everything into
  that single state tensor.
- ACT checkpoints trained from this converter should have `state_dim = 63`.
- Use `run_act_v2.py` to deploy those checkpoints.

### Recommended starting hyperparameters

| Parameter | Suggested value | Notes |
|---|---:|---|
| `chunk_size` | 100 | matches the current ACT runner style |
| `n_action_steps` | 10 | replan every 10 actions |
| `batch_size` | 8 | raise if GPU memory allows |
| `num_steps` | 200000 | starting point |
| `num_workers` | 4 | adjust to CPU / storage |

## 6. Deploy the trained V2 checkpoint

### Terminal 1: simulation for deployment

```bash
distrobox enter -r aic_eval -- /entrypoint.sh \
    ground_truth:=false \
    start_aic_engine:=true \
    gazebo_gui:=true
```

### Terminal 2: run the combined planner

V2 deployment needs:

- fused YOLO topic
- per-camera YOLO topics

So the planner must be running:

```bash
cd $AIC_ROOT
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=team_policy.planner.combined_yolo_depth_pose_planner
```

### Terminal 3: run the V2 ACT policy

```bash
export CHECKPOINT=./outputs/aic_v2_63d_run1/pretrained_model

ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=team_policy.run_act_v2 \
    -p checkpoint_path:=$CHECKPOINT
```

### What `run_act_v2.py` currently does

`run_act_v2.py` subclasses the V1 runner and only overrides the state-building
part. It still inherits the motion logic from [run_act.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/run_act.py:1):

1. ACT approach phase
2. YOLO lateral fine-alignment guard
3. force-guided insertion phase

It additionally:

- subscribes to the three per-camera YOLO JSON topics
- rebuilds the 7D per-camera features online
- includes `tcp_error`, `wrist_force`, and all per-camera YOLO features in the state
- raises an error if the checkpoint state dimension is not `63`

---

## ROS topics used by V2

| Topic | Used by | Purpose |
|---|---|---|
| `/fused_yolo/detections_json` | collector + deployment | fused `port_xyz` hint |
| `/left_camera/yolo/detections_json` | collector + deployment | left-camera YOLO feature vector |
| `/center_camera/yolo/detections_json` | collector + deployment | center-camera YOLO feature vector |
| `/right_camera/yolo/detections_json` | collector + deployment | right-camera YOLO feature vector |
| `Observation.left_image` | collector + deployment | left image |
| `Observation.center_image` | collector + deployment | center image |
| `Observation.right_image` | collector + deployment | right image |
| `Observation.wrist_wrench` | collector + deployment | force/torque |

### Per-camera detection JSON format

The camera-specific JSON comes from the detector path and looks like:

```json
[
  {
    "class_id": 0,
    "class_name": "sc_port_0",
    "confidence": 0.87,
    "bbox_xyxy": [412, 308, 598, 512]
  }
]
```

Those coordinates are in the image pixel space published by the detector topic.

### Fused detection JSON format

The fused JSON additionally contains `pose_base_link` and metadata such as:

- `instance_name`
- `confidence`
- `bbox_xyxy`
- `anchor_uv`
- `camera_name`
- `pose_source`
- `pose_base_link.position`
- `pose_base_link.orientation`

---

## Troubleshooting

## "RunACTV2 expects a 63D checkpoint"

You are trying to deploy:

- a V1 30D checkpoint, or
- an older experimental checkpoint with a different state layout

Use:

- `run_act.py` for 30D checkpoints
- `run_act_v2.py` only for checkpoints trained from `convert_to_lerobot_v2.py`

## Per-camera YOLO vectors are always `[0,0,0,0,0,0,10]`

Check whether the planner is publishing the per-camera topics:

```bash
ros2 topic hz /left_camera/yolo/detections_json
ros2 topic hz /center_camera/yolo/detections_json
ros2 topic hz /right_camera/yolo/detections_json
```

Also confirm that the task target naming matches what YOLO is publishing, since
the collector and runner both filter detections by target port type/name.

## `validate_episode_v2.py` warns about missing image attrs

Schema v6 expects `observations/images` to carry `height` and `width` attrs.
If they are missing, the episode may be:

- a v5 episode
- or a partially written / older experimental file

## Force readings are all near zero

That usually means the wrist wrench path is not active in the observation stream.
Check the simulation setup and FT sensor plugin path.

## Training works but deployment behaves differently than expected

Be aware of the current normalization mismatch:

- collection-time per-camera YOLO normalization uses the raw observation image size
- deployment-time `run_act_v2.py` normalization uses `480x640`

That is part of the current implementation and can affect generalization.

## Final insertion still depends heavily on force phase

That is expected in the current design. `run_act_v2.py` does not replace the
force-guided insertion logic from `run_act.py`; it improves the approach state,
then still hands off to the inherited insertion controller.

---

## Known implementation caveats

These are worth keeping in mind when reading logs or comparing old notes:

1. Several comments and print strings still say `57D`, but the actual V2 state is `63D`.
2. `convert_to_lerobot_v2.py` docstrings and some messages still mix `57D` and `63D`.
3. `run_act_v2.py` uses a schema constant named `_SCHEMA_V6_57D`, but it enforces `state_dim == 63`.
4. The current V2 state includes `tcp_error`, even though some older planning notes described a 57D variant without it.
5. Deployment-time bbox normalization does not exactly match collection-time normalization.

This README reflects the **actual current code behavior**, not the older intended
57D design.

---

## Recommended workflow summary

If you want the current V2 pipeline exactly as implemented today:

1. Collect schema v6 episodes with `cheatcode_collector_v2.py`
2. Validate them with `validate_episode_v2.py`
3. Convert them with `convert_to_lerobot_v2.py`
4. Train ACT on the resulting `63D` dataset
5. Deploy with `run_act_v2.py` while running `combined_yolo_depth_pose_planner.py`

If you want the older stable baseline instead:

1. Collect with `cheatcode_collector.py`
2. Convert with `convert_to_lerobot.py`
3. Train 30D ACT
4. Deploy with `run_act.py`

---

## V1 to V2 migration summary

| Step | V1 | V2 |
|---|---|---|
| Collect | `cheatcode_collector.py` | `cheatcode_collector_v2.py` |
| Record schema | `episode_recorder.py` (`v5`) | `episode_recorder_v2.py` (`v6`) |
| Validate | `validate_episode.py` | `validate_episode_v2.py` |
| Convert | `convert_to_lerobot.py` | `convert_to_lerobot_v2.py` |
| State dim | `30` | `63` |
| Deploy | `run_act.py` | `run_act_v2.py` |

V1 and V2 checkpoints are not interchangeable.
