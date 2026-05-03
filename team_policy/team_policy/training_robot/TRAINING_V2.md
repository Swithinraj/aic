# AIC Cable-Insertion Training Pipeline — V2

**Current repo status: Schema v9 episodes, 77D ACT state, synced per-camera YOLO + tared force/torque + plug type + target module encoding + fused YOLO freshness**

---

## Overview

This document describes the **current V2 pipeline as implemented in this repo**
for collecting, validating, converting, training, and deploying the AIC
cable-insertion policy.

This guide is intentionally code-accurate. Older notes may still mention
experimental **57D**, **63D**, **68D**, or **75D** variants, but the default
synced V2 pipeline in the current repo is now **77D**.

## Workflow Order

Read this file in this order:

1. Precheck
2. Exports
3. Collection
4. Validation
5. Conversion
6. Training
7. Deployment

The detailed state-layout and schema reference comes after these workflow steps.

## 1. Precheck

If someone else is running this pipeline on a different machine, these are the
things they must verify first.

### Host paths

By default this guide assumes:

```bash
export AIC_ROOT=/home/$USER/ros2_ws/src/aic
export TRAIN_ROOT=$AIC_ROOT/team_policy/team_policy/training_robot
export SEAGATE=/media/$USER/seagate/aic_episodes
export DATASET_DIR=$TRAIN_ROOT/lerobot_datasets/aic_v3_77d
```

If their workspace or mount point is different, they must change these first.
The helper script defaults are based on the same assumptions:

- `AIC_ROOT_DEFAULT="/home/${USER}/ros2_ws/src/aic"`
- `EPISODES_DIR_DEFAULT="/media/${USER}/seagate/aic_episodes"`

See [aic_collect_v2.sh](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/aic_collect_v2.sh:64).

### Seagate precheck

If episodes must be stored on Seagate, verify the drive is mounted and writable:

```bash
mkdir -p /media/$USER/seagate/aic_episodes
touch /media/$USER/seagate/aic_episodes/.write_test
rm /media/$USER/seagate/aic_episodes/.write_test
ls -ld /media/$USER/seagate/aic_episodes
```

For manual collection, pass the Seagate path explicitly through
`-p output_dir:=...`.

For the helper script, episodes are intended to be written under:

```bash
/media/$USER/seagate/aic_episodes
```

### Software and environment precheck

Run these in a normal host shell before collection:

```bash
pixi --version
tmux -V
distrobox list | grep aic_eval
ls $AIC_ROOT
```

If `tmux` is missing:

```bash
sudo apt install tmux
```

### One-time sudo setup — remove password prompt for distrobox

The helper script runs `distrobox enter -r` which internally calls `sudo podman`. Without this setup it will prompt for a password on every session, blocking automation.

Run once per machine (each user runs this themselves — `$USER` expands to their own username automatically):

```bash
echo "$USER ALL=(ALL) NOPASSWD: /usr/bin/podman" | sudo tee /etc/sudoers.d/podman-nopasswd
sudo chmod 440 /etc/sudoers.d/podman-nopasswd
```

Verify it works — this should print `OK, no password needed` with no prompt:

```bash
distrobox enter -r aic_eval -- echo "OK, no password needed"
```

### Monitoring the script with tmux

The script runs inside a tmux session named `aic_collect_v2`. To attach from any terminal:

```bash
env LD_LIBRARY_PATH="" tmux attach -t aic_collect_v2
```

The `env LD_LIBRARY_PATH=""` is required because the pixi environment sets a library path that conflicts with the system tmux binary. Without it you will see:

```
tmux: .pixi/envs/default/lib/libtinfo.so.6: version NCURSES6_TINFO_6.4.current not found
```

Always open a **fresh terminal window** (not one already inside a pixi shell) before running this attach command.

To detach from tmux without stopping the script:

```
Ctrl+B  then  D
```

### Where to run each command

- Run the helper script `bash aic_collect_v2.sh` on the host, not inside the distrobox.
- Terminal 1 launches the evaluation container through `distrobox enter -r aic_eval -- /entrypoint.sh ...`.
- Terminal 2 is the host shell where `pixi` and `ros2 run aic_model ...` are launched.
- Validation, conversion, training, and deployment commands are also run from the host shell.

### Minimal preflight

```bash
export AIC_ROOT=/home/$USER/ros2_ws/src/aic
export TRAIN_ROOT=$AIC_ROOT/team_policy/team_policy/training_robot
export SEAGATE=/media/$USER/seagate/aic_episodes
export DATASET_DIR=$TRAIN_ROOT/lerobot_datasets/aic_v3_77d

mkdir -p $SEAGATE
touch $SEAGATE/.write_test && rm $SEAGATE/.write_test
pixi --version
tmux -V
distrobox list | grep aic_eval
```

## 2. Exports

For repeat use, add these to `~/.bashrc`:

```bash
export AIC_ROOT=/home/$USER/ros2_ws/src/aic
export TRAIN_ROOT=$AIC_ROOT/team_policy/team_policy/training_robot
export EPISODES_DIR=/media/$USER/seagate/aic_episodes
```

Then reload:

```bash
source ~/.bashrc
```

For the current V2 pipeline, the working session exports are:

```bash
export AIC_ROOT=~/ros2_ws/src/aic
export TRAIN_ROOT=$AIC_ROOT/team_policy/team_policy/training_robot
export SEAGATE=/media/$USER/seagate/aic_episodes
export DATASET_DIR=$TRAIN_ROOT/lerobot_datasets/aic_v3_77d
```

## 3. Collection

### Launch simulation for collection

For collection, Terminal 1 needs an `aic_engine_config_file:=...` session YAML.

- If you use `bash aic_collect_v2.sh`, the helper script launches Terminal 1 and passes the session YAML automatically.
- If you launch the sim manually, you must pass the YAML yourself.

Manual example:

```bash
distrobox enter -r aic_eval -- /entrypoint.sh \
    ground_truth:=true \
    start_aic_engine:=true \
    gazebo_gui:=false \
    aic_engine_config_file:=$TRAIN_ROOT/configs/sessions/session_01.yaml
```

### Generate session YAMLs if they do not exist

Deterministic generation, same files on every machine for the same code/version:

```bash
cd $AIC_ROOT
python3 team_policy/team_policy/training_robot/configs/generate_competition_sessions.py \
    --sessions 50 \
    --out-dir sessions
```

Randomized generation with a reproducible seed:

```bash
cd $AIC_ROOT
python3 team_policy/team_policy/training_robot/configs/generate_competition_sessions.py \
    --sessions 50 \
    --out-dir sessions \
    --randomize \
    --seed 42
```

If you use `--randomize` without `--seed`, the generator uses a fresh
system-random seed and writes the chosen settings to:

```bash
team_policy/team_policy/training_robot/configs/sessions/generation_manifest.txt
```

### Collect schema v9 episodes

Helper script:

```bash
cd $TRAIN_ROOT
bash aic_collect_v2.sh
```

The helper script handles the session YAML automatically. It launches:

```bash
aic_engine_config_file:=${SESSIONS_DIR}/${SESSION}
```

inside the distrobox for each selected session file, so you do not need to pass
the YAML manually in helper-script mode.

Manual collection:

Terminal 1:

```bash
distrobox enter -r aic_eval -- /entrypoint.sh \
    ground_truth:=true \
    start_aic_engine:=true \
    gazebo_gui:=false \
    aic_engine_config_file:=$TRAIN_ROOT/configs/sessions/session_01.yaml
```

Terminal 2:

```bash
export RUN=run_001

cd $AIC_ROOT && pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=team_policy.training_robot.cheatcode_collector_v2 \
    -p output_dir:=$SEAGATE/$RUN \
    -p num_episodes:=3 \
    -p success_only:=true
```

## 4. Validation

Always validate before converting. The validator checks shapes, ranges, timestamps, images, YOLO features, and schema version.

### Full check — all episodes, all fields (recommended)

```bash
cd $AIC_ROOT
pixi run python team_policy/team_policy/training_robot/check_episodes.py
```

For a specific run directory:

```bash
pixi run python team_policy/team_policy/training_robot/check_episodes.py $SEAGATE/run_session_01
```

This prints per-episode:
- Schema version, frame count, duration, success flag
- Port type / plug type / target module
- Final error (mm)
- `insertion_success` dataset — how many frames are 0 vs 1, which frame the event fired
- `insertion_event_data` — the exact port string from `/scoring/insertion_event`
- All schema v9 dataset checks (shapes, ranges, timestamps, images, YOLO features)

### Schema-only validation

```bash
pixi run python -m team_policy.training_robot.validate_episode_v2 \
    $SEAGATE/run_session_*/
```

### Check episode count

```bash
find $SEAGATE -name "episode_*.hdf5" | wc -l
```

## 5. Conversion

```bash
pixi run python -m team_policy.training_robot.convert_to_lerobot_v2 \
    --input  $SEAGATE/run_001 \
    --output $DATASET_DIR \
    --success_only
```

## 6. Training

```bash
pixi run lerobot-train \
    --dataset.repo_id=local/aic_v3_77d \
    --dataset.root=$DATASET_DIR \
    --policy=act \
    --policy.chunk_size=100 \
    --policy.n_action_steps=10 \
    --training.num_workers=4 \
    --training.batch_size=8 \
    --training.num_steps=200000 \
    --output_dir=./outputs/aic_v3_77d_run1
```

## 7. Deployment

Start the combined planner:

```bash
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=team_policy.planner.combined_yolo_depth_pose_planner
```

Then run the trained V2 checkpoint:

```bash
export CHECKPOINT=./outputs/aic_v3_77d_run1/pretrained_model

cd $AIC_ROOT && pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=team_policy.run_act_v2 \
    -p checkpoint_path:=$CHECKPOINT
```

### What changed from V1

| Feature | V1 (`run_act.py` / `convert_to_lerobot.py`) | V2 (`run_act_v2.py` / `convert_to_lerobot_v2.py`) |
|---|---|---|
| TCP pose | 7D | 7D |
| TCP velocity | 6D | 6D |
| TCP error | not in state | 6D |
| Joint positions | 7D | 7D |
| Joint velocity | 7D | 7D |
| Fused YOLO `port_xyz` | 3D | 3D |
| `port_delta_tcp` | not in state | 3D |
| Tared wrist force/torque | not in state | 6D |
| YOLO left camera | not in state | 7D |
| YOLO center camera | not in state | 7D |
| YOLO right camera | not in state | 7D |
| `plug_type_onehot` | not in state | 2D |
| `target_module_onehot` | not in state | 7D |
| fused `yolo_valid` + `yolo_age` | not in state | 2D |
| Total | 30D | 77D |

### Why V2 exists

V1 gave the policy:

- robot proprioception
- 3 camera images
- one fused `port_xyz` hint in `base_link`

V2 adds three missing signal families:

- **Tared force/torque** from `observations/tared_wrist_force_torque`
- **Relative port hint** from `port_delta_tcp = yolo_port_xyz - tcp_position`
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

## Current V2 state layout (77D)

The implemented V2 state is:

```text
[0:7]    tcp_pose
[7:13]   tcp_velocity
[13:19]  tcp_error
[19:26]  joint_positions
[26:33]  joint_velocity
[33:36]  yolo_port_xyz
[36:37]  yolo_valid
[37:38]  yolo_age
[38:41]  port_delta_tcp
[41:47]  tared_wrist_force_torque
[47:54]  yolo_left
[54:61]  yolo_center
[61:68]  yolo_right
[68:70]  plug_type_onehot
[70:77]  target_module_onehot
```

### Detailed layout

| Indices | Dims | Field | Notes |
|---|---:|---|---|
| `0:7` | 7 | `tcp_pose` | `x y z qx qy qz qw` |
| `7:13` | 6 | `tcp_velocity` | `vx vy vz wx wy wz` |
| `13:19` | 6 | `tcp_error` | from `controller_state.tcp_error` |
| `19:26` | 7 | `joint_positions` | first 7 joints |
| `26:33` | 7 | `joint_velocity` | first 7 joint velocities |
| `33:36` | 3 | `yolo_port_xyz` | held fused YOLO target position in `base_link` |
| `36:37` | 1 | `yolo_valid` | fresh target detection flag, not hold-last existence |
| `37:38` | 1 | `yolo_age` | seconds since last valid target detection |
| `38:41` | 3 | `port_delta_tcp` | `yolo_port_xyz - tcp_position`, using held target xyz |
| `41:47` | 6 | `tared_wrist_force_torque` | tare-subtracted `fx fy fz tx ty tz` |
| `47:54` | 7 | `yolo_left` | per-camera YOLO vector |
| `54:61` | 7 | `yolo_center` | per-camera YOLO vector |
| `61:68` | 7 | `yolo_right` | per-camera YOLO vector |
| `68:70` | 2 | `plug_type_onehot` | `[is_sfp, is_sc]` |
| `70:77` | 7 | `target_module_onehot` | exact task target module onehot |

### How the 77D state is used

The important point is that the same `77D` `observation.state` layout is used in
both places:

- during training, after conversion by `convert_to_lerobot_v2.py`
- during deployment, rebuilt online by `run_act_v2.py`

So these dimensions are not just recorded metadata. They are part of the actual
policy input seen by ACT at train time and inference time.

### What each group contributes

- `tcp_pose(7)` gives the policy the current end-effector position and orientation, so image features can be grounded in robot coordinates.
- `tcp_velocity(6)` tells the policy how the robot is already moving, which helps with smooth corrections instead of overreacting frame to frame.
- `tcp_error(6)` gives controller tracking error, which helps the policy distinguish intended motion from lag, overshoot, or under-tracking.
- `joint_positions(7)` gives arm configuration, which matters because the same TCP pose can come from different robot postures.
- `joint_velocity(7)` gives joint motion context, helping the policy reason about current momentum and configuration change.
- `yolo_port_xyz(3)` gives a coarse held 3D target hint in `base_link`, which is useful for approach even when close-up insertion still depends on images.
- `yolo_valid(1)` tells the model whether that fused 3D hint is fresh right now or stale.
- `yolo_age(1)` tells the model how stale the held fused target is, so trust can decay smoothly instead of acting like vision is either fully on or fully off.
- `port_delta_tcp(3)` gives the relative vector from current TCP to held target, which is easier for the policy to use than repeatedly learning that subtraction internally.
- `tared_wrist_force_torque(6)` gives contact-sensitive feedback with bias removed, which is especially useful near contact and insertion.
- `yolo_left(7)`, `yolo_center(7)`, `yolo_right(7)` give target-specific 2D detection evidence in each camera, helping the model use image-space localization even when fused 3D is noisy.
- `plug_type_onehot(2)` tells the policy whether the task is `sfp` or `sc`, so one checkpoint can condition behavior on plug family.
- `target_module_onehot(7)` tells the policy which rail/module target is intended, so the same visual scene can be interpreted with task context.

### Training versus inference

- During training, `convert_to_lerobot_v2.py` packs these features into one `77D` tensor per frame and ACT learns to map `images + state -> action`.
- During inference, `run_act_v2.py` reconstructs the same `77D` tensor from live ROS observations and feeds it into the trained checkpoint.
- If the train-time and run-time layouts disagree, the checkpoint is effectively seeing a different problem than it was trained on, which is why this README keeps emphasizing synchronization.

### Why `tcp_error` is included in V2

Unlike V1, the current V2 converter and deployment runner both include
`tcp_error` in the state:

- [convert_to_lerobot_v2.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/convert_to_lerobot_v2.py:171)
- [run_act_v2.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/run_act_v2.py:164)

So the current V2 checkpoint format is **not** just "30D base + F/T + YOLO"; it is
"30D base + `tcp_error` + held fused `yolo_port_xyz` + fresh `yolo_valid` + `yolo_age` + `port_delta_tcp` + tared F/T + per-camera YOLO + plug type + target module" = **77D**.

### Fused YOLO design

The current fused target design is intentionally:

- `yolo_port_xyz`: held last known target port position
- `yolo_valid`: whether the target detection is fresh right now
- `yolo_age`: how stale that held target position is

This is important because:

- the model still gets a stable held 3D target hint
- the model can tell fresh vision from stale remembered position
- stale fused YOLO can be trusted less when the port is occluded or the scene has changed

### No module fallback in fused target matching

The current V2 path does **not** use `target_module_name` as a fallback label for
fused target-port matching. Module/card detections are intentionally excluded
from the fused YOLO target state so the policy does not learn to align to a
coarse surrounding object instead of the actual socket mouth.

One caveat remains: if upstream detections only provide generic labels such as
`sfp_port` or `sc_port` instead of exact target-port identity, the fused target
selection can still be ambiguous across multiple same-type ports. Removing
module fallback prevents coarse wrong-object supervision, but exact port
disambiguation still depends on the detector/planner publishing port-specific
names.

### Why this matters

These design choices are deliberate:

- module fallback can point the policy at the card/module instead of the socket mouth
- socket insertion needs millimeter-level alignment, so coarse surrounding objects are noisy supervision
- if `yolo_valid` stayed true during hold-last, the policy could not distinguish fresh vision from stale memory
- exposing `yolo_age` lets the policy reduce trust in fused 3D when the target is occluded or outdated

### Target module encoding (7D)

`target_module_onehot` uses this exact deterministic order:

```text
[nic_card_mount_0,
 nic_card_mount_1,
 nic_card_mount_2,
 nic_card_mount_3,
 nic_card_mount_4,
 sc_port_0,
 sc_port_1]
```

Examples:

- `nic_card_mount_0` -> `[1,0,0,0,0,0,0]`
- `nic_card_mount_4` -> `[0,0,0,0,1,0,0]`
- `sc_port_0` -> `[0,0,0,0,0,1,0]`
- `sc_port_1` -> `[0,0,0,0,0,0,1]`
- unknown / missing -> `[0,0,0,0,0,0,0]`

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
[cheatcode_collector_v2.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/cheatcode_collector_v2.py:307).

At deployment time, `run_act_v2.py` now uses the same convention: it rebuilds
the 7D per-camera YOLO features using the live observation image size for each
camera, not a hardcoded resized-video size.

That keeps collection and deployment synchronized. The converter may still
resize RGB videos to `480x640` for LeRobot export, but the YOLO features remain
correct because they are already stored as normalized values.

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

### What action data is recorded

The collector stores the expert command stream that was actually sent during
collection:

- `actions/commanded_pose (T, 7)`: expert TCP pose command sent via `MotionUpdate.pose`
- `actions/delta_pose (T, 6)`: `[dx, dy, dz, drx, dry, drz]` from current TCP pose to commanded pose
- `actions/velocity (T, 6)`: finite-difference target velocity derived from `commanded_pose`

So the training target is the expert robot control command sequence, not the ROS
task-goal message itself.

### What task-goal metadata is recorded

Each episode also stores the ROS task-goal fields that define what the
evaluation container asked the robot to do:

- `cable_name`
- `plug_type`
- `plug_name`
- `port_type`
- `port_name`
- `target_module_name`

In the current V2 pipeline, these are preserved in episode metadata and
converted episode-level metadata. For model input:

- `plug_type` is used directly through `plug_type_onehot`
- `target_module_name` is used directly through `target_module_onehot`
- the other fields remain metadata, and some of them are also used live by the
  runner for target-specific YOLO filtering

---

## File inventory

| File | Role |
|---|---|
| [training_robot/cheatcode_collector_v2.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/cheatcode_collector_v2.py:1) | V2 data collection policy |
| [training_robot/episode_recorder_v2.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/episode_recorder_v2.py:1) | Schema v9 HDF5 recorder |
| [training_robot/validate_episode_v2.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/validate_episode_v2.py:1) | Schema v5-v9 validator with V2 checks |
| [training_robot/convert_to_lerobot_v2.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/convert_to_lerobot_v2.py:1) | Schema v5-v9 HDF5 to LeRobot converter |
| [run_act_v2.py](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/run_act_v2.py:1) | V2 deployment runner for 77D checkpoints with 75D, 68D, and 63D fallback |
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
    -> records images, proprioception, tcp_error, raw wrist_force
    -> records tared_wrist_force_torque, plug_type_onehot, target_module_onehot,
       and fused yolo freshness/staleness
    -> records fused yolo_port_xyz
    -> records raw port_delta_tcp
    -> records per-camera YOLO 7D features
    -> writes Schema v9 HDF5 via EpisodeRecorderV2

validate_episode_v2.py
    -> checks shapes, ranges, timestamps, images, and v6-v9 feature groups

convert_to_lerobot_v2.py
    -> reads Schema v5-v9 HDF5
    -> applies hold-last to fused yolo_port_xyz
    -> preserves fresh fused yolo_valid + yolo_age
    -> recomputes port_delta_tcp from held yolo_port_xyz
    -> builds 77D observation.state
    -> writes LeRobot v3.0 parquet + videos

lerobot-train --policy=act
    -> trains ACT on observation.state + images -> action

run_act_v2.py
    -> rebuilds same 77D state at inference
    -> runs inherited ACT approach + YOLO lateral guard + force-guided insertion
```

---

## HDF5 Schema v9

Schema v9 is written by
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
│   ├── tared_wrist_force_torque     (T, 6) float32
│   ├── timestamps                   (T,) float64
│   ├── task_id                      (T,) string
│   ├── relative_pose                (T, 7) float32
│   ├── relative_pose_valid          (T,) bool
│   ├── yolo_port_xyz                (T, 3) float32
│   ├── yolo_port_valid              (T,) bool      fresh detection flag
│   ├── yolo_port_age                (T,) float32   fused target staleness seconds
│   ├── port_delta_tcp               (T, 3) float32
│   ├── plug_type_onehot             (T, 2) float32
│   ├── target_module_onehot         (T, 7) float32
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
    └── attrs: schema_version="9", episode_id, task_id,
               cable_type, cable_name, plug_type, plug_name,
               port_type, port_name, target_module_name, time_limit_s,
               target_module_onehot_encoding,
               wrist_force_tare,
               success, num_frames, duration_s, max_force, final_error,
               insertion_time, contact_duration, sustained_penalty_duration_s,
               force_baseline_n, yolo_valid_fraction, yolo_fresh_valid_fraction,
               image_height, image_width
```

### Backward compatibility

Schema v9 preserves the legacy fused YOLO keys:

- `observations/yolo_port_xyz`
- `observations/yolo_port_valid`

So the V1 converter and validator can still open these episodes as a fallback.

---

## Quick-start paths

```bash
export AIC_ROOT=~/ros2_ws/src/aic
export TRAIN_ROOT=$AIC_ROOT/team_policy/team_policy/training_robot
export SEAGATE=/media/$USER/seagate/aic_episodes
export DATASET_DIR=$TRAIN_ROOT/lerobot_datasets/aic_v3_77d
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
export RUN=run_001

cd $AIC_ROOT && pixi run ros2 run aic_model aic_model --ros-args \
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

Useful variants:

```bash
cd $TRAIN_ROOT

# all sessions from the default configs directory
bash aic_collect_v2.sh

# all sessions from a different named sessions directory
bash aic_collect_v2.sh --sessions-dir sessions_nic_nic_sc

# one specific session file
bash aic_collect_v2.sh --sessions-dir sessions_nic_nic_sc session_01.yaml

# a numeric range of session files
bash aic_collect_v2.sh --sessions-dir sessions_nic_nic_sc 1 50
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
cd $AIC_ROOT && pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=team_policy.training_robot.cheatcode_collector_v2 \
    -p output_dir:=$SEAGATE/run_sc_001 \
    -p num_episodes:=3
```

### NIC-focused collection example

```bash
cd $AIC_ROOT && pixi run ros2 run aic_model aic_model --ros-args \
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

- accepts schema v5, v6, v7, v8, and v9 episodes together
- filters failed episodes if `--success_only`
- filters by `final_error`
- downsamples to `target_hz`
- applies hold-last to fused `yolo_port_xyz`
- recomputes `port_delta_tcp` from held `yolo_port_xyz`
- prefers `tared_wrist_force_torque`, falling back to raw `wrist_force` for older episodes
- carries `plug_type_onehot`, inferring it from metadata when needed
- carries `target_module_onehot`, inferring it from metadata when needed
- zero-fills per-camera YOLO features for v5 episodes
- builds one 77D `observation.state` tensor
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

- `meta/info.json` with `observation.state.shape = [77]`
- `meta/stats.json` with per-dimension mean/std/min/max

### V5 compatibility behavior

If an input episode is schema v5:

- `tcp_error` and `wrist_force` are still read if present
- `yolo_per_camera` is missing, so V2 zero-fills it as:
  `[0,0,0,0,0,0,10]`

That allows mixed v5-v9 conversion, but the full richer fused-freshness signal
only comes from the newer schema v9 episodes that actually record those fields.

## 5. Train with ACT

### Minimal training command

```bash
pixi run lerobot-train \
    --dataset.repo_id=local/aic_v3_77d \
    --dataset.root=$DATASET_DIR \
    --policy=act \
    --policy.chunk_size=100 \
    --policy.n_action_steps=10 \
    --training.num_workers=4 \
    --training.batch_size=8 \
    --training.num_steps=200000 \
    --output_dir=./outputs/aic_v3_77d_run1
```

### Practical notes

- The dataset feature name is still `observation.state`; V2 packs everything into
  that single state tensor.
- ACT checkpoints trained from this converter should have `state_dim = 77`.
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
export CHECKPOINT=./outputs/aic_v3_77d_run1/pretrained_model

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
- rebuilds `port_delta_tcp` from the held fused YOLO point
- tares the live 6D wrist wrench at episode start
- includes `tcp_error`, held fused `yolo_port_xyz`, fresh `yolo_valid`, `yolo_age`,
  `tared_wrist_force_torque`, all per-camera YOLO features, `plug_type_onehot`,
  and `target_module_onehot` in the 77D state
- still supports older 63D V2 checkpoints for backward compatibility

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

## "RunACTV2 expects a 63D, 68D, 75D, or 77D checkpoint"

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

Schema v6 and v7 expect `observations/images` to carry `height` and `width` attrs.
If they are missing, the episode may be:

- a v5 episode
- or a partially written / older experimental file

## Force readings are all near zero

That usually means the wrist wrench path is not active in the observation stream.
Check the simulation setup and FT sensor plugin path.

## Training works but deployment behaves differently than expected

The first thing to check is whether the deployment runner is receiving the same
kind of per-camera detections and image sizes that were present during
collection. In the synced V2 path, collection and deployment both normalize the
per-camera YOLO features with the live observation image size.

## Final insertion still depends heavily on force phase

That is expected in the current design. `run_act_v2.py` does not replace the
force-guided insertion logic from `run_act.py`; it improves the approach state,
then still hands off to the inherited insertion controller.

---

## Known implementation caveats

These are worth keeping in mind when reading logs or comparing old notes:

1. Some older notes may still refer to experimental `57D`, `63D`, `68D`, or `75D` variants, but the default synced V2 state in this repo is `77D`.
2. The current V2 state includes `tcp_error`, even though some older planning notes described a 57D variant without it.

This README reflects the **actual current code behavior**, not the older intended
57D design.

---

## Recommended workflow summary

If you want the current V2 pipeline exactly as implemented today:

1. Collect schema v9 episodes with `cheatcode_collector_v2.py`
2. Validate them with `validate_episode_v2.py`
3. Convert them with `convert_to_lerobot_v2.py`
4. Train ACT on the resulting `77D` dataset
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
| State dim | `30` | `77` |
| Deploy | `run_act.py` | `run_act_v2.py` |

V1 and V2 checkpoints are not interchangeable.
