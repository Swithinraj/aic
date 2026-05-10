# Training Robot — YOLO-Guided Imitation Learning Pipeline

For the current V2 training/collection/deployment pipeline, use
[TRAINING_V2.md](/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/TRAINING_V2.md:1).

That document contains the current:

- schema v9 episode format
- 77D state layout
- Seagate storage setup
- host/distrobox prechecks
- helper script commands
- convert/train/deploy commands

This is the single source of truth for collecting demonstrations, checking
episode quality, converting data, training ACT, and running inference.

The full workflow is:

1. Set up the environment
2. Collect HDF5 episodes with CheatCode + embedded YOLO
3. Inspect and validate the saved episodes
4. Merge all runs into one folder
5. Convert HDF5 to LeRobot format
6. Train an ACT policy
7. Run inference with YOLO + the trained checkpoint

---

## What this pipeline does

CheatCode is the privileged expert used during data collection. It performs
successful insertions in simulation while the recorder saves:

- 3 RGB camera streams
- robot state
- wrist force/torque
- commanded actions
- privileged TF data for analysis
- YOLO-detected port positions

The trained policy does not use ground truth. At inference time it uses only:

- camera images
- robot state
- YOLO port estimates

### Why YOLO is part of the state

Ground-truth TF is available only in simulation. If we trained on hidden TF
data and deployed with YOLO, the policy would see a different input
distribution at test time. Instead, we record and train with YOLO-style port
position so collection and inference match.

### Training state (30D)

```text
tcp_pose(7) + tcp_velocity(6) + joint_positions(7) + joint_velocity(7) + port_xyz_in_base(3) = 30
```

`port_xyz_in_base` follows the same logic in conversion and inference:

- before the first valid YOLO detection: `[0, 0, 0]`
- while YOLO is valid: current YOLO position
- if YOLO drops temporarily: hold the last known valid position

---

## Architecture overview

```text
Terminal 1: Gazebo + aic_engine
  - one session YAML per run
  - 3 trials, then exits

Terminal 2: cheatcode_collector
  - starts embedded YOLO planner
  - runs CheatCode
  - records observations/actions
  - applies quality gates
  - saves episode_*.hdf5
```

At inference time:

```text
Terminal 1: Gazebo + aic_engine (ground_truth:=false)
Terminal 2: YOLO planner
Terminal 3: ACT policy (run_act.py)
```

---

## Terminology

| Term | Meaning |
|------|---------|
| Session | One `aic_engine` launch with one `session_XX.yaml` |
| Run | One output folder such as `run_001` |
| Trial | One insertion attempt inside a session |
| Episode | One saved `.hdf5` file |
| Quality gate | Automatic checks that reject bad episodes |

---

## Prerequisites

Verify these before starting:

```bash
# 1. pixi exists
pixi --version

# 2. distrobox container exists
distrobox list | grep aic_eval

# 3. YOLO weights exist
ls $(git rev-parse --show-toplevel)/.pixi/envs/default/lib/python3.12/site-packages/team_policy/models/yolov12.pt

# 4. Session YAMLs exist
ls $(git rev-parse --show-toplevel)/team_policy/team_policy/training_robot/configs/sessions/ | wc -l
```

---

## Step 0 — Set environment variables

Add these to `~/.bashrc` so every terminal has the same paths:

```bash
export AIC_ROOT=$(git -C ~/ros2_ws/src/aic rev-parse --show-toplevel)
export TRAIN_ROOT=$AIC_ROOT/team_policy/team_policy/training_robot
export EPISODES_DIR="$TRAIN_ROOT/episodes"
export DATASET_ROOT="$TRAIN_ROOT/lerobot_datasets"
export FASTRTPS_DEFAULT_PROFILES_FILE=$AIC_ROOT/team_policy/fastdds_no_shm.xml
```

Reload the shell:

```bash
source ~/.bashrc
```

If you save episodes on an external drive such as Seagate, override only
`EPISODES_DIR` in the terminal where you collect/analyze data:

```bash
export EPISODES_DIR="/media/$USER/seagate/aic_episodes"
```

Create the folder if needed and verify it is writable:

```bash
mkdir -p "$EPISODES_DIR"
touch "$EPISODES_DIR/.write_test" && rm "$EPISODES_DIR/.write_test"
```

Validate that the export points where you expect:

```bash
echo $EPISODES_DIR
ls -ld "$EPISODES_DIR"
```

If you want to keep both local and external options handy:

```bash
export EPISODES_DIR_LOCAL="$TRAIN_ROOT/episodes"
export EPISODES_DIR_SEAGATE="/media/$USER/seagate/aic_episodes"
```

Important:

- Terminal 1 uses a hardcoded absolute path for `aic_engine_config_file`
- Terminals 2 and 3 can safely use `$AIC_ROOT`, `$TRAIN_ROOT`, and `$EPISODES_DIR`

---

## Step 1 — Prepare the environment

### 1.1 Sync code into the pixi environment after edits

The collector and inference code run from the pixi-installed package. After
editing these files, sync them into site-packages:

```bash
SITE=$AIC_ROOT/.pixi/envs/default/lib/python3.12/site-packages/team_policy
cp $TRAIN_ROOT/episode_recorder.py    $SITE/training_robot/episode_recorder.py
cp $TRAIN_ROOT/cheatcode_collector.py $SITE/training_robot/cheatcode_collector.py
cp $TRAIN_ROOT/convert_to_lerobot.py  $SITE/training_robot/convert_to_lerobot.py
cp $AIC_ROOT/team_policy/team_policy/run_act.py $SITE/run_act.py
```

Quick check:

```bash
python3 -c "from team_policy.training_robot.episode_recorder import SCHEMA_VERSION; print('schema:', SCHEMA_VERSION)"
```

Expected:

```text
schema: 5
```

If you want a full reinstall instead of targeted copies:

```bash
cd $AIC_ROOT
pixi reinstall ros-kilted-team-policy
```

### 1.2 Optional: regenerate session YAMLs

If the session files are missing or you want to regenerate them:

```bash
cd $AIC_ROOT
python3 team_policy/team_policy/training_robot/configs/generate_competition_sessions.py --sessions 50
```

### 1.3 Session groups

Use the session folder that matches the dataset you want:

| Path | What it contains | Trials per YAML | Collector setting |
|------|------------------|-----------------|-------------------|
| `configs/sessions/` | SC-only multi-trial sessions | 3 | default `num_episodes=3` |
| `configs/sessions_nic_3trial/` | NIC-only multi-trial sessions | 3 | default `num_episodes=3` |
| `configs/sessions_sc/` | SC-only single-task sessions | 1 | pass `-p num_episodes:=1` |
| `configs/sessions_nic/` | NIC-only single-task sessions | 1 | pass `-p num_episodes:=1` |

The rest of this README uses the current default flow:

- `configs/sessions/`
- 3 trials per run
- SC-only collection

---

## Step 2 — Collect episodes

Open 2 terminals.

### 2.1 Terminal 1 — start simulation

Restart Terminal 1 at the beginning of every run. Use a hardcoded absolute
path for the session YAML:

```bash
distrobox enter -r aic_eval -- /entrypoint.sh \
    ground_truth:=true \
    start_aic_engine:=true \
    gazebo_gui:=false \
    launch_rviz:=false \
    aic_engine_config_file:=/home/swithin/official_aic/aic/team_policy/team_policy/training_robot/configs/sessions/session_010.yaml
```

Wait for this line before starting Terminal 2:

```text
[aic_engine] No node with name 'aic_model' found. Retrying...
```

These startup messages are normal and can be ignored:

```text
[joint_state_broadcaster_spawner]: Waiting for controller_manager ...
[gz_sim]: Unable to create renderer ...
[spawner_joint_state_broadcaster]: Controller spawner couldn't activate controllers ...
```

Why the absolute path matters:

- distrobox receives the path after shell expansion
- if `$TRAIN_ROOT` is empty in that terminal, the path becomes invalid
- `aic_engine` then fails with very little diagnostic output

### 2.2 Terminal 2 — start collector + embedded YOLO

Restart Terminal 2 at the beginning of every run and increment the run folder:

```bash
cd $AIC_ROOT && pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=team_policy.training_robot.cheatcode_collector \
    -p output_dir:=$EPISODES_DIR/run_001
```

Then change:

- `run_001` to `run_002`, `run_003`, ...
- `session_01.yaml` to `session_02.yaml`, `session_03.yaml`, ...

Defaults during collection:

- `num_episodes=3`
- `success_only=true`
- embedded YOLO planner starts automatically

Do not run `combined_yolo_depth_pose_planner` separately during collection.

Expected startup output:

```text
[INFO] Combined planner node started: YOLO + metric depth + CAD registration...
[INFO] Embedded YOLO planner started
[INFO] DataCollectionPolicy ready — target=3 episodes, output=.../run_001, success_only=True
```

Expected saved-episode flow:

```text
[INFO] collector/episode=0 port=sc/sc_port_base
[INFO] [1/3] Saved .../run_001/episode_00000.hdf5
[INFO] collector/episode=1 port=sc/sc_port_base
[INFO] [2/3] Saved .../run_001/episode_00001.hdf5
[INFO] collector/episode=2 port=sc/sc_port_base
[INFO] [3/3] Saved .../run_001/episode_00002.hdf5
```

Rejected trials look like:

```text
collector/episode=N rejected by quality gate — discarded
collector/episode=N FAILED — discarded
collector/episode=N too short — discarded
```

Rejected episodes are not written to disk. The collector proceeds to the next
trial automatically.

### 2.3 Quality gates

Every episode must pass these checks before it is saved:

| Gate | SFP threshold | SC threshold | Rejection reason |
|------|---------------|--------------|------------------|
| Minimum frames | 10 | 10 | crash or immediate timeout |
| Final insertion error | < 3 mm | < 10 mm | plug did not finish at the port |
| Sustained contact force | < 20 N for < 0.5 s | < 20 N for < 0.5 s | robot pushed hard for too long |

Notes:

- force is baseline-subtracted at episode start
- `yolo_valid_fraction` is recorded but does not gate saving
- an episode with low YOLO coverage can still be valid for training

### 2.4 End-of-run checklist

After trial 3:

1. Wait for Terminal 2 to stop printing new collector output
2. Stop Terminal 1 with `Ctrl+C`
3. Verify the saved files
4. Increment the session YAML and run folder
5. Start the next run

Check one run:

```bash
ls -lh $EPISODES_DIR/run_001/
```

Example mapping:

| Session YAML | Output folder |
|--------------|---------------|
| `session_01.yaml` | `run_001` |
| `session_02.yaml` | `run_002` |
| `session_03.yaml` | `run_003` |

### 2.5 Alternative session groups

NIC-only 3-trial sessions:

Terminal 1:

```bash
distrobox enter -r aic_eval -- /entrypoint.sh \
    ground_truth:=true \
    start_aic_engine:=true \
    gazebo_gui:=false \
    launch_rviz:=false \
    aic_engine_config_file:=/home/YOUR_USER/ros2_ws/src/aic/team_policy/team_policy/training_robot/configs/sessions_nic_3trial/session_01.yaml
```

Terminal 2:

```bash
cd $AIC_ROOT && pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=team_policy.training_robot.cheatcode_collector \
    -p output_dir:=$EPISODES_DIR/run_001
```

Single-task SC sessions:

Terminal 1:

```bash
distrobox enter -r aic_eval -- /entrypoint.sh \
    ground_truth:=true \
    start_aic_engine:=true \
    gazebo_gui:=false \
    launch_rviz:=false \
    aic_engine_config_file:=/home/YOUR_USER/ros2_ws/src/aic/team_policy/team_policy/training_robot/configs/sessions_sc/session_001.yaml
```

Terminal 2:

```bash
cd $AIC_ROOT && pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=team_policy.training_robot.cheatcode_collector \
    -p num_episodes:=1 \
    -p output_dir:=$EPISODES_DIR/run_001
```

Single-task NIC sessions work the same way, but use
`configs/sessions_nic/session_001.yaml`.

### 2.6 Visual notes during Gazebo runs

These are normal:

- the free cable end hanging in the air or flinging around during startup
- extra SC-looking background hardware
- a pink or magenta board outline from a visualization panel
- steep board yaw in some sessions

The session YAMLs intentionally vary the board pose to create training
diversity.

---

## Step 3 — Inspect and validate the collected episodes

### 3.1 Count all saved episodes

```bash
find $EPISODES_DIR -name 'episode_*.hdf5' | wc -l
```

If you are validating data from Seagate, first point `EPISODES_DIR` there:

```bash
export EPISODES_DIR="/media/$USER/seagate/aic_episodes"
find $EPISODES_DIR -name 'episode_*.hdf5' | wc -l
```

### 3.2 Quick quality summary across all runs

```bash
cd $AIC_ROOT && pixi run python3 -c "
import h5py, numpy as np, pathlib

base = pathlib.Path('$EPISODES_DIR')
runs = sorted(base.glob('run_*'))
total = 0
for run in runs:
    eps = sorted(run.glob('episode_*.hdf5'))
    if not eps: continue
    print(f'\n=== {run.name} ({len(eps)} eps) ===')
    for ep in eps:
        with h5py.File(ep) as f:
            m = f['metadata']
            port_t = m.attrs.get('port_type','?')
            port   = m.attrs.get('port_name','?')
            frames = f['observations/tcp_pose'].shape[0]
            err    = m.attrs.get('final_error', float('nan'))
            force  = m.attrs.get('max_force', float('nan'))
            sust   = m.attrs.get('sustained_penalty_duration_s', float('nan'))
            yolo   = m.attrs.get('yolo_valid_fraction', float('nan'))
            ok     = '✓' if m.attrs.get('success',0) else '✗'
            print(f'  {ok} {ep.name}  {port_t}/{port:<14} frames={frames:3d}  err={err*1000:4.1f}mm  force={force:.2f}N  sust={sust:.2f}s  yolo={yolo:.0%}')
            total += 1
print(f'\nTotal saved: {total} episodes across {len(runs)} runs')
"
```

Typical good output:

```text
✓ episode_00000.hdf5  sfp/sfp_port_0   frames=447  err= 1.1mm  force=0.38N  sust=0.00s  yolo=100%
✓ episode_00001.hdf5  sfp/sfp_port_0   frames=492  err= 1.1mm  force=0.28N  sust=0.00s  yolo=100%
✓ episode_00002.hdf5  sc/sc_port_base  frames=496  err= 0.6mm  force=1.88N  sust=0.00s  yolo=100%
```

### 3.3 Validate all episodes structurally

Validate every file and stop on the first failure:

```bash
cd $AIC_ROOT
for f in $EPISODES_DIR/run_*/episode_*.hdf5; do
  echo "--- $f ---"
  pixi run python -m team_policy.training_robot.validate_episode --file "$f" || break
done
```

Validate one episode:

```bash
cd $AIC_ROOT
pixi run python -m team_policy.training_robot.validate_episode \
  --file $EPISODES_DIR/run_001/episode_00000.hdf5
```

External drive example:

```bash
export EPISODES_DIR="/media/$USER/seagate/aic_episodes"
cd $AIC_ROOT
pixi run python -m team_policy.training_robot.validate_episode \
  --file $EPISODES_DIR/run_001/episode_00000.hdf5
```

Expected ending:

```text
PASS — episode looks valid (N frames, Xs, success=True)
```

Quality interpretation:

| Quality | Rule |
|---------|------|
| Excellent | `success == 1` and `final_error <= 0.005` m |
| Good | `success == 1` and `final_error <= 0.02` m |
| Skip | `success == 0` or `final_error > 0.02` m |

### 3.4 Analyze YOLO coverage by run and session

```bash
cd $AIC_ROOT
pixi run python -m team_policy.training_robot.analyze_yolo_sessions \
    --episodes-dir "$EPISODES_DIR" \
    --sessions-dir "$AIC_ROOT/team_policy/team_policy/training_robot/configs/sessions" \
    --yolo-threshold 0.98
```

This maps `run_NNN` back to session YAMLs and helps separate:

- startup cold-start YOLO gaps
- intermittent mid-episode YOLO loss
- weak sessions caused by pose or clutter

---

## Step 4 — Merge all runs into one folder

Before conversion, merge every `run_*` folder into one renumbered folder with
symlinks. This avoids duplicating large HDF5 files.

```bash
MERGED="$EPISODES_DIR/all"
rm -rf "$MERGED" && mkdir -p "$MERGED"
N=0
for f in $(ls $EPISODES_DIR/run_*/*.hdf5 | sort); do
    ln -s "$f" "$MERGED/episode_$(printf '%05d' $N).hdf5"
    N=$((N+1))
done
echo "Total episodes merged: $N"
```

If your episodes live on Seagate, this merged folder will also be created on
that drive because it uses `$EPISODES_DIR`.

---

## Step 5 — Convert HDF5 to LeRobot format

```bash
cd $AIC_ROOT
pixi run python -m team_policy.training_robot.convert_to_lerobot \
    --input "$EPISODES_DIR/all" \
    --output "$DATASET_ROOT/aic_act_yolo_v2" \
    --success_only
```

What the converter does:

- builds the 30D state vector
- reproduces inference-time YOLO behavior
- writes LeRobot-format parquet data
- writes encoded camera videos

Dataset naming:

- use `aic_act_yolo_v2` for datasets converted after the `port_xyz` fix
- do not create new training runs from older `aic_act_yolo_v1` data

Expected output layout:

```text
lerobot_datasets/aic_act_yolo_v2/
  meta/
  data/
  videos/
```

---

## Step 6 — Train ACT

### 6.1 One-time lerobot 0.5.1 patch

`lerobot` 0.5.1 ships with a broken GR00T policy file that can crash import
even when training ACT. Apply this patch before the first training run:

```bash
sed -i \
  's/backbone_cfg: dict = field(init=False,/backbone_cfg: dict = field(default=None, init=False,/g;
   s/action_head_cfg: dict = field(init=False,/action_head_cfg: dict = field(default=None, init=False,/g;
   s/action_horizon: int = field(init=False,/action_horizon: int = field(default=None, init=False,/g;
   s/action_dim: int = field(init=False,/action_dim: int = field(default=None, init=False,/g' \
  $AIC_ROOT/.pixi/envs/default/lib/python3.12/site-packages/lerobot/policies/groot/groot_n1.py
```

Verify the patch:

```bash
grep "default=None, init=False" \
  $AIC_ROOT/.pixi/envs/default/lib/python3.12/site-packages/lerobot/policies/groot/groot_n1.py | head -1
```

If pixi reinstalls the environment later, re-apply this patch.

### 6.2 Start training inside tmux

```bash
tmux new -s training
```

Useful tmux shortcuts:

- detach: `Ctrl+B` then `D`
- reattach: `tmux attach -t training`

### 6.3 Run the training command

```bash
cd $AIC_ROOT
pixi run lerobot-train \
    --dataset.repo_id=local/aic_act_yolo_v2 \
    --dataset.root=$DATASET_ROOT/aic_act_yolo_v2 \
    --policy.type=act \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/aic_act_yolo_v2 \
    --job_name=aic_act_yolo_v2 \
    --policy.device=cuda \
    --wandb.enable=false \
    --steps=100000 \
    --save_freq=20000
```

Checkpoints are written to:

```text
outputs/train/aic_act_yolo_v2/checkpoints/XXXXXX/pretrained_model/
```

Guidance:

- for about 100 episodes, loss often plateaus near 80k to 100k steps
- for about 300 episodes, 150k to 200k steps is common
- for CPU-only training, replace `--policy.device=cuda` with `--policy.device=cpu`

Resume training example:

```bash
cd $AIC_ROOT
pixi run lerobot-train \
  --config_path=outputs/train/aic_act_yolo_v2/checkpoints/020000/pretrained_model/train_config.json \
  --resume=true \
  --steps=100000
```

---

## Step 7 — Run inference

Open 3 terminals.

### 7.1 Terminal 1 — simulation without ground truth

```bash
distrobox enter -r aic_eval -- /entrypoint.sh \
    ground_truth:=false \
    start_aic_engine:=true \
    gazebo_gui:=false \
    launch_rviz:=false
```

### 7.2 Terminal 2 — YOLO planner

```bash
cd $AIC_ROOT && pixi run ros2 run team_policy combined_yolo_depth_pose_planner
```

### 7.3 Terminal 3 — ACT policy

```bash
cd $AIC_ROOT && pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=team_policy.run_act \
    -p checkpoint_path:=$AIC_ROOT/outputs/train/aic_act_yolo_v2/checkpoints/100000/pretrained_model
```

If you edited `run_act.py`, sync it before running inference:

```bash
cp $AIC_ROOT/team_policy/team_policy/run_act.py \
   $AIC_ROOT/.pixi/envs/default/lib/python3.12/site-packages/team_policy/run_act.py
```

The deployed policy should now attempt insertion using only:

- cameras
- robot state
- YOLO detections

---

## Folder layout

```text
team_policy/team_policy/training_robot/
  README.md
  configs/
    sessions/
    sessions_nic/
    sessions_nic_3trial/
    sessions_sc/
  episodes/
    run_001/
    run_002/
    all/
  lerobot_datasets/
    aic_act_yolo_v2/
  cheatcode_collector.py
  episode_recorder.py
  convert_to_lerobot.py
  analyze_yolo_sessions.py

outputs/train/
  aic_act_yolo_v2/
    checkpoints/
      020000/pretrained_model/
      040000/pretrained_model/
      ...
```

If you use Seagate, the `episodes/` tree may live under your external mount
instead of inside `training_robot/`.

---

## HDF5 episode structure

Each episode stores:

```text
observations/
  tcp_pose
  tcp_velocity
  joint_positions
  joint_velocity
  wrist_force
  tcp_error
  relative_pose
  yolo_port_xyz
  yolo_port_valid
  privileged_tf/
  images/

actions/
  commanded_pose
  delta_pose
  velocity

metadata/
  schema_version
  port_type
  port_name
  success
  final_error
  max_force
  sustained_penalty_duration_s
  force_baseline_n
  yolo_valid_fraction
```

Important notes:

- `tcp_error` is controller tracking error, not GT port distance
- `relative_pose` is privileged and is kept for analysis/debugging
- `yolo_valid_fraction` is the fraction of frames with real YOLO detections

---

## Troubleshooting

### `aic_engine` never prints `Retrying...`

The session YAML path is wrong. Use a hardcoded absolute path in Terminal 1.

### Collector exits immediately with `Reached target of N episodes`

That run folder already contains episodes. Use a new `run_NNN` folder or remove
the old files first.

### `TF_OLD_DATA` spam appears in Terminal 2

This is usually harmless:

- at sim restart, the sim clock resets and stale TF is flushed
- during active runs, cable TF often lags slightly behind the main sim clock

If saved episodes still show good `yolo_valid_fraction`, YOLO is fine.

### All SC episodes are rejected

Some SC poses are simply harder for CheatCode because of how the SC port frame
is defined. This can be session-dependent and is not unusual.

### Episode count is stuck and files are not appearing

Check:

```bash
echo $EPISODES_DIR
```

If it is empty or wrong, export it again and restart Terminal 2.

### `/fused_yolo/detections_json` looks like it only contains `task_board`

Console output may be truncated. The topic publishes one long JSON list, so a
shortened printout can hide later detections. Check full output or inspect the
saved episode metrics instead.

### Force gate rejects every episode

Check the logged `force_baseline=XX.XN`. A healthy value is usually about
18 to 22 N. If it is 0 N, observations were likely unavailable when the
baseline was measured.

### Gazebo is slow

Close side panels in the GUI or use `gazebo_gui:=false` for maximum speed.

### `checkpoint_path` is not set correctly

`run_act.py` expects the path to the `pretrained_model/` folder, not the parent
checkpoint directory and not `training_state/`.
