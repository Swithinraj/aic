# CheatCode Training Data Pipeline

This folder contains the full imitation-learning data pipeline:
collect expert demonstrations → validate → convert → train an ACT policy.

```
aic_engine runs N trials
  → aic_model loads cheatcode_collector
    → CheatCode (ground-truth oracle) executes each trial
      → EpisodeRecorder saves observations + expert actions → episode_XXXXX.hdf5
        → validate_episode.py checks every file
          → convert_to_lerobot.py converts HDF5 → parquet + MP4 videos
            → lerobot-train trains an ACT neural network policy
```

> **Important:** `CheatCode` requires `ground_truth:=true`.
> Ground truth is only allowed for training data collection.
> The final learned policy must rely only on sensors and robot state — no hidden TF frames.

---

## Files

| File | Purpose |
|---|---|
| `cheatcode_collector.py` | Policy wrapper used by `aic_model` to collect data |
| `episode_recorder.py` | Buffers and saves one episode as HDF5 (schema v4) |
| `validate_episode.py` | Validates one HDF5 episode |
| `convert_to_lerobot.py` | Converts HDF5 episodes to LeRobot parquet + MP4 videos |
| `configs/orientation_sweep_50_trials.yaml` | **50-trial engine config** (40 SFP + 10 SC, 10 varied poses) |
| `configs/test_5_trials.yaml` | **5-trial smoke test config** — use this first to verify the pipeline |
| `configs/orientation_sweep_3_trials.yaml` | Original 3-trial config (legacy, kept for reference) |
| `configs/generate_50_trials.py` | Script that regenerates the 50-trial YAML |

Generated data is git-ignored:

```
training_robot/episodes/
training_robot/lerobot_datasets/
training_robot/dataset/
```

---

## What Gets Recorded (HDF5 Schema v4)

Each episode saves at 20 Hz:

| Field | Shape | Description |
|---|---|---|
| `observations/images/{left,center,right}` | `(T, H, W, 3)` | Camera images, uint8 RGB, gzip-compressed |
| `observations/tcp_pose` | `(T, 7)` | Tool position + quaternion in base frame |
| `observations/tcp_velocity` | `(T, 6)` | Cartesian tool velocity (linear + angular) |
| `observations/tcp_error` | `(T, 6)` | Pose error to current target |
| `observations/joint_positions` | `(T, 7)` | Joint angles (rad) |
| `observations/joint_velocity` | `(T, 7)` | Per-joint velocity (rad/s) — **added in v4** |
| `observations/wrist_force` | `(T, 6)` | F/T sensor readings |
| `observations/relative_pose` | `(T, 7)` | Target port pose in plug-tip frame |
| `observations/privileged_tf/transforms` | `(T, N, 7)` | Selected TF snapshots (debug/analysis only) |
| `actions/commanded_pose` | `(T, 7)` | Absolute TCP target commanded by CheatCode |
| `actions/delta_pose` | `(T, 6)` | Position delta + axis-angle rotation delta |
| `actions/velocity` | `(T, 6)` | Finite-difference velocity of commanded pose |

**Why `joint_velocity` matters:**
`tcp_velocity` only tells you how the tool tip moves in Cartesian space — it cannot tell you *which joints* are moving or how fast. For example, a fast wrist-3 rotation and a slow shoulder rotation can produce the same TCP velocity but require completely different motor commands. Including `joint_velocity` gives the policy full observability of the robot's dynamic state.

**Robot state vector for training (33D):**
```
tcp_pose (7) + tcp_velocity (6) + tcp_error (6) + joint_positions (7) + joint_velocity (7) = 33
```

---

## Complete Workflow

### Step 0 — One-Time Setup

Run this in every shell before using any of the commands below:

```bash
cd ~/ros2_ws/src/aic

export AIC_ROOT="$(pwd)"
export TRAIN_ROOT="$AIC_ROOT/team_policy/team_policy/training_robot"
export CONFIG="$TRAIN_ROOT/configs/orientation_sweep_50_trials.yaml"
export EPISODES="$TRAIN_ROOT/episodes/orientation_sweep"
export LEROBOT="$TRAIN_ROOT/lerobot_datasets"
```

After any code change to `episode_recorder.py`, `cheatcode_collector.py`, `validate_episode.py`, or `convert_to_lerobot.py`:

```bash
pixi reinstall ros-kilted-team-policy
```

---

### Step 1A — Quick Smoke Test (5 Trials) — **Do This First**

Before committing to a full 50-episode run, use the 5-trial config to verify the entire pipeline
end-to-end: collection → validation → conversion → training.

#### Terminal 1 — Start Simulation + Engine (5 trials)

```bash
cd ~/ros2_ws/src/aic
export AIC_ROOT="$(pwd)"
export TRAIN_ROOT="$AIC_ROOT/team_policy/team_policy/training_robot"
export DBX_CONTAINER_MANAGER=docker

distrobox enter -r aic_eval -- /entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=true \
  gazebo_gui:=false \
  launch_rviz:=false \
  aic_engine_config_file:="$TRAIN_ROOT/configs/test_5_trials.yaml"
```

Wait until you see:
```
[aic_engine] Waiting for model...
```

#### Terminal 2 — Start Collector (5 trials)

```bash
cd ~/ros2_ws/src/aic
export AIC_ROOT="$(pwd)"
export TRAIN_ROOT="$AIC_ROOT/team_policy/team_policy/training_robot"
export RUN_ID="test_run_001"
export OUTPUT_DIR="$TRAIN_ROOT/episodes/test/$RUN_ID"

pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=team_policy.training_robot.cheatcode_collector \
  -p output_dir:="$OUTPUT_DIR" \
  -p num_episodes:=5 \
  -p success_only:=true
```

Expected output:
```
DataCollectionPolicy ready — target=5 episodes, output=.../test_run_001, success_only=True
collector/episode=0 port=sfp/sfp_port_0
[1/5] Saved .../episode_00000.hdf5
...
[5/5] Saved .../episode_00004.hdf5
on_shutdown(...)
```

#### Validate the 5 test episodes

```bash
cd ~/ros2_ws/src/aic
export AIC_ROOT="$(pwd)"
export TRAIN_ROOT="$AIC_ROOT/team_policy/team_policy/training_robot"

for f in $TRAIN_ROOT/episodes/test/test_run_001/episode_*.hdf5; do
  echo "Validating $f"
  pixi run python -m team_policy.training_robot.validate_episode --file "$f" || break
done
```

#### Convert the 5 test episodes

```bash
pixi run python -m team_policy.training_robot.convert_to_lerobot \
  --input  $TRAIN_ROOT/episodes/test/test_run_001 \
  --output $TRAIN_ROOT/lerobot_datasets/test_run_001 \
  --success_only
```

#### Train on the 5 test episodes (smoke test only)

```bash
pixi run lerobot-train \
  --dataset.repo_id=local/test_run_001 \
  --policy.type=act \
  --output_dir=outputs/train/aic_act_test \
  --job_name=aic_act_test \
  --policy.device=cuda \
  --wandb.enable=false
```

If all 5 steps complete without errors, the pipeline is working. Proceed to Step 1B.

---

### Step 1B — Collect 50 Episodes (Full Run)

**This is the main data collection step.**
The engine runs 50 trials back-to-back; CheatCode completes each one using ground-truth TF frames;
the collector records everything to HDF5.

#### Terminal 1 — Start Simulation + Engine (50 trials)

```bash
cd ~/ros2_ws/src/aic
export AIC_ROOT="$(pwd)"
export TRAIN_ROOT="$AIC_ROOT/team_policy/team_policy/training_robot"
export DBX_CONTAINER_MANAGER=docker

distrobox enter -r aic_eval -- /entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=true \
  gazebo_gui:=false \
  launch_rviz:=false \
  aic_engine_config_file:="$TRAIN_ROOT/configs/orientation_sweep_50_trials.yaml"
```

Wait until you see:
```
[aic_engine] Waiting for model...
```

#### Terminal 2 — Start Collector (50 trials)

```bash
cd ~/ros2_ws/src/aic
export AIC_ROOT="$(pwd)"
export TRAIN_ROOT="$AIC_ROOT/team_policy/team_policy/training_robot"
export RUN_ID="run_001"
export OUTPUT_DIR="$TRAIN_ROOT/episodes/orientation_sweep/$RUN_ID"

pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=team_policy.training_robot.cheatcode_collector \
  -p output_dir:="$OUTPUT_DIR" \
  -p num_episodes:=50 \
  -p success_only:=true
```

Expected output (one line per successful episode):
```
DataCollectionPolicy ready — target=50 episodes, output=.../run_001, success_only=True
collector/episode=0 port=sfp/sfp_port_0
[1/50] Saved .../episode_00000.hdf5
[2/50] Saved .../episode_00001.hdf5
...
[50/50] Saved .../episode_00049.hdf5
on_shutdown(...)
```

After all 50 trials finish, Terminal 1 will exit on its own (or press Ctrl+C).

---

### Step 2 — Collect More Data (Additional Batches)

Each run of the 50-trial config produces up to 50 episodes.
To collect more, rerun both terminals with a new `RUN_ID`:

```bash
export RUN_ID="run_002"
export OUTPUT_DIR="$TRAIN_ROOT/episodes/orientation_sweep/$RUN_ID"
# ... then run Terminal 1 and Terminal 2 again
```

**Do not reuse the same `RUN_ID`** — it will silently overwrite `episode_00000.hdf5` through `episode_00049.hdf5`.

To also vary the task board poses between batches:
1. Edit `configs/orientation_sweep_50_trials.yaml` (or edit parameters in `configs/generate_50_trials.py` and regenerate)
2. Restart Terminal 1 so the engine reloads the config
3. Start Terminal 2 with a new `RUN_ID`

To regenerate the YAML after changing parameters in the generator:
```bash
.pixi/envs/default/bin/python $TRAIN_ROOT/configs/generate_50_trials.py
```

---

### Step 3 — Count and Validate Episodes

Count all collected episodes:
```bash
find $TRAIN_ROOT/episodes/orientation_sweep -name 'episode_*.hdf5' | wc -l
```

Validate a single episode:
```bash
pixi run python -m team_policy.training_robot.validate_episode \
  --file $TRAIN_ROOT/episodes/orientation_sweep/run_001/episode_00000.hdf5
```

Validate all episodes (stops on first failure):
```bash
for f in $TRAIN_ROOT/episodes/orientation_sweep/run_*/episode_*.hdf5; do
  echo "Validating $f"
  pixi run python -m team_policy.training_robot.validate_episode --file "$f" || break
done
```

Expected ending for a good episode:
```
--- Schema v4 keys ---
  [OK ] key exists: observations/joint_velocity
Schema v4 keys — joint_velocity present
...
PASS — episode looks valid (N frames, Xs, success=True)
```

---

### Step 4 — Convert to LeRobot Format

Convert one run:
```bash
pixi run python -m team_policy.training_robot.convert_to_lerobot \
  --input  $TRAIN_ROOT/episodes/orientation_sweep/run_001 \
  --output $LEROBOT/orientation_sweep_run_001 \
  --success_only
```

Convert all runs together into one dataset:
```bash
# Merge all episodes into a single folder first
MERGED="$TRAIN_ROOT/episodes/orientation_sweep/merged"
mkdir -p "$MERGED"
idx=0
for f in $TRAIN_ROOT/episodes/orientation_sweep/run_*/episode_*.hdf5; do
  cp "$f" "$MERGED/episode_$(printf '%05d' $idx).hdf5"
  idx=$((idx + 1))
done

pixi run python -m team_policy.training_robot.convert_to_lerobot \
  --input  "$MERGED" \
  --output "$LEROBOT/orientation_sweep_all" \
  --success_only
```

Expected output structure:
```
lerobot_datasets/orientation_sweep_run_001/
  meta/
    info.json         (dataset metadata, 33D state, features schema)
    stats.json        (mean/std/min/max for normalization)
    episodes.jsonl    (per-episode metadata)
    tasks.jsonl       (task definitions)
  data/chunk-000/
    episode_000000.parquet
    episode_000001.parquet
    ...
  videos/chunk-000/
    observation.images.left/episode_000000.mp4
    observation.images.center/episode_000000.mp4
    observation.images.right/episode_000000.mp4
    ...
```

---

### Step 5 — Train ACT Policy

```bash
pixi run lerobot-train \
  --dataset.repo_id=local/aic_orientation_sweep \
  --policy.type=act \
  --output_dir=outputs/train/aic_act_orientation_sweep \
  --job_name=aic_act_orientation_sweep \
  --policy.device=cuda \
  --wandb.enable=false
```

For CPU-only machines:
```bash
pixi run lerobot-train \
  --dataset.repo_id=local/aic_orientation_sweep \
  --policy.type=act \
  --output_dir=outputs/train/aic_act_orientation_sweep \
  --job_name=aic_act_orientation_sweep \
  --policy.device=cpu \
  --wandb.enable=false
```

---

## What is ACT? (Action Chunking with Transformers)

ACT is the neural network architecture we train to become the final cable-insertion policy.
It is an imitation learning model — it learns to copy the expert behavior recorded by CheatCode,
but using only sensors (cameras + robot state), not the hidden TF frames CheatCode used.

### The Core Idea: Action Chunking

A naive policy predicts **one action per step**: at each 20 Hz tick, it looks at the current
image and state and outputs the next 6D delta move. This works but is brittle — any single
bad prediction propagates immediately.

ACT instead predicts a **chunk of K future actions at once** (default K=100, i.e. 5 seconds).
At each step it outputs the next 100 moves, executes them one by one, then re-plans.
This has two benefits:
1. **Temporal smoothness** — the chunk is generated by a single forward pass through the
   transformer, so consecutive actions are internally consistent and smooth.
2. **Robustness to compounding errors** — replanning every K steps lets the policy correct
   drift before it accumulates.

### Architecture

```
Inputs at each step:
  ├─ 3× camera images (left, center, right)   → ResNet-18 visual encoder → image tokens
  └─ 33D robot state (tcp_pose + velocities + joints)  → linear projection → state token

Transformer encoder:
  Takes image tokens + state token → context representation

Transformer decoder:
  Attends to context → predicts K=100 future actions (6D delta TCP each)

Output:
  100 × [dx, dy, dz, drx, dry, drz]   (position delta + axis-angle rotation delta)
  Executed at 20 Hz → ~5 seconds of planned motion per inference call
```

### Why ACT Works for Cable Insertion

Cable insertion requires **sub-millimetre precision** and **smooth, continuous motion**.
The standard imitation learning failure mode is "compounding errors" — small prediction
mistakes stack up and the robot drifts off course. ACT's chunking directly addresses this:

- **Chunking reduces error compounding** — instead of 100 independent 1-step decisions,
  you make 1 coherent 100-step plan, so errors don't compound within the chunk.
- **Re-planning every 5 seconds** — if the robot drifts slightly, the next chunk corrects it
  before the drift becomes unrecoverable.
- **Transformer attention over image + state** — the model can attend to the exact pixel
  region of the port in the camera image simultaneously with the robot's joint configuration,
  giving it the spatial precision needed for insertion.

### Training vs Deployment

| | Training (CheatCode) | Deployment (ACT) |
|---|---|---|
| Expert source | CheatCode oracle with ground-truth TF | None — learned from demonstrations |
| Inputs used | ground-truth TF frames + all sensors | cameras + robot state only |
| `ground_truth:=true` required | Yes | No |
| Action type | absolute TCP targets | 6D delta TCP (position + axis-angle) |

At deployment, RunACT.py loads the trained checkpoint and calls it as a standard `aic_model`
policy — it receives the same `Observation` message and returns a `MotionUpdate`, exactly like
any other policy in the system.

### How Many Demos You Need

| Episodes | Expected Outcome |
|---|---|
| 5 | Smoke test only — model will overfit, not generalise |
| 50 | Minimum viable — works for narrow pose distribution |
| 200–500 | Recommended for robust generalisation across varied board poses |
| 1000+ | Research-grade; covers full pose distribution reliably |

The 50-trial config provides visual diversity across 10 board poses and 5 NIC rails to make
50 episodes more useful than 50 identical insertions would be.

---

## Complete Flow Explained

```
┌─────────────────────────────────────────────────────────────────────┐
│  TERMINAL 1: aic_eval container                                     │
│                                                                     │
│  Gazebo simulation + aic_engine                                     │
│    - Spawns robot, task board, cable per trial                      │
│    - Uses orientation_sweep_50_trials.yaml (50 varied scenes)       │
│    - ground_truth:=true  → publishes hidden TF frames               │
│      (task_board/nic_card_mount_0/sfp_port_0_link etc.)             │
│    - Calls insert_cable() on the model for each trial               │
└────────────────────────┬────────────────────────────────────────────┘
                         │ ROS 2 topics + services
┌────────────────────────▼────────────────────────────────────────────┐
│  TERMINAL 2: cheatcode_collector (aic_model)                        │
│                                                                     │
│  DataCollectionPolicy.insert_cable() called once per trial          │
│    │                                                                │
│    ├─ Wraps move_robot() to capture CheatCode's commanded poses     │
│    │                                                                │
│    ├─ Spawns background thread recording at ~20 Hz:                 │
│    │    • 3× camera images (left, center, right)                    │
│    │    • tcp_pose, tcp_velocity, tcp_error                         │
│    │    • joint_positions, joint_velocity  ← per-joint motion       │
│    │    • wrist_force (F/T sensor)                                  │
│    │    • relative_pose (plug → port, from ground truth TF)         │
│    │    • privileged_tf snapshots (5 TF pairs, debug use only)      │
│    │                                                                │
│    └─ Runs CheatCode.insert_cable() as the expert                   │
│         CheatCode looks up ground-truth TF frames to navigate       │
│         precisely to the port and insert the cable                  │
│                                                                     │
│  On success → EpisodeRecorder.end_episode() → episode_XXXXX.hdf5   │
└────────────────────────┬────────────────────────────────────────────┘
                         │ HDF5 files
┌────────────────────────▼────────────────────────────────────────────┐
│  validate_episode.py                                                │
│    Checks shapes, quaternion norms, image quality, timing,         │
│    joint_velocity non-zero, relative_pose valid, etc.              │
└────────────────────────┬────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────────┐
│  convert_to_lerobot.py                                              │
│                                                                     │
│  Per episode:                                                       │
│    • Reads HDF5 images → writes MP4 videos (20 FPS, mp4v)          │
│    • Builds 33D robot state:                                        │
│        tcp_pose(7) + tcp_vel(6) + tcp_err(6)                       │
│        + joint_pos(7) + joint_vel(7)                               │
│    • Reads actions/delta_pose (6D) as expert action labels          │
│    • Writes parquet row per frame                                   │
│                                                                     │
│  Across all episodes:                                               │
│    • Computes mean/std/min/max → stats.json (for normalization)     │
│    • Writes info.json, episodes.jsonl, tasks.jsonl                  │
└────────────────────────┬────────────────────────────────────────────┘
                         │ LeRobot dataset
┌────────────────────────▼────────────────────────────────────────────┐
│  lerobot-train (ACT policy)                                         │
│                                                                     │
│  Input:  3× camera images + 33D robot state                         │
│  Output: chunk of 100 × 6D delta TCP actions                        │
│                                                                     │
│  Trains ResNet-18 image encoder + Transformer (encoder-decoder)     │
│  Loss: MSE between predicted and expert delta actions               │
│                                                                     │
│  Result: a checkpoint deployed via RunACT.py as the live policy     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Why 50 Trials in One Config

The original 3-trial config required running the two-terminal setup ~17 times to collect 50 episodes.
The 50-trial config covers:
- **10 different task board positions and yaw angles** — robot sees the board from many viewpoints
- **5 NIC rails × 8 positions = 40 SFP trials** — varied port location in image space
- **2 SC rails × 5 positions = 10 SC trials** — different cable type and connector
- **3 background mount arrangements** — varied visual clutter

More visual diversity → better generalization of the trained policy.

---

## Common Issues

### `on_shutdown` appears but fewer than 50 episodes saved

Some trials failed (CheatCode returned `False`). This is normal for difficult poses.
Run again with a new `RUN_ID` — the engine will replay all 50 trials from scratch.
Increase diversity by editing pose values in the YAML and regenerating.

### `Collector cannot find ground truth TF`

Terminal 1 must have `ground_truth:=true`. The collector depends on these TF frames
for `relative_pose` and `privileged_tf` snapshots. Without them, those fields save as zeros
and the episode is flagged during validation.

### `joint_velocity` is all zeros after validation

This means the episode was recorded with schema v3 (before this update) or `js.velocity`
was not yet populated by the controller. Re-collect with the updated package installed:
```bash
pixi reinstall ros-kilted-team-policy
```

### Reusing the same `RUN_ID`

Do not reuse `output_dir` between runs. The recorder always starts numbering from
`episode_00000.hdf5` and will silently overwrite existing files.

### Merging runs for conversion

Pass `--input` to a folder that contains only `episode_XXXXX.hdf5` files.
If mixing runs, copy them into a single folder and renumber as shown in Step 4.
