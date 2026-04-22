# CheatCode Training Data Pipeline

Full imitation-learning pipeline: collect expert demos → validate → convert → train ACT → deploy.

```
aic_engine runs 3 trials per session (fixed board pose)
  → aic_model loads cheatcode_collector
    → CheatCode (ground-truth oracle) executes each trial
      → EpisodeRecorder saves observations + expert actions → episode_XXXXX.hdf5
        → validate_episode.py checks every file
          → convert_to_lerobot.py converts HDF5 → LeRobot v3.0 dataset
            → lerobot-train trains an ACT neural network policy
              → team_policy.run_act deploys the checkpoint
```

> **Important:** `CheatCode` requires `ground_truth:=true` on the eval side.
> The final deployed policy (`run_act.py`) uses only cameras + robot state — no hidden TF frames.

---

## Why Competition Sessions (not the old 50-trial config)

The old `orientation_sweep_50_trials.yaml` ran 50 trials in one Gazebo session with all 5 NIC
cards + 2 SC mounts spawned simultaneously every trial. This caused two problems:

- **RAM fills up** → PC freezes mid-run as Gazebo accumulates 50 × 7+ entities
- **Score drops to ~30 after trial 1** → The board pose (x/y/z + yaw) jumped to a completely
  different position every trial. CheatCode's TF lookups got stale during entity respawn,
  causing it to move to the wrong position and miss the insertion

The fix is **competition-format sessions**:

| | Old 50-trial config | Competition sessions |
|---|---|---|
| Trials per Gazebo launch | 50 | **3** |
| Board pose within session | Changes every trial | **Fixed** — same for all 3 trials |
| Entities spawned per trial | All 5 NICs + 2 SCs | **1 target entity only** |
| RAM behaviour | Fills up, PC freezes | Stable — restarted between sessions |
| CheatCode score | 30/100 after trial 1 | Consistent across all 3 trials |

Each session = **2 NIC insertions + 1 SC insertion** at one fixed board pose.
Board pose varies **between** sessions, giving training diversity across 50 relaunches.

---

## Files

| File | Purpose |
|---|---|
| `cheatcode_collector.py` | Policy wrapper — records expert demos via CheatCode |
| `episode_recorder.py` | Buffers and saves one episode as HDF5 (schema v4) |
| `validate_episode.py` | Validates one HDF5 episode file |
| `convert_to_lerobot.py` | Converts HDF5 episodes → LeRobot v3.0 parquet + MP4 videos |
| `configs/generate_competition_sessions.py` | **Generates competition session YAMLs** |
| `configs/sessions/session_01.yaml … session_50.yaml` | **50 session configs (3 trials each)** |
| `configs/test_5_trials.yaml` | 5-trial smoke test config |
| `../run_act.py` | **Deployment policy** — loads local ACT checkpoint, runs at 20 Hz |

Generated data is git-ignored:
```
training_robot/episodes/
training_robot/lerobot_datasets/
```

---

## Session Config Design

Each session YAML in `configs/sessions/` shares one fixed board pose across all 3 trials:

| Trial | Task | Entity spawned |
|---|---|---|
| trial_1 | NIC insertion (sfp) | 1 NIC card on target rail + background mounts |
| trial_2 | NIC insertion (sfp) | 1 NIC card on different rail + background mounts |
| trial_3 | SC insertion | 1 SC mount on target rail + background mounts |

Across 50 sessions:
- Board pose cycles through 10 positions (A–J) → full spatial diversity
- NIC rail targets cycle across all 5 rails (0–4) evenly
- SC rail targets alternate between 0 and 1
- Background mounts cycle through 3 arrangements for visual variety

To regenerate sessions (e.g. to change the number or tweak poses):
```bash
cd ~/ros2_ws/src/aic
python3 team_policy/team_policy/training_robot/configs/generate_competition_sessions.py --sessions 50
```

For 100 sessions:
```bash
python3 team_policy/team_policy/training_robot/configs/generate_competition_sessions.py --sessions 100
```

---

## HDF5 Schema v4 — What Gets Recorded

Each episode is saved at ~20 Hz:

| Field | Shape | Description |
|---|---|---|
| `observations/images/{left,center,right}` | `(T, H, W, 3)` | Camera images, uint8 RGB, gzip-compressed |
| `observations/tcp_pose` | `(T, 7)` | Tool position + quaternion in base frame |
| `observations/tcp_velocity` | `(T, 6)` | Cartesian tool velocity (linear + angular) |
| `observations/tcp_error` | `(T, 6)` | Pose error to current target |
| `observations/joint_positions` | `(T, 7)` | Joint angles (rad) |
| `observations/joint_velocity` | `(T, 7)` | Per-joint velocity (rad/s) |
| `observations/wrist_force` | `(T, 6)` | F/T sensor readings |
| `observations/relative_pose` | `(T, 7)` | Target port pose in plug-tip frame (ground truth) |
| `observations/privileged_tf/transforms` | `(T, 5, 7)` | TF snapshots (debug only) |
| `actions/commanded_pose` | `(T, 7)` | Absolute TCP target commanded by CheatCode |
| `actions/delta_pose` | `(T, 6)` | Position delta + axis-angle rotation delta |
| `actions/velocity` | `(T, 6)` | Finite-difference velocity of commanded pose |

**Robot state vector used for training (33D):**
```
tcp_pose (7) + tcp_velocity (6) + tcp_error (6) + joint_positions (7) + joint_velocity (7) = 33
```

---

## Complete Workflow

### Step 0 — Shell Environment

Set these once in Terminal 2. `TRAIN_ROOT` stays exported for the whole day.

```bash
export TRAIN_ROOT=/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot
```

After any code change to `cheatcode_collector.py`, `episode_recorder.py`, or `run_act.py`:
```bash
cd ~/ros2_ws/src/aic
pixi reinstall ros-kilted-team-policy
```

---

### Step 1 — Collect Episodes (50 sessions × 3 trials = 150 episodes)

Run one session at a time. Each session takes ~10–15 minutes.

#### Terminal 1 — Start Simulation + Engine

```bash
distrobox enter -r aic_eval -- /entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=true \
  aic_engine_config_file:=/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/configs/sessions/session_01.yaml
```

Wait until Gazebo opens and you see:
```
No node with name 'aic_model' found. Retrying...
```

#### Terminal 2 — Start CheatCode Collector

```bash
export TRAIN_ROOT=/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot
export RUN_ID=run_001
export OUTPUT_DIR="$TRAIN_ROOT/episodes/$RUN_ID"

cd ~/ros2_ws/src/aic && pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=team_policy.training_robot.cheatcode_collector \
  -p output_dir:="$OUTPUT_DIR" \
  -p num_episodes:=3 \
  -p success_only:=true
```

Expected output per episode:
```
[INFO] DataCollectionPolicy ready — target=3 episodes
[INFO] collector/episode=0 port=sfp/sfp_port_0
[INFO] [1/3] Saved .../run_001/episode_00000.hdf5
[INFO] collector/episode=1 port=sfp/sfp_port_0
[INFO] [2/3] Saved .../run_001/episode_00001.hdf5
[INFO] collector/episode=2 port=sc/sc_port_base
[INFO] [3/3] Saved .../run_001/episode_00002.hdf5
```

Terminal 2 exits on its own after 3 episodes. Press `Ctrl+C` in Terminal 1.

#### Next session — only two things change

**Terminal 1** — increment session number:
```bash
distrobox enter -r aic_eval -- /entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=true \
  aic_engine_config_file:=/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/configs/sessions/session_02.yaml
```

**Terminal 2** — increment RUN_ID (`TRAIN_ROOT` stays set, no need to re-export):
```bash
export RUN_ID=run_002
export OUTPUT_DIR="$TRAIN_ROOT/episodes/$RUN_ID"

cd ~/ros2_ws/src/aic && pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=team_policy.training_robot.cheatcode_collector \
  -p output_dir:="$OUTPUT_DIR" \
  -p num_episodes:=3 \
  -p success_only:=true
```

#### Session reference

| Session | Terminal 1 config | Terminal 2 RUN_ID |
|---------|-------------------|-------------------|
| 1 | `session_01.yaml` | `run_001` |
| 2 | `session_02.yaml` | `run_002` |
| 3 | `session_03.yaml` | `run_003` |
| … | … | … |
| 50 | `session_50.yaml` | `run_050` |

#### Expected folder layout after all 50 sessions

```
training_robot/episodes/
  run_001/
    episode_00000.hdf5   ← NIC0, pose A
    episode_00001.hdf5   ← NIC2, pose A
    episode_00002.hdf5   ← SC0,  pose A
  run_002/
    episode_00000.hdf5   ← NIC1, pose B
    episode_00001.hdf5   ← NIC3, pose B
    episode_00002.hdf5   ← SC1,  pose B
  ...
  run_050/
    episode_00000.hdf5
    episode_00001.hdf5
    episode_00002.hdf5
```

**50 sessions × 3 trials = 150 HDF5 files total.**

If a trial fails (CheatCode returns False), `success_only:=true` discards it — no file saved.
In that case just rerun that session with a new `RUN_ID`.

---

### Step 2 — Count and Validate

Count all collected episodes across all runs:
```bash
find $TRAIN_ROOT/episodes -name 'episode_*.hdf5' | wc -l
```

Validate all episodes (stops on first failure):
```bash
for f in $TRAIN_ROOT/episodes/run_*/episode_*.hdf5; do
  echo "--- $f ---"
  pixi run python -m team_policy.training_robot.validate_episode --file "$f" || break
done
```

Validate a single episode:
```bash
pixi run python -m team_policy.training_robot.validate_episode \
  --file $TRAIN_ROOT/episodes/run_001/episode_00000.hdf5
```

A good episode ends with:
```
PASS — episode looks valid (N frames, Xs, success=True)
```

#### Episode quality thresholds

| Quality | Rule |
|---|---|
| Excellent | `success == 1` and `final_error <= 0.005` m |
| Good | `success == 1` and `final_error <= 0.02` m |
| Skip | `success == 0` or `final_error > 0.02` m |

---

### Step 3 — Convert to LeRobot Format

Merge all runs into one renumbered folder, then convert:

```bash
export TRAIN_ROOT=/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot
export LEROBOT=$TRAIN_ROOT/lerobot_datasets

# Merge all runs into one folder (renumber episode files sequentially)
MERGED="$TRAIN_ROOT/episodes/merged"
rm -rf "$MERGED" && mkdir -p "$MERGED"
idx=0
for f in $TRAIN_ROOT/episodes/run_*/episode_*.hdf5; do
  cp "$f" "$MERGED/episode_$(printf '%05d' $idx).hdf5"
  idx=$((idx + 1))
done
echo "Merged $idx episodes"

# Convert to LeRobot format
cd ~/ros2_ws/src/aic
pixi run python -m team_policy.training_robot.convert_to_lerobot \
  --input  "$MERGED" \
  --output "$LEROBOT/aic_dataset_v1" \
  --success_only \
  --max_final_error 0.02
```

Output structure:
```
lerobot_datasets/aic_dataset_v1/
  meta/
    info.json           ← 33D state, 6D action, video feature schema
    stats.json          ← mean/std/min/max for normalisation
    tasks.parquet
    episodes/chunk-000/file-000.parquet
  data/chunk-000/
    file-000.parquet
  videos/
    observation.images.left/chunk-000/file-000.mp4
    observation.images.center/chunk-000/file-000.mp4
    observation.images.right/chunk-000/file-000.mp4
```

---

### Step 4 — Train ACT Policy

```bash
export TRAIN_ROOT=/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot
export LEROBOT=$TRAIN_ROOT/lerobot_datasets

cd ~/ros2_ws/src/aic
pixi run lerobot-train \
  --dataset.repo_id=local/aic_dataset_v1 \
  --dataset.root="$LEROBOT/aic_dataset_v1" \
  --policy.type=act \
  --output_dir=outputs/train/aic_act_run_001 \
  --job_name=aic_act_run_001 \
  --policy.device=cuda \
  --wandb.enable=false \
  --steps=100000 \
  --save_freq=20000
```

For CPU-only machines replace `--policy.device=cuda` with `--policy.device=cpu`.

Training writes checkpoints every 20 000 steps:
```
outputs/train/aic_act_run_001/checkpoints/
  020000/pretrained_model/   ← deploy this folder
    config.json
    model.safetensors
    policy_preprocessor_step_3_normalizer_processor.safetensors
    policy_postprocessor_step_0_unnormalizer_processor.safetensors
  020000/training_state/     ← resume training only
```

Resume training from a checkpoint:
```bash
pixi run lerobot-train \
  --config_path=outputs/train/aic_act_run_001/checkpoints/020000/pretrained_model/train_config.json \
  --resume=true \
  --steps=100000
```

---

### Step 5 — Deploy and Test the Trained Policy

#### Terminal 1 — Start Simulation (no ground_truth needed for deployment)

```bash
distrobox enter -r aic_eval -- /entrypoint.sh \
  ground_truth:=false \
  start_aic_engine:=true \
  aic_engine_config_file:=/home/ibrahim/ros2_ws/src/aic/team_policy/team_policy/training_robot/configs/sessions/session_01.yaml
```

#### Terminal 2 — Run the Trained Policy

```bash
export CKPT=/home/ibrahim/ros2_ws/src/aic/outputs/train/aic_act_run_001/checkpoints/100000/pretrained_model

cd ~/ros2_ws/src/aic
pixi reinstall ros-kilted-team-policy

pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=team_policy.run_act \
  -p checkpoint_path:="$CKPT"
```

The robot will attempt cable insertion using only cameras and robot state — no ground truth.
Watch Gazebo to see whether it succeeds.

---

## How Many Episodes You Need

| Episodes | Expected outcome |
|---|---|
| 5–10 | Smoke test only — overfits, does not generalise |
| 50 | Minimum viable — works on seen board poses |
| 150 | Good starting point — covers all 10 poses + all rails |
| 300–500 | Recommended for robust generalisation |
| 1000+ | Research-grade, full pose and rail distribution |

---

## Architecture Summary (ACT)

```
Inputs at each 20 Hz step:
  ├─ 3× camera images (left, center, right)  → ResNet-18 encoder → image tokens
  └─ 33D robot state                         → linear projection → state token

Transformer encoder-decoder:
  Attends over tokens → predicts chunk of K=100 future 6D actions

Output per step:
  100 × [dx, dy, dz, drx, dry, drz]   applied at 20 Hz = 5 s of planned motion
```

Action chunking reduces compounding errors: instead of 100 independent predictions,
one coherent 100-step plan is generated. The policy replans every 5 seconds to correct drift.

---

## Complete Data Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│  TERMINAL 1 — aic_eval container                                     │
│                                                                      │
│  Gazebo + aic_engine                                                 │
│    • Config: configs/sessions/session_XX.yaml                        │
│    • Spawns: robot + 1 target entity (NIC or SC) per trial           │
│    • Board pose FIXED for all 3 trials in the session                │
│    • ground_truth:=true → publishes hidden TF frames                 │
│    • Calls insert_cable() once per trial, 3 trials total             │
└─────────────────────────────┬────────────────────────────────────────┘
                              │ ROS 2 topics + services
┌─────────────────────────────▼────────────────────────────────────────┐
│  TERMINAL 2 — cheatcode_collector (aic_model)                        │
│                                                                      │
│  DataCollectionPolicy.insert_cable():                                │
│    • Wraps move_robot() to capture commanded poses                   │
│    • Records at 20 Hz in background thread:                          │
│        3× camera images, tcp_pose, tcp_vel, tcp_err                  │
│        joint_positions, joint_velocity, wrist_force                  │
│        relative_pose (plug→port), privileged_tf snapshots            │
│    • Runs CheatCode as the expert (uses ground-truth TF)             │
│    • On success → episodes/run_XXX/episode_XXXXX.hdf5                │
│    • Exits after 3 episodes                                          │
└─────────────────────────────┬────────────────────────────────────────┘
                              │  repeat × 50 sessions (Ctrl+C T1, relaunch both)
┌─────────────────────────────▼────────────────────────────────────────┐
│  validate_episode.py  — checks shapes, quaternions, timing, etc.     │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────────┐
│  convert_to_lerobot.py                                               │
│    • Merges run_001 … run_050 → single renumbered folder             │
│    • HDF5 images → MP4 videos                                        │
│    • Builds 33D state + 6D delta action per frame                    │
│    • Writes parquet + stats.json + info.json                         │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────────┐
│  lerobot-train (ACT)                                                 │
│    Input:  3× images + 33D state                                     │
│    Output: 100 × 6D delta TCP actions                                │
│    Result: pretrained_model/ checkpoint                              │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────────┐
│  team_policy.run_act  (deployment)                                   │
│    • Loads local checkpoint via checkpoint_path ROS param            │
│    • Runs at 20 Hz: image + state → 6D delta → MODE_POSITION command │
│    • No ground_truth required                                        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Common Issues

### RAM fills up / PC freezes mid-run

You are using the old `orientation_sweep_50_trials.yaml` which spawns all 5 NICs + 2 SC mounts
for 50 trials without restarting Gazebo. Switch to the session YAMLs in `configs/sessions/`.
Each session is only 3 trials with 1 entity spawned per trial.

### Score drops to ~30 after trial 1

Same cause as above — board pose was jumping between trials, causing stale TF lookups.
The session YAMLs fix this by keeping the board pose fixed for all 3 trials.

### Fewer than 3 episodes saved after a session

Some trials failed (CheatCode returned False). Normal for difficult poses.
Start a new session with an incremented `RUN_ID` and the same session YAML to retry,
or skip to the next session YAML — the missed episodes are not critical.

### `Collector cannot find ground truth TF`

Terminal 1 must have `ground_truth:=true`. Without it, `relative_pose` and `privileged_tf`
save as zeros and validation will flag the episode.

### `joint_velocity` is all zeros after validation

Recorded before schema v4 or controller was not publishing joint velocities. Re-collect after:
```bash
pixi reinstall ros-kilted-team-policy
```

### `success=1` but `final_error` is large (e.g. 0.045 m)

Episode is suspicious. Keep `--max_final_error 0.02` in the converter to skip it.

### Reusing the same `RUN_ID`

Do not. The recorder always starts from `episode_00000.hdf5` and silently overwrites existing
files. Always increment `RUN_ID` for each new collection run.

### `checkpoint_path` parameter not set

`team_policy.run_act` requires:
```bash
-p checkpoint_path:=/absolute/path/to/pretrained_model
```
The path must point to the `pretrained_model/` subfolder, not the checkpoint root or
`training_state/` folder.
