# YOLO-Guided Training Pipeline — run_003 (30D state)

## Why run_003 exists

| Run | State dim | Problem |
|-----|-----------|---------|
| run_001 | 33D | `tcp_error` was CheatCode ground-truth during training — near-zero at inference → robot hit the ground |
| run_002 | 27D | No port signal — policy had no idea where the port was |
| **run_003** | **30D** | **YOLO-detected port xyz — same signal at training and inference** |

**30D state breakdown:**
```
tcp_pose(7) + tcp_velocity(6) + joint_positions(7) + joint_velocity(7) + port_xyz(3) = 30
```
`port_xyz` = port position in `base_link` estimated by YOLO.
During training YOLO runs live inside the sim. At inference YOLO runs on the real robot — no ground truth ever used.

---

## One-time setup

### Environment variables

Paste these in **every terminal** you open for this pipeline:

```bash
export AIC_ROOT=$(git -C ~/ros2_ws/src/aic rev-parse --show-toplevel)
export TRAIN_ROOT=$AIC_ROOT/team_policy/team_policy/training_robot
export FASTRTPS_DEFAULT_PROFILES_FILE=$AIC_ROOT/team_policy/fastdds_no_shm.xml
```

### Sync source files after pixi reinstall

Only needed after a `pixi reinstall`. Copies working files into the pixi install location:

```bash
cd $AIC_ROOT
SITE=$AIC_ROOT/.pixi/envs/default/lib/python3.12/site-packages/team_policy
cp team_policy/team_policy/run_act.py                              $SITE/run_act.py
cp team_policy/team_policy/training_robot/episode_recorder.py      $SITE/training_robot/episode_recorder.py
cp team_policy/team_policy/training_robot/cheatcode_collector.py   $SITE/training_robot/cheatcode_collector.py
cp team_policy/team_policy/training_robot/convert_to_lerobot.py    $SITE/training_robot/convert_to_lerobot.py
```

### Open a pixi shell

Run all subsequent commands from inside a pixi shell — this avoids triggering a package rebuild on every command:

```bash
cd $AIC_ROOT
pixi shell
```

---

## Step 1 — Collect episodes

**Target: 200 episodes** saved to `$TRAIN_ROOT/episodes/run_003/`

### How a session works

One run of `aic_engine` = **one session** = **3 trials**:
- Trial 1: SFP NIC insertion
- Trial 2: SFP NIC insertion
- Trial 3: SC fiber insertion

After trial 3, `aic_engine` exits automatically. You then restart the sim and collector and run the next session. YOLO stays running the whole time.

You need **3 terminals** open, all with the env vars set and inside `pixi shell`.

---

### Terminal 1 — Simulation

**Start at the beginning of every session. Kill and restart after each session.**

```bash
distrobox enter -r aic_eval -- /entrypoint.sh \
    ground_truth:=true \
    start_aic_engine:=true \
    gazebo_gui:=false \
    launch_rviz:=false
```

Wait until you see this line before doing anything else:
```
[aic_engine] No node with name 'aic_model' found. Retrying...
```
This means the sim is up and waiting for a policy. Only then start Terminal 2 (first session only) and Terminal 3.

> **Controller errors at startup are normal.** You will see joint controller spawner errors for the first few seconds. Ignore them — they resolve on their own. The `Retrying...` message confirms the sim is fully ready.

---

### Terminal 2 — YOLO planner

**Start once at the beginning of your first session. Leave it running for all sessions.**

```bash
ros2 run team_policy combined_yolo_depth_pose_planner
```

Wait until you see:
```
[yolov12_multicamera_detector]: Combined planner node started: YOLO + metric depth + CAD registration...
```
That line means YOLO loaded the model and is ready. **It does not print xyz to the terminal** — detections are published silently to `/fused_yolo/detections_json`. You can verify it is detecting by running `ros2 topic echo /fused_yolo/detections_json` in a spare terminal, but you do not need to do this every session.

Once you see "Combined planner node started", proceed to Terminal 3.

> Always use `combined_yolo_depth_pose_planner` — **not** `yolov12_detector`.
> Only the combined node fuses depth + YOLO and publishes 3D port positions in `base_link`.
> The collector subscribes to `/fused_yolo/detections_json` which only this node publishes.

---

### Terminal 3 — Collector (CheatCode + recorder)

**Start at the beginning of every session. Kill and restart after each session.**

```bash
ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=team_policy.training_robot.cheatcode_collector \
    -p output_dir:=$TRAIN_ROOT/episodes/run_003
```

Expected output when it connects and starts:
```
[INFO] DataCollectionPolicy ready — target=50 episodes, output=.../episodes/run_003
```

Expected output per saved trial:
```
[INFO] [4/50] Saved .../episodes/run_003/episode_00004.hdf5
```

If a trial fails (CheatCode couldn't insert), you will see:
```
collector/episode=N FAILED — discarded
```
Failed episodes are **not saved** — only successful insertions are kept. This is by design (`success_only=True`).

---

### After 3 trials — end of session

When the engine finishes trial 3 it exits on its own. You will see Gazebo physics cleanup messages in Terminal 1 — **this is normal, not an error.**

**Steps to end a session and start the next one:**

1. `Ctrl+C` in Terminal 3 (collector exits on its own, but Ctrl+C is safe)
2. `Ctrl+C` in Terminal 1 (kills the sim — Gazebo may already have printed cleanup messages)
3. Check how many episodes you have so far:
   ```bash
   ls $TRAIN_ROOT/episodes/run_003/*.hdf5 | wc -l
   ```
4. Relaunch Terminal 1 — wait for `Retrying...`
5. Relaunch Terminal 3 — the collector resumes from where it left off automatically
6. Terminal 2 (YOLO) keeps running — **do not restart it**

Repeat until you have 200 episodes.

---

### Tracking progress

```bash
ls $TRAIN_ROOT/episodes/run_003/*.hdf5 | wc -l
```

Each session gives you up to 3 episodes (one per successful trial). Sessions where CheatCode fails a trial give fewer. Typical pace: **2–3 episodes per session**, so expect ~70–100 sessions for 200 episodes.

---

### What YOLO coverage looks like (good vs bad)

While the collector is running, YOLO-detected xyz is recorded per frame. When you later convert (Step 2), it reports coverage per episode:

```
yolo_port_xyz: 90%+ frames detected by YOLO   ← good — keep this episode
yolo_port_xyz: no YOLO recorded — using GT+noise fallback  ← YOLO was not running
```

If you see many episodes with low YOLO coverage it means YOLO was not running when you collected those episodes. Recollect them (or run Step 2 with `--success_only` to drop them).

---

## Step 2 — Convert to LeRobot dataset

Run this once you have 200 episodes (or whenever you want to check a partial dataset):

```bash
cd $AIC_ROOT
pixi run python -m team_policy.training_robot.convert_to_lerobot \
    --input  $TRAIN_ROOT/episodes/run_003 \
    --output $TRAIN_ROOT/lerobot_datasets/aic_act_run_003 \
    --success_only
```

Expected final output:
```
Done — 200 episodes, XXXXX frames → .../lerobot_datasets/aic_act_run_003
state: 30D (tcp_pose 7 + tcp_vel 6 + joint_pos 7 + joint_vel 7 + port_xyz 3)
```

> **Do not mix run_003 with run_001 or run_002 datasets.** The state dimensions are different and the normalizer stats are incompatible.

---

## Step 3 — Train

Run inside `tmux` so training survives closing the terminal:

```bash
tmux new -s training
# Detach: Ctrl+B then D
# Reattach later: tmux attach -t training
```

```bash
cd $AIC_ROOT
pixi run lerobot-train \
    --dataset.repo_id=local/aic_act_run_003 \
    --dataset.root=$TRAIN_ROOT/lerobot_datasets/aic_act_run_003 \
    --policy.type=act \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/aic_act_run_003 \
    --job_name=aic_act_run_003 \
    --policy.device=cuda \
    --wandb.enable=false \
    --steps=100000 \
    --save_freq=20000
```

Checkpoints are saved every 20k steps:
```
outputs/train/aic_act_run_003/checkpoints/
  020000/pretrained_model/
  040000/pretrained_model/
  060000/pretrained_model/
  080000/pretrained_model/
  100000/pretrained_model/   ← use this for inference
```

---

## Step 4 — Run inference

Open 3 terminals. Set env vars and enter `pixi shell` in each.

**Terminal 1 — Simulation (no ground truth this time)**

```bash
distrobox enter -r aic_eval -- /entrypoint.sh \
    ground_truth:=false \
    start_aic_engine:=true \
    gazebo_gui:=false \
    launch_rviz:=false
```

**Terminal 2 — YOLO planner**

```bash
ros2 run team_policy combined_yolo_depth_pose_planner
```

Wait for "Combined planner node started" in the output, then start Terminal 3.

**Terminal 3 — ACT policy**

```bash
export CKPT=$AIC_ROOT/outputs/train/aic_act_run_003/checkpoints/100000/pretrained_model

ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=team_policy.run_act \
    -p checkpoint_path:="$CKPT"
```

Watch Terminal 3 for:
```
port_xyz=[x.xxx, y.xxx, z.xxx]   ← YOLO feeding port position correctly
port_xyz=[0.0, 0.0, 0.0]         ← YOLO not detecting — check Terminal 2
```

---

## Folder layout

```
team_policy/team_policy/training_robot/
  episodes/
    run_001/   ← old 33D data — do not use
    run_002/   ← old 27D data — do not use
    run_003/   ← current 30D YOLO data ★
      episode_00000.hdf5
      episode_00001.hdf5
      ...
  lerobot_datasets/
    aic_act_run_003/   ← converted LeRobot dataset ★

outputs/train/
  aic_act_run_003/
    checkpoints/
      100000/pretrained_model/   ← checkpoint to deploy
```

---

## Common issues

**Controller spawner errors at sim startup**
Normal. Wait for `Retrying...` — that is the only signal that matters.

**`port_xyz=[0.0, 0.0, 0.0]` during collection or inference**
YOLO planner is not detecting the port. Check Terminal 2 is printing detections. If not, restart Terminal 2.

**Collector shows `resuming from episode 0` but files already exist**
`$TRAIN_ROOT` is not set correctly. Verify the env vars are set in that terminal and relaunch.

**Episode count not going up after restart**
The collector counts existing files at startup — if `output_dir` is wrong it starts from 0 and overwrites. Always double-check `$TRAIN_ROOT` is set before launching Terminal 3.

**`TF_OLD_DATA ignoring data from the past` flood in YOLO terminal after sim restart**
Normal. When the sim restarts, the Gazebo clock resets to ~5 seconds. YOLO's TF buffer still holds transforms from the previous session at higher timestamps, so TF complains for a few seconds until the new sim time catches up. Ignore these — do not Ctrl+C YOLO.

**Gazebo cleanup messages after trial 3**
Normal — the engine exits cleanly. Not an error.

**`final_error` NaN or very large in convert output**
CheatCode failed to get close to the port. Episode is still valid data unless you pass `--success_only`, which drops it.
