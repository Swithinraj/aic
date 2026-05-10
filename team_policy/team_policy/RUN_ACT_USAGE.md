# `run_act.py` — operator manual

Production-grade ACT inference for the AIC cable-insertion task. This file is
the single source of truth for **how to run, tune, and debug**
`team_policy.run_act` against any locally-trained ACT checkpoint.

The file `run_act.py` has the short code-level summary; this manual keeps the
operator commands and tuning notes.

---

## TL;DR — test `trained_model_V2` right now

Start with the local checks, then run the trial.

```bash
cd /home/zaid/ws_aic/src/aic/team_policy
pixi run python -m unittest discover -s test -p 'test_*.py' -v
pixi run python -m unittest test.test_trained_model_v2_smoke -v
```

The smoke test loads
`/home/zaid/ws_aic/src/aic/team_policy/team_policy/models/trained_model_V2/pretrained_model`,
builds a V2-shaped 30D observation plus three `480x640` images, and verifies
the ACT policy returns a finite 6D delta action.

Then use three terminals.

### 1. Simulation (GUI on, no ground-truth — inference mode)

```bash
distrobox enter -r aic_eval -- /entrypoint.sh \
    ground_truth:=false \
    start_aic_engine:=true \
    gazebo_gui:=true \
    launch_rviz:=true \
    model_discovery_timeout_seconds:=180 \
    model_configure_timeout_seconds:=120 \
    aic_engine_config_file:=$TRAIN_ROOT/configs/sessions/session_01.yaml
```

### 2. YOLO planner (publishes `/fused_yolo/detections_json`)

```bash
cd ~/ws_aic/src/aic && pixi run ros2 run team_policy combined_yolo_depth_pose_planner
```

### 3. ACT policy

```bash
cd ~/ws_aic/src/aic && pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=team_policy.run_act \
    -p checkpoint_path:=/home/zaid/ws_aic/src/aic/team_policy/team_policy/models/trained_model_V2/pretrained_model \
    -p replan_every:=10 \
    -p ema_alpha:=0.7 \
    -p max_translation_delta_m:=0.15 \
    -p max_rotation_delta_rad:=0.35
```

Then activate the model and send a goal:

```bash
ros2 lifecycle set /aic_model configure
ros2 lifecycle set /aic_model activate

ros2 action send_goal /insert_cable aic_task_interfaces/action/InsertCable \
  "{task: {id: 'planner_test', cable_type: 'sfp_sc', cable_name: 'cable_0',
            plug_type: 'sfp', plug_name: 'sfp_tip',
            port_type: 'sfp', port_name: 'sfp_port_0',
            target_module_name: 'nic_card_mount_0', time_limit: 1000}}" \
  --feedback
```

For `session_01.yaml`, the three built-in trials are NIC0/SFP, NIC2/SFP, then
SC0/SC. If you manually send goals instead of letting the engine dispatch the
trial task, update `target_module_name`, `port_type`, `port_name`, `cable_name`,
and `plug_name` for the active trial; otherwise trial 2/3 will look "off" even
when the policy code is doing what it was asked.

If you edited `run_act.py` since the last `pixi reinstall`, sync first:

```bash
install -m 644 ~/ws_aic/src/aic/team_policy/team_policy/run_act.py \
    ~/ws_aic/src/aic/.pixi/envs/default/lib/python3.12/site-packages/team_policy/run_act.py
```

Or do a package reinstall:

```bash
pixi reinstall ros-kilted-team-policy
```

---

## What you should see

Startup log (one-shot, on `configure`):

```text
RunACT loaded (clean hybrid rollback):
  path             = .../trained_model_V2/pretrained_model
  device           = cuda
  state_dim/schema = 30 / v2_30d_with_port_xyz
  action_dim       = 6
  chunk/actions    = 100 / 10
  replan_every     = 10
  delta_pose_scale = 1.0
  max_delta        = 0.15m / 0.35rad
  insert           = step 1.0mm, depth 40.0mm
```

During the action goal:

```text
=== PHASE 1: ACT approach (clean hybrid) ===
ACT step=   0 | tcp=(+0.1...) | Δ=(...) | F=0.4N | port_d=83.2mm | travel=0.000m
ACT step=  20 | ...                                             | port_d=42.1mm
...
Stall detected at step 173 (vel=0.21mm/step) — switching to insertion
=== PHASE 2: Force-guided insertion ===
Insertion direction: (+0.012,+0.018,+0.998)
INSERT step=  1 | ...
Contact detected at F=6.3N — continuing insertion
INSERT step= 30 | push=30.0mm | moved=27.3mm
Insertion depth reached: 40.0mm
Holding position for connector stabilization (3.0s)...
RunACT clean hybrid finished — ACT:174 + INSERT:42 in 24.7s
```

---

## Architecture

```
ros2 action /insert_cable
        │
        ▼
┌─────────────────────────────────────────┐
│              insert_cable()             │
│                                         │
│  _reset_task_target(task)               │  remembers port_name/port_type/module_name
│        │                                │  zeros _yolo_port_*
│        ▼                                │
│  Phase 1: _run_act_phase()              │
│    every 100 ms (sim time):             │
│      observe → batch → ACTPolicy        │
│      → unnormalize → EMA                │
│      → rotation_gain                    │
│      → translation/rotation clip        │
│      → delta_pose_scale                 │
│      → MotionUpdate(MODE_POSITION)      │
│      → stall check                      │
│      → pace to 10 Hz sim-time           │
│        │                                │
│        ▼ (stall, time, or hard-stop)    │
│  Phase 2: _run_insertion_phase()        │
│    pick insertion axis:                 │
│      held YOLO port vector              │
│      → recent_actions → gripper_z       │
│    every 100 ms:                        │
│      target_pos += dir * step_size      │
│      step_size shrinks under contact    │
│      MotionUpdate                       │
│    hold for hold_after_insert_s         │
│        │                                │
│        ▼                                │
│  return True                            │
└─────────────────────────────────────────┘
        │
        ▼
ROS action result
```

### YOLO subscription

`run_act.py` subscribes to `/fused_yolo/detections_json` (a `std_msgs/String`
publishing a JSON list of `{instance_name, class_name, confidence,
pose_base_link.position, ...}` dicts). For each message it picks the
**best** detection for the active task using `_target_match_rank`:

| Rank | Match                                              |
| ---- | -------------------------------------------------- |
| 0    | exact `port_name` or `target_module_name`          |
| 1    | port-type family (`sc_port*` for SC, `sfp_port*` for SFP) |
| 2    | substring overlap with `port_name`                 |
| 3    | substring overlap with `target_module_name`        |
| —    | no match → ignored                                 |

When no detection in the **current message** matches, the previously latched
`port_xyz` is preserved (hold-last). This matches `convert_to_lerobot.py`'s
training-time semantics and is essential for stable behaviour at close range
where YOLO often briefly drops.

The clean rollback does not steer from YOLO and does not use YOLO orientation.
The held `port_xyz` is only part of the 30D V2 observation state. Phase 2 picks
its insertion direction from the vector toward the held YOLO port when the TCP
is already close enough for that vector to be credible. If that is unavailable,
it falls back to recent ACT translation deltas, then the gripper local +Z axis.

---

## ROS parameters

All optional. Defaults shown after `=`. Defaults match the V2 training config.

### Required

| Param | Description |
| ----- | ----------- |
| `checkpoint_path` | Path to a `pretrained_model/` directory. May also point at the parent — the loader walks down. |

### Action shaping

| Param | Default | What it does |
| ----- | ------- | ------------ |
| `action_scale` | `1.0` | Global multiplier on the unnormalized 6-D delta. `<1.0` softens, `>1.0` accelerates. |
| `delta_pose_scale` | auto | Final pose-integration multiplier. Auto is `1.0` for V2 30D delta-pose checkpoints and `0.1` for legacy 33D velocity-style checkpoints. |
| `rotation_gain` | `1.0` | Multiplier applied **only** to action[3:6] (rotation). Set to `1.5–2.0` if the robot consistently lags in orientation correction relative to translation. |
| `max_translation_delta_m` | `0.150` | Per-step translation clip. Kept loose to avoid fighting the model. |
| `max_rotation_delta_rad` | `0.350` | Per-step rotation clip (~20°). |

### Smoothing

| Param | Default | What it does |
| ----- | ------- | ------------ |
| `ema_alpha` | `0.7` | EMA blend between the new action and the previous one. Larger = more responsive, less smooth. |
| `replan_every` | `10` | How many steps between forcibly clearing `_action_queue`. 10 (= 1 s sim-time) is closed-loop enough to react to YOLO updates. |

### Insertion phase

| Param | Default | What it does |
| ----- | ------- | ------------ |
| `insert_step_m` | `0.001` | Per-step push along the chosen insertion axis. |
| `insert_depth_m` | `0.040` | Total push budget. Phase 2 ends when this is reached. |
| `insert_force_thresh_n` | `5.0` | Force above which we log "in contact". |
| `hold_after_insert_s` | `3.0` | Hold time after the push, so the connector latches. |
| `prefer_port_axis_for_insertion` | `true` | During Phase 2 only, prefer the vector from TCP to held YOLO port if it is close enough. |
| `insert_port_axis_max_dist_m` | `0.180` | Max TCP-to-held-port distance for trusting that Phase 2 vector. |

### Time / safety budgets

| Param | Default | What it does |
| ----- | ------- | ------------ |
| `time_limit_s` | `150.0` | Total seconds for the whole goal (must be ≤ task.time_limit). |
| `act_timeout_s` | `80.0` | Max time in Phase 1. |
| `force_hard_stop_n` | `80.0` | Emergency stop on `‖F‖`. |
| `torque_hard_stop_nm` | `12.0` | Emergency stop on `‖τ‖`. |

### Schema overrides

| Param | Default | What it does |
| ----- | ------- | ------------ |
| `prev_action_in_state_override` | `auto` | `auto` uses V2 30-D for V2 checkpoints and legacy 33-D tcp-error layout for old checkpoints. |

---

## Symptom → tuning recipe

These are the small knobs worth touching before changing code again.

### Overshoot near the target

* First lower `max_translation_delta_m` from `0.15` to `0.10`.
* If still overshooting, lower `ema_alpha` from `0.7` to `0.5`.
* Avoid adding YOLO steering back into Phase 1 unless the model is genuinely
  moving in the wrong workspace direction for many consecutive trials.

### Jumping past the exact pose

Almost always caused by a YOLO drop near the port resetting `port_xyz` to
zeros mid-trial — except this version of the policy holds the last valid
detection, so this should not happen anymore. If you still see it:

1. Verify with `ros2 topic echo --once /fused_yolo/detections_json` that some
   detection is still being published while close.
2. If detections are missing entirely, inspect the YOLO planner; this policy
   no longer waits for YOLO or steers from YOLO, it only feeds held `port_xyz`
   to the V2 model.

### Stale motion near the target

Usually the ACT phase is not making measurable TCP progress after it has
already had time to approach. The rollback uses the old simple stall detector:
after `MIN_ACT_STEPS=100`, if the average movement over the recent window is
below `0.3 mm/step`, it switches to insertion. If that fires too early:

* Increase `_MIN_ACT_STEPS` in code if the model needs more approach time.
* Increase `_STALL_VEL_THRESH` only if you want Phase 2 to start sooner.
* Lower `act_timeout_s` to `40` when testing bad checkpoints so a trial fails
  fast instead of spending the full budget wandering.

### Translation correction stronger than orientation correction

Set `rotation_gain` to `1.5` first, `2.0` if needed. The action heads in
both V2 (loss 0.049) and chunk50 (loss 0.037) have ~3-4× smaller `std` for
rotation than translation, so unnormalized rotation deltas naturally come
out smaller. `rotation_gain` is the principled compensation.

---

## Troubleshooting

### `RunACT requires the 'checkpoint_path' ROS parameter.`

You forgot `-p checkpoint_path:=...`. The loader will accept either the
parent (`.../trained_model_V2`) or the child (`.../trained_model_V2/pretrained_model`).

### `Checkpoint folder is missing [...]`

The folder does not contain all four required files:

```text
config.json
model.safetensors
policy_preprocessor_step_3_normalizer_processor.safetensors
policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

If only the `_preprocessor`/`_postprocessor` files are missing, your training
run probably wrote them under different names. The model will still load but
state/action normalization will be wrong (you'll see logged warnings about
default values being used, and observed actions will be noisy).

### `Built observation.state with dim X, but checkpoint expects Y`

The schema autodetector picked the wrong layout. Override:

```bash
-p prev_action_in_state_override:=true     # force 33D chunk50 layout
-p prev_action_in_state_override:=false    # force 30D V2 layout
```

### Robot does nothing for several seconds, then jumps

Sim time is paused or running very slowly. The new pacing loop has a
1.0-second wall-clock cap precisely to avoid spinning forever in this case.
Check Gazebo: it may be CPU-starved by the YOLO process. Try
`gazebo_gui:=false` or move the YOLO planner to a different machine.

### Hard-stop fires immediately

`force_hard_stop_n=80` and `torque_hard_stop_nm=12` are conservative defaults.
On the first run after Gazebo restart, the wrist sensor may not be tared —
you'll see startup forces of 15–25 N. If the policy then commands a small
move, |F| can exceed 30 N immediately. Two options:

1. Wait for tare. The `aic_engine` controller usually tares on `activate`.
2. Loosen `force_hard_stop_n` to `120` (only if you trust the gripper not to
   crash).

### Success is reported but the cable visibly didn't seat

This clean rollback returns `True` after the hybrid attempt, matching the older
working policy. Judge success visually/with the simulator score while tuning
approach and insertion separately.

---

## Native iteration loop

```bash
# 1. Edit team_policy/team_policy/run_act.py
# 2. Sync to pixi env (fast)
install -m 644 ~/ws_aic/src/aic/team_policy/team_policy/run_act.py \
   ~/ws_aic/src/aic/.pixi/envs/default/lib/python3.12/site-packages/team_policy/run_act.py

# 3. Run unit tests
cd ~/ws_aic/src/aic/team_policy
~/ws_aic/src/aic/.pixi/envs/default/bin/python3 -m pytest test/test_run_act.py -v

# 4. Re-launch the policy node (terminal 3 from the TL;DR)
```

For a full rebuild (when you've changed `setup.py` or `package.xml`):

```bash
cd ~/ws_aic/src/aic
pixi reinstall ros-kilted-team-policy
```

---

## Swapping in the loss=0.037 model later

When you have access to the better-loss checkpoint:

1. Drop it under `team_policy/team_policy/models/<name>/pretrained_model/`.
2. Inspect its `config.json` to read `state_dim`. The current code handles
   30 (V2) and 33 (chunk50-style with prev-action feedback) automatically.
3. Run the same launch command, just changing `checkpoint_path:=`.
4. The startup log will print `schema = ...` — verify it matches what your
   training pipeline assembled.
5. If `state_dim` is something we haven't seen before, the loader will raise
   a clear `ValueError`. Add the new layout in `_build_state` and a
   detection rule in `detect_state_schema`, and add a test for it. Both are
   small, mechanical changes.

---

## Tests

```bash
cd ~/ws_aic/src/aic/team_policy
pixi run python -m unittest discover -s test -p 'test_*.py' -v
```

Coverage:

* `TestCheckpointResolution` — parent vs `pretrained_model` path resolution.
* `TestSchemaDetection` — V2 30D, legacy 33D, unknown dim.
* `TestStateBuilding` — V2 30D layout with held `port_xyz`, legacy 33D tcp-error layout.
* `TestYoloCallback` — hold-last semantics and SFP aliases.
* `TestActionAndImages` — translation/rotation clipping, NaN/inf cleanup, V2/legacy delta scale, BGR to RGB.
* `TestInsertionAxis` — held-port vector, recent-action insertion axis, and gripper-Z fallback.
* `TestTrainedModelV2Smoke` — loads the real V2 checkpoint and verifies one finite 6D inference action from a correctly shaped synthetic observation.

All tests should pass locally; the smoke test takes a few seconds because it
loads the ACT weights.
