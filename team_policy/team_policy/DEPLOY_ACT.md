# ACT Model Deployment Runbook

## What This Does
Deploys the V2 trained ACT (Action Chunking with Transformers) model for cable insertion in the AIC competition simulation. The model runs at 10Hz, taking camera images + robot state + the held YOLO port estimate as input and outputting 6D delta-TCP actions.

## Architecture
```
Container: aic_eval (host network mode, distrobox)
  ├── Zenoh Router (rmw_zenohd) on port 7447
  ├── Gazebo Simulation (headless)
  ├── aic_engine (trial orchestrator)
  ├── aic_adapter (ROS bridges)
  ├── combined_yolo_depth_pose_planner (publishes /fused_yolo/detections_json)
  └── aic_model (pixi env, loads team_policy.run_act → RunACT on CUDA)
```

## CRITICAL: Startup Order Matters

The aic_engine has a discovery timeout. If the model isn't fully loaded when the engine tries to query its lifecycle state, the service call times out and the engine declares failure. **The model MUST be loaded before the engine starts looking.**

### Step-by-Step Launch

```bash
# === STEP 1: Kill any existing processes ===
docker exec aic_eval bash -c "
  ps aux | grep -E 'ros2|gz|aic_|zenoh|component|robot_state|create|spawner|pixi' \
    | grep -v grep | grep -v gnome | awk '{print \$2}' \
    | xargs -r kill -9 2>/dev/null
"
sleep 2

# === STEP 2: Start Zenoh Router ===
docker exec -d aic_eval bash -c '
source /ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
ZENOH_CONFIG_OVERRIDE="mode=\"router\";listen/endpoints=[\"tcp/[::]:7447\"];connect/endpoints=[];routing/router/peers_failover_brokering=true;transport/shared_memory/enabled=false"
export ZENOH_CONFIG_OVERRIDE
export ZENOH_ROUTER_CONFIG_URI=/aic_zenoh_config.json5
ros2 run rmw_zenoh_cpp rmw_zenohd > /tmp/zenoh_router.log 2>&1
'
sleep 2

# === STEP 3: Start YOLO planner for V2 port_xyz state ===
docker exec -d aic_eval bash -c '
cd /home/zaid/ws_aic/src/aic
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE=";transport/shared_memory/enabled=false"
/home/zaid/.pixi/bin/pixi run ros2 run team_policy combined_yolo_depth_pose_planner \
    > /tmp/aic_yolo_planner.log 2>&1
'
sleep 2

# === STEP 4: Start ACT Model (pre-load before engine) ===
docker exec -d aic_eval bash -c '
cd /home/zaid/ws_aic/src/aic
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE=";transport/shared_memory/enabled=false"
/home/zaid/.pixi/bin/pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=team_policy.run_act \
    -p checkpoint_path:=/home/zaid/ws_aic/src/aic/team_policy/team_policy/models/trained_model_V2/pretrained_model \
    > /tmp/aic_model_inside.log 2>&1
'

# Wait for model to finish loading (check for "Using policy: run_act")
for i in $(seq 1 30); do
    if docker exec aic_eval grep -q "Using policy: run_act" /tmp/aic_model_inside.log 2>/dev/null; then
        echo "Model loaded after ${i}s"
        break
    fi
    sleep 1
done

# Verify model is responsive
docker exec aic_eval bash -c "
  source /ws_aic/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp
  ros2 lifecycle get /aic_model
"
# Should print: "unconfigured [1]"

# === STEP 5: Start Simulation + Engine ===
docker exec -d aic_eval bash -c '
source /ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export DISPLAY=:0
export ZENOH_CONFIG_OVERRIDE=";transport/shared_memory/enabled=false"
ros2 launch aic_bringup aic_gz_bringup.launch.py \
    ground_truth:=false \
    start_aic_engine:=true \
    aic_engine_config_file:=/home/zaid/ws_aic/src/aic/team_policy/team_policy/training_robot/configs/sessions/session_01.yaml \
    model_discovery_timeout_seconds:=180 \
    model_configure_timeout_seconds:=120 \
    gazebo_gui:=false \
    launch_rviz:=false \
    > /tmp/aic_sim_launch.log 2>&1
'
echo "Simulation launched. Engine will discover model, configure, activate, and run trials."
```

## Monitoring

```bash
# Watch model execution
docker exec aic_eval tail -f /tmp/aic_model_inside.log

# Watch YOLO port estimates
docker exec aic_eval tail -f /tmp/aic_yolo_planner.log

# Watch engine progress
docker exec aic_eval tail -f /tmp/aic_sim_launch.log

# Check scoring after completion
cat /home/zaid/aic_results/scoring.yaml
```

## Key Engine Log Lines to Watch For
- `Found 1 node(s) with name 'aic_model'` — engine discovered model
- `on_configure(...)` — model being configured (loads PyTorch weights)
- `RunACT loaded: ... device = cuda` — model on GPU, ready
- `on_activate()` — model activated
- `Sending InsertCable goal` — trial starting
- `Task [task_1] succeeded` — plug inserted
- `✓ All Tasks Completed` — trial done
- `Complete Scoring Results` — all trials finished

## Session Config
The session YAML at `training_robot/configs/sessions/session_01.yaml` defines:
- **Trial 1**: SFP plug → NIC card 0 (NIC rail 0)
- **Trial 2**: SFP plug → NIC card 2 (NIC rail 2)
- **Trial 3**: SC plug → SC port 0 (SC rail 0) — cable is reversed

Board pose: x=0.160, y=-0.180, z=1.14, yaw=2.80

## Results from April 28, 2026 Run
```
Total Score: 15.21
Trial 1: Score 1.0   (model validated, plug not inserted — 0.15m from port)
Trial 2: Score 25.21 (model validated, good trajectory, plug 0.13m from port)
Trial 3: Score -11.0 (model validated, force penalty -12, plug 0.07m from port)
```

## Known Issues
1. **Trial 1 no insertion**: The model moves but doesn't reach the port (0.15m away). May need the model to run longer or the 60s TIME_LIMIT_S in run_act.py is too short.
2. **Trial 3 force penalty**: SC port insertion with reversed cable hits 464N force (limit is 20N for 1s). The trained model may not have enough SC port training data.
3. **No actual insertions detected**: All three trials show "No insertion detected" in tier_3. The plug gets close but doesn't physically insert.

## Model Details
- **Checkpoint**: `models/trained_model_V2/pretrained_model/`
- **Architecture**: ACT (Action Chunking with Transformers) via LeRobot
- **Input**: 3 cameras (left/center/right) + 30D state vector
- **State**: `tcp_pose(7) + tcp_velocity(6) + joint_positions(7) + joint_velocity(7) + port_xyz_in_base(3)`
- **YOLO behavior**: `[0,0,0]` before the first matching port detection, then hold the last valid matching `/fused_yolo/detections_json` port estimate
- **Output**: 6D delta-TCP `(dx,dy,dz,drx,dry,drz)` at 10Hz, not velocity
- **Safety**: 5cm max translation delta, 0.35rad max rotation delta per step, 80N force hard stop, 12Nm torque hard stop
- **Impedance**: Stiffness diag(90,90,90,50,50,50), Damping diag(50,50,50,20,20,20)

## Quick Validation

```bash
cd /home/zaid/ws_aic/src/aic/team_policy
pixi run python -m unittest discover -s test -p 'test_*.py'
```

Expected:

```text
Ran 5 tests
OK
```
