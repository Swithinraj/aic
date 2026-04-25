#!/usr/bin/env bash
# Data collection loop: sessions 95-100 (runs 095-100)
# Uses docker exec on the aic_eval container (ghcr.io/intrinsic-dev/aic/aic_eval)
# which is equivalent to distrobox enter -r aic_eval but without sudo.
set -euo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
AIC_ROOT=$(git -C "$(dirname "$(realpath "$0")")" rev-parse --show-toplevel)
TRAIN_ROOT="$AIC_ROOT/team_policy/team_policy/training_robot"
SESSIONS_DIR="$TRAIN_ROOT/configs/sessions"
EPISODES_DIR="$TRAIN_ROOT/episodes"
LOG_FILE="$TRAIN_ROOT/collection_log_resume_095.txt"
export FASTRTPS_DEFAULT_PROFILES_FILE="$AIC_ROOT/team_policy/fastdds_no_shm.xml"
DOCKER_CONTAINER="aic_eval"

mkdir -p "$EPISODES_DIR"

# Redirect stdout+stderr to log AND terminal
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "========================================================"
log "AIC CheatCode Collection — Resume from session_95"
log "AIC_ROOT   = $AIC_ROOT"
log "TRAIN_ROOT = $TRAIN_ROOT"
log "FASTRTPS   = $FASTRTPS_DEFAULT_PROFILES_FILE"
log "CONTAINER  = $DOCKER_CONTAINER"
log "========================================================"

# ---------------------------------------------------------------------------
# Ensure session YAMLs exist (95-100)
# ---------------------------------------------------------------------------
MISSING=0
for i in 95 96 97 98 99 100; do
    SFILE="$SESSIONS_DIR/session_${i}.yaml"
    if [[ ! -f "$SFILE" ]]; then
        log "WARNING: $SFILE missing — regenerating sessions..."
        MISSING=1
        break
    fi
done
if [[ $MISSING -eq 1 ]]; then
    log "CMD: python3 $TRAIN_ROOT/configs/generate_competition_sessions.py --sessions 100"
    python3 "$TRAIN_ROOT/configs/generate_competition_sessions.py" --sessions 100
    log "Session regeneration complete."
fi

# ---------------------------------------------------------------------------
# Ensure aic_eval Docker container is running
# ---------------------------------------------------------------------------
ensure_container_running() {
    local status
    status=$(docker inspect -f '{{.State.Status}}' "$DOCKER_CONTAINER" 2>/dev/null || echo "missing")
    if [[ "$status" != "running" ]]; then
        log "  Container $DOCKER_CONTAINER is '$status' — starting..."
        docker start "$DOCKER_CONTAINER" > /dev/null
        sleep 3
        log "  Container started."
    else
        log "  Container $DOCKER_CONTAINER already running."
    fi
}

# ---------------------------------------------------------------------------
# Helper: stop simulation inside container
# ---------------------------------------------------------------------------
kill_sim() {
    log "Stopping simulation processes..."
    docker exec "$DOCKER_CONTAINER" bash -c "
        pkill -f 'gz sim'        2>/dev/null || true
        pkill -f 'gzserver'      2>/dev/null || true
        pkill -f 'ros2.*launch'  2>/dev/null || true
        pkill -f 'aic_engine'    2>/dev/null || true
        pkill -f 'rmw_zenohd'    2>/dev/null || true
        true
    " 2>/dev/null || true
    pkill -f "docker exec.*aic_eval" 2>/dev/null || true
    sleep 4
    log "Simulation stopped."
}

# ---------------------------------------------------------------------------
# Summary counters
# ---------------------------------------------------------------------------
TOTAL_SESSIONS=0
TOTAL_EPISODES=0
FAILED_RUNS=()

# ---------------------------------------------------------------------------
# Main loop: sessions 95 to 100
# ---------------------------------------------------------------------------
for SESSION_NUM in 95 96 97 98 99 100; do

    RUN_ID=$(printf "run_%03d" $SESSION_NUM)
    SESSION_FILE="$SESSIONS_DIR/session_${SESSION_NUM}.yaml"
    RUN_DIR="$EPISODES_DIR/$RUN_ID"

    log "--------------------------------------------------------"
    log "SESSION $SESSION_NUM | run=$RUN_ID | file=$(basename "$SESSION_FILE")"
    log "  output_dir = $RUN_DIR"

    if [[ ! -f "$SESSION_FILE" ]]; then
        log "ERROR: $SESSION_FILE not found — skipping."
        FAILED_RUNS+=("$RUN_ID:no_session_file")
        continue
    fi

    # Skip if already complete
    if [[ -d "$RUN_DIR" ]]; then
        N=$(find "$RUN_DIR" -name "episode_*.hdf5" 2>/dev/null | wc -l)
        if [[ $N -ge 3 ]]; then
            log "  SKIP: $RUN_ID already has $N episodes."
            TOTAL_SESSIONS=$((TOTAL_SESSIONS + 1))
            TOTAL_EPISODES=$((TOTAL_EPISODES + N))
            continue
        fi
    fi

    mkdir -p "$RUN_DIR"

    # ------------------------------------------------------------------
    # STEP A — Start simulation inside Docker container (background)
    # ------------------------------------------------------------------
    ensure_container_running

    log "STEP A: Launching simulation for session_${SESSION_NUM}..."
    SIM_CMD="/entrypoint.sh \
        ground_truth:=true \
        start_aic_engine:=true \
        launch_rviz:=false \
        gazebo_gui:=false \
        aic_engine_config_file:=$SESSION_FILE"
    log "CMD: docker exec $DOCKER_CONTAINER $SIM_CMD"

    docker exec "$DOCKER_CONTAINER" bash -c "$SIM_CMD" \
        > "$RUN_DIR/sim.log" 2>&1 &
    SIM_PID=$!
    log "  docker exec PID=$SIM_PID — waiting 30s for simulation startup..."
    sleep 30

    # Check sim is still alive
    if ! kill -0 "$SIM_PID" 2>/dev/null; then
        log "ERROR: Simulation process (PID=$SIM_PID) died early. Sim log tail:"
        tail -20 "$RUN_DIR/sim.log" | while read -r line; do log "  SIM: $line"; done
        FAILED_RUNS+=("$RUN_ID:sim_died")
        kill_sim
        continue
    fi
    log "  Simulation running (PID=$SIM_PID)."

    # ------------------------------------------------------------------
    # STEP B — Run CheatCode collector inside Pixi shell
    # ------------------------------------------------------------------
    log "STEP B: Running CheatCode collector (pixi shell)..."
    COLLECTOR_CMD="ros2 run aic_model aic_model --ros-args \
        -p use_sim_time:=true \
        -p policy:=team_policy.training_robot.cheatcode_collector \
        -p output_dir:=$RUN_DIR \
        -p num_episodes:=3 \
        -p success_only:=true"
    log "CMD (pixi): $COLLECTOR_CMD"

    cd "$AIC_ROOT"
    pixi shell -c "$COLLECTOR_CMD" \
        > "$RUN_DIR/collector.log" 2>&1
    COLLECTOR_EXIT=$?
    log "  Collector exited with code $COLLECTOR_EXIT."

    if [[ $COLLECTOR_EXIT -ne 0 ]]; then
        log "  WARNING: Collector returned non-zero exit ($COLLECTOR_EXIT). Collector log tail:"
        tail -10 "$RUN_DIR/collector.log" | while read -r line; do log "  COL: $line"; done
    fi

    # ------------------------------------------------------------------
    # STEP C — Stop simulation
    # ------------------------------------------------------------------
    log "STEP C: Stopping simulation..."
    kill_sim
    kill "$SIM_PID" 2>/dev/null || true
    wait "$SIM_PID" 2>/dev/null || true

    # ------------------------------------------------------------------
    # STEP D — Verify output
    # ------------------------------------------------------------------
    log "STEP D: Verifying episodes in $RUN_DIR..."
    N_EPISODES=$(ls "$RUN_DIR"/episode_*.hdf5 2>/dev/null | wc -l)
    log "  Found $N_EPISODES episode(s)."
    ls "$RUN_DIR"/episode_*.hdf5 2>/dev/null | while read -r f; do
        log "    $(basename "$f") — $(du -h "$f" | cut -f1)"
    done

    TOTAL_SESSIONS=$((TOTAL_SESSIONS + 1))
    TOTAL_EPISODES=$((TOTAL_EPISODES + N_EPISODES))

    if [[ $N_EPISODES -lt 3 ]]; then
        log "  WARNING: Expected 3 episodes, got $N_EPISODES for $RUN_ID."
        FAILED_RUNS+=("$RUN_ID:only_${N_EPISODES}_episodes")
    else
        log "  OK: $RUN_ID complete — $N_EPISODES episodes saved."
    fi

    # Pause between sessions to let Docker settle
    sleep 5
done

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
log "========================================================"
log "COLLECTION COMPLETE"
log "  Total sessions executed : $TOTAL_SESSIONS"
log "  Total episodes collected: $TOTAL_EPISODES"
if [[ ${#FAILED_RUNS[@]} -eq 0 ]]; then
    log "  Failed runs             : none"
else
    log "  Failed runs             : ${FAILED_RUNS[*]}"
fi
log "  Log: $LOG_FILE"
log "========================================================"
