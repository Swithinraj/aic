#!/usr/bin/env bash
# =============================================================================
# aic_collect_v2.sh — Automated multi-trial episode collection (v2 pipeline)
# =============================================================================
#
# Uses cheatcode_collector_v2 which records Schema v9 episodes:
#   - Per-camera YOLO features (7D × 3 cameras)
#   - Tared wrist force/torque (6D)
#   - Held fused YOLO port_xyz (3D)
#   - Fresh fused yolo_valid (1D)
#   - Fused yolo_age staleness (1D)
#   - target_module_onehot (7D)
# Episodes are saved to EPISODES_DIR (Seagate hard drive by default).
#
# HOW IT WORKS
# ------------
# Drives two tmux panes in parallel:
#   Pane 0 (left)  — distrobox / aic_engine  (Terminal 1)
#   Pane 1 (right) — pixi / aic_model        (Terminal 2)
#
# For each session YAML it:
#   1. Launches aic_engine with the session YAML in pane 0
#   2. Waits until aic_engine emits the "Retrying…" ready signal
#   3. Launches cheatcode_collector_v2 in pane 1
#   4. Waits for "process has finished cleanly" (all trials done)
#   5. Sends Ctrl+C to both panes, kills lingering processes,
#      waits for Zenoh port 7447 to clear, then starts the next session
#
# Score lines are parsed and appended to ~/aic_scores.csv.
#
# PREREQUISITES
# -------------
#   sudo apt install tmux
#   ~/.bashrc must export:
#     AIC_ROOT=/home/$USER/ros2_ws/src/aic
#     EPISODES_DIR=/media/$USER/seagate/aic_episodes
#   distrobox container 'aic_eval' must be running
#
# USAGE
# -----
#   bash aic_collect_v2.sh
#       # all sessions in default configs/sessions
#   bash aic_collect_v2.sh --sessions-dir sessions_nic_nic_sc
#       # all sessions in configs/sessions_nic_nic_sc
#   bash aic_collect_v2.sh --sessions-dir sessions_nic_nic_sc session_01.yaml
#       # single named session file from that directory
#   bash aic_collect_v2.sh 3
#       # sessions 03 … last from the default directory
#   bash aic_collect_v2.sh --sessions-dir sessions_nic_nic_sc 1 50
#       # sessions 01 … 50 from configs/sessions_nic_nic_sc
#
# Monitor:
#   tmux attach -t aic_collect_v2
#
# =============================================================================

set -uo pipefail
trap 'echo "[ERROR] Script failed at line $LINENO: $BASH_COMMAND" >&2' ERR

# =============================================================================
# CONFIG — edit these paths if your workspace layout changes
# =============================================================================

AIC_ROOT_DEFAULT="/home/${USER}/official_aic/aic"
EPISODES_DIR_DEFAULT="/mnt/seagate/intrinsic_swithin"
CONFIGS_ROOT_DEFAULT="${AIC_ROOT_DEFAULT}/team_policy/team_policy/training_robot/configs"
DEFAULT_SESSIONS_SUBDIR="sessions"

SESSIONS_DIR="${CONFIGS_ROOT_DEFAULT}/${DEFAULT_SESSIONS_SUBDIR}"
DISTROBOX_CONTAINER="aic_eval"
TMUX_SESSION="aic_collect_v2"

# Collector policy (v2)
COLLECTOR_POLICY="team_policy.training_robot.cheatcode_collector_v2"

# Log patterns
TRIGGER_READY="aic_engine-[0-9]+.*No node with name 'aic_model' found. Retrying"
TRIGGER_DONE="aic_engine-[0-9]+\]: process has finished cleanly"

# Timeouts (seconds)
READY_TIMEOUT=180
DONE_TIMEOUT=900     # 3 trials × ~3 min + margin
READY_POLL=2
DONE_POLL=3
POST_CTRLC_WAIT=20   # Zenoh needs ~15 s to release port 7447

STATUS_DIR="/tmp/aic_collect_v2_status"
SCRIPT_DIR="/tmp/aic_collect_v2_scripts"
SCORES_CSV="${HOME}/aic_scores_v2.csv"

# =============================================================================
# PREFLIGHT
# =============================================================================

usage() {
    cat <<EOF
Usage:
  bash aic_collect_v2.sh
  bash aic_collect_v2.sh --sessions-dir sessions_nic_nic_sc
  bash aic_collect_v2.sh --sessions-dir sessions_nic_nic_sc 1 50
  bash aic_collect_v2.sh --sessions-dir /abs/path/to/session_dir session_01.yaml

Positional modes:
  (no args)          -> all *.yaml files in the selected sessions dir
  (single .yaml arg) -> just that one file
  (1 or 2 integers)  -> range of session_NN.yaml files

Options:
  --sessions-dir DIR -> session directory name under configs/ or absolute path
  -h, --help         -> show this help
EOF
}

SESSIONS_DIR_OVERRIDE=""
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sessions-dir)
            [[ $# -ge 2 ]] || { echo "[ERROR] --sessions-dir requires a value."; exit 1; }
            SESSIONS_DIR_OVERRIDE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL_ARGS[@]}"

if [[ -n "$SESSIONS_DIR_OVERRIDE" ]]; then
    if [[ -d "$SESSIONS_DIR_OVERRIDE" ]]; then
        SESSIONS_DIR="$SESSIONS_DIR_OVERRIDE"
    elif [[ -d "${CONFIGS_ROOT_DEFAULT}/${SESSIONS_DIR_OVERRIDE}" ]]; then
        SESSIONS_DIR="${CONFIGS_ROOT_DEFAULT}/${SESSIONS_DIR_OVERRIDE}"
    else
        echo "[ERROR] Sessions dir not found: $SESSIONS_DIR_OVERRIDE"
        echo "        Tried: $SESSIONS_DIR_OVERRIDE"
        echo "               ${CONFIGS_ROOT_DEFAULT}/${SESSIONS_DIR_OVERRIDE}"
        exit 1
    fi
fi

if ! command -v tmux &>/dev/null; then
    echo "[ERROR] tmux not installed: sudo apt install tmux"; exit 1
fi
if [[ ! -d "$SESSIONS_DIR" ]]; then
    echo "[ERROR] Sessions dir not found: $SESSIONS_DIR"; exit 1
fi

# =============================================================================
# ARGUMENT PARSING
# Three modes:
#   (no args)          → all *.yaml files sorted alphabetically
#   (single .yaml arg) → just that one file
#   (1 or 2 integers)  → range of session_NN.yaml files
# =============================================================================

YAML_FILES=()

if [[ $# -eq 1 && "$1" == *.yaml ]]; then
    # Named file
    if [[ ! -f "$SESSIONS_DIR/$1" ]]; then
        echo "[ERROR] File not found: $SESSIONS_DIR/$1"; exit 1
    fi
    YAML_FILES=("$1")
elif [[ $# -ge 1 && "$1" =~ ^[0-9]+$ ]]; then
    # Numeric range
    TOTAL=$(ls "$SESSIONS_DIR"/*.yaml 2>/dev/null | wc -l)
    START=$1
    END=${2:-$TOTAL}
    if (( START < 1 || END > TOTAL || START > END )); then
        echo "[ERROR] Range $START–$END invalid (total: $TOTAL)"; exit 1
    fi
    for i in $(seq "$START" "$END"); do
        YAML_FILES+=("$(printf 'session_%03d.yaml' "$i")")
    done
else
    # All yaml files
    while IFS= read -r f; do
        YAML_FILES+=("$(basename "$f")")
    done < <(ls -1 "$SESSIONS_DIR"/*.yaml 2>/dev/null | sort)
fi

if (( ${#YAML_FILES[@]} == 0 )); then
    echo "[ERROR] No session files selected."; exit 1
fi

mkdir -p "$STATUS_DIR" "$SCRIPT_DIR"

if [[ ! -f "$SCORES_CSV" ]]; then
    echo "run_id,trial_number,score" > "$SCORES_CSV"
    echo "Score log created: $SCORES_CSV"
else
    echo "Appending scores to: $SCORES_CSV"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  AIC Episode Collector v2"
echo "  Sessions dir   : $SESSIONS_DIR"
echo "  Episodes dir   : ${EPISODES_DIR_DEFAULT}"
echo "  Policy         : $COLLECTOR_POLICY"
echo "  Sessions queued: ${#YAML_FILES[@]}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# =============================================================================
# TMUX SETUP
# =============================================================================

tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
T1_PANE=$(tmux new-session  -d -s "$TMUX_SESSION" -x 240 -y 55 -P -F "#{pane_id}")
T2_PANE=$(tmux split-window -h -t "$T1_PANE" -P -F "#{pane_id}")
tmux select-pane  -t "$T1_PANE"
echo "T1 pane: $T1_PANE  T2 pane: $T2_PANE"

echo "Monitor with: tmux attach -t $TMUX_SESSION"
echo ""

# =============================================================================
# HELPERS
# =============================================================================

wait_for() {
    local file="$1" pattern="$2" timeout="$3" poll="$4"
    local elapsed=0
    while ! grep -qE "$pattern" "$file" 2>/dev/null; do
        sleep "$poll"; elapsed=$(( elapsed + poll ))
        (( elapsed >= timeout )) && return 1
    done; return 0
}

ts() { date "+[%H:%M:%S]"; }

AIC_RESULTS_DIR="${HOME}/aic_results"

clear_aic_results() {
    if [[ ! -d "$AIC_RESULTS_DIR" ]]; then return 0; fi
    local bags; bags=$(find "$AIC_RESULTS_DIR" -maxdepth 1 -name "bag_trial_*" -type d 2>/dev/null)
    if [[ -n "$bags" ]]; then
        echo "$(ts) Clearing rosbags from $AIC_RESULTS_DIR ..."
        find "$AIC_RESULTS_DIR" -maxdepth 1 -name "bag_trial_*" -type d -exec rm -rf {} + 2>/dev/null || true
        echo "$(ts) Rosbags cleared."
    fi
}

kill_lingering() {
    local patterns="component_container|aic_engine|ros_gz|aic_adapter|robot_state_publisher|zenoh"
    local pids; pids=$(pgrep -f "$patterns" 2>/dev/null) || pids=""
    if [[ -n "$pids" ]]; then
        echo "$(ts) SIGINT to lingering: $(echo $pids | tr '\n' ' ')"
        kill -SIGINT $pids 2>/dev/null || true; sleep 3
        pids=$(pgrep -f "$patterns" 2>/dev/null) || pids=""
        [[ -n "$pids" ]] && { kill -SIGKILL $pids 2>/dev/null || true; }
    fi
    local model_pids; model_pids=$(pgrep -f "aic_model" 2>/dev/null) || model_pids=""
    [[ -n "$model_pids" ]] && kill -SIGKILL $model_pids 2>/dev/null || true
}

wait_for_port() {
    local port=7447 timeout=40 elapsed=0
    echo "$(ts) Waiting for Zenoh port ${port} to clear..."
    while ss -tlnp 2>/dev/null | grep -q ":${port}"; do
        sleep 2; elapsed=$(( elapsed + 2 ))
        (( elapsed >= timeout )) && {
            echo "$(ts) [WARN] Port still occupied after ${timeout}s. Proceeding."; return 0; }
    done; echo "$(ts) Port ${port} clear."
}

log_scores() {
    local status_file="$1" run="$2"
    local score_lines; score_lines=$(grep -E "Score:" "$status_file" 2>/dev/null) || score_lines=""
    [[ -z "$score_lines" ]] && { echo "$(ts) [WARN] No score lines for ${run}."; return 0; }
    local found=0
    while IFS= read -r line; do
        local trial score
        trial=$(echo "$line" | grep -oE "trial_[0-9]+" | grep -oE "[0-9]+") || trial=""
        score=$(echo "$line" | grep -oE "Score: [0-9.]+" | grep -oE "[0-9.]+") || score=""
        [[ -n "$trial" && -n "$score" ]] && {
            echo "${run},${trial},${score}" >> "$SCORES_CSV"
            echo "$(ts) Score: run=${run} trial=${trial} score=${score}"
            found=$(( found + 1 ))
        }
    done <<< "$score_lines"
    (( found == 0 )) && echo "$(ts) [WARN] No scores parsed for ${run}."
    return 0
}

# =============================================================================
# MAIN LOOP
# =============================================================================

FAILED=()

for idx in "${!YAML_FILES[@]}"; do
    SESSION="${YAML_FILES[$idx]}"
    # Derive RUN name: strip .yaml, replace dots with underscores, prefix run_
    RUN="run_$(basename "$SESSION" .yaml)"
    T1_STATUS="$STATUS_DIR/${SESSION%.yaml}.status"
    T1_SCRIPT="$SCRIPT_DIR/t1_${SESSION%.yaml}.sh"
    T2_SCRIPT="$SCRIPT_DIR/t2_${SESSION%.yaml}.sh"

    echo "══════════════════════════════════════════════════"
    echo "  [$(( idx+1 ))/${#YAML_FILES[@]}]  $SESSION  →  $RUN"
    echo "══════════════════════════════════════════════════"

    > "$T1_STATUS"

    # T1: launch aic_engine (absolute paths only — distrobox ignores caller env vars)
    cat > "$T1_SCRIPT" << EOF
#!/usr/bin/env bash
distrobox enter -r ${DISTROBOX_CONTAINER} \\
    -a "--env __NV_PRIME_RENDER_OFFLOAD=1 --env __GLX_VENDOR_LIBRARY_NAME=nvidia" \\
    -- /entrypoint.sh \\
    ground_truth:=true \\
    start_aic_engine:=true \\
    gazebo_gui:=false \\
    launch_rviz:=false \\
    aic_engine_config_file:=${SESSIONS_DIR}/${SESSION} \\
    2>&1 | tee >(grep --line-buffered -E \\
        "aic_engine-[0-9]+.*Retrying|aic_engine-[0-9]+\]: process has finished cleanly|Score:" \\
        >> "${T1_STATUS}")
EOF
    chmod +x "$T1_SCRIPT"

    # T2: launch v2 collector
    # \${AIC_ROOT} and \${EPISODES_DIR} expand in the PANE shell (from ~/.bashrc)
    # ${RUN} expands here from this outer script
    cat > "$T2_SCRIPT" << EOF
#!/usr/bin/env bash
AIC_ROOT="\${AIC_ROOT:-${AIC_ROOT_DEFAULT}}"
EPISODES_DIR="\${EPISODES_DIR:-${EPISODES_DIR_DEFAULT}}"
if [[ ! -d "\$AIC_ROOT" ]]; then
    echo "[ERROR] AIC_ROOT not found: \$AIC_ROOT"; exit 1
fi
mkdir -p "\${EPISODES_DIR}/${RUN}"
cd "\$AIC_ROOT" && pixi run ros2 run aic_model aic_model --ros-args \\
    -p use_sim_time:=true \\
    -p policy:=${COLLECTOR_POLICY} \\
    -p output_dir:="\${EPISODES_DIR}/${RUN}"
EOF
    chmod +x "$T2_SCRIPT"

    # ── Launch T1 ──────────────────────────────────────────────────────
    tmux send-keys -t "$T1_PANE" "bash ${T1_SCRIPT}" Enter
    echo "$(ts) T1 started. Waiting for aic_engine ready (timeout: ${READY_TIMEOUT}s)..."

    if ! wait_for "$T1_STATUS" "$TRIGGER_READY" "$READY_TIMEOUT" "$READY_POLL"; then
        echo "$(ts) [ERROR] aic_engine never became ready. Skipping $SESSION."
        FAILED+=("$SESSION (aic_engine never ready)")
        tmux send-keys -t "$T1_PANE" C-c
        sleep 5; continue
    fi
    echo "$(ts) aic_engine ready."

    # ── Launch T2 ──────────────────────────────────────────────────────
    tmux send-keys -t "$T2_PANE" "bash ${T2_SCRIPT}" Enter
    echo "$(ts) T2 started. Collecting (timeout: ${DONE_TIMEOUT}s)..."

    if ! wait_for "$T1_STATUS" "$TRIGGER_DONE" "$DONE_TIMEOUT" "$DONE_POLL"; then
        echo "$(ts) [WARN] No clean-exit signal within ${DONE_TIMEOUT}s."
        FAILED+=("$SESSION (timeout)")
    else
        echo "$(ts) Clean exit."
        log_scores "$T1_STATUS" "$RUN"
    fi

    # ── Shutdown ───────────────────────────────────────────────────────
    echo "$(ts) Ctrl+C → T2 (collector)..."
    tmux send-keys -t "$T2_PANE" C-c
    sleep 2
    echo "$(ts) Ctrl+C → T1 (aic_engine)..."
    tmux send-keys -t "$T1_PANE" C-c
    sleep 3
    kill_lingering
    wait_for_port
    clear_aic_results
    sleep "$POST_CTRLC_WAIT"
    echo "$(ts) Session done: $SESSION"
    echo ""
done

# =============================================================================
# SUMMARY
# =============================================================================

echo "══════════════════════════════════════════════════"
echo "  v2 Collection complete. ${#YAML_FILES[@]} sessions."
if (( ${#FAILED[@]} > 0 )); then
    echo "  Failed:"; for e in "${FAILED[@]}"; do echo "    - $e"; done
else
    echo "  All sessions completed without errors."
fi
echo "══════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  Count episodes : find \${EPISODES_DIR:-$EPISODES_DIR_DEFAULT} -name 'episode_*.hdf5' | wc -l"
echo "  Validate       : pixi run python -m team_policy.training_robot.validate_episode_v2 \$EPISODES_DIR/run_*/"
echo "  Convert        : see TRAINING_V2.md Step 4"
