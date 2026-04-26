#!/usr/bin/env bash
set -Eeuo pipefail

RUNS_TO_DO=44
COMPLETED_RUNS=6
NUM_EPISODES=3

POST_SUCCESS_WAIT=15
SIM_READY_TIMEOUT=420
SIM_STABILIZE_AFTER_READY=20
COLLECTOR_SUCCESS_TIMEOUT=1800

GENERATE_SESSIONS_IF_NEEDED=1
TRIALS_PER_SESSION=3

SESSION_PAD=2
RUN_PAD=3

DELETE_AIC_RESULTS_DIR="/home/swithin/aic_results"
SIM_READY_PATTERN="No node with name 'aic_model' found. Retrying..."
SUCCESS_PATTERN="[3/3] Saved"

AIC_ROOT="${AIC_ROOT:-$(git rev-parse --show-toplevel)}"
TRAIN_ROOT="${TRAIN_ROOT:-$AIC_ROOT/team_policy/team_policy/training_robot}"
FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-$AIC_ROOT/team_policy/fastdds_no_shm.xml}"

LOG_ROOT="$TRAIN_ROOT/automation_logs"
STATE_FILE="$LOG_ROOT/last_progress.env"
mkdir -p "$LOG_ROOT"

START_RUN=$((COMPLETED_RUNS + 1))
END_RUN=$((COMPLETED_RUNS + RUNS_TO_DO))

COLLECTOR_PID=""
SHUTTING_DOWN=0

DBX_SHELL_PID=""
DBX_IN=""
DBX_OUT=""
DBX_LAST_LINE=""
ENTRY_INNER_PID=""

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

session_name() {
  printf "session_%0${SESSION_PAD}d.yaml" "$1"
}

run_name() {
  printf "run_%0${RUN_PAD}d" "$1"
}

safe_clear_dir_contents() {
  local target="$1"
  [[ -n "$target" ]] || { log "Refusing to clear empty path"; return 1; }
  [[ "$target" != "/" ]] || { log "Refusing to clear /"; return 1; }
  mkdir -p "$target"
  find "$target" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
}

kill_process_group() {
  local pid="${1:-}"
  [[ -n "$pid" ]] || return 0

  if kill -0 "$pid" 2>/dev/null; then
    local pgid
    pgid="$(ps -o pgid= "$pid" 2>/dev/null | tr -d ' ')"
    if [[ -n "$pgid" ]]; then
      kill -TERM -- "-$pgid" 2>/dev/null || true
      sleep 5
      kill -KILL -- "-$pgid" 2>/dev/null || true
    else
      kill -TERM "$pid" 2>/dev/null || true
      sleep 2
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
}

dump_log_tail() {
  local file="$1"
  local lines="${2:-60}"
  if [[ -f "$file" ]]; then
    echo "--------------------"
    echo "Last $lines lines of $file"
    tail -n "$lines" "$file"
    echo "--------------------"
  else
    echo "Log file not found: $file"
  fi
}

write_state() {
  local current_run="$1"
  local status="$2"

  cat > "$STATE_FILE" <<EOF
AIC_ROOT="$AIC_ROOT"
TRAIN_ROOT="$TRAIN_ROOT"
LAST_ATTEMPTED_RUN="$current_run"
LAST_COMPLETED_RUN="${LAST_COMPLETED_RUN:-0}"
NEXT_RUN="$((current_run + 1))"
STATUS="$status"
UPDATED_AT="$(timestamp)"
EOF
}

dbx_send() {
  local cmd="$1"
  [[ -n "${DBX_IN:-}" ]] || { log "Distrobox stdin fd missing"; return 1; }
  printf '%s\n' "$cmd" >&$DBX_IN
}

dbx_read_until() {
  local marker="$1"
  local timeout_s="$2"
  local label="$3"

  local start_time
  start_time="$(date +%s)"
  DBX_LAST_LINE=""

  while true; do
    if read -r -t 1 -u $DBX_OUT line; then
      DBX_LAST_LINE="$line"
      if [[ "$line" == *"$marker"* ]]; then
        return 0
      fi
    fi

    if [[ -n "${DBX_SHELL_PID:-}" ]] && ! kill -0 "$DBX_SHELL_PID" 2>/dev/null; then
      log "Persistent distrobox shell exited while waiting for $label"
      return 1
    fi

    local now
    now="$(date +%s)"
    if (( now - start_time >= timeout_s )); then
      log "Timeout waiting for $label"
      return 1
    fi
  done
}

start_distrobox_shell() {
  log "Starting persistent distrobox shell"

  coproc DBX_PROC { stdbuf -oL -eL distrobox enter aic_eval -- bash --noprofile --norc; }
  DBX_SHELL_PID=$!
  DBX_IN=${DBX_PROC[1]}
  DBX_OUT=${DBX_PROC[0]}

  dbx_send "export FASTRTPS_DEFAULT_PROFILES_FILE='$FASTRTPS_DEFAULT_PROFILES_FILE'; export PYTHONUNBUFFERED=1; export RCUTILS_LOGGING_BUFFERED_STREAM=0; echo __DBX_READY__"
  dbx_read_until "__DBX_READY__" 30 "persistent distrobox shell ready"
}

stop_distrobox_shell() {
  [[ -n "${DBX_SHELL_PID:-}" ]] || return 0

  if kill -0 "$DBX_SHELL_PID" 2>/dev/null; then
    dbx_send "exit" || true
    sleep 2
    kill -TERM "$DBX_SHELL_PID" 2>/dev/null || true
    sleep 1
    kill -KILL "$DBX_SHELL_PID" 2>/dev/null || true
  fi

  DBX_SHELL_PID=""
  DBX_IN=""
  DBX_OUT=""
  DBX_LAST_LINE=""
}

stop_sim_in_distrobox() {
  [[ -n "${DBX_SHELL_PID:-}" ]] || return 0

  dbx_send "pkill -TERM -f 'zenoh|rmw_zenohd|/entrypoint.sh|aic_engine|gz sim|gazebo|ign gazebo|controller_manager|spawner|robot_state_publisher|joint_state_broadcaster|aic_model' >/dev/null 2>&1 || true; sleep 5; pkill -KILL -f 'zenoh|rmw_zenohd|/entrypoint.sh|aic_engine|gz sim|gazebo|ign gazebo|controller_manager|spawner|robot_state_publisher|joint_state_broadcaster|aic_model' >/dev/null 2>&1 || true; echo __SIM_STOPPED__"
  dbx_read_until "__SIM_STOPPED__" 30 "simulation stop" || true
  ENTRY_INNER_PID=""
}

start_sim_in_distrobox() {
  local session_file="$1"
  local entry_log="$2"

  stop_sim_in_distrobox

  dbx_send "stdbuf -oL -eL /entrypoint.sh ground_truth:=true start_aic_engine:=true gazebo_gui:=false launch_rviz:=false aic_engine_config_file:='$session_file' >'$entry_log' 2>&1 & echo __SIM_STARTED__:\$!"
  dbx_read_until "__SIM_STARTED__:" 30 "simulation start"

  ENTRY_INNER_PID="${DBX_LAST_LINE##*:}"
  log "Inner simulation PID=$ENTRY_INNER_PID"
}

sim_pid_alive_in_distrobox() {
  [[ -n "${ENTRY_INNER_PID:-}" ]] || return 1

  dbx_send "if kill -0 $ENTRY_INNER_PID 2>/dev/null; then echo __PID_ALIVE__; else echo __PID_DEAD__; fi"
  dbx_read_until "__PID_" 10 "simulation pid state" || return 1

  [[ "$DBX_LAST_LINE" == *"__PID_ALIVE__"* ]]
}

wait_for_log_pattern_or_failure() {
  local file="$1"
  local pattern="$2"
  local timeout_s="$3"
  local label="$4"

  local start_time
  start_time="$(date +%s)"
  local last_report=0

  while true; do
    if [[ -f "$file" ]] && grep -Fq "$pattern" "$file"; then
      return 0
    fi

    if ! sim_pid_alive_in_distrobox; then
      log "$label simulation process died before ready pattern"
      dump_log_tail "$file" 120
      return 1
    fi

    local now
    now="$(date +%s)"

    if (( now - start_time >= timeout_s )); then
      log "Timeout waiting for $label pattern: $pattern"
      dump_log_tail "$file" 120
      return 1
    fi

    if (( now - last_report >= 15 )); then
      log "Still waiting for $label ..."
      if [[ -f "$file" ]]; then
        tail -n 5 "$file" | sed 's/^/[log] /'
      fi
      last_report=$now
    fi

    sleep 2
  done
}

wait_for_sim_stabilization() {
  local file="$1"
  local seconds="$2"

  log "Ready pattern found. Stabilizing for $seconds seconds"

  local i
  for (( i=1; i<=seconds; i++ )); do
    if ! sim_pid_alive_in_distrobox; then
      log "Simulation process died during stabilization"
      dump_log_tail "$file" 120
      return 1
    fi
    sleep 1
  done

  return 0
}

wait_for_collector_success_or_failure() {
  local file="$1"
  local pattern="$2"
  local pid="$3"
  local timeout_s="$4"
  local label="$5"
  local entry_log="$6"

  local start_time
  start_time="$(date +%s)"
  local last_report=0

  while true; do
    if [[ -f "$file" ]] && grep -Fq "$pattern" "$file"; then
      return 0
    fi

    if ! sim_pid_alive_in_distrobox; then
      log "Simulation died while collector was running"
      dump_log_tail "$entry_log" 120
      dump_log_tail "$file" 120
      return 1
    fi

    if ! kill -0 "$pid" 2>/dev/null; then
      log "$label process exited before success pattern"
      dump_log_tail "$entry_log" 120
      dump_log_tail "$file" 120
      return 1
    fi

    local now
    now="$(date +%s)"

    if (( now - start_time >= timeout_s )); then
      log "Timeout waiting for $label success pattern"
      dump_log_tail "$entry_log" 120
      dump_log_tail "$file" 120
      return 1
    fi

    if (( now - last_report >= 15 )); then
      log "Still waiting for $label ..."
      if [[ -f "$file" ]]; then
        tail -n 5 "$file" | sed 's/^/[log] /'
      fi
      last_report=$now
    fi

    sleep 2
  done
}

cleanup_current_run() {
  kill_process_group "$COLLECTOR_PID"
  COLLECTOR_PID=""
  stop_sim_in_distrobox || true
}

cleanup_all() {
  if (( SHUTTING_DOWN == 1 )); then
    return 0
  fi

  SHUTTING_DOWN=1
  log "Stopping all processes started by this script"
  cleanup_current_run || true
  stop_distrobox_shell || true
}

on_interrupt() {
  trap - INT TERM
  log "Interrupted. Cleaning up everything."
  cleanup_all
  exit 130
}

on_exit() {
  local rc=$?
  trap - EXIT
  cleanup_all || true
  exit "$rc"
}

trap on_interrupt INT TERM
trap on_exit EXIT

log "AIC_ROOT=$AIC_ROOT"
log "TRAIN_ROOT=$TRAIN_ROOT"
log "RUNS_TO_DO=$RUNS_TO_DO"
log "COMPLETED_RUNS=$COMPLETED_RUNS"
log "Will execute runs $START_RUN to $END_RUN"

if (( COMPLETED_RUNS > 0 )); then
  log "Skipping generate_competition_sessions.py because COMPLETED_RUNS > 0"
else
  if (( GENERATE_SESSIONS_IF_NEEDED == 1 )); then
    log "Generating session YAML files"
    (
      cd "$AIC_ROOT"
      python3 team_policy/team_policy/training_robot/configs/generate_competition_sessions.py \
        --sessions "$END_RUN" \
        --trials-per-session "$TRIALS_PER_SESSION"
    )
  fi
fi

start_distrobox_shell

LAST_COMPLETED_RUN="$COMPLETED_RUNS"

for run_idx in $(seq "$START_RUN" "$END_RUN"); do
  session_file="$TRAIN_ROOT/configs/sessions/$(session_name "$run_idx")"
  run_id="$(run_name "$run_idx")"
  output_dir="$TRAIN_ROOT/episodes/$run_id"

  [[ -f "$session_file" ]] || { log "Missing session file: $session_file"; exit 1; }

  if compgen -G "$output_dir/episode_*.hdf5" > /dev/null; then
    log "Output dir already contains episode files: $output_dir"
    log "Refusing to continue to avoid overwrite"
    exit 1
  fi

  run_log_dir="$LOG_ROOT/$run_id"
  mkdir -p "$run_log_dir"
  entry_log="$run_log_dir/entrypoint.log"
  collector_log="$run_log_dir/collector.log"

  : > "$entry_log"
  : > "$collector_log"

  log "============================================================"
  log "Starting $run_id using $(basename "$session_file")"
  log "Output dir: $output_dir"
  write_state "$run_idx" "starting"

  mkdir -p "$output_dir"

  log "Clearing $DELETE_AIC_RESULTS_DIR before run"
  safe_clear_dir_contents "$DELETE_AIC_RESULTS_DIR"

  start_sim_in_distrobox "$session_file" "$entry_log"

  log "Waiting for simulation readiness pattern"
  write_state "$run_idx" "waiting_for_sim_ready"

  if ! wait_for_log_pattern_or_failure "$entry_log" "$SIM_READY_PATTERN" "$SIM_READY_TIMEOUT" "entrypoint"; then
    log "Simulation did not become ready"
    exit 1
  fi

  if ! wait_for_sim_stabilization "$entry_log" "$SIM_STABILIZE_AFTER_READY"; then
    log "Simulation failed stabilization"
    exit 1
  fi

  log "Simulation ready and stable. Starting collector."

  setsid bash -lc "
    export AIC_ROOT='$AIC_ROOT'
    export TRAIN_ROOT='$TRAIN_ROOT'
    export FASTRTPS_DEFAULT_PROFILES_FILE='$FASTRTPS_DEFAULT_PROFILES_FILE'
    export RUN_ID='$run_id'
    export OUTPUT_DIR='$output_dir'
    export PYTHONUNBUFFERED=1
    export RCUTILS_LOGGING_BUFFERED_STREAM=0
    cd '$AIC_ROOT'
    stdbuf -oL -eL pixi run ros2 run aic_model aic_model --ros-args \
      -p use_sim_time:=true \
      -p policy:=team_policy.training_robot.cheatcode_collector \
      -p output_dir:=\"\$OUTPUT_DIR\" \
      -p num_episodes:=$NUM_EPISODES \
      -p success_only:=true
  " >"$collector_log" 2>&1 &
  COLLECTOR_PID=$!

  log "Collector PID=$COLLECTOR_PID"
  write_state "$run_idx" "collector_started"

  if ! wait_for_collector_success_or_failure "$collector_log" "$SUCCESS_PATTERN" "$COLLECTOR_PID" "$COLLECTOR_SUCCESS_TIMEOUT" "collector" "$entry_log"; then
    log "Collector failed before reaching success pattern"
    exit 1
  fi

  log "$run_id reached success pattern: $SUCCESS_PATTERN"
  write_state "$run_idx" "success_pattern_seen"

  log "Waiting $POST_SUCCESS_WAIT seconds before shutdown"
  sleep "$POST_SUCCESS_WAIT"

  log "Stopping collector and simulation"
  cleanup_current_run

  log "Clearing $DELETE_AIC_RESULTS_DIR after run"
  safe_clear_dir_contents "$DELETE_AIC_RESULTS_DIR"

  LAST_COMPLETED_RUN="$run_idx"
  write_state "$run_idx" "completed"

  log "Completed $run_id"
  log "Runs completed in this batch: $((run_idx - START_RUN + 1)) / $RUNS_TO_DO"
  log "Total completed overall: $LAST_COMPLETED_RUN"
done

log "============================================================"
log "Batch finished successfully"
log "Last completed run: $LAST_COMPLETED_RUN"
log "Next run to continue from: $((LAST_COMPLETED_RUN + 1))"
write_state "$LAST_COMPLETED_RUN" "batch_finished"