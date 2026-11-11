#!/usr/bin/env bash
# Gazebo acceptance run (docs/PICAR_GAZEBO_KO.md section 4), unattended:
#   setup (if ROS/workspace missing) -> calibrate.launch.py -> course.launch.py x seeds -> beam-fault run -> verdict
#
#   bash picar_gazebo/scripts/run_acceptance.sh                 # 3 cars, 10 min, seeds 1 2 3 (~40 min + setup)
#   MINUTES=2 SEEDS="1" bash picar_gazebo/scripts/run_acceptance.sh     # quick smoke run
#   GUI=false ...                                               # no Gazebo window (needs EGL headless rendering)
#   CONTROLLERS="rule dqn" bash ...                             # same seeds with oldest_first and the trained DQN (active)
#   CARS=6 FLEET=runs/rehearsal_v1/fleet_truth.yaml bash ...    # six different cars, each with its own calibration
# Results: ~/picar_gz_runs/acceptance_<time>/{seed1,seed2,seed3,beam}/ + summary.md
set -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CARS="${CARS:-3}"; MINUTES="${MINUTES:-10}"; SEEDS="${SEEDS:-1 2 3}"; GUI="${GUI:-true}"
CONTROLLERS="${CONTROLLERS:-rule}"          # rule (oldest_first) and/or dqn, e.g. CONTROLLERS="rule dqn"
POLICY="${POLICY:-runs/picar_v9_photo_rt/seed42/best_policy.npz}"
TURN_RATE="${TURN_RATE:-45}"                 # junction spin rate (deg/s), calibration and drive nodes; 45 is repeatable in Gazebo, 90 is not
FLEET="${FLEET:-}"                           # fleet_truth.yaml: six different cars (virtual real fleet), calibrated one by one
BEAM_START="${BEAM_START:-60}"; BEAM_STOP="${BEAM_STOP:-80}"; BEAM_SECONDS="${BEAM_SECONDS:-130}"
WS="${PICAR_WS:-$HOME/picar_ws}"

if [ ! -f /opt/ros/humble/setup.bash ] || [ ! -f "$WS/install/setup.bash" ]; then
  echo "== ROS 2 Humble / $WS not found: running setup_ubuntu2204.sh (sudo password, 20-40 min)"
  bash "$REPO/picar_gazebo/scripts/setup_ubuntu2204.sh" || { echo "setup failed" >&2; exit 2; }
fi
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash; source "$WS/install/setup.bash"
export PICAR_REPO="$REPO"

OUT="$HOME/picar_gz_runs/acceptance_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
printf '{"cars": %s, "minutes": %s, "seeds": "%s", "controllers": "%s", "policy": "%s", "gui": "%s", "beam_fault_start": %s, "beam_fault_stop": %s, "turn_rate_dps": %s, "commit": "%s"}\n' \
  "$CARS" "$MINUTES" "$SEEDS" "$CONTROLLERS" "$POLICY" "$GUI" "$BEAM_START" "$BEAM_STOP" "$TURN_RATE" "$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)" > "$OUT/config.json"
echo "== results: $OUT"

# Background jobs of a non-interactive bash ignore SIGINT, so stop them with SIGTERM (ros2 launch shuts down
# its children on SIGTERM too) and fall back to SIGKILL.
stop_pid() {   # pid
  kill -TERM "$1" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$1" 2>/dev/null || break; sleep 1; done
  kill -KILL "$1" 2>/dev/null; wait "$1" 2>/dev/null
}
cleanup() {
  pkill -TERM -f "ros2 launch picar_gazebo" 2>/dev/null; sleep 3
  pkill -f "ign gazebo" 2>/dev/null; pkill -f "picar_sim.central_server" 2>/dev/null
  pkill -f "ros2_drive_node" 2>/dev/null; pkill -f "pose_logger.py" 2>/dev/null
  pkill -f "parameter_bridge" 2>/dev/null; pkill -f "calibrate --ros-args|motor_model|ir_camera|course_monitor" 2>/dev/null
  sleep 2; pkill -KILL -f "ign gazebo|ros2 launch picar_gazebo" 2>/dev/null; true
}
trap 'cleanup; exit 130' INT TERM
cleanup

calibrate_one() {   # out-file log-file extra-launch-args...
  local cal=$1 log=$2; shift 2
  local before now pid
  before=$(stat -c %Y "$cal" 2>/dev/null || echo 0)
  ros2 launch picar_gazebo calibrate.launch.py gui:="$GUI" turn_rate_dps:="$TURN_RATE" out_file:="$cal" "$@" > "$log" 2>&1 &
  pid=$!
  for _ in $(seq 1 180); do
    now=$(stat -c %Y "$cal" 2>/dev/null || echo 0)
    [ "$now" -gt "$before" ] && break
    kill -0 $pid 2>/dev/null || break
    sleep 1
  done
  sleep 2; stop_pid $pid; cleanup
  if [ "$(stat -c %Y "$cal" 2>/dev/null || echo 0)" -le "$before" ]; then
    echo "calibration file was not written; see $log" >&2; tail -30 "$log" >&2; return 3
  fi
}

FLEET_ARGS=()
if [ -n "$FLEET" ]; then
  # Virtual real fleet: every car has its own hidden hardware, so every car is measured on its own, as the real
  # cars will be. Motors: measured here in Gazebo. Camera mount: the one-photo result (camera_calib.py) is taken
  # from CAM_CALIB_DIR (2D rehearsal of the same fleet), because it only depends on how the camera is mounted.
  [[ "$FLEET" = /* ]] || FLEET="$REPO/$FLEET"
  CAM_CALIB_DIR="${CAM_CALIB_DIR:-$(dirname "$FLEET")/calib}"
  CALDIR="$OUT/calib"; mkdir -p "$CALDIR"
  echo "== 1) calibrate.launch.py for each of the $CARS cars of $FLEET"
  for i in $(seq 0 $((CARS - 1))); do
    if [ -n "$CALIB_FROM" ] && [ -f "$CALIB_FROM/car$i.yaml" ]; then      # reuse an earlier run's calibration
      cp "$CALIB_FROM/car$i.yaml" "$CALDIR/car$i.yaml"; echo "   car$i.yaml reused from $CALIB_FROM"; continue
    fi
    calibrate_one "$CALDIR/car$i.yaml" "$OUT/calibrate_car$i.log" fleet:="$FLEET" car_index:="$i" || exit 3
    python3 - "$CALDIR/car$i.yaml" "$CAM_CALIB_DIR/car$i.yaml" <<'PY'
import sys, yaml
dst, src = sys.argv[1:3]
cal = yaml.safe_load(open(dst))
if cal.get("measured", {}).get("straight_m", 1.0) < 0.3 or cal.get("turn_scale", 1.0) < 0.3:
    sys.exit(f"{dst}: the car did not move during calibration (straight {cal['measured']['straight_m']} m): "
             "motor_model or the bridge is not running - see the calibrate_car log")
try:
    cam = yaml.safe_load(open(src))
    cal.update({k: cam[k] for k in ("cam_pitch_deg", "cam_yaw_deg", "cam_y_m") if k in cam})
except FileNotFoundError:
    print("   no camera calibration", src, "- nominal mount assumed")
yaml.safe_dump(cal, open(dst, "w"), sort_keys=False)
print("  ", dst.split("/")[-1], {k: v for k, v in cal.items() if k not in ("source", "measured")})
PY
    [ $? -eq 0 ] || { tail -25 "$OUT/calibrate_car$i.log" >&2; exit 3; }
  done
  FLEET_ARGS=(fleet:="$FLEET" calibration_dir:="$CALDIR")
else
  echo "== 1) calibrate.launch.py"
  CAL="$HOME/picar_gz_runs/car_calibration.yaml"
  calibrate_one "$CAL" "$OUT/calibrate.log" || exit 3
  cp "$CAL" "$OUT/car_calibration.yaml"; cat "$CAL"
fi

run_course() {   # name seconds extra-args...
  local name=$1 secs=$2; shift 2
  local dir="$OUT/$name" try
  for try in 1 2; do
    rm -rf "$dir"; mkdir -p "$dir"
    [ -n "$DUMP_FRAMES" ] && export PICAR_DUMP_DIR="$dir"
    echo "== $name: ${secs}s  $*"
    python3 "$REPO/picar_gazebo/scripts/pose_logger.py" "$dir" > "$dir/pose_logger.log" 2>&1 &
    local lp=$!
    timeout -s INT -k 30 "$secs" ros2 launch picar_gazebo course.launch.py cars:="$CARS" gui:="$GUI" out_dir:="$dir" turn_rate_dps:="$TURN_RATE" "${FLEET_ARGS[@]}" "$@" \
      > "$dir/launch.log" 2>&1
    stop_pid $lp
    cleanup
    # Gazebo sometimes comes up without running the world (no poses at all, seen once on the PC): run it again
    [ "$(wc -l < "$dir/poses.csv" 2>/dev/null || echo 0)" -gt 10 ] && break
    [ $try = 1 ] && { echo "   no car poses: Gazebo did not run, retrying once"; sleep 5; }
  done
  python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print({k: m.get(k) for k in ('elapsed_s','collisions','lane_departures','stop_error_p95_abs_m')})" \
    "$dir/metrics.json" 2>/dev/null || echo "   (no metrics.json)"
}

echo "== 2) course.launch.py: $CARS cars x $MINUTES min, seeds: $SEEDS, controllers: $CONTROLLERS"
case "$POLICY" in /*) ;; *) POLICY="$REPO/$POLICY" ;; esac
for s in $SEEDS; do
  for c in $CONTROLLERS; do
    case "$c" in
      rule) extra=() ;;
      dqn)  [ -f "$POLICY" ] || { echo "policy not found: $POLICY" >&2; exit 4; }
            extra=(policy:="$POLICY" mode:=active) ;;
      *)    echo "unknown controller $c (rule|dqn)" >&2; exit 4 ;;
    esac
    name="seed$s"; [ "$CONTROLLERS" = "rule" ] || name="seed${s}_$c"
    run_course "$name" $(( MINUTES * 60 + 20 )) seed:="$s" port:=$(( 5100 + s )) "${extra[@]}"
  done
done
echo "== 3) beam fault ${BEAM_START}-${BEAM_STOP}s"
[ -n "$SKIP_BEAM" ] || run_course beam "$BEAM_SECONDS" seed:=1 port:=5199 beam_fault_start:="$BEAM_START" beam_fault_stop:="$BEAM_STOP"

echo "== 4) verdict"
python3 "$REPO/picar_gazebo/scripts/check_acceptance.py" "$OUT"
rc=$?
echo "== summary: $OUT/summary.md"
exit $rc
