#!/usr/bin/env bash
# Gazebo acceptance run (docs/PICAR_GAZEBO_KO.md section 4), unattended:
#   setup (if ROS/workspace missing) -> calibrate.launch.py -> course.launch.py x seeds -> beam-fault run -> verdict
#
#   bash picar_gazebo/scripts/run_acceptance.sh                 # 3 cars, 10 min, seeds 1 2 3 (~40 min + setup)
#   MINUTES=2 SEEDS="1" bash picar_gazebo/scripts/run_acceptance.sh     # quick smoke run
#   GUI=false ...                                               # no Gazebo window (needs EGL headless rendering)
# Results: ~/picar_gz_runs/acceptance_<time>/{seed1,seed2,seed3,beam}/ + summary.md
set -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CARS="${CARS:-3}"; MINUTES="${MINUTES:-10}"; SEEDS="${SEEDS:-1 2 3}"; GUI="${GUI:-true}"
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
printf '{"cars": %s, "minutes": %s, "seeds": "%s", "gui": "%s", "beam_fault_start": %s, "beam_fault_stop": %s, "commit": "%s"}\n' \
  "$CARS" "$MINUTES" "$SEEDS" "$GUI" "$BEAM_START" "$BEAM_STOP" "$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)" > "$OUT/config.json"
echo "== results: $OUT"

cleanup() {
  pkill -INT -f "ros2 launch picar_gazebo" 2>/dev/null; sleep 3
  pkill -f "ign gazebo" 2>/dev/null; pkill -f "picar_sim.central_server" 2>/dev/null
  pkill -f "ros2_drive_node" 2>/dev/null; pkill -f "pose_logger.py" 2>/dev/null
  pkill -f "parameter_bridge" 2>/dev/null; sleep 2
}
trap 'cleanup; exit 130' INT TERM
cleanup

echo "== 1) calibrate.launch.py"
CAL="$HOME/picar_gz_runs/car_calibration.yaml"
before=$(stat -c %Y "$CAL" 2>/dev/null || echo 0)
ros2 launch picar_gazebo calibrate.launch.py gui:="$GUI" > "$OUT/calibrate.log" 2>&1 &
pid=$!
for _ in $(seq 1 180); do
  now=$(stat -c %Y "$CAL" 2>/dev/null || echo 0)
  [ "$now" -gt "$before" ] && break
  kill -0 $pid 2>/dev/null || break
  sleep 1
done
sleep 2; kill -INT $pid 2>/dev/null; wait $pid 2>/dev/null; cleanup
if [ "$(stat -c %Y "$CAL" 2>/dev/null || echo 0)" -le "$before" ]; then
  echo "calibration file was not written; see $OUT/calibrate.log" >&2; tail -30 "$OUT/calibrate.log" >&2; exit 3
fi
cp "$CAL" "$OUT/car_calibration.yaml"; cat "$CAL"

run_course() {   # name seconds extra-args...
  local name=$1 secs=$2; shift 2
  local dir="$OUT/$name"; mkdir -p "$dir"
  echo "== $name: ${secs}s  $*"
  python3 "$REPO/picar_gazebo/scripts/pose_logger.py" "$dir" > "$dir/pose_logger.log" 2>&1 &
  local lp=$!
  timeout -s INT -k 30 "$secs" ros2 launch picar_gazebo course.launch.py cars:="$CARS" gui:="$GUI" out_dir:="$dir" "$@" \
    > "$dir/launch.log" 2>&1
  kill -INT $lp 2>/dev/null; wait $lp 2>/dev/null
  cleanup
  python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print({k: m.get(k) for k in ('elapsed_s','collisions','lane_departures','stop_error_p95_abs_m')})" \
    "$dir/metrics.json" 2>/dev/null || echo "   (no metrics.json)"
}

echo "== 2) course.launch.py: $CARS cars x $MINUTES min, seeds: $SEEDS"
for s in $SEEDS; do
  run_course "seed$s" $(( MINUTES * 60 + 20 )) seed:="$s" port:=$(( 5100 + s ))
done
echo "== 3) beam fault ${BEAM_START}-${BEAM_STOP}s"
run_course beam "$BEAM_SECONDS" seed:=1 port:=5199 beam_fault_start:="$BEAM_START" beam_fault_stop:="$BEAM_STOP"

echo "== 4) verdict"
python3 "$REPO/picar_gazebo/scripts/check_acceptance.py" "$OUT"
rc=$?
echo "== summary: $OUT/summary.md"
exit $rc
