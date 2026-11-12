#!/bin/bash
# Headless end-to-end check of the exit zone verdicts (sub-project 2 of the emergency exit inspection).
#
#   exit room (robot at the origin facing +x) + acados controller + FAST-LIO/SC-PGO (people:=false) +
#   exit_zone_check_launch.xml (person detector + zone checker); the robot stops 4.5 m in front of each door, facing
#   it, for DWELL s; zone_checks.yaml is copied after each stop and check_zones_gt.py compares the verdicts with
#   worlds/exit_room_gt.yaml (clear CLEAR, warning WARNING, blocked BLOCKED, person CLEAR).
#
# Usage: test/sim_exit_zones.sh <work_dir>
#   EXITS=<exits.yaml>: check those exits (e.g. from test/sim_exit_mapping.sh, same robot start pose) instead of the
#   ground truth ones (test/gt_to_exits.py through the FAST-LIO start pose).
#   DWELL (default 8 s sim time), ROS_DOMAIN_ID (default 83), SIM_ARGS as in sim_exit_mapping.sh.
#   The person detector runs on the GPU when the .venv-yolo nvidia wheels are there (ONNX Runtime CUDA libraries).
#   Exit code: the one of check_zones_gt.py (0 = pass).

W=$(realpath -m "${1:?work dir}")
DWELL=${DWELL:-8}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-83}
HERE=$(dirname "$(realpath "$0")")
source /opt/ros/jazzy/setup.bash
source "$HOME/RaNAV/install/setup.bash"
mkdir -p "$W/zones"
cd "$W" || exit 2
PKG=$(ros2 pkg prefix exit_inspection)/share/exit_inspection
AC=$(ros2 pkg prefix acados_legged_controller)/share/acados_legged_controller
PIDS=()
cleanup() {
  for p in "${PIDS[@]}"; do kill -INT -"$p" 2>/dev/null; done
  sleep 5
  for p in "${PIDS[@]}"; do kill -9 -"$p" 2>/dev/null; done
}
trap cleanup EXIT

echo "[$(date +%T)] sim"
setsid ros2 launch exit_inspection exit_room_launch.xml server:=true $SIM_ARGS \
  controllers_yaml:="$AC/config/ros2_controllers.yaml" > sim.log 2>&1 &
PIDS+=($!)
sleep 15
echo "[$(date +%T)] controller (CppAD code generation can take minutes on a first run)"
setsid ros2 launch acados_legged_controller load_controller_launch.xml rviz:=false > ctrl.log 2>&1 &
PIDS+=($!)
for _ in $(seq 1 120); do
  grep -q "Configured and activated acados_legged_controller" ctrl.log && break
  grep -q "Failed loading controller" ctrl.log && { echo "controller failed, see ctrl.log"; exit 2; }
  sleep 5
done
grep -q "Configured and activated acados_legged_controller" ctrl.log || { echo "controller not active"; exit 2; }
sleep 15  # stand up

BASE0=$(gz model -m aliengo -p | python3 -c "
import re, sys
num = r'[-+0-9.eE]+'
groups = re.findall(r'\[\s*(' + num + r')\s+(' + num + r')\s+(' + num + r')\s*\]', sys.stdin.read())
(x, y, z), (_, _, yaw) = [tuple(map(float, g)) for g in groups[:2]]
print(x, y, z, yaw)")
echo "[$(date +%T)] base at FAST-LIO start (world): $BASE0"
echo "$BASE0" > base0.txt

setsid ros2 launch fast_lio fast_lio_slam_launch.xml rviz:=false people:=false > lio.log 2>&1 &
PIDS+=($!)
sleep 10

if [ -n "$EXITS" ]; then
  cp "$EXITS" zones/exits.yaml
else
  python3 "$HERE/gt_to_exits.py" "$PKG/worlds/exit_room_gt.yaml" zones/exits.yaml --base0 $BASE0
fi
NVLIBS=$(ls -d "$HOME"/RaNAV/.venv-yolo/lib/python3.*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
LD_LIBRARY_PATH="$NVLIBS$LD_LIBRARY_PATH" setsid ros2 launch exit_inspection exit_zone_check_launch.xml \
  exits_file:="$W/zones/exits.yaml" debug_dump:=true > zones.log 2>&1 &
PIDS+=($!)
sleep 10

# viewpoints in odom (= world at the start): 4.5 m in front of each door, facing it
declare -A VIEW=(
  [clear]="-5 1.5 1.5708"
  [warning]="4 1.5 1.5708"
  [person]="5 -1.5 -1.5708"
  [blocked]="-4 -1.5 -1.5708"
)
SNAPS=()
for case in clear warning person blocked; do
  echo "[$(date +%T)] $case: ${VIEW[$case]}"
  python3 "$HERE/walk_waypoints.py" "${VIEW[$case]}" --dwell "$DWELL" | tee -a walk.log
  cp zones/zone_checks.yaml "zones/zone_checks_$case.yaml" 2>/dev/null || echo "no zone_checks.yaml yet"
  cp -r zones/zone_debug "zones/zone_debug_$case" 2>/dev/null
  SNAPS+=("$case=zones/zone_checks_$case.yaml")
done
python3 "$HERE/walk_waypoints.py" "0 0 0" --dwell 1 | tee -a walk.log

echo "[$(date +%T)] check"
ros2 run exit_inspection check_zones_gt.py "$PKG/worlds/exit_room_gt.yaml" "${SNAPS[@]}" --base0 $BASE0 \
  | tee check.log
grep -q "^PASS" check.log
