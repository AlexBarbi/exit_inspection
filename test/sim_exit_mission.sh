#!/bin/bash
# Headless end-to-end check of the exit inspection pass (sub-project 3 of the emergency exit inspection).
#
#   exit room (robot at the origin facing +x) + acados controller + FAST-LIO/SC-PGO (people:=false) +
#   exit_mission_launch.xml far:=false planner:=direct auto_start:=true (person detector, zone checker, pass): the
#   pass plans the viewpoints, walks to each door, takes the verdicts, comes back and writes the report;
#   check_zones_gt.py compares the report's verdicts with worlds/exit_room_gt.yaml.
#
# Usage: test/sim_exit_mission.sh <work_dir>
#   EXITS=<exits.yaml>: inspect those exits (same robot start pose) instead of the ground truth ones; MAP=<pcd>: check
#   the viewpoints against that map (default: none). TIMEOUT (default 1200 s wall), ROS_DOMAIN_ID (default 83),
#   SIM_ARGS as in sim_exit_mapping.sh. Exit code: the one of check_zones_gt.py (0 = pass).

W=$(realpath -m "${1:?work dir}")
TIMEOUT=${TIMEOUT:-1200}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-83}
HERE=$(dirname "$(realpath "$0")")
source /opt/ros/jazzy/setup.bash
source "$HOME/RaNAV/install/setup.bash"
mkdir -p "$W/save/exit_inspection"
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
  cp "$EXITS" save/exit_inspection/exits.yaml
else
  python3 "$HERE/gt_to_exits.py" "$PKG/worlds/exit_room_gt.yaml" save/exit_inspection/exits.yaml --base0 $BASE0
fi
[ -n "$MAP" ] && cp "$MAP" save/optimized_map.pcd
NVLIBS=$(ls -d "$HOME"/RaNAV/.venv-yolo/lib/python3.*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
echo "[$(date +%T)] mission"
LD_LIBRARY_PATH="$NVLIBS$LD_LIBRARY_PATH" setsid ros2 launch exit_inspection exit_mission_launch.xml \
  save_directory:="$W/save" far:=false planner:=direct auto_start:=true debug_dump:=true > mission.log 2>&1 &
PIDS+=($!)
for _ in $(seq 1 $((TIMEOUT / 5))); do
  [ -f save/exit_inspection/exit_report.yaml ] && break
  sleep 5
done
[ -f save/exit_inspection/exit_report.yaml ] || { echo "no report after $TIMEOUT s, see mission.log"; exit 2; }
grep -E "exit_inspection_pass.*(exit [0-9]+:|attempt|report|not reached|order)" mission.log | sed 's/^.*\]: //'

echo "[$(date +%T)] check"
ros2 run exit_inspection check_zones_gt.py "$PKG/worlds/exit_room_gt.yaml" \
  clear=save/exit_inspection/exit_report.yaml warning=save/exit_inspection/exit_report.yaml \
  blocked=save/exit_inspection/exit_report.yaml person=save/exit_inspection/exit_report.yaml --base0 $BASE0 \
  | tee check.log
grep -q "^PASS" check.log
