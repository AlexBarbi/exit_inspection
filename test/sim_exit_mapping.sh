#!/bin/bash
# Headless end-to-end check of exit sign mapping (sub-project 1 of the emergency exit inspection).
#
#   exit room (robot at the origin facing +x) + acados controller + FAST-LIO/SC-PGO (people:=false) +
#   exit_detection_launch.xml; the robot walks past the five cases (walk_waypoints.py), then finalize_exits.py writes
#   exits.yaml from SC-PGO's pose files and check_exits_gt.py compares it with worlds/exit_room_gt.yaml.
#
# Usage: test/sim_exit_mapping.sh <work_dir> [model.onnx]
#   ROS_DOMAIN_ID defaults to 83. Everything goes to <work_dir>, which is the launches' working directory: SC-PGO
#   saves to ./src/FAST_LIO_SLAM_ROS2/save_data/ there (fast_lio_slam_launch.xml sets alaserPGO's save_directory twice,
#   the hard-coded relative one wins), and the mapper asks alaserPGO for that same folder.
#   SIM_ARGS: extra exit_room_launch.xml arguments (e.g. camera:=false for a dry run of the chain without the camera);
#   DETECTOR=false: mapper only (no detector).
#   Exit code: the one of check_exits_gt.py (0 = pass).

W=$(realpath -m "${1:?work dir}")
MODEL=${2:-$HOME/RaNAV/models/exit_sign/exit_sign_736x1280.onnx}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-83}
HERE=$(dirname "$(realpath "$0")")
source /opt/ros/jazzy/setup.bash
source "$HOME/RaNAV/install/setup.bash"
mkdir -p "$W"
PGO="$W/src/FAST_LIO_SLAM_ROS2/save_data"
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
setsid ros2 launch exit_inspection exit_detection_launch.xml model:="$MODEL" detector:="${DETECTOR:-true}" \
  > exits.log 2>&1 &
EXITS=$!
PIDS+=($EXITS)
sleep 15

echo "[$(date +%T)] walk"
python3 "$HERE/walk_waypoints.py" \
  "0 0 0" "0 -3 1.5708" "-2 -3 1.5708" "2 -3 1.5708" "0 -1 1.5708" \
  "0 3 -1.5708" "-2 3 -1.5708" "2 3 -1.5708" "0 0 0" | tee walk.log

sleep 4  # SC-PGO rewrites its pose files at 1 Hz
kill -INT -"$EXITS"
sleep 3
echo "[$(date +%T)] finalize"
ros2 run exit_inspection finalize_exits.py "$PGO" | tee finalize.log
ros2 run exit_inspection check_exits_gt.py "$PGO/exit_inspection/exits.yaml" \
  "$PKG/worlds/exit_room_gt.yaml" --base0 $BASE0 | tee check.log
exit "${PIPESTATUS[0]}"
