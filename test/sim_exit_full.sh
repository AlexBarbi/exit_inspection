#!/bin/bash
# Headless end-to-end check of the whole emergency exit inspection: exploration, then the FAR mission.
#
#   1. exploration: exit room + acados controller + FAST-LIO/SC-PGO (people:=false) + explore_warehouse_launch.xml
#      auto_start:=true exits:=true (TARE, FAR graph, exit sign detector + landmark mapper; at the end the robot
#      returns home, the map is merged, the graph saved and exits.yaml written); check_exits_gt.py on exits.yaml;
#   2. mission: everything restarted (robot back at the spawn, FAST-LIO started there like in the exploration),
#      exit_mission_launch.xml far:=true planner:=far (far_planner on the saved graph, zone checker, pass) with the
#      exploration's save_data, started once the graph is loaded; check_zones_gt.py on the report.
#
# Usage: test/sim_exit_full.sh <work_dir>
#   <work_dir>/explore: exploration (SC-PGO save_data in ./src/FAST_LIO_SLAM_ROS2/save_data there), <work_dir>/mission:
#   the mission's own FAST-LIO/SC-PGO output. EXPLORE_TIMEOUT (default 2400 s wall), MISSION_TIMEOUT (1800 s),
#   ROS_DOMAIN_ID (default 83). SKIP_EXPLORE=1 reuses an earlier exploration in <work_dir>/explore.
#   Exit code: 0 when both checks pass.

W=$(realpath -m "${1:?work dir}")
EXPLORE_TIMEOUT=${EXPLORE_TIMEOUT:-2400}
MISSION_TIMEOUT=${MISSION_TIMEOUT:-1800}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-83}
HERE=$(dirname "$(realpath "$0")")
source /opt/ros/jazzy/setup.bash
source "$HOME/RaNAV/install/setup.bash"
PKG=$(ros2 pkg prefix exit_inspection)/share/exit_inspection
AC=$(ros2 pkg prefix acados_legged_controller)/share/acados_legged_controller
NVLIBS=$(ls -d "$HOME"/RaNAV/.venv-yolo/lib/python3.*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
SAVE="$W/explore/src/FAST_LIO_SLAM_ROS2/save_data"
PIDS=()
stop_all() {
  for p in "${PIDS[@]}"; do kill -INT -"$p" 2>/dev/null; done
  sleep 8
  for p in "${PIDS[@]}"; do kill -9 -"$p" 2>/dev/null; done
  PIDS=()
  sleep 3
  # Fast DDS shared memory files of the killed processes would hang the next launches
  pgrep -f "ros2|gz sim|ruby" > /dev/null || rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null
}
trap stop_all EXIT

start_robot() {  # sim + controller, robot standing at the spawn; prints the base pose (world) to base0.txt
  echo "[$(date +%T)] sim"
  setsid ros2 launch exit_inspection exit_room_launch.xml server:=true \
    controllers_yaml:="$AC/config/ros2_controllers.yaml" > sim.log 2>&1 &
  PIDS+=($!)
  sleep 15
  echo "[$(date +%T)] controller"
  setsid ros2 launch acados_legged_controller load_controller_launch.xml rviz:=false > ctrl.log 2>&1 &
  PIDS+=($!)
  for _ in $(seq 1 120); do
    grep -q "Configured and activated acados_legged_controller" ctrl.log && break
    grep -q "Failed loading controller" ctrl.log && { echo "controller failed, see $PWD/ctrl.log"; return 1; }
    sleep 5
  done
  grep -q "Configured and activated acados_legged_controller" ctrl.log || { echo "controller not active"; return 1; }
  sleep 15
  gz model -m aliengo -p | python3 -c "
import re, sys
num = r'[-+0-9.eE]+'
groups = re.findall(r'\[\s*(' + num + r')\s+(' + num + r')\s+(' + num + r')\s*\]', sys.stdin.read())
(x, y, z), (_, _, yaw) = [tuple(map(float, g)) for g in groups[:2]]
print(x, y, z, yaw)" > base0.txt
  echo "[$(date +%T)] base at FAST-LIO start (world): $(cat base0.txt)"
  setsid ros2 launch fast_lio fast_lio_slam_launch.xml rviz:=false people:=false > lio.log 2>&1 &
  PIDS+=($!)
  sleep 10
}

# ------------------------------------------------------------------ 1. exploration
if [ -z "$SKIP_EXPLORE" ]; then
  rm -rf "$W/explore"
  mkdir -p "$W/explore" && cd "$W/explore" || exit 2
  start_robot || exit 2
  echo "[$(date +%T)] exploration (TARE + exits)"
  LD_LIBRARY_PATH="$NVLIBS$LD_LIBRARY_PATH" setsid ros2 launch tare_planner explore_warehouse_launch.xml \
    auto_start:=true exits:=true rviz:=false > explore.log 2>&1 &
  PIDS+=($!)
  for _ in $(seq 1 $((EXPLORE_TIMEOUT / 10))); do
    grep -q "exits written" explore.log && break
    grep -q "tare_finish_merge_map.*process has died" explore.log && break
    sleep 10
  done
  grep -E "Exploration completed|robot back home|sweep (plan|done|stop)|not reached|map saved|FAR graph (saved|pruned)|exits written|confirmed" explore.log \
    | sed "s/^.*\]: //" | tail -30
  stop_all
  grep -q "exits written" explore.log || { echo "exploration did not finish, see $W/explore/explore.log"; exit 2; }
fi
cd "$W/explore" || exit 2
echo "[$(date +%T)] check exits.yaml"
ros2 run exit_inspection check_exits_gt.py "$SAVE/exit_inspection/exits.yaml" "$PKG/worlds/exit_room_gt.yaml" \
  --base0 $(cat base0.txt) | tee check_exits.log
for f in optimized_map.pcd warehouse.vgh exit_inspection/exits.yaml; do
  [ -s "$SAVE/$f" ] || { echo "missing $SAVE/$f"; exit 2; }
done

# ------------------------------------------------------------------ 2. mission
rm -rf "$W/mission"
mkdir -p "$W/mission" && cd "$W/mission" || exit 2
start_robot || exit 2
echo "[$(date +%T)] mission (far_planner on the saved graph)"
LD_LIBRARY_PATH="$NVLIBS$LD_LIBRARY_PATH" setsid ros2 launch exit_inspection exit_mission_launch.xml \
  save_directory:="$SAVE" far:=true planner:=far auto_start:=false debug_dump:=true > mission.log 2>&1 &
PIDS+=($!)
for _ in $(seq 1 30); do
  grep -q "nodes read from" mission.log && break
  sleep 5
done
grep "nodes read from\|far_planner up" mission.log | sed 's/^.*\]: //'
rm -f "$SAVE/exit_inspection/exit_report.yaml"
sleep 5
ros2 topic pub -1 /start_exit_inspection std_msgs/msg/String "{data: ''}" > /dev/null
for _ in $(seq 1 $((MISSION_TIMEOUT / 10))); do
  [ -f "$SAVE/exit_inspection/exit_report.yaml" ] && break
  sleep 10
done
grep -E "exit_inspection_pass.*(exit [0-9]+:|attempt|report|not reached|order|back)" mission.log | sed 's/^.*\]: //'
[ -f "$SAVE/exit_inspection/exit_report.yaml" ] || { echo "no report after $MISSION_TIMEOUT s, see $W/mission/mission.log"; exit 2; }

echo "[$(date +%T)] check the report"
R="$SAVE/exit_inspection/exit_report.yaml"
ros2 run exit_inspection check_zones_gt.py "$PKG/worlds/exit_room_gt.yaml" clear=$R warning=$R blocked=$R person=$R \
  --base0 $(cat "$W/explore/base0.txt") | tee check_report.log
grep -q "^PASS" "$W/explore/check_exits.log" && grep -q "^PASS" check_report.log
