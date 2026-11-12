# exit_inspection

Emergency exit inspection for the quadruped (D435i + RoboSense Airy). Sub-project 1, implemented here: exit signs
(ISO 7010 E001/E002) found by the camera during a TARE exploration become exit landmarks in the optimized map,
`exits.yaml`. Sub-project 2, implemented here: the floor zone in front of every door exit gets a verdict CLEAR /
WARNING / BLOCKED / UNVERIFIED from the D435i depth and the lidar, people excluded. Sub-project 3, implemented here:
an inspection mission visits every door, takes its verdict and writes a report; map_point_picker shows the exits and
starts the mission. Designs in the RaNAV workspace: `docs/superpowers/specs/2026-11-11-exit-detection-mapping-design.md`,
`docs/superpowers/specs/2026-11-11-exit-zone-checker-design.md`, `docs/superpowers/specs/2026-11-11-exit-inspection-pass-design.md`.

## Data flow

```
D435i 1920x1080 @ 15 Hz
   │ /camera/color/image_raw + /camera/aligned_depth_to_color/image_raw
   ▼
exit_sign_detector  (opti_pessi_perception Yolo26RGBD, exit sign ONNX, publish_obstacles:=false)
   │ /exit_sign_detector/detections_2d   vision_msgs/Detection2DArray
   ▼
exit_landmark_mapper.py ◄── /registered_scan_static (camera_init), TF camera_init <- lio_base, base <- camera
   │  ray through the box ∩ vertical wall plane = sign center + wall normal; sightings merged into landmarks
   ├─► /exit_inspection/exit_markers  (rviz: confirmed door exits green, unconfirmed grey, route signs blue)
   └─► <SC-PGO save_directory>/exit_inspection/sightings.csv
                    │  end of the exploration (tare_finish_merge_map.py exits:=true, after the map merge)
                    ▼
finalize_exits.py  ◄── times.txt, odom_poses.txt, optimized_poses.txt (SC-PGO)
   └─► <save_directory>/exit_inspection/exits.yaml   (in the frame of optimized_map.pcd)
```

Sign sweep (`exit_sign_sweep.py`, `exit_inspection/sweep.py`, on with `exits:=true`, `exit_sweep:=false` turns it
off): TARE stops once the lidar has seen everything, but the camera sees a sign 2.2-2.5 m up only from ~6.8 m and only
where the robot faces. A 2D grid of walls, free space and camera looks is kept during the exploration; at the end
the finish node stops TARE and the sweep walks the robot to up to 8 viewpoints facing the walls not looked at (two
stops 0.5 m apart each), then home. Spec: `docs/superpowers/specs/2026-11-12-exit-sign-sweep-design.md`.

Classes: `exit_plain` (no arrow), `exit_arrow_down`, `exit_arrow_side`. `door_classes` (default plain and arrow down)
mark a door; the others are route signs, mapped but never inspected. A door exit is confirmed after 5 sightings from
robot positions at least 0.5 m apart, or 8 from at least 0.25 m.

## Zone verdicts (sub-project 2)

```
exits.yaml (confirmed door exits)
   │
exit_zone_checker.py ◄── /camera/depth/image_rect_raw (raw D435i depth, 86 x 58 deg), /registered_scan_static,
   │                     /person_detector/detections (Yolo26RGBD, COCO person), TF as the mapper
   │  per exit: frames of the last 3 s -> 5 cm grid of the zone (door width + 2 x 0.5 m, 3 m deep, floor + 5 cm to
   │  2 m): free (floor seen) / object / unseen / person / behind a person; a 0.9 m disk must get out from the door
   ├─► /exit_inspection/zone_verdicts  (std_msgs/String, YAML, transient local)
   ├─► /exit_inspection/zone_markers   (rviz: zone outline colored by verdict, object / unseen / people cells)
   └─► <exits.yaml folder>/zone_checks.yaml
```

| verdict | |
|---|---|
| CLEAR | a corridor >= w_min through cells seen free, no object, >= 90 % of the zone seen |
| WARNING | such a corridor, objects in the zone |
| BLOCKED | no corridor even if every unseen cell were free |
| UNVERIFIED | the corridor needs unseen cells, or the zone is partly unseen, or no data |

Why depth and lidar: the Airy (dome pitched 5 deg) sees nothing below its 0.42 m to the sides and low objects ahead only
from 4.2 m; the raw depth sees the floor from 0.5 m. From about 4.5 m in front of a door, facing it, the zone is seen
whole. The node keeps the latest verdict of every exit; the inspection pass (sub-project 3) decides when it counts.

```bash
ros2 launch exit_inspection exit_zone_check_launch.xml exits_file:=<save_directory>/exit_inspection/exits.yaml
```

Parameters: `config/exit_zone_checker.yaml` (`w_min`, `margin`, `depth`, heights, window, ...).

## Inspection mission (sub-project 3)

After an exploration with `exits:=true` (exits.yaml, optimized_map.pcd, warehouse.vgh in SC-PGO's save_data), with
FAST-LIO restarted at the exploration's spawn:

```bash
ros2 launch fast_lio fast_lio_slam_launch.xml
ros2 launch exit_inspection exit_mission_launch.xml          # far_planner + zone checker + exit_inspection_pass.py
python3 src/FAST_LIO_SLAM_ROS2/SC_PGO_ROS2/utils/map_point_picker/map_point_picker.py --port 8770
# "Start exit inspection" in the page, or: ros2 topic pub -1 /start_exit_inspection std_msgs/msg/String "{data: ''}"
```

`exit_inspection_pass.py`: viewpoints 4.5 m in front of each door (`exit_inspection/viewpoints.py`, checked against
the map: the robot fits, the line of sight to the zone is clear), nearest door first; goal on `/goal_point` plus
`/goal_yaw_hint` (the goal bridge turns the robot to the door there); verdict = majority of the checker's evaluations
once its 3 s window holds only frames of the stop; UNVERIFIED or unreached: the next viewpoint (2 attempts); back to
the start; report in `save_data/exit_inspection/reports/<stamp>/report.html` (+ report.yaml, exit_report.yaml).
`planner:=direct far:=false`: straight goals to the controller (open rooms, tests). Parameters:
`config/exit_inspection_pass.yaml`.

## Running

With an exploration:

```bash
ros2 launch tare_planner explore_warehouse_launch.xml exits:=true   # + exit_sign_model:=<onnx>
```

Alone, next to FAST-LIO/SC-PGO and the camera:

```bash
ros2 launch exit_inspection exit_detection_launch.xml model:=$HOME/RaNAV/models/exit_sign/exit_sign_736x1280.onnx
ros2 run exit_inspection finalize_exits.py <save_directory>     # at the end
```

Parameters: `config/exit_landmark_mapper.yaml`.

## Simulation

- `launch/exit_room_launch.xml`: the exit test room (`worlds/exit_room.world`, ground truth `worlds/exit_room_gt.yaml`),
  same arguments as `empty_room_launch.xml` plus `yaw`. Five cases: clear, warning, blocked, person in front, route
  sign without a door.
- The world, the sign models and the camera rig are written by `sim/generate_exit_room_world.py`: never edit them.
- `test/sim_exit_mapping.sh <work_dir>`: headless end to end (sim, acados controller, FAST-LIO/SC-PGO, detector,
  mapper, scripted walk, finalize, `check_exits_gt.py`).
- `test/sim_exit_full.sh <work_dir>`: the whole chain headless: TARE exploration with `exits:=true` (sign sweep at the
  end), `check_exits_gt.py`; then everything restarted at the spawn and the FAR mission on the saved graph,
  `check_zones_gt.py` on the report (`SKIP_EXPLORE=1` reuses an exploration). About 11 min.
- `test/sim_exit_mission.sh <work_dir>`: headless mission (`planner:=direct`, ground-truth exits, auto start), report
  checked with `check_zones_gt.py`.
- `test/sim_exit_zones.sh <work_dir>`: headless zone verdicts (exits from the ground truth via `test/gt_to_exits.py`, or
  `EXITS=<exits.yaml>`), the robot stops 4.5 m in front of each door, `check_zones_gt.py` against the expected
  verdicts.

## Detector training (`training/`, `.venv-yolo` python)

1. `ros2 launch exit_inspection harvest_renders_launch.xml`, then `sim/harvest_renders.py` (system python): labeled
   frames of the exit room for the evaluation (`-n 600 --seed 1`) and sign-free backgrounds (`--aim 0.15 --seed 2`).
2. `training/make_synthetic.py`: signs from `exit_inspection/sign_art.py` pasted on background photos.
3. `training/train.py`: YOLO26n fine-tune at 1280, ONNX export at 736x1280 (raw head, static shape).
4. `training/evaluate.py <onnx> <harvested frames>`: recall by distance / box size / view angle; gate recall >= 0.9
   up to 10 m.

Pictograms: `exit_inspection/assets/iso7010/` (Wikimedia Commons, public domain).

## Tests

```bash
python3 -m pytest src/exit_inspection/test -q
```
