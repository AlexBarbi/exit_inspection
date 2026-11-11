#!/usr/bin/env python3
"""
finalize_exits.py

End of an exploration: re-anchors the exit sign sightings that exit_landmark_mapper logged
(<save_directory>/exit_inspection/sightings.csv) on SC-PGO's optimized poses (times.txt, odom_poses.txt,
optimized_poses.txt in <save_directory>), merges them into exits and writes <save_directory>/exit_inspection/exits.yaml.
Without usable pose files the exits keep the live odometry coordinates (anchored: odometry, with a warning).

tare_finish_merge_map.py runs it after the map merge (parameter exits); by hand:
  ros2 run exit_inspection finalize_exits.py ~/RaNAV/src/FAST_LIO_SLAM_ROS2/save_data
"""

import argparse
import sys

from exit_inspection.finalize import finalize
from exit_inspection.landmarks import MapperParams


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("save_directory", help="SC-PGO save directory (holds times.txt and the pose files)")
    ap.add_argument("--door-classes", nargs="+", default=["exit_plain", "exit_arrow_down"],
                    help="detector classes that mark a door (the others are route signs)")
    ap.add_argument("--door-width", type=float, default=MapperParams.door_width)
    ap.add_argument("--door-height", type=float, default=MapperParams.door_height)
    ap.add_argument("--confirm-sightings", type=int, default=MapperParams.confirm_sightings)
    ap.add_argument("--confirm-baseline", type=float, default=MapperParams.confirm_baseline)
    args = ap.parse_args()

    params = MapperParams(door_width=args.door_width, door_height=args.door_height,
                          confirm_sightings=args.confirm_sightings, confirm_baseline=args.confirm_baseline)
    data = finalize(args.save_directory, params, args.door_classes)
    doors = [e for e in data["exits"] if e["type"] == "door"]
    confirmed = [e for e in doors if e["confirmed"]]
    print(f"{len(confirmed)} door exits confirmed ({len(doors)} door landmarks, "
          f"{len(data['exits']) - len(doors)} route signs), anchored: {data['anchored']}")
    if "warning" in data:
        print("warning:", data["warning"], file=sys.stderr)
    for e in confirmed:
        print(f"  exit {e['id']}: sign {e['sign_center']} normal {e['normal']} ({e['n_sightings']} sightings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
