#!/usr/bin/env python3
"""
gt_to_exits.py (test tool)

exits.yaml of the exit room's ground truth (worlds/exit_room_gt.yaml) in camera_init, the lidar frame where FAST-LIO
started (--base0 x y z yaw of the robot base then, --lidar-offset as in check_exits_gt.py): tests the zone checker
without the sign detector and the mapping.

Usage: gt_to_exits.py exit_room_gt.yaml out/exit_inspection/exits.yaml --base0 0 0 0.38 0
"""

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from check_exits_gt import world_to_map  # noqa: E402


def exits_from_gt(gt_doc: dict, T: np.ndarray) -> dict:
    out = []
    for i, g in enumerate(gt_doc["exits"]):
        r3 = lambda a: [round(float(x), 4) for x in a]
        c = T[:3, :3] @ np.array(g["sign_center"], float) + T[:3, 3]
        n = T[:3, :3] @ np.array(g["normal"], float)
        e = {"id": i, "type": g["type"], "cls": g["cls"], "confirmed": True, "sign_center": r3(c), "normal": r3(n),
             "n_sightings": 0, "name": g["name"]}
        if g["type"] == "door":
            f = T[:3, :3] @ np.array(g["door_floor_center"], float) + T[:3, 3]
            e.update(door_floor_center=r3(f), door_width=float(gt_doc.get("door_width", 1.2)),
                     door_height=float(gt_doc.get("door_height", 2.1)))
        out.append(e)
    return {"frame": "camera_init (from the ground truth)", "anchored": "ground_truth", "exits": out}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gt")
    ap.add_argument("out")
    ap.add_argument("--base0", nargs=4, type=float, required=True, metavar=("X", "Y", "Z", "YAW"))
    ap.add_argument("--lidar-offset", nargs=3, type=float, default=[0.2835, 0.0, 0.066])
    args = ap.parse_args()
    with open(args.gt) as f:
        gt_doc = yaml.safe_load(f)
    doc = exits_from_gt(gt_doc, world_to_map(args.base0, args.lidar_offset))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=None)
    print(f"{len(doc['exits'])} exits to {args.out}")


if __name__ == "__main__":
    main()
