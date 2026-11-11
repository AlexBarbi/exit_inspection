#!/usr/bin/env python3
"""
check_exits_gt.py

Compares an exits.yaml (finalize_exits.py) with the ground truth of the exit room (worlds/exit_room_gt.yaml).

exits.yaml is in camera_init, the lidar frame where FAST-LIO started: its world pose is the robot base pose at that
moment (--base0 x y z yaw, from `gz model -m aliengo -p` with the robot standing still) times the static base -> lidar
offset of fast_lio_slam_launch.xml (--lidar-offset, lio_map -> camera_init).

Pass when every door of the ground truth has a confirmed door exit within --max-distance (sign center, 3D) and
--max-angle (normal), every route sign that was mapped is typed route, and no confirmed door exit is left unmatched.

Usage: check_exits_gt.py exits.yaml exit_room_gt.yaml --base0 0 0 0.38 0
"""

import argparse
import math
import sys

import numpy as np
import yaml


def world_to_map(base0, lidar_offset):
    x, y, z, yaw = base0
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = [x, y, z]
    L = np.eye(4)
    L[:3, 3] = lidar_offset
    return np.linalg.inv(T @ L)  # world -> camera_init


def check(exits_doc, gt_doc, T, max_distance, max_angle):
    """(passed, report lines)."""
    lines, ok = [], True
    exits = exits_doc.get("exits", [])
    cos_max = math.cos(math.radians(max_angle))
    matched = set()
    for g in gt_doc["exits"]:
        c = T[:3, :3] @ np.array(g["sign_center"]) + T[:3, 3]
        n = T[:3, :3] @ np.array(g["normal"])
        best, best_d = None, max_distance
        for e in exits:
            d = float(np.linalg.norm(np.array(e["sign_center"]) - c))
            if d < best_d and np.array(e["normal"]) @ n >= cos_max:
                best, best_d = e, d
        if g["type"] == "door":
            good = best is not None and best["type"] == "door" and best["confirmed"]
            ok &= good
            if best is not None:
                matched.add(best["id"])
                ang = math.degrees(math.acos(min(1.0, float(np.array(best["normal"]) @ n))))
                lines.append(f"{'OK  ' if good else 'FAIL'} door {g['name']}: exit {best['id']} ({best['type']}, "
                             f"{'confirmed' if best['confirmed'] else 'unconfirmed'}, {best['n_sightings']} sightings) "
                             f"{best_d:.2f} m, {ang:.1f} deg off")
            else:
                lines.append(f"FAIL door {g['name']}: no exit within {max_distance} m / {max_angle} deg")
        else:
            if best is None:
                lines.append(f"OK   route sign {g['name']}: not mapped")
            else:
                matched.add(best["id"])
                good = best["type"] == "route"
                ok &= good
                lines.append(f"{'OK  ' if good else 'FAIL'} route sign {g['name']}: exit {best['id']} typed "
                             f"{best['type']} ({best['n_sightings']} sightings), {best_d:.2f} m")
    for e in exits:
        if e["type"] == "door" and e["confirmed"] and e["id"] not in matched:
            ok = False
            lines.append(f"FAIL false exit {e['id']} at {e['sign_center']} ({e['n_sightings']} sightings)")
    unconfirmed = [e["id"] for e in exits if e["id"] not in matched and not e["confirmed"]]
    if unconfirmed:
        lines.append(f"     unconfirmed, unmatched landmarks (not failures): {unconfirmed}")
    return ok, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exits")
    ap.add_argument("gt")
    ap.add_argument("--base0", nargs=4, type=float, required=True, metavar=("X", "Y", "Z", "YAW"))
    ap.add_argument("--lidar-offset", nargs=3, type=float, default=[0.2835, 0.0, 0.066])
    ap.add_argument("--max-distance", type=float, default=0.3)
    ap.add_argument("--max-angle", type=float, default=15.0)
    args = ap.parse_args()
    with open(args.exits) as f:
        exits_doc = yaml.safe_load(f)
    with open(args.gt) as f:
        gt_doc = yaml.safe_load(f)
    ok, lines = check(exits_doc, gt_doc, world_to_map(args.base0, args.lidar_offset), args.max_distance,
                      args.max_angle)
    print(f"exits.yaml anchored: {exits_doc.get('anchored')}")
    print("\n".join(lines))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
