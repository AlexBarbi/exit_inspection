#!/usr/bin/env python3
"""
check_zones_gt.py

Compares zone verdicts (zone_checks.yaml of exit_zone_checker.py, one snapshot per case taken while the robot stood in
front of that door) with the ground truth of the exit room (worlds/exit_room_gt.yaml, `expected` of each door).

The verdicts are in camera_init, the lidar frame where FAST-LIO started: its world pose is the robot base pose at that
moment (--base0 x y z yaw) times the static base -> lidar offset (--lidar-offset), as in check_exits_gt.py. In each
snapshot the exit whose door floor center is within --max-distance (horizontal) of the case's door is taken.

Pass when every door case has a snapshot and its verdict is the expected one.

Usage: check_zones_gt.py exit_room_gt.yaml clear=zc_clear.yaml warning=zc_warning.yaml ... --base0 0 0 0.38 0
"""

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_exits_gt import world_to_map  # noqa: E402


def door_floor_world(g: dict) -> np.ndarray:
    if "door_floor_center" in g:
        return np.array(g["door_floor_center"], dtype=float)
    c = np.array(g["sign_center"], dtype=float)
    return np.array([c[0], c[1], 0.0])


def check(snapshots: dict, gt_doc: dict, T: np.ndarray, max_distance: float):
    """(passed, report lines); snapshots: case name -> zone_checks document (or None when missing)."""
    ok, lines = True, []
    for g in gt_doc["exits"]:
        if g["type"] != "door":
            continue
        name, expected = g["name"], g.get("expected")
        doc = snapshots.get(name)
        if doc is None:
            ok = False
            lines.append(f"FAIL {name}: no snapshot")
            continue
        door = T[:3, :3] @ door_floor_world(g) + T[:3, 3]
        best, best_d = None, max_distance
        for e in doc.get("exits") or []:
            d = float(np.hypot(*(np.array(e["door_floor_center"][:2]) - door[:2])))
            if d <= best_d:
                best, best_d = e, d
        if best is None:
            ok = False
            lines.append(f"FAIL {name}: no verdict for its door (expected {expected})")
            continue
        good = best["verdict"] == expected
        ok &= good
        lines.append(f"{'OK  ' if good else 'FAIL'} {name}: exit {best['id']} {best['verdict']} (expected {expected}; "
                     f"{best['reason']}; coverage {best['coverage']}, {best['frames']} frames, {best['people']} people)")
    return ok, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gt")
    ap.add_argument("snapshots", nargs="+", metavar="CASE=ZONE_CHECKS")
    ap.add_argument("--base0", nargs=4, type=float, required=True, metavar=("X", "Y", "Z", "YAW"))
    ap.add_argument("--lidar-offset", nargs=3, type=float, default=[0.2835, 0.0, 0.066])
    ap.add_argument("--max-distance", type=float, default=0.5)
    args = ap.parse_args()
    with open(args.gt) as f:
        gt_doc = yaml.safe_load(f)
    snapshots = {}
    for item in args.snapshots:
        name, _, path = item.partition("=")
        try:
            with open(path) as f:
                snapshots[name] = yaml.safe_load(f) or {}
        except OSError:
            snapshots[name] = None
    ok, lines = check(snapshots, gt_doc, world_to_map(args.base0, args.lidar_offset), args.max_distance)
    print("\n".join(lines))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
