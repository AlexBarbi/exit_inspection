"""End of the exploration: sightings re-anchored on SC-PGO's optimized poses, merged into exits.yaml.

The mapper logs its sightings in camera_init as FAST-LIO had it live. SC-PGO's keyframes have both poses, the live
one (odom_poses.txt, /aft_mapped_to_init) and the optimized one (optimized_poses.txt), with their stamps (times.txt).
Each sighting takes the correction T_opt * T_odom^-1 of the keyframe nearest in time, so the exits line up with
optimized_map.pcd, the map the inspection pass, far_planner's graph and map_point_picker use.
"""

import copy
import os
from dataclasses import asdict
from typing import List, Sequence, Tuple

import numpy as np
import yaml

from exit_inspection import records
from exit_inspection.landmarks import LandmarkSet, MapperParams, Sighting

SUBDIR = "exit_inspection"
SIGHTINGS = "sightings.csv"
EXITS = "exits.yaml"


def reanchor(sightings: Sequence[Sighting], times: np.ndarray, odom: np.ndarray,
             optimized: np.ndarray) -> List[Sighting]:
    out = []
    for s in sightings:
        k = int(np.argmin(np.abs(times - s.stamp)))
        C = optimized[k] @ np.linalg.inv(odom[k])
        R, t = C[:3, :3], C[:3, 3]
        floor_point = np.array([s.hit[0], s.hit[1], s.floor_z])
        n = R @ s.normal
        n[2] = 0.0
        r = copy.copy(s)
        r.hit = R @ s.hit + t
        r.normal = n / np.linalg.norm(n)
        r.robot = R @ s.robot + t
        r.floor_z = float((R @ floor_point + t)[2])
        out.append(r)
    return out


def load_poses(save_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """(times, odom, optimized, problem); problem is "" when the three files are there and agree."""
    paths = [os.path.join(save_dir, n) for n in ("times.txt", "odom_poses.txt", "optimized_poses.txt")]
    missing = [p for p in paths if not os.path.exists(p) or os.path.getsize(p) == 0]
    if missing:
        return None, None, None, "missing " + ", ".join(os.path.basename(p) for p in missing)
    times, odom, opt = records.load_times(paths[0]), records.load_kitti_poses(paths[1]), records.load_kitti_poses(
        paths[2])
    if not len(times) == len(odom) == len(opt):
        return None, None, None, f"row counts differ: times {len(times)}, odom {len(odom)}, optimized {len(opt)}"
    return times, odom, opt, ""


def _r(v, digits=4):
    return [round(float(x), digits) for x in v]


def finalize(save_dir: str, params: MapperParams, door_classes: Sequence[str]) -> dict:
    """Writes <save_dir>/exit_inspection/exits.yaml and returns its content."""
    out_dir = os.path.join(save_dir, SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, SIGHTINGS)
    sightings = records.read_sightings(path) if os.path.exists(path) else []
    sightings.sort(key=lambda s: s.stamp)

    times, odom, opt, problem = load_poses(save_dir)
    anchored = "odometry"
    if sightings and not problem:
        sightings = reanchor(sightings, times, odom, opt)
        anchored = "optimized"

    lms = LandmarkSet(params, door_classes)
    for s in sightings:
        lms.add(s)
    exits = []
    for l in lms.landmarks():
        exits.append({
            "id": l.id,
            "type": l.group,
            "cls": l.cls,
            "confirmed": bool(l.confirmed),
            "sign_center": _r(l.center),
            "normal": _r(l.normal),
            "sign_width": round(l.width, 3),
            "door_floor_center": _r(l.door_floor_center),
            "door_width": params.door_width,
            "door_height": params.door_height,
            "n_sightings": len(l.sightings),
            "mean_conf": round(l.mean_conf, 3),
            "low_quality_fraction": round(float(np.mean([s.low_quality for s in l.sightings])), 3),
        })
    data = {
        "frame": "camera_init of the exploration (optimized_map.pcd)" if anchored == "optimized" else
                 "camera_init of the exploration, live odometry (not aligned with optimized_map.pcd)",
        "anchored": anchored,
        "door_classes": list(door_classes),
        "params": asdict(params),
        "exits": exits,
    }
    if problem and sightings:
        data["warning"] = f"not re-anchored: {problem}"
    with open(os.path.join(out_dir, EXITS), "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return data
