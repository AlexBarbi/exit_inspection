"""Files: the sightings CSV written live by the mapper, SC-PGO's pose files (times.txt, KITTI 3x4 poses)."""

import csv
import os
from typing import List

import numpy as np

from exit_inspection.landmarks import Sighting

COLUMNS = ["stamp", "cls", "conf", "hit_x", "hit_y", "hit_z", "normal_x", "normal_y", "normal_z", "robot_x", "robot_y",
           "robot_z", "range", "width", "floor_z", "low_quality"]


class SightingWriter:
    """Appends sightings to a CSV, one flushed row each, the header only when the file is new."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._file = open(path, "a", newline="")
        self._csv = csv.writer(self._file)
        if new:
            self._csv.writerow(COLUMNS)
            self._file.flush()

    def write(self, s: Sighting) -> None:
        self._csv.writerow([f"{s.stamp:.6f}", s.cls, f"{s.conf:.4f}", *(f"{v:.4f}" for v in s.hit),
                            *(f"{v:.5f}" for v in s.normal), *(f"{v:.4f}" for v in s.robot), f"{s.range_m:.3f}",
                            f"{s.width_m:.4f}", f"{s.floor_z:.4f}", int(s.low_quality)])
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def read_sightings(path: str) -> List[Sighting]:
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            vec = lambda p: np.array([float(row[p + "_x"]), float(row[p + "_y"]), float(row[p + "_z"])])
            out.append(Sighting(stamp=float(row["stamp"]), cls=row["cls"], conf=float(row["conf"]), hit=vec("hit"),
                                normal=vec("normal"), robot=vec("robot"), range_m=float(row["range"]),
                                width_m=float(row["width"]), floor_z=float(row["floor_z"]),
                                low_quality=row["low_quality"] == "1"))
    return out


def load_kitti_poses(path: str) -> np.ndarray:
    """(N, 4, 4) poses of a KITTI file: one row of r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz per pose."""
    rows = np.atleast_2d(np.loadtxt(path))
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    poses[:, :3, :] = rows.reshape(-1, 3, 4)
    return poses


def load_times(path: str) -> np.ndarray:
    return np.atleast_1d(np.loadtxt(path))
