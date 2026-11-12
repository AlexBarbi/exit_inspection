"""
Viewpoints in front of the exit doors for the inspection pass (sub-project 3 of the emergency exit inspection,
docs/superpowers/specs/2026-11-11-exit-inspection-pass-design.md).

Candidates per door: standoffs along the door normal from the door floor center, each with lateral offsets along the
wall, yaw facing the door center. With a map (2D occupancy of its points in the robot's height band) a candidate is
kept when the robot fits there and the line of sight to the zone's far edge is clear; objects inside the zone do not
count (they are what is inspected).
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ViewpointParams:
    standoffs: Sequence[float] = (4.5, 4.0, 5.0, 3.5, 5.5, 3.0)  # m from the door, in order of preference
    lateral: Sequence[float] = (0.0, 0.6, -0.6, 1.2, -1.2)        # m along the wall
    robot_radius: float = 0.45
    sight_clearance: float = 0.2
    band_low: float = 0.15      # occupancy band above the floor
    band_high: float = 1.0
    resolution: float = 0.1
    min_points: int = 2         # map points for an occupied cell
    zone_depth: float = 3.0     # the line of sight ends at the zone's far edge


@dataclass
class Viewpoint:
    x: float
    y: float
    yaw: float
    standoff: float
    lateral: float

    def as_dict(self) -> dict:
        return {"x": round(self.x, 3), "y": round(self.y, 3), "yaw": round(self.yaw, 4), "standoff": self.standoff,
                "lateral": self.lateral}


class Occupancy:
    """2D occupancy grid (bool, [ix, iy]) with origin at the corner of cell (0, 0)."""

    def __init__(self, origin: np.ndarray, resolution: float, grid: np.ndarray):
        self.origin, self.resolution, self.grid = np.asarray(origin, float), float(resolution), grid

    @classmethod
    def from_points(cls, xyz: np.ndarray, floor_z: float, p: ViewpointParams) -> "Occupancy":
        xyz = np.asarray(xyz, float)
        band = xyz[(xyz[:, 2] >= floor_z + p.band_low) & (xyz[:, 2] <= floor_z + p.band_high), :2]
        if len(band) == 0:
            return cls(np.zeros(2), p.resolution, np.zeros((1, 1), bool))
        origin = band.min(0) - p.resolution
        idx = np.floor((band - origin) / p.resolution).astype(int)
        shape = idx.max(0) + 2
        counts = np.zeros(shape, int)
        np.add.at(counts, (idx[:, 0], idx[:, 1]), 1)
        return cls(origin, p.resolution, counts >= p.min_points)

    def occupied_near(self, xy, radius: float) -> bool:
        """An occupied cell center within radius of xy (outside the grid is free)."""
        r = self.resolution
        lo = np.floor((np.asarray(xy) - radius - self.origin) / r).astype(int)
        hi = np.floor((np.asarray(xy) + radius - self.origin) / r).astype(int)
        lo, hi = np.maximum(lo, 0), np.minimum(hi, np.array(self.grid.shape) - 1)
        if (hi < lo).any():
            return False
        sub = self.grid[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1]
        if not sub.any():
            return False
        ix, iy = np.nonzero(sub)
        centers = self.origin + (np.column_stack([ix + lo[0], iy + lo[1]]) + 0.5) * r
        return bool((np.hypot(*(centers - xy).T) <= radius).any())

    def segment_clear(self, a, b, clearance: float) -> bool:
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / (self.resolution / 2))))
        return not any(self.occupied_near(a + (b - a) * t, clearance) for t in np.linspace(0.0, 1.0, n + 1))


def door_frame(exit_: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(door floor center xy, unit normal into the room, unit along the wall)."""
    c = np.array(exit_["door_floor_center"], float)[:2]
    n = np.array(exit_["normal"], float)[:2]
    n /= np.linalg.norm(n)
    return c, n, np.array([-n[1], n[0]])


def candidates(exit_: dict, p: ViewpointParams) -> List[Viewpoint]:
    c, n, t = door_frame(exit_)
    out = []
    for d in p.standoffs:
        for s in p.lateral:
            q = c + d * n + s * t
            out.append(Viewpoint(float(q[0]), float(q[1]), math.atan2(*(c - q)[::-1]), float(d), float(s)))
    return out


def check(exit_: dict, vp: Viewpoint, occ: Optional[Occupancy], p: ViewpointParams) -> Tuple[bool, str]:
    if occ is None:
        return True, "no map"
    q = np.array([vp.x, vp.y])
    if occ.occupied_near(q, p.robot_radius):
        return False, "robot does not fit"
    c, n, _ = door_frame(exit_)
    far_edge = c + p.zone_depth * n
    if vp.standoff > p.zone_depth and not occ.segment_clear(q, far_edge, p.sight_clearance):
        return False, "line of sight blocked"
    return True, "ok"


@dataclass
class ExitPlan:
    exit: dict
    viewpoints: List[Viewpoint] = field(default_factory=list)  # kept candidates, best first
    rejected: dict = field(default_factory=dict)                # reason -> count


def plan(exits: Sequence[dict], p: ViewpointParams, occ: Optional[Occupancy] = None) -> List[ExitPlan]:
    out = []
    for e in exits:
        ep = ExitPlan(e)
        for vp in candidates(e, p):
            ok, why = check(e, vp, occ, p)
            if ok:
                ep.viewpoints.append(vp)
            else:
                ep.rejected[why] = ep.rejected.get(why, 0) + 1
        out.append(ep)
    return out


def tour(start_xy, plans: Sequence[ExitPlan]) -> List[int]:
    """Visiting order (indices into plans): nearest first viewpoint from where the robot is; exits without a
    viewpoint last."""
    todo = [i for i, ep in enumerate(plans) if ep.viewpoints]
    order, here = [], np.asarray(start_xy, float)
    while todo:
        i = min(todo, key=lambda k: np.hypot(plans[k].viewpoints[0].x - here[0], plans[k].viewpoints[0].y - here[1]))
        order.append(i)
        todo.remove(i)
        here = np.array([plans[i].viewpoints[0].x, plans[i].viewpoints[0].y])
    return order + [i for i, ep in enumerate(plans) if not ep.viewpoints]
