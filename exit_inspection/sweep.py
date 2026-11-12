"""
Exit sign sweep at the end of an exploration (emergency exit inspection,
docs/superpowers/specs/2026-11-12-exit-sign-sweep-design.md).

TARE explores until the lidar has seen everything; the D435i (0.35 m high, pitched 5 deg down) sees a sign 2.2-2.5 m up
only from about 6.8 m, and only where the robot happens to face. A 2D grid built from the lidar scans during the
exploration holds the walls (points in the sign band), the low obstacles, the free space the lidar rays crossed, and
how often the camera looked at each wall cell from a distance and angle a sign would be detected at. At the end the
wall cells not looked at are covered greedily by viewpoints (position, yaw) the robot then visits.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class SweepParams:
    cell: float = 0.2
    size: float = 160.0            # grid side [m], centred on the start
    wall_low: float = 1.5          # sign band above the floor: walls (low obstacles excluded)
    wall_high: float = 2.8
    obstacle_low: float = 0.15     # robot body band: obstacles for the viewpoints
    obstacle_high: float = 1.0
    min_hits: int = 2              # points for an occupied cell
    clearance: float = 0.6         # viewpoint distance to obstacles and walls
    d_min: float = 6.8             # a sign at ~2.25 m enters the camera image from ~6.6 m
    d_max: float = 10.0            # detector recall falls beyond
    half_fov_deg: float = 28.0     # of the 69 deg of the colour image, with a margin
    max_incidence_deg: float = 50.0
    candidate_step: float = 1.0
    yaw_count: int = 16
    covered_looks: int = 2         # camera looks for a wall cell to count as seen
    min_new_cells: int = 10        # a viewpoint must cover at least this many new wall cells
    max_viewpoints: int = 8
    target_coverage: float = 0.9   # of the wall cells some viewpoint can cover
    max_rays: int = 3000           # free-space rays per scan
    bin_deg: float = 0.5           # angular resolution of the visibility z-buffer


def _box_sum(a: np.ndarray, r: int) -> np.ndarray:
    """Sum of a over the (2r+1)^2 window around each cell (zero padding)."""
    p = np.pad(a.astype(np.int64), r + 1)
    c = p.cumsum(0).cumsum(1)
    n0, n1 = a.shape
    k = 2 * r + 1
    return (c[k:k + n0, k:k + n1] - c[:n0, k:k + n1] - c[k:k + n0, :n1] + c[:n0, :n1])


class SweepGrid:
    def __init__(self, center_xy, p: SweepParams):
        self.p = p
        self.n = int(round(p.size / p.cell))
        self.origin = np.asarray(center_xy, float) - p.size / 2
        shape = (self.n, self.n)
        self.wall = np.zeros(shape, np.int32)
        self.obstacle = np.zeros(shape, np.int32)
        self.free = np.zeros(shape, np.int32)
        self.looks = np.zeros(shape, np.int32)

    # ------------------------------------------------------------------ indexing
    def index(self, xy: np.ndarray):
        ij = np.floor((np.asarray(xy, float).reshape(-1, 2) - self.origin) / self.p.cell).astype(int)
        ok = (ij >= 0).all(1) & (ij < self.n).all(1)
        return ij, ok

    def center(self, ij: np.ndarray) -> np.ndarray:
        return self.origin + (np.asarray(ij) + 0.5) * self.p.cell

    def _add(self, arr: np.ndarray, xy: np.ndarray):
        ij, ok = self.index(xy)
        np.add.at(arr, (ij[ok, 0], ij[ok, 1]), 1)

    # ------------------------------------------------------------------ building
    def add_scan(self, origin_xy, pts: np.ndarray, floor_z: float, rng=None):
        """One scan (map frame): walls, obstacles, and free cells along the rays to the low points."""
        p = self.p
        pts = np.asarray(pts, float).reshape(-1, 3)
        h = pts[:, 2] - floor_z
        self._add(self.wall, pts[(h >= p.wall_low) & (h <= p.wall_high), :2])
        self._add(self.obstacle, pts[(h >= p.obstacle_low) & (h <= p.obstacle_high), :2])
        low = pts[h < p.obstacle_high, :2]
        if len(low) > p.max_rays:
            low = low[(rng or np.random.default_rng(0)).choice(len(low), p.max_rays, replace=False)]
        if len(low) == 0:
            return
        o = np.asarray(origin_xy, float)
        d = low - o
        dist = np.hypot(d[:, 0], d[:, 1])
        step = p.cell / 2
        t = np.arange(int(math.ceil(dist.max() / step))) * step
        ok = t[None, :] < (dist[:, None] - 0.3)  # stop short of the return
        q = o + d[:, None, :] / np.maximum(dist, 1e-6)[:, None, None] * t[None, :, None]
        ij, valid = self.index(q[ok])
        ij = ij[valid]
        if len(ij):
            keys = np.unique(ij[:, 0].astype(np.int64) * self.n + ij[:, 1])
            self.free[keys // self.n, keys % self.n] += 1

    def walls(self) -> np.ndarray:
        return self.wall >= self.p.min_hits

    def normals(self) -> np.ndarray:
        """(n, n, 2) unit normal of the wall cells, pointing to the free side (zero where unknown)."""
        free = (self.free > 0) & ~self.walls()
        blur = _box_sum(free, 3).astype(float)
        gx, gy = np.gradient(blur)
        g = np.stack([gx, gy], -1)
        norm = np.linalg.norm(g, axis=-1, keepdims=True)
        return np.where(norm > 1e-6, g / np.maximum(norm, 1e-6), 0.0)

    # ------------------------------------------------------------------ visibility
    def visible(self, xy, walls_ij: np.ndarray, normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Of the wall cells walls_ij (m, 2): which are seen from xy (first wall along their bearing, d_min..d_max,
        facing xy within max_incidence) and their bearings."""
        p = self.p
        c = self.center(walls_ij)
        d = c - np.asarray(xy, float)
        dist = np.hypot(d[:, 0], d[:, 1])
        bearing = np.arctan2(d[:, 1], d[:, 0])
        near = dist <= p.d_max + p.cell
        nbins = int(round(360.0 / p.bin_deg))
        b = ((bearing + math.pi) / (2 * math.pi) * nbins).astype(int) % nbins
        first = np.full(nbins, np.inf)
        # each cell blocks the bins of its angular width (a 0.2 m cell 3 m away spans ~4 deg, 8 bins)
        half = np.ceil(np.degrees(np.arctan2(0.6 * p.cell, np.maximum(dist, 0.5))) / p.bin_deg).astype(int)
        bn, dn, hn = b[near], dist[near], half[near]
        for off in range(-int(hn.max(initial=0)), int(hn.max(initial=0)) + 1):
            m = hn >= abs(off)
            np.minimum.at(first, (bn[m] + off) % nbins, dn[m])
        nrm = normals[walls_ij[:, 0], walls_ij[:, 1]]
        facing = (-(d * nrm).sum(1)) >= np.cos(np.radians(p.max_incidence_deg)) * np.maximum(dist, 1e-6)
        seen = near & (dist <= first[b] + 1.5 * p.cell) & (dist >= p.d_min) & (dist <= p.d_max) & facing
        return seen, bearing

    def mark_look(self, cam_xy, cam_yaw: float):
        """The camera at cam_xy looking along cam_yaw: its wall cells get one more look."""
        walls_ij = np.argwhere(self.walls())
        if len(walls_ij) == 0:
            return
        seen, bearing = self.visible(cam_xy, walls_ij, self.normals())
        ahead = np.abs((bearing - cam_yaw + math.pi) % (2 * math.pi) - math.pi) <= math.radians(self.p.half_fov_deg)
        hit = walls_ij[seen & ahead]
        self.looks[hit[:, 0], hit[:, 1]] += 1

    # ------------------------------------------------------------------ planning
    def candidates(self) -> np.ndarray:
        """Cell indices (k, 2) on the candidate lattice where the robot can stand and the lidar saw free space."""
        p = self.p
        occupied = (self.obstacle >= p.min_hits) | self.walls()
        r = int(math.ceil(p.clearance / p.cell))
        blocked = _box_sum(occupied, r) > 0
        stride = max(1, int(round(p.candidate_step / p.cell)))
        mask = np.zeros_like(blocked)
        mask[::stride, ::stride] = True
        return np.argwhere(mask & (self.free > 0) & ~blocked)

    def plan(self, start_xy) -> Tuple[List[Tuple[float, float, float, int]], dict]:
        """Viewpoints (x, y, yaw, new wall cells) covering the wall cells not looked at yet, nearest first."""
        p = self.p
        walls = self.walls()
        normals = self.normals()
        has_normal = np.linalg.norm(normals, axis=-1) > 0.5
        targets_ij = np.argwhere(walls & has_normal & (self.looks < p.covered_looks))
        stats = {"wall_cells": int(walls.sum()), "looked_at": int((walls & (self.looks >= p.covered_looks)).sum()),
                 "targets": int(len(targets_ij)), "coverable": 0, "planned": 0}
        if len(targets_ij) == 0:
            return [], stats
        cands = self.candidates()
        tc = self.center(targets_ij)
        yaws = np.linspace(-math.pi, math.pi, p.yaw_count, endpoint=False)
        half = math.radians(p.half_fov_deg)
        options = []  # (xy, yaw, target indices)
        for ij in cands:
            xy = self.center(ij)
            dd = np.hypot(*(tc - xy).T)
            near = np.flatnonzero((dd >= p.d_min) & (dd <= p.d_max))
            if len(near) < p.min_new_cells:
                continue
            seen, bearing = self.visible(xy, targets_ij[near], normals)
            idx, bearing = near[seen], bearing[seen]
            if len(idx) < p.min_new_cells:
                continue
            for yaw in yaws:
                sel = idx[np.abs((bearing - yaw + math.pi) % (2 * math.pi) - math.pi) <= half]
                if len(sel) >= p.min_new_cells:
                    options.append((xy, float(yaw), sel))
        coverable = np.unique(np.concatenate([o[2] for o in options])) if options else np.zeros(0, int)
        stats["coverable"] = int(len(coverable))
        covered = np.zeros(len(targets_ij), bool)
        chosen = []
        while len(coverable) and len(chosen) < p.max_viewpoints and covered[coverable].mean() < p.target_coverage:
            gains = [int((~covered[o[2]]).sum()) for o in options]
            k = int(np.argmax(gains))
            if gains[k] < p.min_new_cells:
                break
            xy, yaw, sel = options[k]
            # face the middle of what it covers (the yaw lattice is coarse)
            mid = tc[sel[~covered[sel]]].mean(0) - xy
            covered[sel] = True
            chosen.append((float(xy[0]), float(xy[1]), float(math.atan2(mid[1], mid[0])), gains[k]))
        stats["planned"] = int(covered.sum())
        # nearest neighbour order from the start
        order, here, todo = [], np.asarray(start_xy, float), list(range(len(chosen)))
        while todo:
            k = min(todo, key=lambda i: math.hypot(chosen[i][0] - here[0], chosen[i][1] - here[1]))
            order.append(chosen[k])
            here = np.array(chosen[k][:2])
            todo.remove(k)
        return order, stats
