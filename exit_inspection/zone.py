"""
Zone in front of an exit door and its verdict (sub-project 2 of the emergency exit inspection,
docs/superpowers/specs/2026-11-11-exit-zone-checker-design.md).

Exit frame: origin the door floor center, n the horizontal normal into the room, t = (-n_y, n_x, 0) along the wall.
A map point p has u = t.(p - c), v = n.(p - c); the zone is |u| <= door_width / 2 + margin, wall_gap <= v <= depth, a
grid of `cell` cells. Depth and lidar points of the last seconds (Frame) vote per cell:

  free     a floor point, or a ray to a floor point passes over it lower than object_min_height (an object that high
           would have stopped the ray: at a grazing view the floor points alone leave rows of gaps)
  object   points in the height band, from several frames
  hidden   unseen, within object_depth behind an object (seen from the front, objects are assumed that deep)
  person   within person_radius of a person detection (its points dropped); behind: unseen, hidden by a person
  unknown  the rest

People and what they hide count as free. A disk of diameter w_min must get from the door to the zone's open boundary
(far edge or sides); the wall beside the door (v < 0, |u| > door_width / 2) stops it:

  UNVERIFIED  fewer than min_camera_frames depth frames: people cannot be told from objects (they come from the
              camera), whatever the lidar shows
  BLOCKED     no corridor even through unseen cells (hidden ones stay blocked)
  UNVERIFIED  the corridor needs unseen cells, or the zone is partly unseen and has no object
  WARNING     corridor through seen cells, objects in the zone
  CLEAR       corridor through seen cells, no object, the zone seen (clear_coverage)
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

UNKNOWN, FREE, OBJECT, PERSON, BEHIND, HIDDEN = 0, 1, 2, 3, 4, 5
VERDICTS = ("CLEAR", "WARNING", "BLOCKED", "UNVERIFIED")


@dataclass
class ZoneParams:
    cell: float = 0.05
    margin: float = 0.5               # zone width = door width + 2 margin
    depth: float = 3.0                # zone depth from the wall
    wall_gap: float = 0.1             # the first wall_gap m are the wall/door (not checked)
    object_min_height: float = 0.05   # above the floor
    object_max_height: float = 2.0
    floor_tolerance: float = 0.03     # |h| of a floor point
    floor_search: float = 0.3         # floor searched within +- this of the stored one
    floor_bin: float = 0.02
    floor_min_points: int = 50
    wall_band: float = 0.3            # wall points beside the door within +- this of v = 0
    wall_min_points: int = 30
    min_points: int = 3               # object points per cell
    min_frames: int = 2               # frames with object points per cell
    min_object_area: float = 0.01     # m2, smaller 8-connected objects are noise
    object_depth: float = 0.5         # unseen cells this far behind an object (from the sensor) count as object
    w_min: float = 0.9                # corridor width
    clear_coverage: float = 0.9
    min_camera_frames: int = 3        # depth frames in the window for any verdict but UNVERIFIED
    person_radius: float = 0.4
    person_extent: float = 0.9        # object parts touching a person disk and within this of its center: the person


@dataclass
class Frame:
    stamp: float
    sensor: np.ndarray   # sensor origin, map frame
    points: np.ndarray   # (N, 3) map frame, cropped to the zone box
    kind: str = "depth"  # depth or lidar


@dataclass
class Zone:
    exit_id: int
    center: np.ndarray   # door floor center, map frame
    normal: np.ndarray   # horizontal unit normal into the room
    door_width: float
    params: ZoneParams = field(default_factory=ZoneParams)

    @classmethod
    def from_exit(cls, exit_: dict, params: ZoneParams) -> "Zone":
        n = np.array(exit_["normal"], dtype=float)
        n[2] = 0.0
        n /= np.linalg.norm(n)
        return cls(int(exit_["id"]), np.array(exit_["door_floor_center"], dtype=float), n,
                   float(exit_.get("door_width", 1.2)), params)

    @property
    def along(self) -> np.ndarray:
        return np.array([-self.normal[1], self.normal[0], 0.0])

    @property
    def door_half(self) -> float:
        return self.door_width / 2

    @property
    def u_half(self) -> float:
        return self.door_width / 2 + self.params.margin

    @property
    def shape(self):
        p = self.params
        return (int(math.ceil(2 * self.u_half / p.cell - 1e-9)), int(math.ceil((p.depth - p.wall_gap) / p.cell - 1e-9)))

    def local(self, pts: np.ndarray) -> np.ndarray:
        """(N, 3) map points -> (u, v, z)."""
        d = np.asarray(pts, dtype=float).reshape(-1, 3) - self.center
        return np.column_stack([d @ self.along, d @ self.normal, d[:, 2] + self.center[2]])

    def to_map(self, u, v, z) -> np.ndarray:
        u, v, z = (np.asarray(a, dtype=float).ravel() for a in (u, v, z))
        out = self.center + u[:, None] * self.along + v[:, None] * self.normal
        out[:, 2] = z
        return out

    def cell_centers(self):
        """(u, v) of the cell centers, arrays of shape `shape` (v without the wall offset)."""
        nu, nv = self.shape
        c = self.params.cell
        return np.meshgrid((np.arange(nu) + 0.5) * c - nu * c / 2, self.params.wall_gap + (np.arange(nv) + 0.5) * c,
                           indexing="ij")

    def crop(self, pts: np.ndarray, floor_z: float) -> np.ndarray:
        """The map points that can matter to this zone (zone box, wall band, floor search to the band top)."""
        p = self.params
        loc = self.local(pts)
        keep = ((np.abs(loc[:, 0]) <= self.u_half) & (loc[:, 1] >= -p.wall_band) & (loc[:, 1] <= p.depth + p.wall_band)
                & (loc[:, 2] >= floor_z - p.floor_search - p.floor_tolerance)
                & (loc[:, 2] <= floor_z + p.floor_search + p.object_max_height))
        return np.asarray(pts)[keep]


@dataclass
class ZoneResult:
    verdict: str
    reason: str
    coverage: float = 0.0
    object_area: float = 0.0
    corridor_width: float = 0.0             # widest through seen cells
    corridor_width_optimistic: float = 0.0  # widest if unseen cells were free
    behind_person_area: float = 0.0
    person_area: float = 0.0
    people: int = 0
    floor_z: float = float("nan")
    wall_offset: float = 0.0
    frames: int = 0
    points: int = 0
    state: np.ndarray = None                # cell states, Zone.shape

    def summary(self) -> dict:
        r3 = lambda x: round(float(x), 3)
        return {"verdict": self.verdict, "reason": self.reason, "coverage": r3(self.coverage),
                "object_area": r3(self.object_area), "corridor_width": r3(self.corridor_width),
                "corridor_width_optimistic": r3(self.corridor_width_optimistic),
                "behind_person_area": r3(self.behind_person_area), "person_area": r3(self.person_area),
                "people": int(self.people), "floor_z": r3(self.floor_z), "wall_offset": r3(self.wall_offset),
                "frames": int(self.frames), "points": int(self.points)}


def _free_centers(blocked: np.ndarray, cell: float, width: float, door_half: float, wall_gap: float) -> np.ndarray:
    """Cells whose center is at least width / 2 from every blocked cell center. Outside the grid is free, but for the
    wall beside the door (v < 0, |u| > door_half)."""
    r = width / 2 / cell
    k = int(math.ceil(r))
    nu, nv = blocked.shape
    padded = np.pad(blocked, k, constant_values=False)
    u = (np.arange(nu + 2 * k) - k + 0.5) * cell - nu * cell / 2
    v = wall_gap + (np.arange(nv + 2 * k) - k + 0.5) * cell
    padded[np.ix_(np.abs(u) > door_half, v < 0)] = True
    near = blocked.copy()
    for di in range(-k, k + 1):
        for dj in range(-k, k + 1):
            if (di or dj) and di * di + dj * dj < r * r - 1e-6:
                near |= padded[k + di:k + di + nu, k + dj:k + dj + nv]
    return ~near


def corridor_exists(blocked: np.ndarray, cell: float, width: float, door_half: float, wall_gap: float = 0.1) -> bool:
    """A disk of diameter `width` gets from the door (first row) to the last row or a side column."""
    nu, nv = blocked.shape
    ok = _free_centers(blocked, cell, width, door_half, wall_gap)
    u = (np.arange(nu) + 0.5) * cell - nu * cell / 2
    start_cols = np.flatnonzero(np.abs(u) <= max(0.0, door_half - width / 2) + 1e-9)
    if start_cols.size == 0:  # disk wider than the door: from its middle
        start_cols = np.flatnonzero(np.abs(u) <= np.abs(u).min() + 1e-9)
    seen = np.zeros_like(ok)
    queue = deque()
    for i in start_cols:
        if ok[i, 0]:
            seen[i, 0] = True
            queue.append((i, 0))
    while queue:
        i, j = queue.popleft()
        if j == nv - 1 or i == 0 or i == nu - 1:
            return True
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < nu and 0 <= b < nv and ok[a, b] and not seen[a, b]:
                seen[a, b] = True
                queue.append((a, b))
    return False


def widest_corridor(blocked: np.ndarray, cell: float, door_half: float, max_width: float, wall_gap: float = 0.1,
                    step: float = 0.05) -> float:
    """Widest corridor (multiple of step, up to max_width); 0 when not even a point gets through."""
    widths = np.arange(0.0, max_width + 1e-9, step)
    if not corridor_exists(blocked, cell, 0.0, door_half, wall_gap):
        return 0.0
    lo, hi = 0, len(widths) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if corridor_exists(blocked, cell, widths[mid], door_half, wall_gap):
            lo = mid
        else:
            hi = mid - 1
    return float(widths[lo])


def _components(mask: np.ndarray) -> List[np.ndarray]:
    """8-connected components of mask, each an (n, 2) array of cell indices."""
    seen = np.zeros_like(mask)
    nu, nv = mask.shape
    out = []
    for i0, j0 in zip(*np.nonzero(mask)):
        if seen[i0, j0]:
            continue
        comp, queue = [], deque([(i0, j0)])
        seen[i0, j0] = True
        while queue:
            i, j = queue.popleft()
            comp.append((i, j))
            for a in (i - 1, i, i + 1):
                for b in (j - 1, j, j + 1):
                    if 0 <= a < nu and 0 <= b < nv and mask[a, b] and not seen[a, b]:
                        seen[a, b] = True
                        queue.append((a, b))
        out.append(np.array(comp))
    return out


def _drop_small(mask: np.ndarray, min_cells: int) -> np.ndarray:
    """mask without its 8-connected components smaller than min_cells."""
    out = mask.copy()
    for comp in _components(mask):
        if len(comp) < min_cells:
            out[comp[:, 0], comp[:, 1]] = False
    return out


def _person_parts(obj: np.ndarray, cu: np.ndarray, cv: np.ndarray, persons: np.ndarray, p: ZoneParams) -> np.ndarray:
    """Object components that touch a person disk and lie within person_extent of its center (limbs, a walking
    person's swing): they belong to the person."""
    out = np.zeros_like(obj)
    for comp in _components(obj):
        cells = np.column_stack([cu[comp[:, 0], comp[:, 1]], cv[comp[:, 0], comp[:, 1]]])
        for q in persons:
            d = np.hypot(*(cells - q).T)
            if d.min() <= p.person_radius + 2 * p.cell and d.max() <= p.person_extent:
                out[comp[:, 0], comp[:, 1]] = True
                break
    return out


def estimate_floor(loc: np.ndarray, zone: Zone) -> float:
    """Lowest dense z layer of the zone (away from the wall) near the stored floor, else the stored floor."""
    p = zone.params
    hint = float(zone.center[2])
    z = loc[(np.abs(loc[:, 0]) <= zone.u_half) & (loc[:, 1] >= p.wall_band) & (loc[:, 1] <= p.depth)
            & (np.abs(loc[:, 2] - hint) <= p.floor_search), 2]
    if z.size < p.floor_min_points:
        return hint
    edges = np.arange(hint - p.floor_search, hint + p.floor_search + p.floor_bin, p.floor_bin)
    counts, _ = np.histogram(z, edges)
    dense = np.flatnonzero(counts >= max(p.floor_min_points, 0.2 * counts.max()))
    if dense.size == 0:
        return hint
    lo = edges[dense[0]]
    sel = z[(z >= lo - p.floor_bin) & (z <= lo + 2 * p.floor_bin)]
    return float(np.median(sel))


def estimate_wall_offset(loc: np.ndarray, h: np.ndarray, zone: Zone) -> float:
    """Median v of the wall beside the door (0 when too few points)."""
    p = zone.params
    au = np.abs(loc[:, 0])
    sel = ((au >= zone.door_half + 0.1) & (au <= zone.u_half) & (np.abs(loc[:, 1]) <= p.wall_band) & (h >= 0.3)
           & (h <= 2.0))
    if sel.sum() < p.wall_min_points:
        return 0.0
    return float(np.clip(np.median(loc[sel, 1]), -p.wall_band, p.wall_band))


def _count_people(persons_uv: np.ndarray, link: float = 0.6) -> int:
    clusters: List[np.ndarray] = []
    for q in persons_uv:
        for c in clusters:
            if np.linalg.norm(c - q) <= link:
                break
        else:
            clusters.append(q)
    return len(clusters)


def _cells_of(u: np.ndarray, v: np.ndarray, zone: Zone) -> np.ndarray:
    """Flat cell index of (u, v) points, -1 outside the grid."""
    p = zone.params
    nu, nv = zone.shape
    iu = np.floor((u + nu * p.cell / 2) / p.cell).astype(int)
    iv = np.floor((v - p.wall_gap) / p.cell).astype(int)
    return np.where((iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv), iu * nv + iv, -1)


def _march(start: np.ndarray, direction: np.ndarray, length: np.ndarray, step: float, zone: Zone) -> np.ndarray:
    """Flat cells visited from start (N, 2) along unit direction (N, 2) for length (N,) m."""
    if len(start) == 0:
        return np.zeros(0, int)
    t = np.arange(int(math.ceil(float(length.max()) / step)) + 1) * step
    ok = t[None, :] <= length[:, None]
    pts = start[:, None, :] + t[None, :, None] * direction[:, None, :]
    cells = _cells_of(pts[..., 0][ok], pts[..., 1][ok], zone)
    return np.unique(cells[cells >= 0])


def _sensor_groups(frames: Sequence[Frame], zone: Zone, offset: float, floor_z: float, fid: np.ndarray):
    """Frames grouped by sensor position (0.1 m): [(sensor u, v, height above the floor), point mask]."""
    s = zone.local(np.array([f.sensor for f in frames]))
    s[:, 1] -= offset
    s[:, 2] -= floor_z
    keys = np.round(s / 0.1).astype(int)
    _, group = np.unique(keys, axis=0, return_inverse=True)
    group = np.asarray(group).ravel()
    out = []
    for gi in range(group.max() + 1):
        members = np.flatnonzero(group == gi)
        out.append((s[members].mean(0), np.isin(fid, members)))
    return out


def _ray_free(u, v, groups, zone: Zone) -> np.ndarray:
    """Flat cells a ray from the sensor to a floor point (u, v) crosses lower than object_min_height."""
    p = zone.params
    step = p.cell / 2
    out = []
    for (su, sv, sh), mask in groups:
        if not mask.any():
            continue
        k = np.round(np.column_stack([u[mask], v[mask]]) / step).astype(np.int64)
        k = np.unique((k[:, 0] + (1 << 20)) * (1 << 21) + k[:, 1] + (1 << 20))  # 1D keys: unique(axis=0) is 10x slower
        q = np.column_stack([k // (1 << 21) - (1 << 20), k % (1 << 21) - (1 << 20)]) * step
        d = np.array([su, sv]) - q
        dist = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)
        length = dist * min(1.0, p.object_min_height / max(sh, 1e-6))
        out.append(_march(q, d / dist[:, None], length, step, zone))
    return np.unique(np.concatenate(out)) if out else np.zeros(0, int)


def _hidden(obj: np.ndarray, groups, zone: Zone, length: float) -> np.ndarray:
    """Flat cells within `length` behind the object cells, seen from each sensor."""
    p = zone.params
    cu, cv = zone.cell_centers()
    q = np.column_stack([cu[obj], cv[obj]])
    out = []
    for (su, sv, _), _mask in groups:
        d = q - [su, sv]
        dist = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)
        out.append(_march(q, d / dist[:, None], np.full(len(q), length), p.cell / 2, zone))
    return np.unique(np.concatenate(out)) if out else np.zeros(0, int)


def _behind(cells: np.ndarray, sensors: np.ndarray, persons: np.ndarray, radius: float) -> np.ndarray:
    """Cells whose segment to one of the sensors crosses a person disk."""
    out = np.zeros(len(cells), bool)
    for s in sensors:
        d = cells - s
        dd = np.maximum((d ** 2).sum(1), 1e-12)
        for q in persons:
            t = np.clip(((q - s) @ d.T) / dd, 0.0, 1.0)
            closest = s + t[:, None] * d
            out |= np.hypot(*(closest - q).T) <= radius
    return out


def evaluate(zone: Zone, frames: Sequence[Frame], persons: np.ndarray) -> ZoneResult:
    """Verdict of the zone from the frames (map points) and the person positions (K, 2) map xy."""
    p = zone.params
    nu, nv = zone.shape
    if not frames:
        return ZoneResult("UNVERIFIED", "no data", state=np.zeros((nu, nv), np.uint8))
    pts = np.vstack([f.points for f in frames]) if frames else np.zeros((0, 3))
    fid = np.concatenate([np.full(len(f.points), i) for i, f in enumerate(frames)])
    loc = zone.local(pts)
    persons = np.asarray(persons, dtype=float).reshape(-1, 2)
    persons_uv = zone.local(np.column_stack([persons, np.zeros(len(persons))]))[:, :2]
    r = p.person_radius
    near_zone = ((np.abs(persons_uv[:, 0]) <= zone.u_half + r) & (persons_uv[:, 1] >= -r)
                 & (persons_uv[:, 1] <= p.depth + r))
    persons_uv = persons_uv[near_zone]
    if len(persons_uv):
        dist = np.min(np.hypot(loc[:, 0, None] - persons_uv[:, 0], loc[:, 1, None] - persons_uv[:, 1]), axis=1)
        keep = dist > r
        loc, fid = loc[keep], fid[keep]

    floor_z = estimate_floor(loc, zone)
    h = loc[:, 2] - floor_z
    offset = estimate_wall_offset(loc, h, zone)
    u, v = loc[:, 0], loc[:, 1] - offset
    flat = _cells_of(u, v, zone)
    inside = flat >= 0
    floor_pt = np.abs(h) <= p.floor_tolerance
    is_obj = inside & (h >= p.object_min_height) & (h <= p.object_max_height)
    groups = _sensor_groups(frames, zone, offset, floor_z, fid)
    free = np.zeros(nu * nv, bool)
    free[flat[inside & floor_pt]] = True
    free[_ray_free(u[floor_pt], v[floor_pt], [(s, m[floor_pt]) for s, m in groups], zone)] = True
    obj_count = np.bincount(flat[is_obj], minlength=nu * nv).reshape(nu, nv)
    pairs = np.unique(flat[is_obj].astype(np.int64) * len(frames) + fid[is_obj])
    obj_frames = np.bincount(pairs // len(frames), minlength=nu * nv).reshape(nu, nv)
    obj = (obj_count >= p.min_points) & (obj_frames >= p.min_frames)
    cu, cv = zone.cell_centers()
    pu = persons_uv - [0.0, offset]
    limbs = _person_parts(obj, cu, cv, pu, p) if len(pu) else np.zeros_like(obj)
    obj = _drop_small(obj & ~limbs, int(math.ceil(p.min_object_area / p.cell ** 2 - 1e-9)))

    state = np.full((nu, nv), UNKNOWN, np.uint8)
    state[free.reshape(nu, nv)] = FREE
    if len(persons_uv):
        in_person = limbs | (np.min(np.hypot(cu[..., None] - pu[:, 0], cv[..., None] - pu[:, 1]), axis=-1) <= r)
        state[in_person & ~obj] = PERSON
        unknown = state == UNKNOWN
        if unknown.any():
            sensors = zone.local(np.array([f.sensor for f in frames]))[:, :2] - [0.0, offset]
            sensors = np.unique(np.round(sensors / 0.25) * 0.25, axis=0)
            cells = np.column_stack([cu[unknown], cv[unknown]])
            idx = np.argwhere(unknown)[_behind(cells, sensors, pu, r)]
            state[idx[:, 0], idx[:, 1]] = BEHIND
            behind_limbs = np.zeros(nu * nv, bool)
            behind_limbs[_hidden(limbs, groups, zone, p.depth)] = True
            state[behind_limbs.reshape(nu, nv) & (state == UNKNOWN)] = BEHIND
    hidden = np.zeros(nu * nv, bool)
    hidden[_hidden(obj, groups, zone, p.object_depth)] = True
    state[hidden.reshape(nu, nv) & (state == UNKNOWN)] = HIDDEN
    state[obj] = OBJECT

    cell_area = p.cell ** 2
    coverage = float(np.mean((state != UNKNOWN) & (state != HIDDEN)))
    object_area = float(obj.sum() * cell_area)
    blocked_obj = (state == OBJECT) | (state == HIDDEN)
    blocked_pes = blocked_obj | (state == UNKNOWN)
    dh, wg, max_w = zone.door_half, p.wall_gap, 2 * zone.u_half
    w = min(p.w_min, zone.door_width - 2 * p.cell)  # a door narrower than w_min is the door's problem, not the zone's
    widest_p = widest_corridor(blocked_pes, p.cell, dh, max_w, wg)
    widest_o = widest_corridor(blocked_obj, p.cell, dh, max_w, wg)
    camera_frames = sum(f.kind == "depth" for f in frames)
    if camera_frames < p.min_camera_frames:
        verdict, reason = "UNVERIFIED", (f"camera did not see the zone ({camera_frames} depth frames): people cannot "
                                         f"be told from objects")
    elif not corridor_exists(blocked_obj, p.cell, w, dh, wg):
        verdict, reason = "BLOCKED", f"objects close the door: widest corridor {widest_o:.2f} m < {w:.2f} m"
    elif not corridor_exists(blocked_pes, p.cell, w, dh, wg):
        verdict, reason = "UNVERIFIED", (f"unseen cells on the way: corridor {widest_p:.2f} m through seen cells, "
                                         f"{widest_o:.2f} m if they were free")
    elif object_area > 0:
        verdict, reason = "WARNING", f"objects in the zone ({object_area:.2f} m2), corridor {widest_p:.2f} m"
    elif coverage < p.clear_coverage:
        verdict, reason = "UNVERIFIED", f"zone partly unseen ({coverage:.0%} seen), no object"
    else:
        verdict, reason = "CLEAR", f"zone free ({coverage:.0%} seen), corridor {widest_p:.2f} m"
    return ZoneResult(verdict, reason, coverage, object_area, widest_p, widest_o,
                      float((state == BEHIND).sum() * cell_area), float((state == PERSON).sum() * cell_area),
                      _count_people(persons_uv), floor_z, offset, len(frames), int(len(loc)), state)
