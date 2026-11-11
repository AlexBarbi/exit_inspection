"""Exit sign sightings and landmarks.

A sighting is one detector box cast into the lidar cloud: the ray through the box center hits the wall, a vertical
plane fitted around the hit gives the sign center (ray x plane) and the wall normal. A landmark is the sightings of
one sign merged: median center, mean normal, majority class. Plain numpy, no ROS.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from exit_inspection import geometry as g

# Box from the detector: (class name, confidence, center u, center v, width px, height px).
Box = Tuple[str, float, float, float, float, float]


@dataclass
class MapperParams:
    # first hit of the ray in the cloud
    ray_radius: float = 0.15
    hit_min_points: int = 5
    hit_window: float = 0.3
    # wall plane around the hit
    plane_radius: float = 0.6
    plane_threshold: float = 0.03
    plane_min_inliers: int = 30
    plane_min_height: float = 0.2
    plane_iterations: int = 100
    # floor under the sign: lowest z bin with enough points, within floor_radius (xy), floor_below under the sign and
    # floor_wall_gap in front of the wall (points on the wall plane would make its lowest row the floor)
    floor_radius: float = 1.0
    floor_wall_gap: float = 0.1
    floor_below: float = 1.0
    floor_bin: float = 0.05
    floor_min_points: int = 20
    base_height: float = 0.35  # robot base above the floor, when no floor is seen
    # sanity of a sighting
    sign_min_height: float = 1.5
    sign_max_height: float = 3.5
    sign_min_width: float = 0.15
    sign_max_width: float = 0.8
    # association and confirmation
    assoc_distance: float = 1.0
    assoc_angle_deg: float = 45.0
    confirm_sightings: int = 5
    confirm_baseline: float = 0.5
    # door below a door sign (v1: fixed size)
    door_width: float = 1.2
    door_height: float = 2.0


@dataclass
class Sighting:
    stamp: float
    cls: str
    conf: float
    hit: np.ndarray  # sign center in the map frame
    normal: np.ndarray  # unit, horizontal, from the wall towards the camera
    robot: np.ndarray  # robot base position at the sighting
    range_m: float  # camera to sign center
    width_m: float  # sign width from the box width
    floor_z: float  # floor height under the sign
    low_quality: bool  # no wall plane: center is the raw hit, normal is the reversed ray


def group_of(cls: str, door_classes: Sequence[str]) -> str:
    return "door" if cls in door_classes else "route"


def estimate_floor(cloud: np.ndarray, center: np.ndarray, normal: np.ndarray, robot_z: float,
                   p: MapperParams) -> float:
    """Floor height under the sign: the lowest dense z layer in front of the wall near it, else the robot base height
    minus base_height."""
    near = cloud[(np.linalg.norm(cloud[:, :2] - center[:2], axis=1) < p.floor_radius)
                 & (cloud[:, 2] < center[2] - p.floor_below)
                 & ((cloud - center) @ normal > p.floor_wall_gap)]
    if len(near) >= p.floor_min_points:
        z = np.sort(near[:, 2])
        bins = np.floor((z - z[0]) / p.floor_bin).astype(int)
        counts = np.bincount(bins)
        dense = np.nonzero(counts >= p.floor_min_points)[0]
        if len(dense):
            return float(z[bins == dense[0]].mean())
    return float(robot_z - p.base_height)


def make_sighting(box: Box, K: np.ndarray, T_map_cam: np.ndarray, cloud: np.ndarray, robot_xyz: np.ndarray,
                  stamp: float, p: MapperParams,
                  rng: Optional[np.random.Generator] = None) -> Tuple[Optional[Sighting], str]:
    """Cast one box into the cloud (map frame). Returns (sighting, "ok") or (None, reason)."""
    if rng is None:
        rng = np.random.default_rng()
    cls, conf, u, v, w_px, _ = box
    origin = T_map_cam[:3, 3]
    direction = g.transform_dirs(T_map_cam, g.pixel_ray(u, v, K))[0]
    hit = g.first_hit(origin, direction, cloud, p.ray_radius, p.hit_min_points, p.hit_window)
    if hit is None:
        return None, "no_hit"

    near = cloud[np.linalg.norm(cloud - hit, axis=1) < p.plane_radius]
    plane = g.fit_vertical_plane(near, rng, p.plane_iterations, p.plane_threshold, p.plane_min_inliers,
                                 p.plane_min_height)
    center, normal, low_quality = None, None, True
    if plane is not None:
        n, d = plane
        center = g.ray_plane(origin, direction, n, d)
        if center is not None and np.linalg.norm(center - hit) < p.plane_radius:
            normal = n if n @ (origin - center) >= 0.0 else -n
            low_quality = False
    if low_quality:
        horizontal = np.array([direction[0], direction[1], 0.0])
        if np.linalg.norm(horizontal) < 1e-6:
            return None, "no_normal"
        center, normal = hit, -horizontal / np.linalg.norm(horizontal)

    depth = (center - origin) @ T_map_cam[:3, 2]  # along the optical axis
    width = w_px * depth / K[0, 0]
    floor_z = estimate_floor(cloud, center, normal, robot_xyz[2], p)
    if not p.sign_min_height <= center[2] - floor_z <= p.sign_max_height:
        return None, "height"
    if not p.sign_min_width <= width <= p.sign_max_width:
        return None, "width"
    return Sighting(stamp=stamp, cls=cls, conf=float(conf), hit=center, normal=normal,
                    robot=np.asarray(robot_xyz, float), range_m=float(np.linalg.norm(center - origin)),
                    width_m=float(width), floor_z=floor_z, low_quality=low_quality), "ok"


@dataclass
class Landmark:
    id: int
    group: str  # "door" or "route"
    params: MapperParams
    sightings: List[Sighting] = field(default_factory=list)

    @property
    def center(self) -> np.ndarray:
        return np.median(np.array([s.hit for s in self.sightings]), axis=0)

    @property
    def normal(self) -> np.ndarray:
        n = np.mean(np.array([s.normal for s in self.sightings]), axis=0)
        n[2] = 0.0
        return n / np.linalg.norm(n)

    @property
    def width(self) -> float:
        return float(np.median([s.width_m for s in self.sightings]))

    @property
    def floor_z(self) -> float:
        return float(np.median([s.floor_z for s in self.sightings]))

    @property
    def cls(self) -> str:
        return Counter(s.cls for s in self.sightings).most_common(1)[0][0]

    @property
    def mean_conf(self) -> float:
        return float(np.mean([s.conf for s in self.sightings]))

    @property
    def door_floor_center(self) -> np.ndarray:
        c = self.center.copy()
        c[2] = self.floor_z
        return c

    @property
    def baseline(self) -> float:
        """Largest distance between two robot positions the sign was seen from."""
        r = np.array([s.robot for s in self.sightings])
        return float(np.max(np.linalg.norm(r[:, None, :] - r[None, :, :], axis=2))) if len(r) > 1 else 0.0

    @property
    def confirmed(self) -> bool:
        return len(self.sightings) >= self.params.confirm_sightings and self.baseline >= self.params.confirm_baseline


class LandmarkSet:
    """Sightings merged into landmarks: same group, within assoc_distance of the center and assoc_angle_deg."""

    def __init__(self, params: MapperParams, door_classes: Sequence[str]):
        self.params = params
        self.door_classes = tuple(door_classes)
        self._landmarks: List[Landmark] = []
        self._cos_angle = np.cos(np.radians(params.assoc_angle_deg))

    def add(self, s: Sighting) -> int:
        group = group_of(s.cls, self.door_classes)
        best, best_dist = None, self.params.assoc_distance
        for l in self._landmarks:
            if l.group != group or l.normal @ s.normal < self._cos_angle:
                continue
            dist = float(np.linalg.norm(l.center - s.hit))
            if dist < best_dist:
                best, best_dist = l, dist
        if best is None:
            best = Landmark(id=len(self._landmarks), group=group, params=self.params)
            self._landmarks.append(best)
        best.sightings.append(s)
        return best.id

    def landmarks(self) -> List[Landmark]:
        return list(self._landmarks)
