"""Geometry helpers: camera rays, rigid transforms, first hit of a ray in a cloud, vertical plane fit.

Plain numpy, no ROS, so that the tests run anywhere. Planes are (n, d) with unit n and n . x + d = 0.
"""

from typing import Optional, Sequence, Tuple

import numpy as np


def pixel_ray(u: float, v: float, K: np.ndarray) -> np.ndarray:
    """Unit direction, in the optical frame (z forward, x right, y down), of the ray through pixel (u, v)."""
    d = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    return d / np.linalg.norm(d)


def quat_to_matrix(q_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = q_xyzw
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n if n > 0.0 else 0.0
    return np.array([
        [1.0 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1.0 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1.0 - s * (x * x + y * y)],
    ])


def pose_matrix(translation: Sequence[float], q_xyzw: Sequence[float]) -> np.ndarray:
    """4x4 homogeneous transform from a translation and a quaternion (x, y, z, w)."""
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(q_xyzw)
    T[:3, 3] = translation
    return T


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(pts)
    return pts @ T[:3, :3].T + T[:3, 3]


def transform_dirs(T: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    return np.atleast_2d(dirs) @ T[:3, :3].T


def first_hit(origin: np.ndarray, direction: np.ndarray, points: np.ndarray, radius: float, min_points: int,
              window: float) -> Optional[np.ndarray]:
    """Centroid of the first cluster of points along the ray.

    Points count when they are ahead of the origin and within `radius` of the ray; the first cluster is the nearest
    run of at least `min_points` of them within `window` metres along the ray. None when there is none.
    """
    if len(points) == 0:
        return None
    rel = points - origin
    t = rel @ direction
    perp = np.linalg.norm(rel - np.outer(t, direction), axis=1)
    mask = (t > 0.0) & (perp < radius)
    if np.count_nonzero(mask) < min_points:
        return None
    ts = t[mask]
    order = np.argsort(ts)
    ts, near = ts[order], points[mask][order]
    ends = np.searchsorted(ts, ts + window, side="right")
    counts = ends - np.arange(len(ts))
    ok = np.nonzero(counts >= min_points)[0]
    if len(ok) == 0:
        return None
    i = ok[0]
    return near[i:ends[i]].mean(axis=0)


def fit_vertical_plane(points: np.ndarray, rng: np.random.Generator, iterations: int, threshold: float,
                       min_inliers: int, min_height: float) -> Optional[Tuple[np.ndarray, float]]:
    """RANSAC vertical plane (n_z = 0 exactly): a line in xy, refined on its inliers.

    The map is gravity aligned and walls are vertical, so fitting in xy keeps a door recessed a few cm behind the wall
    from tilting the plane. None unless at least `min_inliers` points lie within `threshold` of it and they span at
    least `min_height` in z (a strip of floor is no wall).
    """
    n_pts = len(points)
    if n_pts < max(2, min_inliers):
        return None
    xy = points[:, :2]
    best_count, best = 0, None
    for _ in range(iterations):
        a, b = xy[rng.choice(n_pts, 2, replace=False)]
        t = b - a
        norm = np.linalg.norm(t)
        if norm < 1e-6:
            continue
        n = np.array([-t[1], t[0]]) / norm
        d = -n @ a
        count = np.count_nonzero(np.abs(xy @ n + d) < threshold)
        if count > best_count:
            best_count, best = count, (n, d)
    if best is None or best_count < min_inliers:
        return None
    n, d = best
    mask = np.abs(xy @ n + d) < threshold
    centroid = xy[mask].mean(axis=0)
    _, _, vt = np.linalg.svd(xy[mask] - centroid, full_matrices=False)
    n_ref = vt[-1] if vt[-1] @ n >= 0.0 else -vt[-1]
    d_ref = -n_ref @ centroid
    mask = np.abs(xy @ n_ref + d_ref) < threshold
    if np.count_nonzero(mask) < min_inliers:
        return None
    z = points[mask, 2]
    if z.max() - z.min() < min_height:
        return None
    return np.array([n_ref[0], n_ref[1], 0.0]), float(d_ref)


def ray_plane(origin: np.ndarray, direction: np.ndarray, normal: np.ndarray, d: float) -> Optional[np.ndarray]:
    """Intersection of the ray with the plane; None when the ray is parallel to it or the plane is behind."""
    denom = normal @ direction
    if abs(denom) < 1e-9:
        return None
    t = -(normal @ origin + d) / denom
    if t <= 0.0:
        return None
    return origin + t * direction
