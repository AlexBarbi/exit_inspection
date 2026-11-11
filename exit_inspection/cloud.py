"""Rolling voxel cloud: the scans of the last few seconds merged on a voxel grid, one point per voxel."""

import numpy as np

_OFFSET = 1 << 20  # voxel index range +-2^20 per axis (+-52 km at 5 cm)


def voxel_keys(pts: np.ndarray, voxel: float) -> np.ndarray:
    k = np.floor(pts / voxel).astype(np.int64) + _OFFSET
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    if len(pts) == 0:
        return pts
    _, idx = np.unique(voxel_keys(pts, voxel), return_index=True)
    return pts[idx]


class RollingCloud:
    """Points of the scans added in the last `window` seconds, one per voxel (the newest)."""

    def __init__(self, window: float, voxel: float):
        self.window, self.voxel = window, voxel
        self.keys = np.empty(0, np.int64)
        self.pts = np.empty((0, 3))
        self.stamps = np.empty(0)

    def add(self, pts: np.ndarray, stamp: float) -> None:
        pts = np.asarray(pts, float).reshape(-1, 3)
        keys = np.concatenate([self.keys, voxel_keys(pts, self.voxel)])
        allpts = np.vstack([self.pts, pts])
        stamps = np.concatenate([self.stamps, np.full(len(pts), stamp)])
        # newest point of each voxel: first occurrence in the reversed arrays
        _, idx = np.unique(keys[::-1], return_index=True)
        idx = len(keys) - 1 - idx
        keep = idx[stamps[idx] >= stamp - self.window]
        self.keys, self.pts, self.stamps = keys[keep], allpts[keep], stamps[keep]

    def points(self, center: np.ndarray = None, radius: float = None) -> np.ndarray:
        if center is None or radius is None:
            return self.pts
        return self.pts[np.linalg.norm(self.pts - center, axis=1) < radius]
