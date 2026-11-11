"""Ground-truth labels of the exit signs in a frame of the simulated camera (render harvester, end-to-end checks).

Gazebo cameras look along +x of their frame: image right = -y, image down = -z. Poses are (x, y, z, roll, pitch,
yaw) with the usual ZYX convention (positive pitch lowers +x).
"""

import math
from typing import Optional, Sequence

import numpy as np


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), \
        math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def intrinsics(width: int, height: int, hfov_rad: float) -> np.ndarray:
    f = (width / 2) / math.tan(hfov_rad / 2)
    return np.array([[f, 0.0, width / 2], [0.0, f, height / 2], [0.0, 0.0, 1.0]])


def project(points: np.ndarray, pose: Sequence[float], K: np.ndarray):
    """(uv (N, 2), depth along the optical axis (N,)) of world points seen from a Gazebo camera at `pose`."""
    R = rpy_matrix(*pose[3:])
    q = (np.atleast_2d(points) - np.asarray(pose[:3])) @ R  # in the camera frame (x forward)
    depth = q[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 2] - K[0, 0] * q[:, 1] / depth
        v = K[1, 2] - K[1, 1] * q[:, 2] / depth
    return np.column_stack([u, v]), depth


def sign_corners(center: Sequence[float], normal: Sequence[float], size: Sequence[float],
                 thickness: float = 0.02) -> np.ndarray:
    """Four corners of the front face of a sign (gt sign_center is the box center, normal points into the room)."""
    c, n = np.asarray(center, float), np.asarray(normal, float)
    along = np.array([-n[1], n[0], 0.0])
    w, h = size
    front = c + n * thickness / 2
    return np.array([front + sa * along * w / 2 + sz * np.array([0.0, 0.0, h / 2])
                     for sa, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def sign_label(exit_gt: dict, pose: Sequence[float], K: np.ndarray, width: int, height: int,
               depth_image: Optional[np.ndarray] = None, min_visible: float = 0.6, depth_tolerance: float = 0.3,
               max_view_angle_deg: float = 80.0) -> Optional[dict]:
    """Box (x1, y1, x2, y2) of a sign in the frame, or None when it is behind, out of the frame (less than
    `min_visible` of its box inside), seen too edge-on, or occluded (the depth at its center is nearer than expected by
    more than `depth_tolerance`)."""
    corners = sign_corners(exit_gt["sign_center"], exit_gt["normal"], exit_gt["sign_size"])
    center = corners.mean(axis=0)
    to_cam = np.asarray(pose[:3]) - center
    dist = float(np.linalg.norm(to_cam))
    cos_view = float(np.asarray(exit_gt["normal"]) @ to_cam / dist)
    if cos_view < math.cos(math.radians(max_view_angle_deg)):
        return None
    uv, depth = project(corners, pose, K)
    if np.any(depth <= 0.1):
        return None
    x1, y1 = uv.min(axis=0)
    x2, y2 = uv.max(axis=0)
    area = (x2 - x1) * (y2 - y1)
    cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)
    if area <= 0 or cx2 <= cx1 or cy2 <= cy1 or (cx2 - cx1) * (cy2 - cy1) < min_visible * area:
        return None
    uvc, dc = project(center[None, :], pose, K)
    (cu, cv), cdepth = uvc[0], dc[0]
    if depth_image is not None:
        iu, iv = int(round(cu)), int(round(cv))
        if not (0 <= iu < width and 0 <= iv < height):
            return None
        d = float(depth_image[iv, iu])
        if math.isfinite(d) and d < cdepth - depth_tolerance:
            return None
    return {"name": exit_gt["name"], "cls": exit_gt["cls"], "box": [float(cx1), float(cy1), float(cx2), float(cy2)],
            "distance": dist, "view_angle_deg": math.degrees(math.acos(min(1.0, cos_view)))}
