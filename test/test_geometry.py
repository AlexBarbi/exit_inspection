"""Geometry helpers of exit_inspection: rays, first hit along a ray, vertical plane fit."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exit_inspection import geometry as g  # noqa: E402

K = np.array([[1386.0, 0.0, 960.0], [0.0, 1386.0, 540.0], [0.0, 0.0, 1.0]])


def wall_cloud(rng, x=5.0, y_range=(-3.0, 3.0), z_range=(0.0, 3.0), step=0.05, noise=0.01):
    """Points of a wall at x (normal along x), with a 5 cm door recess at |y| < 0.6, z < 2.0, and a floor."""
    ys = np.arange(y_range[0], y_range[1], step)
    zs = np.arange(z_range[0], z_range[1], step)
    yy, zz = np.meshgrid(ys, zs)
    xx = np.full_like(yy, x)
    xx[(np.abs(yy) < 0.6) & (zz < 2.0)] += 0.05  # door recessed into the wall
    wall = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    fx = np.arange(0.0, x, step)
    fxx, fyy = np.meshgrid(fx, ys)
    floor = np.stack([fxx.ravel(), fyy.ravel(), np.zeros(fxx.size)], axis=1)
    pts = np.vstack([wall, floor])
    return pts + rng.normal(0.0, noise, pts.shape)


def test_pixel_ray_principal_point_is_optical_axis():
    np.testing.assert_allclose(g.pixel_ray(960.0, 540.0, K), [0.0, 0.0, 1.0], atol=1e-12)


def test_pixel_ray_matches_inverse_intrinsics():
    d = g.pixel_ray(1500.0, 200.0, K)
    expected = np.linalg.inv(K) @ np.array([1500.0, 200.0, 1.0])
    np.testing.assert_allclose(d, expected / np.linalg.norm(expected), atol=1e-12)
    assert math.isclose(np.linalg.norm(d), 1.0)


def test_pose_matrix_and_transforms():
    T = g.pose_matrix([1.0, 2.0, 3.0], [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])  # yaw 90 deg
    np.testing.assert_allclose(g.transform_points(T, np.array([[1.0, 0.0, 0.0]])), [[1.0, 3.0, 3.0]], atol=1e-12)
    np.testing.assert_allclose(g.transform_dirs(T, np.array([[1.0, 0.0, 0.0]])), [[0.0, 1.0, 0.0]], atol=1e-12)


def test_first_hit_picks_nearer_cluster_and_ignores_behind_and_off_ray():
    rng = np.random.default_rng(0)
    near = rng.normal([3.0, 0.0, 0.0], 0.02, (20, 3))
    far = rng.normal([6.0, 0.0, 0.0], 0.02, (50, 3))
    behind = rng.normal([-2.0, 0.0, 0.0], 0.02, (50, 3))
    off_ray = rng.normal([1.5, 1.0, 0.0], 0.02, (50, 3))
    pts = np.vstack([near, far, behind, off_ray])
    hit = g.first_hit(np.zeros(3), np.array([1.0, 0.0, 0.0]), pts, radius=0.15, min_points=5, window=0.3)
    assert hit is not None
    np.testing.assert_allclose(hit, [3.0, 0.0, 0.0], atol=0.03)


def test_first_hit_skips_sparse_cluster():
    rng = np.random.default_rng(1)
    sparse = rng.normal([2.0, 0.0, 0.0], 0.02, (3, 3))
    dense = rng.normal([4.0, 0.0, 0.0], 0.02, (10, 3))
    hit = g.first_hit(np.zeros(3), np.array([1.0, 0.0, 0.0]), np.vstack([sparse, dense]), 0.15, 5, 0.3)
    np.testing.assert_allclose(hit, [4.0, 0.0, 0.0], atol=0.03)


def test_first_hit_none_when_nothing_on_ray():
    pts = np.array([[1.0, 2.0, 0.0], [2.0, -2.0, 0.0]])
    assert g.first_hit(np.zeros(3), np.array([1.0, 0.0, 0.0]), pts, 0.15, 1, 0.3) is None


def test_fit_vertical_plane_on_wall_with_recess_and_floor():
    rng = np.random.default_rng(2)
    pts = wall_cloud(rng)
    near = pts[np.linalg.norm(pts - np.array([5.0, 0.0, 2.2]), axis=1) < 0.6]
    plane = g.fit_vertical_plane(near, rng, iterations=200, threshold=0.03, min_inliers=30, min_height=0.2)
    assert plane is not None
    n, d = plane
    assert n[2] == 0.0
    assert abs(abs(n[0]) - 1.0) < 1 - math.cos(math.radians(2.0))
    assert abs(abs(d) - 5.0) < 0.03


def test_fit_vertical_plane_rejects_floor():
    rng = np.random.default_rng(3)
    floor = np.column_stack([rng.uniform(0, 2, 300), rng.uniform(0, 2, 300), rng.normal(0, 0.01, 300)])
    assert g.fit_vertical_plane(floor, rng, 200, 0.03, 30, 0.2) is None


def test_fit_vertical_plane_needs_enough_inliers():
    rng = np.random.default_rng(4)
    few = np.column_stack([np.full(10, 5.0), rng.uniform(-1, 1, 10), rng.uniform(1, 3, 10)])
    assert g.fit_vertical_plane(few, rng, 200, 0.03, 30, 0.2) is None


def test_ray_plane():
    p = g.ray_plane(np.zeros(3), np.array([1.0, 0.0, 0.5]) / math.sqrt(1.25), np.array([-1.0, 0.0, 0.0]), 5.0)
    np.testing.assert_allclose(p, [5.0, 0.0, 2.5], atol=1e-9)
    assert g.ray_plane(np.zeros(3), np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 5.0) is None
    assert g.ray_plane(np.zeros(3), np.array([-1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 5.0) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
