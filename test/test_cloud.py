"""Rolling voxel cloud."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exit_inspection.cloud import RollingCloud, voxel_downsample  # noqa: E402


def test_voxel_downsample_keeps_one_point_per_voxel_negative_coordinates_too():
    pts = np.array([[0.01, 0.01, 0.01], [0.02, 0.03, 0.04], [-0.01, 0.0, 0.0], [-0.04, 0.01, 0.0], [1.0, 1.0, 1.0]])
    assert len(voxel_downsample(pts, 0.05)) == 3


def test_rolling_cloud_merges_voxels_and_ages_out():
    c = RollingCloud(window=5.0, voxel=0.05)
    c.add(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), 0.0)
    c.add(np.array([[0.01, 0.01, 0.01], [2.0, 0.0, 0.0]]), 3.0)  # first point lands in the voxel of (0, 0, 0)
    assert len(c.points()) == 3
    np.testing.assert_allclose(sorted(c.points()[:, 0]), [0.01, 1.0, 2.0])  # newest point of the shared voxel
    c.add(np.empty((0, 3)), 5.5)  # (1, 0, 0) seen at 0.0 is now older than 5 s; the shared voxel was refreshed at 3.0
    np.testing.assert_allclose(sorted(c.points()[:, 0]), [0.01, 2.0])


def test_rolling_cloud_radius_query():
    c = RollingCloud(window=5.0, voxel=0.05)
    c.add(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]), 0.0)
    np.testing.assert_allclose(c.points(np.zeros(3), 5.0), [[0.0, 0.0, 0.0]])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
