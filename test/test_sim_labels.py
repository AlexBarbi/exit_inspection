"""Labels of the simulated signs in a Gazebo camera frame."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exit_inspection import sim_labels as sl  # noqa: E402

K = sl.intrinsics(1920, 1080, math.radians(69.4))
EXIT = {"name": "clear", "cls": "exit_arrow_down", "sign_center": [-5.0, 5.99, 2.25], "normal": [0.0, -1.0, 0.0],
        "sign_size": [0.4, 0.2]}


def test_intrinsics_match_the_bridged_camera_info():
    assert abs(K[0, 0] - 1386.416) < 0.01


def test_point_ahead_projects_to_the_principal_point():
    uv, depth = sl.project(np.array([[5.0, 0.0, 0.4]]), (0.0, 0.0, 0.4, 0.0, 0.0, 0.0), K)
    np.testing.assert_allclose(uv[0], [960.0, 540.0])
    assert depth[0] == 5.0


def test_left_and_up_in_the_world_are_left_and_up_in_the_image():
    uv, _ = sl.project(np.array([[5.0, 1.0, 1.4]]), (0.0, 0.0, 0.4, 0.0, 0.0, 0.0), K)
    assert uv[0, 0] < 960 and uv[0, 1] < 540


def test_pitch_down_moves_points_up_in_the_image():
    uv0, _ = sl.project(np.array([[5.0, 0.0, 0.4]]), (0.0, 0.0, 0.4, 0.0, 0.0, 0.0), K)
    uv1, _ = sl.project(np.array([[5.0, 0.0, 0.4]]), (0.0, 0.0, 0.4, 0.0, math.radians(10), 0.0), K)
    assert uv1[0, 1] < uv0[0, 1]


def test_sign_seen_from_10m_has_expected_box():
    pose = (-5.0, -4.0, 0.4, 0.0, 0.0, math.pi / 2)  # facing +y, the north wall
    lab = sl.sign_label(EXIT, pose, K, 1920, 1080)
    assert lab is not None
    x1, y1, x2, y2 = lab["box"]
    dist = 5.99 - 0.01 + 4.0
    assert abs((x2 - x1) - 0.4 * K[0, 0] / dist) < 1.0
    assert abs((x1 + x2) / 2 - 960) < 1.0
    assert abs(lab["distance"] - math.hypot(dist, 2.25 - 0.4)) < 0.01


def test_sign_behind_out_of_frame_edge_on_or_occluded_has_no_label():
    assert sl.sign_label(EXIT, (-5.0, -4.0, 0.4, 0.0, 0.0, -math.pi / 2), K, 1920, 1080) is None  # behind
    assert sl.sign_label(EXIT, (-5.0, 3.0, 0.4, 0.0, 0.0, math.pi / 2), K, 1920, 1080) is None  # above the frame
    assert sl.sign_label(EXIT, (5.0, 5.9, 2.25, 0.0, 0.0, math.pi), K, 1920, 1080) is None  # along the wall
    depth = np.full((1080, 1920), 3.0)  # something 3 m in front of the camera
    assert sl.sign_label(EXIT, (-5.0, -4.0, 0.4, 0.0, 0.0, math.pi / 2), K, 1920, 1080, depth) is None
    depth[:] = 30.0
    assert sl.sign_label(EXIT, (-5.0, -4.0, 0.4, 0.0, 0.0, math.pi / 2), K, 1920, 1080, depth) is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
