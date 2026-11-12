"""Sightings (one detector box cast into the cloud) and landmarks (sightings of one sign merged)."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exit_inspection import landmarks as lm  # noqa: E402

K = np.array([[1386.0, 0.0, 960.0], [0.0, 1386.0, 540.0], [0.0, 0.0, 1.0]])
CAM_HEIGHT = 0.35
# Camera at (0, 0, 0.35) looking along +x: optical x = -y, optical y = -z, optical z = +x of the map.
T_MAP_CAM = np.eye(4)
T_MAP_CAM[:3, :3] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
T_MAP_CAM[:3, 3] = [0.0, 0.0, CAM_HEIGHT]
SIGN = np.array([5.0, 0.0, 2.2])
PARAMS = lm.MapperParams()
DOOR_CLASSES = ("exit_plain", "exit_arrow_down")


def scene(rng, wall=True, floor=True, step=0.05, noise=0.01):
    parts = []
    ys = np.arange(-3.0, 3.0, step)
    if wall:
        zs = np.arange(0.0, 3.0, step)
        yy, zz = np.meshgrid(ys, zs)
        parts.append(np.stack([np.full(yy.size, 5.0), yy.ravel(), zz.ravel()], axis=1))
    if floor:
        xx, yy = np.meshgrid(np.arange(0.0, 5.0, step), ys)
        parts.append(np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1))
    pts = np.vstack(parts)
    return pts + rng.normal(0.0, noise, pts.shape)


def project(p):
    """Pixel of a map point in the test camera."""
    q = T_MAP_CAM[:3, :3].T @ (p - T_MAP_CAM[:3, 3])
    return K[0, 0] * q[0] / q[2] + K[0, 2], K[1, 1] * q[1] / q[2] + K[1, 2]


def sign_box(width_m=0.4, cls="exit_plain"):
    u, v = project(SIGN)
    w_px = width_m * K[0, 0] / (SIGN[0] - T_MAP_CAM[0, 3])
    return (cls, 0.9, u, v, w_px, w_px / 2)


def sight(rng, box, cloud):
    return lm.make_sighting(box, K, T_MAP_CAM, cloud, np.array([0.0, 0.0, CAM_HEIGHT]), 12.5, PARAMS, rng)


def test_sighting_on_wall():
    rng = np.random.default_rng(0)
    s, reason = sight(rng, sign_box(), scene(rng))
    assert reason == "ok"
    np.testing.assert_allclose(s.hit, SIGN, atol=0.03)
    np.testing.assert_allclose(s.normal, [-1.0, 0.0, 0.0], atol=0.02)
    assert abs(s.width_m - 0.4) < 0.03
    assert abs(s.floor_z) < 0.03
    assert not s.low_quality
    assert s.cls == "exit_plain" and s.stamp == 12.5
    assert abs(s.range_m - np.linalg.norm(SIGN - T_MAP_CAM[:3, 3])) < 0.05


def test_box_beside_the_wall_has_no_hit():
    rng = np.random.default_rng(1)
    box = ("exit_plain", 0.9, 1900.0, 540.0, 50.0, 25.0)  # ray passes y = -3.4 at x = 5, beyond the wall
    s, reason = sight(rng, box, scene(rng))
    assert s is None and reason == "no_hit"


def test_hit_on_the_floor_is_rejected_by_height():
    rng = np.random.default_rng(2)
    box = ("exit_plain", 0.9, 960.0, 1000.0, 50.0, 25.0)  # ray reaches the floor about 1 m ahead
    s, reason = sight(rng, box, scene(rng))
    assert s is None and reason == "height"


def test_too_wide_is_rejected():
    rng = np.random.default_rng(3)
    s, reason = sight(rng, sign_box(width_m=1.5), scene(rng))
    assert s is None and reason == "width"


def test_sparse_wall_gives_low_quality_sighting():
    rng = np.random.default_rng(4)
    patch = SIGN + np.column_stack([np.zeros(12), rng.uniform(-0.05, 0.05, 12), rng.uniform(-0.05, 0.05, 12)])
    cloud = np.vstack([patch, scene(rng, wall=False)])
    s, reason = sight(rng, sign_box(), cloud)
    assert reason == "ok" and s.low_quality
    np.testing.assert_allclose(s.normal, [-1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(s.hit, SIGN, atol=0.06)


def test_no_floor_falls_back_to_robot_height():
    rng = np.random.default_rng(5)
    s, reason = sight(rng, sign_box(), scene(rng, floor=False))
    assert reason == "ok"
    assert abs(s.floor_z - (CAM_HEIGHT - PARAMS.base_height)) < 1e-9


def mk(hit, normal=(-1.0, 0.0, 0.0), robot=(0.0, 0.0, 0.35), cls="exit_plain", width=0.4, conf=0.8):
    return lm.Sighting(stamp=0.0, cls=cls, conf=conf, hit=np.array(hit, float), normal=np.array(normal, float),
                       robot=np.array(robot, float), range_m=5.0, width_m=width, floor_z=0.0, low_quality=False)


def test_group_of():
    assert lm.group_of("exit_plain", DOOR_CLASSES) == "door"
    assert lm.group_of("exit_arrow_side", DOOR_CLASSES) == "route"


def test_association_merges_close_and_splits_far_or_turned():
    ls = lm.LandmarkSet(PARAMS, DOOR_CLASSES)
    a = ls.add(mk([5.0, 0.0, 2.2]))
    assert ls.add(mk([5.0, 0.6, 2.2])) == a  # 0.6 m away: same sign
    assert ls.add(mk([5.0, 2.5, 2.2])) != a  # 2.5 m away
    turned = (-math.cos(math.radians(60)), math.sin(math.radians(60)), 0.0)
    assert ls.add(mk([5.0, 0.1, 2.2], normal=turned)) != a  # 60 deg off
    assert len(ls.landmarks()) == 3


def test_a_misclassified_sighting_joins_the_sign_and_the_majority_decides_the_group():
    """One physical sign is one landmark: a few boxes of the wrong class must not make a route sign next to a door."""
    ls = lm.LandmarkSet(PARAMS, DOOR_CLASSES)
    a = ls.add(mk([5.0, 0.0, 2.2], cls="exit_arrow_down"))
    assert ls.add(mk([5.0, 0.05, 2.2], cls="exit_arrow_side")) == a
    assert ls.add(mk([5.0, -0.05, 2.2], cls="exit_arrow_down")) == a
    (l,) = ls.landmarks()
    assert l.cls == "exit_arrow_down" and l.group == "door"
    for _ in range(3):
        ls.add(mk([5.0, 0.0, 2.2], cls="exit_arrow_side"))
    assert ls.landmarks()[0].group == "route"  # 4 side against 2 down


def test_landmark_center_normal_width_majority():
    ls = lm.LandmarkSet(PARAMS, DOOR_CLASSES)
    for y, cls in [(0.0, "exit_plain"), (0.1, "exit_arrow_down"), (-0.1, "exit_plain"), (0.05, "exit_plain")]:
        ls.add(mk([5.0, y, 2.2], cls=cls, width=0.4 + y))
    (l,) = ls.landmarks()
    np.testing.assert_allclose(l.center, [5.0, 0.025, 2.2], atol=1e-9)
    np.testing.assert_allclose(l.normal, [-1.0, 0.0, 0.0], atol=1e-9)
    assert abs(l.width - 0.425) < 1e-9
    assert l.cls == "exit_plain" and l.group == "door"
    assert abs(l.mean_conf - 0.8) < 1e-9
    np.testing.assert_allclose(l.door_floor_center, [5.0, 0.025, 0.0], atol=1e-9)


def test_confirmation_needs_count_and_baseline():
    ls = lm.LandmarkSet(PARAMS, DOOR_CLASSES)
    for i in range(5):
        ls.add(mk([5.0, 0.0, 2.2], robot=(0.02 * i, 0.0, 0.35)))
    assert not ls.landmarks()[0].confirmed  # 5 sightings from almost one place
    ls = lm.LandmarkSet(PARAMS, DOOR_CLASSES)
    for i in range(4):
        ls.add(mk([5.0, 0.0, 2.2], robot=(0.3 * i, 0.0, 0.35)))
    assert not ls.landmarks()[0].confirmed  # 4 sightings
    ls.add(mk([5.0, 0.0, 2.2], robot=(1.2, 0.0, 0.35)))
    assert ls.landmarks()[0].confirmed


def test_many_sightings_from_a_short_baseline_confirm_too():
    """A sign glimpsed while the robot turns: 9 sightings within 0.3 m (the exit room's warning door, first full run)."""
    ls = lm.LandmarkSet(PARAMS, DOOR_CLASSES)
    for i in range(7):
        ls.add(mk([5.0, 0.0, 2.2], robot=(0.05 * i, 0.0, 0.35)))
    assert not ls.landmarks()[0].confirmed  # 7 sightings, 0.30 m
    ls.add(mk([5.0, 0.0, 2.2], robot=(0.3, 0.0, 0.35)))
    assert ls.landmarks()[0].confirmed      # 8 sightings, 0.30 m
    ls = lm.LandmarkSet(PARAMS, DOOR_CLASSES)
    for i in range(12):
        ls.add(mk([5.0, 0.0, 2.2], robot=(0.01 * i, 0.0, 0.35)))
    assert not ls.landmarks()[0].confirmed  # many, but from one spot


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
