"""Zone in front of an exit door: grid, corridor and verdict on synthetic clouds."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exit_inspection import zone as z  # noqa: E402

P = z.ZoneParams()
FLOOR = -0.44  # floor height in camera_init (lidar about 0.44 m above the floor at the start)
EXIT = {"id": 3, "type": "door", "confirmed": True, "door_floor_center": [2.0, 6.0, FLOOR], "normal": [0.0, -1.0, 0.0],
        "door_width": 1.2}
SENSOR_UV = (0.0, 4.5)  # robot 4.5 m in front of the door, facing it
STEP = 0.025


def zone(**kw):
    return z.Zone.from_exit(EXIT, z.ZoneParams(**kw) if kw else P)


def grid_uv(u0, u1, v0, v1, step=STEP):
    uu, vv = np.meshgrid(np.arange(u0, u1, step) + step / 2, np.arange(v0, v1, step) + step / 2, indexing="ij")
    return uu.ravel(), vv.ravel()


def floor_pts(zn, u0=-1.2, u1=1.2, v0=-0.05, v1=3.1, hole=None):
    """Floor points (map frame) on [u0,u1] x [v0,v1]; hole = (u0, u1, v0, v1) without points."""
    u, v = grid_uv(u0, u1, v0, v1)
    if hole is not None:
        keep = ~((u >= hole[0]) & (u <= hole[1]) & (v >= hole[2]) & (v <= hole[3]))
        u, v = u[keep], v[keep]
    return zn.to_map(u, v, np.full(u.size, FLOOR))


def box_pts(zn, u0, u1, v0, v1, height):
    """Top and the four sides of a box standing on the floor (map frame)."""
    parts = []
    u, v = grid_uv(u0, u1, v0, v1)
    parts.append(zn.to_map(u, v, np.full(u.size, FLOOR + height)))
    hs = np.arange(0.0, height, STEP) + STEP / 2
    for uu in (u0, u1):
        vv, h = np.meshgrid(np.arange(v0, v1, STEP), hs, indexing="ij")
        parts.append(zn.to_map(np.full(vv.size, uu), vv.ravel(), FLOOR + h.ravel()))
    for vv in (v0, v1):
        uu, h = np.meshgrid(np.arange(u0, u1, STEP), hs, indexing="ij")
        parts.append(zn.to_map(uu.ravel(), np.full(uu.size, vv), FLOOR + h.ravel()))
    return np.vstack(parts)


def wall_pts(zn, offset=0.0):
    """The wall beside the door (|u| from 0.66 to 1.5), from the floor to 2.5 m, at v = offset."""
    parts = []
    for side in (-1, 1):
        u, h = grid_uv(0.66, 1.5, 0.0, 2.5)
        parts.append(zn.to_map(side * u, np.full(u.size, offset), FLOOR + h))
    return np.vstack(parts)


def frames(zn, pts, n=3, rng=None, noise=0.003):
    rng = rng or np.random.default_rng(0)
    sensor = zn.to_map(np.array([SENSOR_UV[0]]), np.array([SENSOR_UV[1]]), np.array([FLOOR + 0.35]))[0]
    return [z.Frame(stamp=float(i), sensor=sensor, points=pts + rng.normal(0.0, noise, pts.shape)) for i in range(n)]


def check(zn, pts, persons=None, **kw):
    return z.evaluate(zn, frames(zn, pts, **kw), np.zeros((0, 2)) if persons is None else np.asarray(persons))


def test_local_and_to_map_round_trip():
    zn = zone()
    p = zn.to_map(np.array([0.5]), np.array([2.0]), np.array([FLOOR + 1.0]))
    np.testing.assert_allclose(p[0], [2.0 + 0.5 * zn.along[0], 6.0 - 2.0, FLOOR + 1.0], atol=1e-9)
    np.testing.assert_allclose(zn.local(p)[0], [0.5, 2.0, FLOOR + 1.0], atol=1e-9)
    assert zn.u_half == pytest.approx(1.1)
    assert zn.shape == (44, 58)


def test_corridor_on_hand_made_grids():
    nu, nv, cell, door_half = 44, 58, 0.05, 0.6
    free = np.zeros((nu, nv), bool)
    assert z.corridor_exists(free, cell, 0.9, door_half)
    across = free.copy()
    across[:, 30] = True  # a wall across the zone 1.6 m from the door: the zone's sides are open
    assert z.corridor_exists(across, cell, 0.9, door_half)
    closed = across.copy()
    closed[:2, :31] = closed[-2:, :31] = True  # and closed at the sides: a U around the door
    assert not z.corridor_exists(closed, cell, 0.9, door_half)
    gap = closed.copy()
    u = (np.arange(nu) + 0.5) * cell - nu * cell / 2
    gap[np.abs(u) < 0.5, 30] = False  # a 1.0 m gap
    assert z.corridor_exists(gap, cell, 0.9, door_half)
    assert not z.corridor_exists(gap, cell, 1.2, door_half)
    assert z.widest_corridor(gap, cell, door_half, 2.2) == pytest.approx(1.0, abs=0.051)
    at_door = free.copy()
    at_door[np.abs(u) < 0.7, 3:8] = True  # an object across the door, 0.25 m from it
    assert not z.corridor_exists(at_door, cell, 0.9, door_half)
    # the wall beside the door: no corridor wider than the door, and none sliding along the wall
    assert z.widest_corridor(free, cell, door_half, 2.2) == pytest.approx(1.2, abs=0.051)
    low = free.copy()
    low[:, 9:] = True  # objects from 0.55 m on: a 0.9 m disk cannot turn along the wall
    assert not z.corridor_exists(low, cell, 0.9, door_half)


def test_empty_zone_seen_whole_is_clear():
    zn = zone()
    r = check(zn, np.vstack([floor_pts(zn), wall_pts(zn)]))
    assert r.verdict == "CLEAR", r.reason
    assert r.coverage > 0.99 and r.object_area == 0.0
    assert r.corridor_width >= 1.15
    assert r.floor_z == pytest.approx(FLOOR, abs=0.01)
    assert r.frames == 3


def test_box_beside_the_door_is_a_warning():
    zn = zone()
    box = (0.8, 1.4, 0.5, 1.1)  # the exit room's warning box, 0.3 m inside the zone
    pts = np.vstack([floor_pts(zn, hole=box), box_pts(zn, *box, 0.8), wall_pts(zn)])
    r = check(zn, pts)
    assert r.verdict == "WARNING", r.reason
    assert 0.05 < r.object_area < 0.25  # 0.3 x 0.6 m of it inside the zone
    assert 1.0 <= r.corridor_width <= 1.25


def test_pallet_across_the_door_is_blocked():
    zn = zone()
    pallet = (-0.6, 0.6, 0.3, 1.1)
    shadow = (-0.7, 0.7, -0.1, 1.1)  # nothing seen between the pallet and the door
    pts = np.vstack([floor_pts(zn, hole=shadow), box_pts(zn, *pallet, 1.15), wall_pts(zn)])
    r = check(zn, pts)
    assert r.verdict == "BLOCKED", r.reason
    assert r.corridor_width_optimistic < 0.9


def test_far_half_unseen_is_unverified():
    zn = zone()
    r = check(zn, np.vstack([floor_pts(zn, v1=1.5), wall_pts(zn)]))
    assert r.verdict == "UNVERIFIED", r.reason
    # the rays to the last floor points cross 0.4 m more of it below 5 cm
    assert "unseen" in r.reason and 0.4 < r.coverage < 0.75


def test_grazing_floor_rows_are_filled_by_the_rays():
    """Far from the camera the floor comes in rows (here every 0.3 m): the rays below 5 cm fill the gaps."""
    zn = zone()
    u, v = grid_uv(-1.2, 1.2, -0.05, 3.1)
    rows = (np.round(v / STEP) % 12) == 0
    floor = zn.to_map(u[rows], v[rows], np.full(rows.sum(), FLOOR))
    r = check(zn, np.vstack([floor, wall_pts(zn)]))
    assert r.verdict == "CLEAR", r.reason
    assert r.coverage > 0.95


def front_face(zn, u0, u1, v, height):
    hs = np.arange(0.0, height, STEP) + STEP / 2
    uu, h = np.meshgrid(np.arange(u0, u1, STEP), hs, indexing="ij")
    return zn.to_map(uu.ravel(), np.full(uu.size, v), FLOOR + h.ravel())


def shadow_hole(zn, u0, u1, v, step=STEP):
    """Floor points not hidden by a face (u0..u1 at v) seen from the sensor."""
    u, vv = grid_uv(-1.2, 1.2, -0.05, 3.1, step)
    su, sv = SENSOR_UV
    t = (v - sv) / (vv - sv)  # fraction of the way to the floor point where the ray crosses v
    cross_u = su + t * (u - su)
    hidden = (vv < v) & (cross_u >= u0) & (cross_u <= u1)
    return zn.to_map(u[~hidden], vv[~hidden], np.full((~hidden).sum(), FLOOR))


def test_pallet_seen_from_the_front_only_is_blocked():
    """The exit room's pallet: cartons 1.15 m high, 0.3 m from the door; the camera sees the front face only."""
    zn = zone()
    pts = np.vstack([shadow_hole(zn, -0.6, 0.6, 1.1), front_face(zn, -0.6, 0.6, 1.1, 1.15), wall_pts(zn)])
    r = check(zn, pts)
    assert r.verdict == "BLOCKED", r.reason


def test_small_box_in_front_of_the_door_is_not_blocked():
    zn = zone()
    pts = np.vstack([shadow_hole(zn, -0.15, 0.15, 1.8), front_face(zn, -0.15, 0.15, 1.8, 0.4), wall_pts(zn)])
    r = check(zn, pts)
    assert r.verdict in ("WARNING", "UNVERIFIED"), r.reason
    assert r.corridor_width_optimistic >= 0.9


def test_unseen_band_at_the_door_is_unverified_not_blocked():
    zn = zone()
    r = check(zn, np.vstack([floor_pts(zn, hole=(-1.2, 1.2, -0.1, 0.8)), wall_pts(zn)]))
    assert r.verdict == "UNVERIFIED", r.reason
    assert r.corridor_width < 0.9 <= r.corridor_width_optimistic


def person_scene(zn):
    """A standing person 0.9 m in front of the door, the floor behind it (as seen from the sensor) unseen."""
    pu, pv = 0.2, 0.9
    ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    hs = np.arange(0.0, 1.7, STEP)
    a, h = np.meshgrid(ang, hs, indexing="ij")
    body = zn.to_map(pu + 0.2 * np.cos(a.ravel()), pv + 0.15 * np.sin(a.ravel()), FLOOR + h.ravel())
    u, v = grid_uv(-1.2, 1.2, -0.05, 3.1)
    # shadow: the floor whose ray from the sensor passes within 0.25 m of the person, behind it
    su, sv = SENSOR_UV
    d = np.stack([u - su, v - sv], 1)
    t = np.clip(((pu - su) * d[:, 0] + (pv - sv) * d[:, 1]) / (d ** 2).sum(1), 0, 1)
    dist = np.hypot(su + t * d[:, 0] - pu, sv + t * d[:, 1] - pv)
    seen = ~((dist < 0.25) & (v < pv)) & ~((np.hypot(u - pu, v - pv) < 0.2))
    floor = zn.to_map(u[seen], v[seen], np.full(seen.sum(), FLOOR))
    person_xy = zn.to_map(np.array([pu]), np.array([pv]), np.array([FLOOR]))[0, :2]
    return np.vstack([floor, body, wall_pts(zn)]), person_xy


def test_person_and_its_shadow_are_excluded():
    zn = zone()
    pts, person_xy = person_scene(zn)
    r = check(zn, pts, persons=[person_xy])
    assert r.verdict == "CLEAR", r.reason
    assert r.people == 1 and r.person_area > 0.3 and r.behind_person_area > 0.0
    without = check(zn, pts)
    assert without.verdict != "CLEAR"
    assert without.object_area > 0.05


def test_limbs_beyond_the_person_disk_belong_to_the_person():
    """A walking actor swings legs and arms up to about 0.5 m from its center, outside the 0.4 m disk."""
    zn = zone()
    pts, person_xy = person_scene(zn)
    leg = box_pts(zn, 0.15, 0.25, 1.35, 1.45, 0.6)  # 0.5 m in front of the person center (0.2, 0.9)
    r = check(zn, np.vstack([pts, leg]), persons=[person_xy])
    assert r.verdict == "CLEAR", r.reason
    # a pallet the person stands next to is still an object: too far from the person's center
    pallet = box_pts(zn, -0.4, 0.8, 1.3, 2.1, 0.6)
    r = check(zn, np.vstack([pts, pallet]), persons=[person_xy])
    assert r.object_area > 0.3


def test_isolated_points_and_one_frame_points_are_ignored():
    zn = zone()
    base = np.vstack([floor_pts(zn), wall_pts(zn)])
    stray = zn.to_map(np.array([0.0, 0.01]), np.array([1.5, 1.51]), np.array([FLOOR + 0.5, FLOOR + 0.5]))
    assert check(zn, np.vstack([base, stray])).verdict == "CLEAR"  # 2 points in one cell
    fs = frames(zn, base)
    fs[0] = z.Frame(fs[0].stamp, fs[0].sensor, np.vstack([fs[0].points, box_pts(zn, -0.3, 0.3, 1.0, 1.4, 0.5)]))
    assert z.evaluate(zn, fs, np.zeros((0, 2))).verdict == "CLEAR"  # a box in one frame only


def test_floor_below_the_stored_one_is_found():
    zn = z.Zone.from_exit(dict(EXIT, door_floor_center=[2.0, 6.0, FLOOR + 0.12]), P)
    box = (0.8, 1.4, 0.5, 1.1)
    # stored floor 12 cm too high: without the estimate the box (0.8 m) would still be an object, the floor too low
    r = z.evaluate(zn, frames(zn, np.vstack([floor_pts(zn, hole=box), box_pts(zn, *box, 0.8), wall_pts(zn)])),
                   np.zeros((0, 2)))
    assert r.floor_z == pytest.approx(FLOOR, abs=0.015)
    assert r.verdict == "WARNING"


def test_wall_offset_is_absorbed():
    zn = zone()
    # the stored wall line is 12 cm in front of the real one: the wall is at v = -0.12 ... and 12 cm behind it
    for off in (-0.12, 0.12):
        pts = np.vstack([floor_pts(zn, v0=off - 0.05), wall_pts(zn, off),
                         # the door panel and its frame across the opening, as a scan sees them
                         zn.to_map(*[a.ravel() for a in np.meshgrid(np.arange(-0.6, 0.6, STEP), [off - 0.05])],
                                   np.full(48, FLOOR + 1.0))])
        r = check(zn, pts)
        assert r.wall_offset == pytest.approx(off, abs=0.02)
        assert r.verdict == "CLEAR", (off, r.reason)


def test_lidar_alone_never_blocks():
    """People are told apart by the camera: without depth frames a lidar-only view stays UNVERIFIED."""
    zn = zone()
    pallet = (-0.6, 0.6, 0.3, 1.1)
    fs = [z.Frame(f.stamp, f.sensor, f.points, "lidar") for f in frames(zn, box_pts(zn, *pallet, 1.15))]
    r = z.evaluate(zn, fs, np.zeros((0, 2)))
    assert r.verdict == "UNVERIFIED" and "camera" in r.reason
    assert r.object_area > 0.5  # the grid is still there (markers)


def test_no_frames_is_unverified():
    r = z.evaluate(zone(), [], np.zeros((0, 2)))
    assert r.verdict == "UNVERIFIED" and r.reason == "no data" and r.frames == 0


def test_summary_is_plain_data():
    zn = zone()
    s = check(zn, np.vstack([floor_pts(zn), wall_pts(zn)])).summary()
    assert s["verdict"] == "CLEAR"
    for v in s.values():
        assert isinstance(v, (str, int, float))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
