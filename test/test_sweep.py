"""Exit sign sweep: wall/free grid from scans, camera looks, viewpoints for the walls not looked at."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exit_inspection import sweep as sw  # noqa: E402

FLOOR = -0.44
P = sw.SweepParams(size=40.0)


def room_points(hx=10.0, hy=6.0, step=0.1):
    """Walls of a hx*2 x hy*2 room, 3 m high, and its floor (the exit room)."""
    zs = np.arange(FLOOR, FLOOR + 3.0, 0.1)
    xs, ys = np.arange(-hx, hx + 1e-9, step), np.arange(-hy, hy + 1e-9, step)
    walls = [np.array([(x, y, z) for x in xs for z in zs]) for y in (-hy, hy)]
    walls += [np.array([(x, y, z) for y in ys for z in zs]) for x in (-hx, hx)]
    fx, fy = np.meshgrid(np.arange(-hx + 0.2, hx - 0.2, 0.25), np.arange(-hy + 0.2, hy - 0.2, 0.25))
    floor = np.column_stack([fx.ravel(), fy.ravel(), np.full(fx.size, FLOOR)])
    return np.vstack(walls + [floor])


def scanned(points, origins=((0, 0), (-6, 0), (6, 0)), p=P):
    g = sw.SweepGrid((0.0, 0.0), p)
    rng = np.random.default_rng(0)
    for o in origins:
        for _ in range(3):
            g.add_scan(o, points, FLOOR, rng)
    return g


def wall_of(g, ij):
    x, y = g.center(ij)
    return "north" if y > 5.5 else "south" if y < -5.5 else "east" if x > 9.5 else "west" if x < -9.5 else "?"


def test_grid_has_walls_free_space_and_normals_into_the_room():
    g = scanned(room_points())
    walls = g.walls()
    assert walls.sum() > 300
    ij, _ = g.index(np.array([[0.0, 6.0], [10.0, 0.0], [0.0, 0.0]]))
    assert walls[tuple(ij[0])] and walls[tuple(ij[1])] and not walls[tuple(ij[2])]
    assert g.free[tuple(ij[2])] > 0
    n = g.normals()
    np.testing.assert_allclose(n[tuple(ij[0])], [0.0, -1.0], atol=0.2)  # north wall faces south
    np.testing.assert_allclose(n[tuple(ij[1])], [-1.0, 0.0], atol=0.2)  # east wall faces west


def test_visibility_respects_distance_and_occlusion():
    g = scanned(room_points())
    walls_ij = np.argwhere(g.walls())
    seen, _ = g.visible((0.0, -1.0), walls_ij, g.normals())
    front = {wall_of(g, ij) for ij in walls_ij[seen] if abs(g.center(ij)[0]) < 3.0}
    assert front == {"north"}  # straight ahead: north 7 m away (seen), south 5 m away (too close)
    blocked = sw.SweepGrid((0.0, 0.0), P)
    pts = room_points()
    pillar = np.array([(x, 2.0, z) for x in np.arange(-1.0, 1.0, 0.1) for z in np.arange(FLOOR, FLOOR + 3.0, 0.1)])
    for o in ((0, 0), (-6, 0), (6, 0)):
        blocked.add_scan(o, np.vstack([pts, pillar]), FLOOR)
    walls_ij = np.argwhere(blocked.walls())
    seen, _ = blocked.visible((0.0, -1.0), walls_ij, blocked.normals())
    hidden = [ij for ij in walls_ij[seen] if wall_of(blocked, ij) == "north" and abs(blocked.center(ij)[0]) < 1.0]
    assert hidden == []  # the pillar 3 m ahead hides the middle of the north wall


def test_plan_covers_every_wall_from_far_enough():
    g = scanned(room_points())
    vps, stats = g.plan((0.0, 0.0))
    assert 2 <= len(vps) <= P.max_viewpoints
    assert stats["planned"] >= 0.9 * stats["coverable"] > 0
    for x, y, yaw, n in vps:
        assert abs(x) < 9.4 and abs(y) < 5.4  # inside the room, clear of the walls
        assert n >= P.min_new_cells
    faced = set()
    for x, y, yaw, _ in vps:
        dx, dy = math.cos(yaw), math.sin(yaw)
        faced.add("north" if dy > 0.7 else "south" if dy < -0.7 else "east" if dx > 0.7 else "west" if dx < -0.7 else "?")
    assert {"north", "south"} <= faced  # the long walls, where the exit room's doors are


def test_walls_already_looked_at_are_not_planned_again():
    g = scanned(room_points())
    vps, _ = g.plan((0.0, 0.0))
    for x, y, yaw, _ in vps:
        for _ in range(P.covered_looks):
            g.mark_look((x, y), yaw)
    again, stats = g.plan((0.0, 0.0))
    assert stats["targets"] < 0.5 * stats["wall_cells"]
    assert len(again) <= 2


def test_a_small_room_has_nothing_to_plan_from_far_enough():
    g = scanned(room_points(hx=3.0, hy=2.5), origins=((0, 0),))
    vps, stats = g.plan((0.0, 0.0))
    assert vps == [] and stats["coverable"] == 0 and stats["targets"] > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
