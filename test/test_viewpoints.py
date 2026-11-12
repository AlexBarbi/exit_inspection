"""Viewpoints in front of the exit doors: candidates, map checks, visiting order; the PCD reader."""

import math
import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exit_inspection import pcd  # noqa: E402
from exit_inspection import viewpoints as vpm  # noqa: E402

P = vpm.ViewpointParams()
FLOOR = -0.44
DOOR = {"id": 0, "door_floor_center": [0.0, 4.0, FLOOR], "normal": [0.0, -1.0, 0.0], "door_width": 1.2}


def wall(x0, y0, x1, y1, step=0.05):
    n = int(math.hypot(x1 - x0, y1 - y0) / step) + 1
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    zs = np.arange(FLOOR, FLOOR + 2.5, 0.1)
    return np.array([(x, y, z) for x, y in zip(xs, ys) for z in zs])


def room(depth):
    """Room from y = 4 (door wall) down to y = 4 - depth, 10 m wide."""
    y0 = 4.0 - depth
    return np.vstack([wall(-5, 4, 5, 4), wall(-5, y0, 5, y0), wall(-5, y0, -5, 4), wall(5, y0, 5, 4)])


def occ(points):
    return vpm.Occupancy.from_points(points, FLOOR, P)


def test_candidates_face_the_door_nominal_first():
    cands = vpm.candidates(DOOR, P)
    assert len(cands) == len(P.standoffs) * len(P.lateral)
    first = cands[0]
    assert (first.x, first.y) == pytest.approx((0.0, -0.5))
    assert first.yaw == pytest.approx(math.pi / 2)  # looking at the door, +y
    side = cands[1]  # lateral +0.6 along the wall
    assert math.atan2(4.0 - side.y, 0.0 - side.x) == pytest.approx(side.yaw)


def test_without_a_map_every_candidate_is_kept():
    (ep,) = vpm.plan([DOOR], P, None)
    assert len(ep.viewpoints) == 30 and ep.viewpoints[0].standoff == 4.5


def test_a_big_room_keeps_the_nominal_viewpoint():
    (ep,) = vpm.plan([DOOR], P, occ(room(8.0)))
    assert (ep.viewpoints[0].standoff, ep.viewpoints[0].lateral) == (4.5, 0.0)


def test_a_shallow_room_moves_the_viewpoint_closer():
    """Room 4.2 m deep: 4.5 m is behind the opposite wall (sight blocked), 4.0 m too close to it."""
    (ep,) = vpm.plan([DOOR], P, occ(room(4.2)))
    assert ep.viewpoints[0].standoff == 3.5
    assert ep.rejected.get("line of sight blocked", 0) > 0 and ep.rejected.get("robot does not fit", 0) > 0


def test_a_pillar_on_the_line_of_sight_moves_the_viewpoint_sideways():
    pillar = wall(-0.1, 0.0, 0.1, 0.0)  # at y = 0, between the nominal viewpoint (y = -0.5) and the zone (y >= 1)
    (ep,) = vpm.plan([DOOR], P, occ(np.vstack([room(8.0), pillar])))
    best = ep.viewpoints[0]
    assert best.standoff == 4.5 and best.lateral != 0.0


def test_an_object_inside_the_zone_does_not_move_the_viewpoint():
    pallet = wall(-0.6, 3.0, 0.6, 3.0)  # 1 m in front of the door: inside the zone
    (ep,) = vpm.plan([DOOR], P, occ(np.vstack([room(8.0), pallet])))
    assert (ep.viewpoints[0].standoff, ep.viewpoints[0].lateral) == (4.5, 0.0)


def test_points_outside_the_height_band_do_not_count():
    ceiling = np.array([(x, y, FLOOR + 3.0) for x in np.arange(-5, 5, 0.1) for y in np.arange(-4, 4, 0.1)])
    floor = ceiling - [0.0, 0.0, 3.0]
    (ep,) = vpm.plan([DOOR], P, occ(np.vstack([room(8.0), ceiling, floor])))
    assert (ep.viewpoints[0].standoff, ep.viewpoints[0].lateral) == (4.5, 0.0)


def test_tour_goes_to_the_nearest_first_and_leaves_unreachable_exits_last():
    def door(x):
        return {"id": x, "door_floor_center": [x, 4.0, FLOOR], "normal": [0.0, -1.0, 0.0]}
    plans = vpm.plan([door(5.0), door(1.0), door(3.0)], P, None)
    plans.insert(1, vpm.ExitPlan({"id": 99}))  # no viewpoint
    assert [plans[i].exit["id"] for i in vpm.tour((0.0, -0.5), plans)] == [1.0, 3.0, 5.0, 99]


def test_pcd_reader_binary_and_ascii(tmp_path):
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [np.nan, 0.0, 0.0]], np.float32)
    head = ("VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 3\nHEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\nPOINTS 3\nDATA {}\n")
    (tmp_path / "b.pcd").write_bytes(head.format("binary").encode() + pts.tobytes())
    (tmp_path / "a.pcd").write_text(head.format("ascii") + "\n".join(" ".join(map(str, p)) for p in pts) + "\n")
    # binary_compressed with the whole payload as one LZF literal run per <= 32 bytes
    cols = pts.T.astype("<f4").tobytes()
    lzf = b"".join(bytes([len(cols[i:i + 32]) - 1]) + cols[i:i + 32] for i in range(0, len(cols), 32))
    (tmp_path / "c.pcd").write_bytes(head.format("binary_compressed").encode() + struct.pack("<II", len(lzf),
                                                                                            len(cols)) + lzf)
    for name in ("a", "b", "c"):
        np.testing.assert_allclose(pcd.read_pcd_xyz(str(tmp_path / f"{name}.pcd")), pts[:2])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
