"""Ground-truth check of exits.yaml against the exit room."""

import importlib.util
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("check_exits_gt", os.path.join(HERE, "..", "scripts",
                                                                                "check_exits_gt.py"))
cg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cg)

GT = {"exits": [
    {"name": "clear", "type": "door", "sign_center": [-5.0, 5.99, 2.25], "normal": [0.0, -1.0, 0.0]},
    {"name": "route", "type": "route", "sign_center": [9.99, 1.0, 2.25], "normal": [-1.0, 0.0, 0.0]},
]}
BASE0 = (1.0, 2.0, 0.38, math.pi / 2)


def to_map(p, T, direction=False):
    return (T[:3, :3] @ np.array(p) + (0 if direction else T[:3, 3])).tolist()


def exit_entry(i, center, normal, type_="door", confirmed=True):
    return {"id": i, "type": type_, "confirmed": confirmed, "sign_center": center, "normal": normal, "n_sightings": 9}


def test_world_to_map_puts_the_lidar_start_at_the_origin():
    T = cg.world_to_map(BASE0, [0.2835, 0.0, 0.066])
    lidar_world = [1.0, 2.0 + 0.2835, 0.38 + 0.066]  # base facing +y, lidar 0.2835 m ahead
    np.testing.assert_allclose(to_map(lidar_world, T), [0.0, 0.0, 0.0], atol=1e-9)


def test_pass_fail_and_false_exits():
    T = cg.world_to_map(BASE0, [0.2835, 0.0, 0.066])
    door = GT["exits"][0]
    good = exit_entry(0, to_map(np.array(door["sign_center"]) + [0.1, 0.0, 0.05], T),
                      to_map(door["normal"], T, True))
    route = exit_entry(1, to_map(GT["exits"][1]["sign_center"], T), to_map(GT["exits"][1]["normal"], T, True),
                       "route")
    ok, _ = cg.check({"exits": [good, route]}, GT, T, 0.3, 15.0)
    assert ok
    far = dict(good, sign_center=to_map(np.array(door["sign_center"]) + [0.5, 0.0, 0.0], T))
    assert not cg.check({"exits": [far, route]}, GT, T, 0.3, 15.0)[0]  # door missed, and a false exit
    unconfirmed = dict(good, confirmed=False)
    assert not cg.check({"exits": [unconfirmed]}, GT, T, 0.3, 15.0)[0]
    route_as_door = dict(route, type="door")
    assert not cg.check({"exits": [good, route_as_door]}, GT, T, 0.3, 15.0)[0]
    assert cg.check({"exits": [good]}, GT, T, 0.3, 15.0)[0]  # route sign never mapped: fine
