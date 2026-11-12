"""Ground-truth check of the zone verdicts, and the ground-truth exits.yaml of the test tool."""

import importlib.util
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cz = load("check_zones_gt", os.path.join(HERE, "..", "scripts", "check_zones_gt.py"))
ge = load("gt_to_exits", os.path.join(HERE, "gt_to_exits.py"))

GT = {"door_width": 1.2, "door_height": 2.1, "exits": [
    {"name": "clear", "type": "door", "cls": "exit_plain", "sign_center": [-5.0, 5.99, 2.25], "normal": [0, -1, 0],
     "door_floor_center": [-5.0, 6.0, 0.0], "expected": "CLEAR"},
    {"name": "blocked", "type": "door", "cls": "exit_plain", "sign_center": [-4.0, -5.99, 2.25], "normal": [0, 1, 0],
     "door_floor_center": [-4.0, -6.0, 0.0], "expected": "BLOCKED"},
    {"name": "route", "type": "route", "cls": "exit_arrow_side", "sign_center": [9.99, 1.0, 2.25],
     "normal": [-1, 0, 0], "expected": "IGNORED"},
]}
BASE0 = (1.0, 2.0, 0.38, math.pi / 2)


def test_gt_exits_and_verdict_matching():
    T = cz.world_to_map(BASE0, [0.2835, 0.0, 0.066])
    doc = ge.exits_from_gt(GT, T)
    doors = [e for e in doc["exits"] if e["type"] == "door"]
    assert len(doors) == 2 and all("door_floor_center" in e for e in doors)
    route = [e for e in doc["exits"] if e["type"] == "route"][0]
    assert "door_floor_center" not in route
    # base faces +y at (1, 2): world (-5, 6) is 4 m ahead (+x of the map) and 6 m to the left (+y) of the base
    np.testing.assert_allclose(doors[0]["door_floor_center"][:2], [4.0 - 0.2835, 6.0], atol=1e-3)

    def snap(exit_, verdict):
        return {"exits": [{"id": exit_["id"], "door_floor_center": exit_["door_floor_center"], "verdict": verdict,
                           "reason": "r", "coverage": 1.0, "frames": 5, "people": 0}]}

    ok, lines = cz.check({"clear": snap(doors[0], "CLEAR"), "blocked": snap(doors[1], "BLOCKED")}, GT, T, 0.5)
    assert ok, lines
    ok, lines = cz.check({"clear": snap(doors[0], "CLEAR"), "blocked": snap(doors[1], "WARNING")}, GT, T, 0.5)
    assert not ok and "FAIL blocked" in lines[1]
    ok, lines = cz.check({"clear": snap(doors[0], "CLEAR"), "blocked": snap(doors[0], "BLOCKED")}, GT, T, 0.5)
    assert not ok and "no verdict" in lines[1]  # the snapshot holds another door only
    ok, lines = cz.check({"clear": snap(doors[0], "CLEAR")}, GT, T, 0.5)
    assert not ok and "no snapshot" in lines[1]
