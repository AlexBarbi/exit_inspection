"""Choosing the verdict of a stop, and the report files."""

import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exit_inspection import report as rp  # noqa: E402


def ev(verdict, i=0):
    return {"verdict": verdict, "reason": f"r{i}", "coverage": 0.9, "stamp": float(i)}


def test_pick_verdict_majority_then_worse_then_latest():
    assert rp.pick_verdict([]) is None
    got = rp.pick_verdict([ev("CLEAR", 0), ev("BLOCKED", 1), ev("CLEAR", 2)])
    assert got["verdict"] == "CLEAR" and got["reason"] == "r2" and got["votes"] == "2/3"
    assert rp.pick_verdict([ev("CLEAR", 0), ev("WARNING", 1)])["verdict"] == "WARNING"
    assert rp.pick_verdict([ev("UNVERIFIED", 0), ev("CLEAR", 1)])["verdict"] == "UNVERIFIED"


def test_evaluations_without_camera_data_do_not_vote():
    gap = {"verdict": "UNVERIFIED", "reason": "camera did not see the zone (1 depth frames): people ...", "stamp": 1.0}
    got = rp.pick_verdict([gap, gap, ev("CLEAR", 2), gap])
    assert got["verdict"] == "CLEAR" and got["votes"] == "1/1"
    assert rp.pick_verdict([gap, gap])["verdict"] == "UNVERIFIED"  # nothing else: it stays unverified


def test_best_attempt_prefers_a_real_verdict():
    a1 = {"verdict": "UNVERIFIED", "reached": True}
    a2 = {"verdict": "WARNING", "reached": True}
    a3 = {"verdict": None, "reached": False, "reason": "not reached"}
    assert rp.best_attempt([a1, a2]) is a2
    assert rp.best_attempt([a1, a3]) is a3
    assert rp.best_attempt([]) is None


def test_exit_entry_and_report_files(tmp_path):
    door = {"id": 2, "cls": "exit_plain", "door_floor_center": [1.0, 4.0, -0.44], "normal": [0.0, -1.0, 0.0],
            "door_width": 1.2}
    vp = {"x": 1.0, "y": -0.5, "yaw": 1.5708, "standoff": 4.5, "lateral": 0.0}
    attempts = [{"viewpoint": vp, "reached": True, "verdict": "BLOCKED", "reason": "pallet <b>", "coverage": 0.78,
                 "object_area": 0.4, "corridor_width": 0.0, "people": 0, "votes": "6/6"}]
    e = rp.exit_entry(door, attempts, 42.0)
    assert e["verdict"] == "BLOCKED" and e["viewpoint"] == vp and e["duration"] == 42.0
    lost = rp.exit_entry(dict(door, id=3), [{"viewpoint": vp, "reached": False, "verdict": None,
                                              "reason": "not reached"}], 10.0)
    assert lost["verdict"] == "UNVERIFIED" and lost["reason"] == "not reached"
    mission = {"started": "2026-11-11T16:00:00", "finished": "2026-11-11T16:05:00", "exits_file": "x/exits.yaml",
               "planner": "direct", "params": {"margin": 0.5, "depth": 3.0}, "path": [[0, 0], [1, -0.5]],
               "exits": [e, lost]}
    walls = np.array([(x, 4.0, z) for x in np.arange(-3, 3, 0.05) for z in (-0.3, 0.5, 1.5)])
    paths = rp.write_report(str(tmp_path), "20261111-160000", mission, walls, floor_z=-0.44)
    doc = yaml.safe_load(open(paths["yaml"]))
    assert doc["summary"] == {"CLEAR": 0, "WARNING": 0, "BLOCKED": 1, "UNVERIFIED": 1}
    assert yaml.safe_load(open(paths["latest"])) == doc
    page = open(paths["html"]).read()
    assert "<svg" in page and "BLOCKED 1" in page and "pallet &lt;b&gt;" in page and "<b>BLOCKED</b>" not in page
    assert page.count("<tr>") == 3  # header + 2 exits


def test_map_cells_keep_the_height_band_only():
    xyz = np.array([[0.0, 0.0, -0.44], [0.05, 0.0, 0.5], [0.06, 0.01, 0.6], [3.0, 3.0, 3.0]])
    cells = rp.map_cells(xyz, -0.44)
    assert cells.shape == (1, 2) and cells[0] == pytest.approx([0.0, 0.0])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
