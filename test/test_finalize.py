"""Sightings CSV, SC-PGO pose files, re-anchoring on the optimized poses, exits.yaml."""

import math
import os
import subprocess
import sys

import numpy as np
import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from exit_inspection import finalize as fz  # noqa: E402
from exit_inspection import records  # noqa: E402
from exit_inspection.landmarks import MapperParams, Sighting  # noqa: E402

DOOR_CLASSES = ("exit_plain", "exit_arrow_down")


def yaw_pose(yaw_deg, t):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    T[:3, 3] = t
    return T


def apply(T, p):
    return T[:3, :3] @ p + T[:3, 3]


def sighting(stamp, hit, normal=(-1.0, 0.0, 0.0), robot=(0.0, 0.0, 0.35), cls="exit_plain", floor_z=0.0):
    return Sighting(stamp=stamp, cls=cls, conf=0.75, hit=np.array(hit, float), normal=np.array(normal, float),
                    robot=np.array(robot, float), range_m=5.0, width_m=0.4, floor_z=floor_z, low_quality=False)


def write_kitti(path, poses):
    with open(path, "w") as f:
        for T in poses:
            f.write(" ".join(f"{v:.9g}" for v in T[:3, :].ravel()) + "\n")


def test_sightings_csv_round_trip(tmp_path):
    path = tmp_path / "exit_inspection" / "sightings.csv"
    w = records.SightingWriter(str(path))
    a = sighting(1.5, [5.0, 0.1, 2.2])
    b = sighting(2.5, [5.0, -0.1, 2.3], cls="exit_arrow_side")
    b.low_quality = True
    w.write(a)
    w.close()
    w = records.SightingWriter(str(path))  # appends, header once
    w.write(b)
    w.close()
    got = records.read_sightings(str(path))
    assert len(got) == 2
    assert got[0].cls == "exit_plain" and got[1].cls == "exit_arrow_side"
    np.testing.assert_allclose(got[1].hit, b.hit)
    assert got[1].low_quality and not got[0].low_quality
    assert got[0].stamp == 1.5 and abs(got[0].width_m - 0.4) < 1e-9


def test_kitti_round_trip(tmp_path):
    poses = [yaw_pose(10.0 * i, [i, 2.0 * i, 0.1]) for i in range(3)]
    write_kitti(tmp_path / "p.txt", poses)
    got = records.load_kitti_poses(str(tmp_path / "p.txt"))
    assert got.shape == (3, 4, 4)
    np.testing.assert_allclose(got, np.array(poses), atol=1e-8)


def drifted_setup(tmp_path):
    """Three keyframes; the odometry frame drifts by D_k from the optimized one. Sightings logged in odometry."""
    times = np.array([0.0, 10.0, 20.0])
    opt = [yaw_pose(3.0 * k, [2.0 * k, 0.0, 0.0]) for k in range(3)]
    drift = [yaw_pose(5.0 * k, [0.4 * k, -0.2 * k, 0.0]) for k in range(3)]
    odom = [drift[k] @ opt[k] for k in range(3)]
    truth = [np.array([5.0, 1.0, 2.2]), np.array([8.0, -2.0, 2.3]), np.array([9.0, 3.0, 2.1])]
    stamps = [1.0, 11.0, 19.0]  # nearest keyframes 0, 1, 2
    sightings = [sighting(st, apply(drift[k], p), normal=drift[k][:3, :3] @ np.array([-1.0, 0.0, 0.0]),
                          robot=apply(drift[k], np.array([0.0, 0.0, 0.35])), floor_z=apply(drift[k], p)[2] - p[2])
                 for k, (st, p) in enumerate(zip(stamps, truth))]
    np.savetxt(tmp_path / "times.txt", times)
    write_kitti(tmp_path / "odom_poses.txt", odom)
    write_kitti(tmp_path / "optimized_poses.txt", opt)
    return sightings, truth


def test_reanchor_recovers_injected_drift(tmp_path):
    sightings, truth = drifted_setup(tmp_path)
    out = fz.reanchor(sightings, records.load_times(str(tmp_path / "times.txt")),
                      records.load_kitti_poses(str(tmp_path / "odom_poses.txt")),
                      records.load_kitti_poses(str(tmp_path / "optimized_poses.txt")))
    for s, p in zip(out, truth):
        np.testing.assert_allclose(s.hit, p, atol=0.01)
        np.testing.assert_allclose(s.normal, [-1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(s.robot, [0.0, 0.0, 0.35], atol=0.01)
        assert abs(s.floor_z) < 0.01


def write_sightings(save_dir, sightings):
    w = records.SightingWriter(os.path.join(save_dir, "exit_inspection", "sightings.csv"))
    for s in sightings:
        w.write(s)
    w.close()


def test_finalize_writes_exits_yaml_anchored_on_optimized_poses(tmp_path):
    sightings, truth = drifted_setup(tmp_path)
    write_sightings(str(tmp_path), sightings)
    out = fz.finalize(str(tmp_path), MapperParams(), DOOR_CLASSES)
    with open(tmp_path / "exit_inspection" / "exits.yaml") as f:
        data = yaml.safe_load(f)
    assert data == out
    assert data["anchored"] == "optimized"
    assert len(data["exits"]) == 3
    e = data["exits"][0]
    for key in ("id", "type", "cls", "confirmed", "sign_center", "normal", "sign_width", "door_floor_center",
                "door_width", "door_height", "n_sightings", "mean_conf", "low_quality_fraction"):
        assert key in e
    np.testing.assert_allclose(e["sign_center"], truth[0], atol=0.01)
    assert e["type"] == "door" and e["n_sightings"] == 1 and e["confirmed"] is False
    np.testing.assert_allclose(e["door_floor_center"], [5.0, 1.0, 0.0], atol=0.01)


def test_finalize_without_pose_files_keeps_odometry(tmp_path):
    write_sightings(str(tmp_path), [sighting(1.0, [5.0, 0.0, 2.2])])
    out = fz.finalize(str(tmp_path), MapperParams(), DOOR_CLASSES)
    assert out["anchored"] == "odometry"
    np.testing.assert_allclose(out["exits"][0]["sign_center"], [5.0, 0.0, 2.2], atol=1e-6)


def test_finalize_with_mismatched_pose_files_keeps_odometry(tmp_path):
    sightings, _ = drifted_setup(tmp_path)
    np.savetxt(tmp_path / "times.txt", np.array([0.0, 10.0]))  # 2 times, 3 poses
    write_sightings(str(tmp_path), sightings)
    assert fz.finalize(str(tmp_path), MapperParams(), DOOR_CLASSES)["anchored"] == "odometry"


def test_finalize_without_sightings_writes_empty_list(tmp_path):
    out = fz.finalize(str(tmp_path), MapperParams(), DOOR_CLASSES)
    assert out["exits"] == []
    assert (tmp_path / "exit_inspection" / "exits.yaml").exists()


def test_finalize_script(tmp_path):
    write_sightings(str(tmp_path), [sighting(1.0 + i, [5.0, 0.0, 2.2], robot=(0.3 * i, 0.0, 0.35))
                                    for i in range(5)])
    env = dict(os.environ, PYTHONPATH=os.path.join(HERE, "..") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, os.path.join(HERE, "..", "scripts", "finalize_exits.py"), str(tmp_path)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "1 door exits confirmed" in r.stdout
    with open(tmp_path / "exit_inspection" / "exits.yaml") as f:
        assert yaml.safe_load(f)["exits"][0]["confirmed"] is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
