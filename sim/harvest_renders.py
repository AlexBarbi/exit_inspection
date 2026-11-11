#!/usr/bin/env python3
"""
harvest_renders.py

Labeled frames of the exit room for the exit sign detector: moves the free camera rig (launch/harvest_renders_launch.xml)
to random poses with `gz service .../set_pose`, waits for a color + depth pair rendered after the move, and writes
  <out>/images/NNNNN.jpg      1920x1080 color frame
  <out>/labels/NNNNN.txt      YOLO labels (class x_center y_center width height, normalized) of the visible signs
  <out>/meta.csv              one row per label: frame, sign, class, distance, view angle, box size
  <out>/clean_backgrounds.txt frames where no part of any sign is in the frame (training backgrounds)
Visibility: in front, at least 60 % of the box in the frame, not seen edge-on, not occluded (depth at the sign center),
see exit_inspection.sim_labels. Classes in exit_inspection.sign_art.CLASSES order.

Poses: camera 0.30-0.45 m high, pitched 0-10 deg down; with probability --aim it looks at a random sign from 2.5-13 m
within 60 deg of its normal, otherwise anywhere. Frames without signs serve as training backgrounds.

Usage (system python, ROS sourced, harvest_renders_launch.xml running):
  python3 sim/harvest_renders.py /path/out -n 400 --seed 1
"""

import argparse
import csv
import math
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "sim"))

from exit_inspection.sign_art import CLASSES  # noqa: E402
from exit_inspection.sim_labels import intrinsics, sign_label  # noqa: E402
import generate_exit_room_world as world  # noqa: E402

W, H = 1920, 1080
K = intrinsics(W, H, math.radians(69.4))


def keep_out():
    """(x, y, radius) of what the camera must not be inside: obstacles and the standing person."""
    person = next(c for c in world.CASES if c.name == "person")
    px, py = world.to_world(person.wall, person.pos + 0.2, 0.9)
    return [(5.3, world.ROOM_Y / 2 - 0.8, 0.8), (-4.0, -world.ROOM_Y / 2 + 0.7, 1.0), (px, py, 0.6)]


def quat(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2),
                              math.cos(yaw / 2), math.sin(yaw / 2))
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def set_pose(world_name, model, pose):
    x, y, z, r, p, yw = pose
    qx, qy, qz, qw = quat(r, p, yw)
    req = (f'name: "{model}", position: {{x: {x}, y: {y}, z: {z}}}, '
           f'orientation: {{x: {qx}, y: {qy}, z: {qz}, w: {qw}}}')
    out = subprocess.run(["gz", "service", "-s", f"/world/{world_name}/set_pose", "--reqtype", "gz.msgs.Pose",
                          "--reptype", "gz.msgs.Boolean", "--timeout", "2000", "--req", req],
                         capture_output=True, text=True)
    return "true" in out.stdout


def sample_pose(rng, exits, aim):
    hx, hy, margin = world.ROOM_X / 2, world.ROOM_Y / 2, 0.4
    for _ in range(1000):
        if rng.random() < aim:
            e = exits[rng.integers(len(exits))]
            n = np.array(e["normal"][:2])
            along = np.array([-n[1], n[0]])
            d, phi = rng.uniform(2.5, 13.0), math.radians(rng.uniform(-60, 60))
            xy = np.array(e["sign_center"][:2]) + d * (math.cos(phi) * n + math.sin(phi) * along)
            to_sign = np.array(e["sign_center"][:2]) - xy
            yaw = math.atan2(to_sign[1], to_sign[0]) + math.radians(rng.normal(0, 15))
        else:
            xy = np.array([rng.uniform(-hx, hx), rng.uniform(-hy, hy)])
            yaw = rng.uniform(-math.pi, math.pi)
        if abs(xy[0]) > hx - margin or abs(xy[1]) > hy - margin:
            continue
        if any(math.hypot(xy[0] - ox, xy[1] - oy) < r for ox, oy, r in keep_out()):
            continue
        return (float(xy[0]), float(xy[1]), rng.uniform(0.30, 0.45), math.radians(rng.normal(0, 1.5)),
                math.radians(rng.uniform(0, 10)), yaw)
    raise RuntimeError("no free pose found")


class Grabber(Node):
    def __init__(self):
        super().__init__("harvest_renders")
        self.sim_time = 0.0
        self.color = self.depth = None
        self.create_subscription(Clock, "/clock", self.on_clock, 10)
        self.create_subscription(Image, "/rig/color", lambda m: setattr(self, "color", m), qos_profile_sensor_data)
        self.create_subscription(Image, "/rig/depth", lambda m: setattr(self, "depth", m), qos_profile_sensor_data)

    def on_clock(self, m):
        self.sim_time = m.clock.sec + m.clock.nanosec * 1e-9

    @staticmethod
    def stamp(m):
        return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9

    def pair_after(self, t, timeout=20.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.color is not None and self.depth is not None and self.stamp(self.color) >= t and \
                    self.stamp(self.depth) >= t:
                c, d = self.color, self.depth
                color = np.frombuffer(c.data, np.uint8).reshape(c.height, c.width, 3)
                depth = np.frombuffer(d.data, np.float32).reshape(d.height, d.width)
                return cv2.cvtColor(color, cv2.COLOR_RGB2BGR), depth
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("-n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--aim", type=float, default=0.7)
    ap.add_argument("--world", default="exit_room")
    ap.add_argument("--gt", default=os.path.join(PKG, "worlds", "exit_room_gt.yaml"))
    args = ap.parse_args()

    with open(args.gt) as f:
        exits = yaml.safe_load(f)["exits"]
    for e in exits:
        e.setdefault("sign_size", [0.4, 0.2])
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rclpy.init()
    node = Grabber()
    t0 = time.time()
    while node.sim_time == 0.0 and time.time() - t0 < 30:
        rclpy.spin_once(node, timeout_sec=0.1)
    meta = open(os.path.join(args.out, "meta.csv"), "w", newline="")
    w = csv.writer(meta)
    w.writerow(["frame", "sign", "cls", "distance", "view_angle_deg", "box_w", "box_h", "x", "y", "z", "roll",
                "pitch", "yaw"])
    clean = open(os.path.join(args.out, "clean_backgrounds.txt"), "w")
    n_labels = 0
    for i in range(args.n):
        pose = sample_pose(rng, exits, args.aim)
        if not set_pose(args.world, "rig", pose):
            print(f"frame {i}: set_pose failed", file=sys.stderr)
            continue
        pair = node.pair_after(node.sim_time + 0.25)
        if pair is None:
            print(f"frame {i}: no frame after the move", file=sys.stderr)
            continue
        color, depth = pair
        name = f"{i:05d}"
        cv2.imwrite(os.path.join(args.out, "images", name + ".jpg"), color, [cv2.IMWRITE_JPEG_QUALITY, 92])
        lines = []
        for e in exits:
            lab = sign_label(e, pose, K, W, H, depth)
            if lab is None:
                continue
            x1, y1, x2, y2 = lab["box"]
            lines.append(f"{CLASSES.index(lab['cls'])} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                         f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            w.writerow([name, lab["name"], lab["cls"], f"{lab['distance']:.2f}", f"{lab['view_angle_deg']:.1f}",
                        f"{x2 - x1:.1f}", f"{y2 - y1:.1f}", *(f"{v:.3f}" for v in pose)])
        with open(os.path.join(args.out, "labels", name + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        # any sliver of a sign, occluded or edge-on included, keeps the frame out of the backgrounds
        if not any(sign_label(e, pose, K, W, H, None, min_visible=1e-6, max_view_angle_deg=89.9) for e in exits):
            clean.write(os.path.join("images", name + ".jpg") + "\n")
            clean.flush()
        n_labels += len(lines)
        if i % 50 == 0:
            print(f"{i}/{args.n} frames, {n_labels} labels", flush=True)
    meta.close()
    clean.close()
    print(f"done: {args.n} frames, {n_labels} labels in {args.out}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
