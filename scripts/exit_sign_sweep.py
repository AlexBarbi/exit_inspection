#!/usr/bin/env python3
"""
exit_sign_sweep.py (ROS 2 Python)

Exit sign sweep at the end of a TARE exploration (docs/superpowers/specs/2026-11-12-exit-sign-sweep-design.md).
TARE stops when the lidar has seen everything; the camera sees a sign 2.2-2.5 m up only from ~6.8 m and only where the
robot faces, so some signs are glimpsed for a second or never. During the exploration this node builds a 2D grid
(exit_inspection.sweep: walls in the sign band, obstacles, free space crossed by the lidar rays) and counts the
camera's looks at each wall cell. When tare_finish_merge_map.py (sweep:=true) has stopped TARE, it sends
/exit_inspection/start_sweep; this node then
  1. plans viewpoints covering the wall cells not looked at (greedy, nearest first);
  2. starts its own goal bridge (tare_path_to_opti_pessi_goal.py following far_planner's route, far_planner's
     /way_point being remapped to /far_planner/way_point in the exploration) and drives the robot through them with
     /goal_point + /goal_yaw_hint: at each viewpoint two stops side_step apart (the landmark mapper confirms a sign
     seen from robot positions at least 0.25-0.5 m apart), dwell s each, facing the wall;
  3. walks the robot home (the start pose the finish node sends), stops the bridge and publishes
     /exit_inspection/sweep_done (std_msgs/String YAML, latched); the finish node then stops the robot and merges.

exit_landmark_mapper keeps recording sightings all along. A viewpoint far_planner cannot reach (no /far_planner/way_point
for idle_timeout s, or goal_timeout s) is skipped. Nothing to plan: straight home.

Usage: started by explore_warehouse_launch.xml exits:=true (exit_sweep:=true).
"""

import math
import os
import signal
import subprocess
import time
from typing import List, Optional

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, ColorRGBA, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from exit_inspection import geometry as g
from exit_inspection import sweep as sw


def transform_matrix(tf) -> np.ndarray:
    t, q = tf.transform.translation, tf.transform.rotation
    return g.pose_matrix([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class ExitSignSweep(Node):
    def __init__(self):
        super().__init__("exit_sign_sweep")
        param = lambda name, default: self.declare_parameter(
            name, default, ParameterDescriptor(dynamic_typing=True)).value
        defaults = sw.SweepParams()
        self.sp = sw.SweepParams(**{k: type(v)(param(k, v)) for k, v in vars(defaults).items()})
        self.map_frame = param("map_frame", "camera_init")
        self.lio_base_frame = param("lio_base_frame", "lio_base")
        self.base_frame = param("base_frame", "base")
        self.camera_frame = param("camera_frame", "camera_color_optical_frame")
        self.lidar_frame = param("lidar_frame", "rslidar_lio")
        self.far_frame = param("far_frame", "map")
        self.base_height = float(param("base_height", 0.35))
        self.reach_tolerance = float(param("reach_tolerance", 0.35))
        self.yaw_tolerance = float(param("yaw_tolerance", 0.3))
        self.turn_timeout = float(param("turn_timeout", 15.0))
        self.dwell = float(param("dwell", 2.0))
        self.side_step = float(param("side_step", 0.5))
        self.goal_timeout = float(param("goal_timeout", 120.0))
        self.idle_timeout = float(param("idle_timeout", 10.0))
        self.ack_timeout = float(param("ack_timeout", 5.0))
        self.start_bridge = bool(param("start_bridge", True))
        self.bridge_node = param("bridge_node", "exit_sweep_goal_bridge")
        cloud_topic = param("cloud_topic", "/registered_scan_static")
        period = float(param("process_period", 1.0))

        self.grid: Optional[sw.SweepGrid] = None
        self.cloud: Optional[PointCloud2] = None
        self.lidar_z = 0.0
        self.state = "collecting"
        self.stops: List[dict] = []
        self.k = -1
        self.bridge = None
        self.scans = 0
        self.rng = np.random.default_rng(0)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.create_subscription(PointCloud2, cloud_topic, lambda m: setattr(self, "cloud", m), 2)
        self.create_subscription(Odometry, "/Odometry", lambda m: setattr(self, "lidar_z", m.pose.pose.position.z), 5)
        self.create_subscription(PointStamped, "/far_planner/way_point", self.on_waypoint, 5)
        self.create_subscription(PointStamped, "/way_point", self.on_waypoint, 5)
        self.create_subscription(Bool, "/far_reach_goal_status", self.on_reach, 5)
        self.create_subscription(String, "/exit_inspection/start_sweep", self.on_start, 5)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.done_pub = self.create_publisher(String, "/exit_inspection/sweep_done", latched)
        self.marker_pub = self.create_publisher(MarkerArray, "/exit_inspection/sweep_markers", latched)
        self.goal_pub = self.create_publisher(PointStamped, "/goal_point", 5)
        self.hint_pub = self.create_publisher(PoseStamped, "/goal_yaw_hint", 5)
        self.create_timer(period, self.process)
        self.create_timer(0.2, self.tick)
        self.get_logger().info(f"collecting {cloud_topic} and the camera's looks; sweep on /exit_inspection/start_sweep")

    # ------------------------------------------------------------------ collecting
    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def pose(self, frame: str) -> Optional[np.ndarray]:
        try:
            return transform_matrix(self.tf_buffer.lookup_transform(self.map_frame, frame, Time()))
        except TransformException:
            return None

    def robot(self):
        T = self.pose(self.lio_base_frame)
        if T is None:
            return None
        return (T[0, 3], T[1, 3], math.atan2(T[1, 0], T[0, 0]))

    def process(self):
        msg, self.cloud = self.cloud, None
        if msg is None or msg.header.frame_id != self.map_frame:
            return
        T_base = self.pose(self.lio_base_frame)
        T_lidar = self.pose(self.lidar_frame)
        if T_base is None or T_lidar is None:
            return
        if self.grid is None:
            self.grid = sw.SweepGrid(T_base[:2, 3], self.sp)
        pts = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True).astype(float)
        floor_z = T_base[2, 3] - self.base_height
        self.floor_z = floor_z
        self.grid.add_scan(T_lidar[:2, 3], pts, floor_z, self.rng)
        self.scans += 1
        try:
            t_base_cam = transform_matrix(self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, Time()))
        except TransformException:
            return
        T_cam = T_base @ t_base_cam
        self.grid.mark_look(T_cam[:2, 3], math.atan2(T_cam[1, 2], T_cam[0, 2]))  # optical z = viewing direction

    # ------------------------------------------------------------------ sweep
    def on_waypoint(self, _msg):
        self.last_waypoint = self.now()

    def on_reach(self, msg: Bool):
        if msg.data and self.state == "sweeping":
            self.far_reached = True

    def on_start(self, msg: String):
        if self.state != "collecting":
            return
        try:
            doc = yaml.safe_load(msg.data) or {}
        except yaml.YAMLError:
            doc = {}
        here = self.robot()
        home = doc.get("home") or (list(here) if here else None)
        if self.grid is None or here is None:
            self.get_logger().error("no grid or no robot pose: no sweep, going home only")
            viewpoints, stats = [], {}
        else:
            t0 = time.monotonic()
            viewpoints, stats = self.grid.plan(here[:2])
            self.get_logger().info(f"sweep plan ({time.monotonic() - t0:.1f} s, {self.scans} scans): "
                                   f"{len(viewpoints)} viewpoints for {stats}")
        self.stats = stats
        self.stops = []
        for i, (x, y, yaw, n) in enumerate(viewpoints):
            side = np.array([-math.sin(yaw), math.cos(yaw)]) * self.side_step
            second = np.array([x, y]) + side
            if not self.standable(second):
                second = np.array([x, y]) - side
            self.stops.append({"x": x, "y": y, "yaw": yaw, "viewpoint": i, "cells": n})
            if self.standable(second):
                self.stops.append({"x": float(second[0]), "y": float(second[1]), "yaw": yaw, "viewpoint": i,
                                   "cells": n})
        if home is not None:
            hyaw = home[2] if len(home) > 2 else here[2] if here else 0.0
            self.stops.append({"x": float(home[0]), "y": float(home[1]), "yaw": float(hyaw), "viewpoint": "home"})
        self.results = []
        self.started = self.now()
        self.state = "sweeping"
        if self.start_bridge:
            self.launch_bridge()
        self.publish_markers()
        self.next_stop()

    def standable(self, xy) -> bool:
        ij, ok = self.grid.index(np.asarray(xy))
        if not ok[0]:
            return False
        r = int(math.ceil(self.sp.clearance / self.sp.cell))
        i, j = ij[0]
        win = (slice(max(0, i - r), i + r + 1), slice(max(0, j - r), j + r + 1))
        occupied = (self.grid.obstacle[win] >= self.sp.min_hits) | (self.grid.wall[win] >= self.sp.min_hits)
        return bool(self.grid.free[i, j] > 0 and not occupied.any())

    def launch_bridge(self):
        sim = "true" if self.get_parameter("use_sim_time").value else "false"
        command = ["ros2", "run", "tare_planner", "tare_path_to_opti_pessi_goal.py", "--ros-args",
                   "-r", f"__node:={self.bridge_node}", "-p", "route_topic:=/viz_path_topic",
                   "-r", "/way_point:=/far_planner/way_point", "-p", f"use_sim_time:={sim}"]
        self.get_logger().info(f"goal bridge: {' '.join(command)}")
        self.bridge = subprocess.Popen(command, start_new_session=True)

    def stop_bridge(self):
        if self.bridge is None or self.bridge.poll() is not None:
            return
        os.killpg(self.bridge.pid, signal.SIGINT)
        try:
            self.bridge.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            os.killpg(self.bridge.pid, signal.SIGKILL)

    def next_stop(self):
        self.k += 1
        if self.k >= len(self.stops):
            self.finish()
            return
        s = self.stops[self.k]
        self.goal_started = self.now()
        self.sent_at, self.at_since, self.dwell_until = -1e9, None, None
        self.last_waypoint, self.far_reached = None, False
        self.get_logger().info(f"sweep stop {self.k + 1}/{len(self.stops)} ({s['viewpoint']}): ({s['x']:.2f}, "
                               f"{s['y']:.2f}) facing {math.degrees(s['yaw']):.0f} deg")
        self.send_goal()

    def send_goal(self):
        s = self.stops[self.k]
        stamp = self.get_clock().now().to_msg()
        goal = PointStamped()
        goal.header.frame_id, goal.header.stamp = self.far_frame, stamp
        goal.point.x, goal.point.y, goal.point.z = s["x"], s["y"], self.lidar_z
        self.goal_pub.publish(goal)
        hint = PoseStamped()
        hint.header.frame_id, hint.header.stamp = self.far_frame, stamp
        hint.pose.position.x, hint.pose.position.y = s["x"], s["y"]
        hint.pose.orientation.z, hint.pose.orientation.w = math.sin(s["yaw"] / 2), math.cos(s["yaw"] / 2)
        self.hint_pub.publish(hint)
        self.sent_at = self.now()

    def tick(self):
        if self.state != "sweeping":
            return
        t = self.now()
        s = self.stops[self.k]
        here = self.robot()
        near = here is not None and math.hypot(here[0] - s["x"], here[1] - s["y"]) <= self.reach_tolerance
        facing = here is not None and abs(wrap(here[2] - s["yaw"])) <= self.yaw_tolerance
        if self.dwell_until is not None:
            if t >= self.dwell_until:
                self.results.append(dict(s, reached=True))
                self.next_stop()
            return
        if near and (facing or (self.at_since is not None and t - self.at_since > self.turn_timeout)):
            self.dwell_until = t + (self.dwell if s["viewpoint"] != "home" else 0.0)
            return
        if near and self.at_since is None:
            self.at_since = t
        if t - self.goal_started > self.goal_timeout:
            self.skip(f"goal_timeout ({self.goal_timeout:.0f} s)")
        elif self.last_waypoint is None or self.last_waypoint < self.goal_started:
            if t - self.sent_at > self.ack_timeout:
                self.send_goal()
        elif t - self.last_waypoint > self.idle_timeout and not near:
            self.skip(f"far_planner stopped planning (no way point for {self.idle_timeout:.0f} s)")

    def skip(self, why: str):
        s = self.stops[self.k]
        self.get_logger().warning(f"sweep stop {self.k + 1} not reached: {why}")
        self.results.append(dict(s, reached=False, reason=why))
        self.next_stop()

    def finish(self):
        self.hint_pub.publish(PoseStamped())
        self.stop_bridge()
        self.state = "done"
        reached = sum(1 for r in self.results if r.get("reached") and r["viewpoint"] != "home")
        doc = {"stats": self.stats, "stops": len(self.stops), "reached": reached,
               "duration": round(self.now() - self.started, 1), "results": self.results}
        self.done_pub.publish(String(data=yaml.safe_dump(doc, sort_keys=False, default_flow_style=None)))
        self.get_logger().info(f"sweep done: {reached} of {len(self.stops) - 1} stops reached in {doc['duration']} s")
        self.publish_markers()

    def publish_markers(self):
        out = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        stamp = self.get_clock().now().to_msg()
        z = getattr(self, "floor_z", 0.0) + 0.1
        for i, s in enumerate(self.stops):
            m = Marker(type=Marker.ARROW, action=Marker.ADD, ns="sweep", id=i)
            m.header.frame_id, m.header.stamp = self.map_frame, stamp
            m.pose.position.x, m.pose.position.y, m.pose.position.z = s["x"], s["y"], z
            m.pose.orientation.z, m.pose.orientation.w = math.sin(s["yaw"] / 2), math.cos(s["yaw"] / 2)
            m.scale.x, m.scale.y, m.scale.z = 0.7, 0.15, 0.15
            m.color = ColorRGBA(r=0.9, g=0.45, b=1.0, a=1.0)
            out.markers.append(m)
        if self.grid is not None:
            todo = np.argwhere(self.grid.walls() & (self.grid.looks < self.sp.covered_looks))
            cells = Marker(type=Marker.CUBE_LIST, action=Marker.ADD, ns="unseen_walls", id=0)
            cells.header.frame_id, cells.header.stamp = self.map_frame, stamp
            cells.pose.orientation.w = 1.0
            cells.scale.x = cells.scale.y = self.sp.cell
            cells.scale.z = 0.05
            cells.color = ColorRGBA(r=1.0, g=0.3, b=0.3, a=0.8)
            for c in self.grid.center(todo):
                cells.points.append(Point(x=float(c[0]), y=float(c[1]), z=z + 2.0))
            out.markers.append(cells)
        self.marker_pub.publish(out)

    def destroy_node(self):
        self.stop_bridge()
        super().destroy_node()


def main():
    rclpy.init()
    node = ExitSignSweep()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:  # noqa: BLE001 - SIGINT can invalidate the context inside spin
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
