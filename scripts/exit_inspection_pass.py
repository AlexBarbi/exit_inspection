#!/usr/bin/env python3
"""
exit_inspection_pass.py (ROS 2 Python)

Exit inspection mission (sub-project 3 of the emergency exit inspection,
docs/superpowers/specs/2026-11-11-exit-inspection-pass-design.md): visits a viewpoint in front of every confirmed door
exit of exits.yaml, faces the door, takes the zone verdict of exit_zone_checker.py once the checker's window holds only
frames of the stop, and writes a report.

  0. waits for /start_exit_inspection (std_msgs/String: exits.yaml, empty = exits_file), from the "Start exit
     inspection" button of map_point_picker or by hand:
       ros2 topic pub -1 /start_exit_inspection std_msgs/msg/String "{data: ''}"
     auto_start: at once. A start command during a pass starts over;
  1. viewpoints (exit_inspection.viewpoints): candidates in front of each door, checked against map_file
     (optimized_map.pcd next to the exit_inspection folder by default) when it exists; nearest exit first;
  2. per exit, per attempt (up to max_attempts candidates):
       planner far:    /goal_point (far_frame) + /goal_yaw_hint (the bridge tare_path_to_opti_pessi_goal.py turns the
                       robot to the door there); reached on /far_reach_goal_status or within reach_tolerance; failed
                       when far_planner stops publishing /way_point for idle_timeout s or after goal_timeout s
       planner direct: /opti_pessi/goal ARROW in odom (open rooms, tests), republished every second through the
                       robot's pose in odom (/odom) and in map_frame (TF); failed after goal_timeout s
     then, within reach_tolerance and yaw_tolerance of the viewpoint, the evaluations of this exit on
     /exit_inspection/zone_verdicts whose newest frame is at least window + window_margin s after the robot got
     there, for observe_time s and at least min_evaluations with camera data ("camera did not see the zone" is a gap
     in the data, not a vote): the most frequent verdict (ties: the worse). None within verdict_timeout s:
     UNVERIFIED. An UNVERIFIED or unreached attempt moves on to the next candidate;
  3. back to the start pose (return_home), report: <exits dir>/reports/<stamp>/report.yaml + report.html and
     <exits dir>/exit_report.yaml (exit_inspection.report).

  out: /exit_inspection/pass_status   std_msgs/String (transient local): YAML, state and results so far
       /exit_inspection/pass_markers  visualization_msgs/MarkerArray: viewpoints colored by result

Frames: exits.yaml and the map are in the exploration's camera_init; the mission's FAST-LIO must start from the same
pose (as for far_planner's saved graph). map_frame is the live camera_init.

Usage: ros2 launch exit_inspection exit_mission_launch.xml   (or ros2 run exit_inspection exit_inspection_pass.py)
"""

import datetime
import math
import os
from typing import List, Optional

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from exit_inspection import report as rp
from exit_inspection import viewpoints as vpm
from exit_inspection.pcd import read_pcd_xyz
from exit_inspection.pgo_dir import PgoSaveDirectory

STATUS_COLORS = {"pending": (0.8, 0.8, 0.8, 0.9), "active": (1.0, 0.3, 0.85, 1.0), "CLEAR": (0.1, 0.6, 0.3, 1.0),
                 "WARNING": (0.9, 0.63, 0.0, 1.0), "BLOCKED": (0.84, 0.19, 0.12, 1.0),
                 "UNVERIFIED": (0.55, 0.55, 0.55, 1.0)}


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def compose(a, b):
    """Planar poses (x, y, yaw): a * b."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2]))


def inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-(c * a[0] + s * a[1]), s * a[0] - c * a[1], -a[2])


class ExitInspectionPass(Node):
    def __init__(self):
        super().__init__("exit_inspection_pass")
        param = lambda name, default: self.declare_parameter(
            name, default, ParameterDescriptor(dynamic_typing=True)).value
        self.exits_file = os.path.expanduser(param("exits_file", ""))
        self.map_file = os.path.expanduser(param("map_file", ""))
        self.pgo_node = param("pgo_node", "alaserPGO")
        self.planner = param("planner", "far")
        self.auto_start = bool(param("auto_start", False))
        self.map_frame = param("map_frame", "camera_init")
        self.far_frame = param("far_frame", "map")
        self.lio_base_frame = param("lio_base_frame", "lio_base")
        self.reach_tolerance = float(param("reach_tolerance", 0.35))
        self.yaw_tolerance = float(param("yaw_tolerance", 0.25))
        self.turn_timeout = float(param("turn_timeout", 20.0))
        self.window = float(param("window", 3.0))
        self.window_margin = float(param("window_margin", 0.5))
        self.observe_time = float(param("observe_time", 3.0))
        self.min_evaluations = int(param("min_evaluations", 3))
        self.verdict_timeout = float(param("verdict_timeout", 20.0))
        self.max_attempts = int(param("max_attempts", 2))
        self.goal_timeout = float(param("goal_timeout", 180.0))
        self.idle_timeout = float(param("idle_timeout", 10.0))
        self.ack_timeout = float(param("ack_timeout", 5.0))
        self.return_home = bool(param("return_home", True))
        vp_defaults = vpm.ViewpointParams()
        self.vp_params = vpm.ViewpointParams(**{
            k: (list(param(k, list(v))) if isinstance(v, tuple) else type(v)(param(k, v)))
            for k, v in vars(vp_defaults).items()})
        self.zone_margin = float(param("zone_margin", 0.5))

        self.state = "idle"
        self.odom: Optional[Odometry] = None
        self.lidar_z = 0.0
        self.last_waypoint = None
        self.far_reached = False
        self.evaluations: dict = {}   # exit id -> {stamp: entry}
        self.plans: List[vpm.ExitPlan] = []
        self.order: List[int] = []
        self.results: List[dict] = []
        self.map_xyz = None
        self.floor_z = 0.0
        self.pgo_dir = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.create_subscription(Odometry, "/odom", lambda m: setattr(self, "odom", m), 5)
        self.create_subscription(Odometry, "/Odometry", self.on_lio, 5)
        self.create_subscription(PointStamped, "/way_point", self.on_waypoint, 5)
        self.create_subscription(Bool, "/far_reach_goal_status", self.on_reach, 5)
        self.create_subscription(String, "/exit_inspection/zone_verdicts", self.on_verdicts, 20)
        self.create_subscription(String, "/start_exit_inspection", self.on_start, 5)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, "/exit_inspection/pass_status", latched)
        self.marker_pub = self.create_publisher(MarkerArray, "/exit_inspection/pass_markers", latched)
        self.far_goal_pub = self.create_publisher(PointStamped, "/goal_point", 5)
        self.hint_pub = self.create_publisher(PoseStamped, "/goal_yaw_hint", 5)
        self.direct_goal_pub = self.create_publisher(MarkerArray, "/opti_pessi/goal", 1)
        self.create_timer(0.2, self.tick)
        self.pending_start = "" if self.auto_start else None
        self.publish_status()
        self.get_logger().info(f"planner {self.planner}; " + ("auto start" if self.auto_start else
                               "waiting for /start_exit_inspection (Start exit inspection in map_point_picker)"))

    # ------------------------------------------------------------------ inputs
    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_lio(self, msg: Odometry):
        self.lidar_z = msg.pose.pose.position.z

    def on_waypoint(self, _msg):
        self.last_waypoint = self.now()

    def on_reach(self, msg: Bool):
        if msg.data and self.state == "going":
            self.far_reached = True

    def on_verdicts(self, msg: String):
        try:
            doc = yaml.safe_load(msg.data) or {}
        except yaml.YAMLError:
            return
        for e in doc.get("exits") or []:
            self.evaluations.setdefault(int(e["id"]), {})[float(e.get("stamp", 0.0))] = e

    def on_start(self, msg: String):
        self.pending_start = msg.data.strip()

    def robot_pose(self):
        """(x, y, yaw) of lio_base in map_frame, or None."""
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.lio_base_frame, Time())
        except TransformException:
            return None
        t = tf.transform.translation
        return (t.x, t.y, yaw_of(tf.transform.rotation))

    # ------------------------------------------------------------------ mission set-up
    def resolve_exits_file(self, requested: str) -> Optional[str]:
        if requested:
            return os.path.abspath(os.path.expanduser(requested))
        if self.exits_file:
            return os.path.abspath(self.exits_file)
        if self.pgo_dir is None:
            self.pgo_dir = PgoSaveDirectory(self, self.pgo_node)
        directory = self.pgo_dir.poll()
        return os.path.join(directory, "exit_inspection", "exits.yaml") if directory else None

    def start(self, path: str) -> bool:
        try:
            with open(path) as f:
                doc = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            self.get_logger().error(f"cannot read {path}: {e}: no inspection")
            return True  # consumed
        doors = [e for e in doc.get("exits") or [] if e.get("type") == "door" and e.get("confirmed")]
        if not doors:
            self.get_logger().warning(f"{path} has no confirmed door exit: no inspection")
            return True
        pose = self.robot_pose()
        if pose is None:
            return False  # retried on the next tick
        self.exits_path = path
        map_file = self.map_file or os.path.join(os.path.dirname(os.path.dirname(path)), "optimized_map.pcd")
        self.floor_z = float(np.median([e["door_floor_center"][2] for e in doors]))
        occupancy, self.map_xyz, self.map_used = None, None, None
        if os.path.isfile(map_file):
            try:
                self.map_xyz = read_pcd_xyz(map_file)
                occupancy = vpm.Occupancy.from_points(self.map_xyz, self.floor_z, self.vp_params)
                self.map_used = map_file
            except (OSError, ValueError) as e:
                self.get_logger().warning(f"map {map_file} not read ({e}): viewpoints not checked")
        else:
            self.get_logger().warning(f"no map {map_file}: viewpoints not checked against obstacles")
        self.plans = vpm.plan(doors, self.vp_params, occupancy)
        self.order = vpm.tour(pose[:2], self.plans)
        self.results = []
        self.home = pose
        self.path = [[round(pose[0], 2), round(pose[1], 2)]]
        self.started_wall = datetime.datetime.now()
        self.started = self.now()
        self.k = -1  # index into order
        for ep in self.plans:
            if not ep.viewpoints:
                self.get_logger().warning(f"exit {ep.exit['id']}: no viewpoint ({ep.rejected})")
        self.get_logger().info(f"inspecting {len(doors)} door exits of {path} in the order "
                               f"{[self.plans[i].exit['id'] for i in self.order]}"
                               + (f", map {self.map_used}" if self.map_used else ""))
        self.next_exit()
        return True

    def next_exit(self):
        self.k += 1
        if self.k >= len(self.order):
            if self.return_home:
                self.get_logger().info("all exits done: going back to the start")
                self.begin_goal(vpm.Viewpoint(self.home[0], self.home[1], self.home[2], 0.0, 0.0), home=True)
            else:
                self.finish()
            return
        self.plan_i = self.order[self.k]
        self.attempts: List[dict] = []
        self.exit_started = self.now()
        self.next_attempt()

    def next_attempt(self):
        ep = self.plans[self.plan_i]
        if len(self.attempts) >= min(self.max_attempts, len(ep.viewpoints)):
            self.close_exit()
            return
        vp = ep.viewpoints[len(self.attempts)]
        self.attempts.append({"viewpoint": vp.as_dict(), "reached": False, "verdict": None, "reason": ""})
        self.get_logger().info(f"exit {ep.exit['id']} attempt {len(self.attempts)}: viewpoint ({vp.x:.2f}, "
                               f"{vp.y:.2f}) facing {math.degrees(vp.yaw):.0f} deg, {vp.standoff} m out, "
                               f"{vp.lateral} m aside")
        self.begin_goal(vp)

    def close_exit(self):
        ep = self.plans[self.plan_i]
        entry = rp.exit_entry(ep.exit, self.attempts, self.now() - self.exit_started)
        self.results.append(entry)
        self.get_logger().info(f"exit {ep.exit['id']}: {entry['verdict']} ({entry['reason']})")
        self.clear_hint()
        self.publish_status()
        self.next_exit()

    # ------------------------------------------------------------------ driving
    def begin_goal(self, vp: vpm.Viewpoint, home: bool = False):
        self.goal, self.home_leg = vp, home
        self.state = "going"
        self.goal_started = self.now()
        self.far_reached = False
        self.last_waypoint = None
        self.at_goal_since = None
        self.sent_at = -1e9
        self.send_goal()
        self.publish_status()

    def send_goal(self):
        vp = self.goal
        stamp = self.get_clock().now().to_msg()
        if self.planner == "far":
            goal = PointStamped()
            goal.header.frame_id, goal.header.stamp = self.far_frame, stamp
            goal.point.x, goal.point.y, goal.point.z = vp.x, vp.y, self.lidar_z
            self.far_goal_pub.publish(goal)
            hint = PoseStamped()
            hint.header.frame_id, hint.header.stamp = self.far_frame, stamp
            hint.pose.position.x, hint.pose.position.y = vp.x, vp.y
            hint.pose.orientation.z, hint.pose.orientation.w = math.sin(vp.yaw / 2), math.cos(vp.yaw / 2)
            self.hint_pub.publish(hint)
        else:
            pose_map = self.robot_pose()
            if self.odom is None or pose_map is None:
                return
            o = self.odom.pose.pose
            pose_odom = (o.position.x, o.position.y, yaw_of(o.orientation))
            g = compose(compose(pose_odom, inverse(pose_map)), (vp.x, vp.y, vp.yaw))
            m = Marker(type=Marker.ARROW, action=Marker.ADD, ns="goal")
            m.header.frame_id, m.header.stamp = self.odom.header.frame_id or "odom", stamp
            m.pose.position.x, m.pose.position.y, m.pose.position.z = g[0], g[1], o.position.z
            m.pose.orientation.z, m.pose.orientation.w = math.sin(g[2] / 2), math.cos(g[2] / 2)
            m.scale.x, m.scale.y, m.scale.z = 0.5, 0.1, 0.1
            m.color.r, m.color.b, m.color.a = 1.0, 0.85, 1.0
            self.direct_goal_pub.publish(MarkerArray(markers=[m]))
        self.sent_at = self.now()

    def clear_hint(self):
        if self.planner == "far":
            self.hint_pub.publish(PoseStamped())

    def attempt_failed(self, why: str):
        self.get_logger().warning(f"viewpoint not reached: {why}")
        if self.home_leg:
            self.finish()
            return
        self.attempts[-1].update(reached=False, verdict=None, reason=f"viewpoint not reached: {why}")
        self.clear_hint()
        self.next_attempt()

    # ------------------------------------------------------------------ tick
    def tick(self):
        t = self.now()
        if t == 0.0:
            return
        if self.pending_start is not None:
            path = self.resolve_exits_file(self.pending_start)
            if path is not None and self.start(path):
                self.pending_start = None
            return
        if self.state in ("idle", "done"):
            return
        pose = self.robot_pose()
        if pose is not None and (not self.path or math.hypot(pose[0] - self.path[-1][0],
                                                             pose[1] - self.path[-1][1]) > 0.25):
            self.path.append([round(pose[0], 2), round(pose[1], 2)])
        if self.state == "going":
            self.tick_going(t, pose)
        elif self.state == "observing":
            self.tick_observing(t, pose)

    def at_viewpoint(self, pose) -> bool:
        vp = self.goal
        return (pose is not None and math.hypot(pose[0] - vp.x, pose[1] - vp.y) <= self.reach_tolerance
                and abs(wrap(pose[2] - vp.yaw)) <= self.yaw_tolerance)

    def tick_going(self, t, pose):
        vp = self.goal
        near = pose is not None and math.hypot(pose[0] - vp.x, pose[1] - vp.y) <= self.reach_tolerance
        if self.at_viewpoint(pose) or (self.far_reached and self.planner == "far" and near):
            if self.at_goal_since is None:
                self.at_goal_since = t
            if not self.at_viewpoint(pose) and t - self.at_goal_since < self.turn_timeout:
                return  # far_planner says reached: wait for the turn to the door
            if self.home_leg:
                self.finish()
                return
            self.attempts[-1]["reached"] = True
            self.attempts[-1]["turned"] = self.at_viewpoint(pose)
            self.state, self.observe_from, self.observe_until = "observing", t + self.window + self.window_margin, None
            self.get_logger().info(f"at the viewpoint: verdicts from t = {self.observe_from:.1f}")
            self.publish_status()
            return
        if near and self.planner == "direct":
            if self.at_goal_since is None:
                self.at_goal_since = t
            if t - self.at_goal_since > self.turn_timeout:
                self.get_logger().warning("turn to the viewpoint yaw not finished: taking the yaw reached")
                self.goal = vpm.Viewpoint(vp.x, vp.y, pose[2], vp.standoff, vp.lateral)
                return
        if t - self.goal_started > self.goal_timeout:
            self.attempt_failed(f"goal_timeout ({self.goal_timeout:.0f} s)")
            return
        if self.planner == "direct":
            if t - self.sent_at > 1.0:
                self.send_goal()
        else:
            if self.last_waypoint is None or self.last_waypoint < self.goal_started:
                if t - self.sent_at > self.ack_timeout:
                    self.send_goal()
            elif t - self.last_waypoint > self.idle_timeout and not near:
                self.attempt_failed(f"far_planner stopped planning (no /way_point for {self.idle_timeout:.0f} s)")

    def tick_observing(self, t, pose):
        exit_id = int(self.plans[self.plan_i].exit["id"])
        if self.planner == "direct" and t - self.sent_at > 1.0:
            self.send_goal()  # keep standing there
        fresh = [e for s, e in sorted(self.evaluations.get(exit_id, {}).items()) if s >= self.observe_from]
        with_data = [e for e in fresh if rp.has_data(e)]  # "camera did not see the zone" is a gap, not a vote
        if with_data and self.observe_until is None:
            self.observe_until = t + self.observe_time
        done = (self.observe_until is not None and t >= self.observe_until
                and len(with_data) >= self.min_evaluations)
        if not done and t - self.observe_from < self.verdict_timeout:
            return
        a = self.attempts[-1]
        chosen = rp.pick_verdict(fresh)
        if chosen is None:
            a.update(verdict="UNVERIFIED", reason="no verdict from exit_zone_checker at the viewpoint")
        else:
            a.update({k: chosen.get(k) for k in ("verdict", "reason", "coverage", "object_area", "corridor_width",
                                                 "corridor_width_optimistic", "people", "behind_person_area",
                                                 "frames", "votes")})
        a["evaluations"] = len(fresh)
        self.get_logger().info(f"exit {exit_id}: {a['verdict']} from {len(fresh)} evaluations ({a['reason']})")
        if a["verdict"] == "UNVERIFIED":
            self.clear_hint()
            self.next_attempt()
        else:
            self.close_exit()

    # ------------------------------------------------------------------ end
    def finish(self):
        self.clear_hint()
        stamp = self.started_wall.strftime("%Y%m%d-%H%M%S")
        mission = {"started": self.started_wall.isoformat(timespec="seconds"),
                   "finished": datetime.datetime.now().isoformat(timespec="seconds"),
                   "duration": round(self.now() - self.started, 1), "exits_file": self.exits_path,
                   "map_file": self.map_used, "planner": self.planner, "frame": self.map_frame,
                   "params": {"margin": self.zone_margin, "depth": self.vp_params.zone_depth,
                              "max_attempts": self.max_attempts, "observe_time": self.observe_time},
                   "home": [round(v, 3) for v in self.home], "path": self.path,
                   "exits": sorted(self.results, key=lambda e: e["id"])}
        try:
            paths = rp.write_report(os.path.dirname(self.exits_path), stamp, mission, self.map_xyz, self.floor_z)
            self.report_paths = paths
            self.get_logger().info(f"report {paths['html']} ({rp.summary(mission['exits'])})")
        except OSError as e:
            self.report_paths = None
            self.get_logger().error(f"report not written: {e}")
        self.state = "done"
        self.publish_status()

    def publish_status(self):
        doc = {"state": self.state, "planner": self.planner,
               "exits_file": getattr(self, "exits_path", None),
               "current_exit": (self.plans[self.plan_i].exit["id"] if self.state in ("going", "observing")
                                and getattr(self, "plan_i", None) is not None and not getattr(self, "home_leg",
                                                                                               False) else None),
               "results": [{k: e[k] for k in ("id", "verdict", "reason")} for e in self.results],
               "viewpoints": [{"id": ep.exit["id"], **(ep.viewpoints[0].as_dict() if ep.viewpoints else {})}
                              for ep in self.plans],
               "report": (getattr(self, "report_paths", None) or {}).get("html")}
        self.status_pub.publish(String(data=yaml.safe_dump(doc, sort_keys=False, default_flow_style=None)))
        self.publish_markers()

    def publish_markers(self):
        done = {e["id"]: e["verdict"] for e in self.results}
        current = None
        if self.state in ("going", "observing") and not getattr(self, "home_leg", False):
            current = self.plans[self.plan_i].exit["id"]
        out = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        stamp = self.get_clock().now().to_msg()
        for i, ep in enumerate(self.plans):
            if not ep.viewpoints:
                continue
            vp = ep.viewpoints[0]
            if ep.exit["id"] == current and self.attempts:
                a = self.attempts[-1]["viewpoint"]
                vp = vpm.Viewpoint(a["x"], a["y"], a["yaw"], a["standoff"], a["lateral"])
            status = done.get(ep.exit["id"], "active" if ep.exit["id"] == current else "pending")
            m = Marker(type=Marker.ARROW, action=Marker.ADD, ns="viewpoint", id=i)
            m.header.frame_id, m.header.stamp = self.map_frame, stamp
            m.pose.position.x, m.pose.position.y, m.pose.position.z = vp.x, vp.y, self.floor_z + 0.1
            m.pose.orientation.z, m.pose.orientation.w = math.sin(vp.yaw / 2), math.cos(vp.yaw / 2)
            m.scale.x, m.scale.y, m.scale.z = 0.6, 0.15, 0.15
            m.color.r, m.color.g, m.color.b, m.color.a = STATUS_COLORS[status]
            out.markers.append(m)
        self.marker_pub.publish(out)


def main():
    rclpy.init()
    node = ExitInspectionPass()
    executor = MultiThreadedExecutor(num_threads=3)
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
