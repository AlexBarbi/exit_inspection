#!/usr/bin/env python3
"""
walk_waypoints.py (test tool)

Walks the simulated robot through waypoints (x y yaw, odom frame) by publishing /opti_pessi/goal (MarkerArray, one
ARROW marker whose orientation is the goal yaw; the acados relay forwards it to /goal_pose). A waypoint is reached
when /odom is within --tolerance m and --yaw-tolerance rad of it; the robot then stays --dwell s (sim time). A
waypoint not reached within --timeout s (sim time) is skipped. Prints one line per waypoint and exits.

Usage: walk_waypoints.py "0 -3 1.5708" "-2 -3 1.5708" ...
"""

import argparse
import math
import sys

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from visualization_msgs.msg import Marker, MarkerArray


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class Walker(Node):
    def __init__(self):
        super().__init__("walk_waypoints", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.odom = None
        self.create_subscription(Odometry, "/odom", lambda m: setattr(self, "odom", m), 10)
        self.pub = self.create_publisher(MarkerArray, "/opti_pessi/goal", 1)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def goal(self, x, y, yaw):
        m = Marker(type=Marker.ARROW, action=Marker.ADD, ns="goal")
        m.header.frame_id = self.odom.header.frame_id if self.odom else "odom"
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.position.x, m.pose.position.y = float(x), float(y)
        m.pose.position.z = self.odom.pose.pose.position.z if self.odom else 0.3
        m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        m.scale.x, m.scale.y, m.scale.z = 0.5, 0.1, 0.1
        m.color.g, m.color.a = 1.0, 1.0
        self.pub.publish(MarkerArray(markers=[m]))

    def spin_for(self, seconds):
        end = self.now() + seconds
        while rclpy.ok() and self.now() < end:
            rclpy.spin_once(self, timeout_sec=0.05)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("waypoints", nargs="+", help='"x y yaw" in the odom frame')
    ap.add_argument("--tolerance", type=float, default=0.3)
    ap.add_argument("--yaw-tolerance", type=float, default=0.25)
    ap.add_argument("--dwell", type=float, default=4.0)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()
    rclpy.init()
    w = Walker()
    while rclpy.ok() and (w.odom is None or w.now() == 0.0):
        rclpy.spin_once(w, timeout_sec=0.1)
    reached_all = True
    for i, text in enumerate(args.waypoints):
        x, y, yaw = map(float, text.split())
        start, last_pub, reached = w.now(), -1.0, False
        while rclpy.ok() and w.now() - start < args.timeout:
            if w.now() - last_pub > 1.0:
                w.goal(x, y, yaw)
                last_pub = w.now()
            rclpy.spin_once(w, timeout_sec=0.05)
            p = w.odom.pose.pose
            if math.hypot(p.position.x - x, p.position.y - y) < args.tolerance and \
                    abs(wrap(yaw_of(p.orientation) - yaw)) < args.yaw_tolerance:
                reached = True
                break
        p = w.odom.pose.pose
        print(f"waypoint {i} ({x:g}, {y:g}, {yaw:.2f}): {'reached' if reached else 'TIMEOUT'} after "
              f"{w.now() - start:.1f} s, robot at ({p.position.x:.2f}, {p.position.y:.2f}, "
              f"{yaw_of(p.orientation):.2f})", flush=True)
        reached_all &= reached
        w.spin_for(args.dwell)
    w.destroy_node()
    rclpy.shutdown()
    return 0 if reached_all else 1


if __name__ == "__main__":
    sys.exit(main())
