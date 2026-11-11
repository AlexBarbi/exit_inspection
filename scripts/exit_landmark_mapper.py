#!/usr/bin/env python3
"""
exit_landmark_mapper.py (ROS 2 Python)

Exit sign landmarks during an exploration. Every box of the exit sign detector (Yolo26RGBD instance
exit_sign_detector, ~/detections_2d) becomes a sighting: the ray through the box center, cast from the camera into the
lidar cloud of the last few seconds, hits the wall; a vertical plane fitted around the hit gives the sign center and the
wall normal (exit_inspection.landmarks.make_sighting). Sightings of one sign merge into a landmark; door landmarks seen
often enough from far enough apart are confirmed.

  in:  detections_topic   vision_msgs/Detection2DArray   (default /exit_sign_detector/detections_2d)
       camera_info_topic  sensor_msgs/CameraInfo          (default /camera/color/camera_info)
       cloud_topic        sensor_msgs/PointCloud2         (default /registered_scan_static, map_frame)
       TF  map_frame <- lio_base_frame at the image stamp (FAST-LIO: camera_init -> rslidar_lio -> lio_base) and
           base_frame <- camera frame (URDF, static). lio_base is FAST-LIO's base, gravity aligned like base; the
           odom -> base TF is not used (it comes from a wall-clock node).
  out: /exit_inspection/exit_markers  visualization_msgs/MarkerArray (transient local): confirmed door exits (green
           sign, normal arrow, door outline), unconfirmed ones (grey), route signs (blue)
       <save_directory>/exit_inspection/sightings.csv  every sighting, appended and flushed. An empty
           save_directory (the default) is SC-PGO's (parameter save_directory of pgo_node, a relative one taken from
           the working directory of that process), the folder tare_finish_merge_map.py runs finalize_exits.py on;
           sightings wait in memory until it is known.

The sightings are in camera_init as FAST-LIO has it live; finalize_exits.py re-anchors them on SC-PGO's optimized poses
at the end of the exploration and writes exits.yaml. This node only reads: if it stops, the exploration goes on.

Usage: ros2 run exit_inspection exit_landmark_mapper.py --ros-args -p save_directory:=/path/save_data
"""

import os
import threading
from typing import List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rcl_interfaces.msg import ParameterType
from rclpy.duration import Duration
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter_client import AsyncParameterClient
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray

from exit_inspection import geometry as g
from exit_inspection.cloud import RollingCloud, voxel_downsample
from exit_inspection.landmarks import LandmarkSet, MapperParams, Sighting, make_sighting
from exit_inspection.records import SightingWriter

def node_pids(name: str) -> List[int]:
    """PIDs of the processes running ROS node `name` (-r __node:=name, or an executable called name)."""
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                args = f.read().decode(errors="replace").split("\0")
        except OSError:
            continue
        if f"__node:={name}" in args or name in {os.path.basename(a) for a in args[:2]}:
            pids.append(int(entry))
    return pids


COLORS = {"confirmed": (0.0, 0.8, 0.3, 1.0), "unconfirmed": (0.6, 0.6, 0.6, 0.8), "route": (0.2, 0.5, 1.0, 0.9)}


def transform_matrix(tf) -> np.ndarray:
    t, q = tf.transform.translation, tf.transform.rotation
    return g.pose_matrix([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])


class ExitLandmarkMapper(Node):
    def __init__(self):
        super().__init__("exit_landmark_mapper")
        p = MapperParams()
        param = lambda name, default: self.declare_parameter(name, default).value
        self.params = MapperParams(**{k: type(v)(param(k, v)) for k, v in vars(p).items()})
        self.door_classes = list(param("door_classes", ["exit_plain", "exit_arrow_down"]))
        self.min_conf = param("min_conf", 0.4)
        self.map_frame = param("map_frame", "camera_init")
        self.lio_base_frame = param("lio_base_frame", "lio_base")
        self.base_frame = param("base_frame", "base")
        self.cloud_window = param("cloud_window", 5.0)  # s of scans kept
        self.cloud_voxel = param("cloud_voxel", 0.05)
        self.cloud_radius = param("cloud_radius", 15.0)
        self.tf_timeout = param("tf_timeout", 0.2)
        save_directory = os.path.expanduser(param("save_directory", ""))
        self.pgo_node = param("pgo_node", "alaserPGO")
        detections_topic = param("detections_topic", "/exit_sign_detector/detections_2d")
        camera_info_topic = param("camera_info_topic", "/camera/color/camera_info")
        cloud_topic = param("cloud_topic", "/registered_scan_static")

        self.lock = threading.Lock()
        self.K = None
        self.cloud = RollingCloud(self.cloud_window, self.cloud_voxel)  # map_frame
        self.landmarks = LandmarkSet(self.params, self.door_classes)
        self.rng = np.random.default_rng(0)
        self.writer: Optional[SightingWriter] = None
        self.pending: List[Sighting] = []  # until the save directory is known
        self.rejects = {}
        self.pgo_params = None
        self.pgo_future = None
        if save_directory:
            self.open_writer(os.path.abspath(save_directory))
        else:
            self.pgo_params = AsyncParameterClient(self, self.pgo_node)
            self.create_timer(2.0, self.resolve_save_directory)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        # Own groups: scans keep merging while a detection message is cast (TF has its own reentrant group).
        self.create_subscription(CameraInfo, camera_info_topic, self.on_info, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, cloud_topic, self.on_cloud, 5,
                                 callback_group=MutuallyExclusiveCallbackGroup())
        self.create_subscription(Detection2DArray, detections_topic, self.on_detections, 2,
                                 callback_group=MutuallyExclusiveCallbackGroup())
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_pub = self.create_publisher(MarkerArray, "/exit_inspection/exit_markers", latched)
        self.create_timer(10.0, self.log_summary)
        self.get_logger().info(f"waiting for {detections_topic}, {cloud_topic}")

    def open_writer(self, directory: str):
        self.csv_path = os.path.join(directory, "exit_inspection", "sightings.csv")
        self.writer = SightingWriter(self.csv_path)
        for s in self.pending:
            self.writer.write(s)
        self.get_logger().info(f"sightings to {self.csv_path} ({len(self.pending)} waiting written)")
        self.pending = []

    def resolve_save_directory(self):
        """SC-PGO's save_directory, asked until it answers (then this timer does nothing)."""
        if self.writer is not None:
            return
        if self.pgo_future is None:
            if not self.pgo_params.services_are_ready():
                self.get_logger().warn(f"no /{self.pgo_node} yet: sightings kept in memory ({len(self.pending)})",
                                       throttle_duration_sec=30.0)
                return
            self.pgo_future = self.pgo_params.get_parameters(["save_directory"])
            return
        if not self.pgo_future.done():
            return
        result, self.pgo_future = self.pgo_future.result(), None
        if result is None or not result.values or result.values[0].type != ParameterType.PARAMETER_STRING:
            self.get_logger().error(f"/{self.pgo_node} has no save_directory: set this node's save_directory",
                                    throttle_duration_sec=30.0)
            return
        directory = result.values[0].string_value
        if not os.path.isabs(directory):
            pids = node_pids(self.pgo_node)
            directory = os.path.join(os.readlink(f"/proc/{pids[0]}/cwd") if pids else os.getcwd(), directory)
        self.open_writer(os.path.normpath(directory))

    def on_info(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)

    def on_cloud(self, msg: PointCloud2):
        if msg.header.frame_id != self.map_frame:
            self.get_logger().warn(f"cloud in '{msg.header.frame_id}', expected '{self.map_frame}': ignored",
                                   throttle_duration_sec=10.0)
            return
        pts = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True).astype(float)
        pts = voxel_downsample(pts, self.cloud_voxel)
        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        with self.lock:
            self.cloud.add(pts, stamp)

    def local_cloud(self, center: np.ndarray) -> np.ndarray:
        with self.lock:
            return self.cloud.points(center, self.cloud_radius)

    def camera_pose(self, frame_id: str, stamp) -> tuple:
        """(T_map_cam, robot xyz) at the stamp, or raises TransformException."""
        t_map_base = self.tf_buffer.lookup_transform(self.map_frame, self.lio_base_frame, stamp,
                                                     Duration(seconds=self.tf_timeout))
        t_base_cam = self.tf_buffer.lookup_transform(self.base_frame, frame_id, Time())
        T_map_base = transform_matrix(t_map_base)
        return T_map_base @ transform_matrix(t_base_cam), T_map_base[:3, 3].copy()

    def on_detections(self, msg: Detection2DArray):
        boxes = []
        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0].hypothesis
            if hyp.score < self.min_conf:
                continue
            b = det.bbox
            boxes.append((hyp.class_id, float(hyp.score), b.center.position.x, b.center.position.y, b.size_x, b.size_y))
        if not boxes:
            return
        if self.K is None:
            self.get_logger().warn("no camera_info yet: detections skipped", throttle_duration_sec=5.0)
            return
        stamp = Time.from_msg(msg.header.stamp)
        try:
            T_map_cam, robot = self.camera_pose(msg.header.frame_id, stamp)
        except TransformException as e:
            self.get_logger().warn(f"no camera pose at the image stamp: {e}", throttle_duration_sec=5.0)
            return
        cloud = self.local_cloud(T_map_cam[:3, 3])
        if len(cloud) == 0:
            self.get_logger().warn("no cloud yet: detections skipped", throttle_duration_sec=5.0)
            return
        changed = False
        for box in boxes:
            s, reason = make_sighting(box, self.K, T_map_cam, cloud, robot, stamp.nanoseconds * 1e-9, self.params,
                                      self.rng)
            if s is None:
                self.rejects[reason] = self.rejects.get(reason, 0) + 1
                continue
            if self.writer is not None:
                self.writer.write(s)
            else:
                self.pending.append(s)
            self.landmarks.add(s)
            changed = True
        if changed:
            self.publish_markers(msg.header.stamp)

    def publish_markers(self, stamp):
        out = MarkerArray()
        out.markers.append(Marker(action=Marker.DELETEALL))
        for l in self.landmarks.landmarks():
            kind = "route" if l.group == "route" else ("confirmed" if l.confirmed else "unconfirmed")
            color = COLORS[kind]
            c, n = l.center, l.normal
            along = np.array([-n[1], n[0], 0.0])  # along the wall

            def marker(mid, mtype, **kw):
                m = Marker(type=mtype, action=Marker.ADD, id=4 * l.id + mid, ns="exits")
                m.header.frame_id, m.header.stamp = self.map_frame, stamp
                m.pose.orientation.w = 1.0
                m.color.r, m.color.g, m.color.b, m.color.a = color
                for k, v in kw.items():
                    setattr(m, k, v)
                return m

            sign = marker(0, Marker.CUBE)
            sign.pose.position = Point(x=c[0], y=c[1], z=c[2])
            yaw = np.arctan2(n[1], n[0])
            sign.pose.orientation.z, sign.pose.orientation.w = np.sin(yaw / 2), np.cos(yaw / 2)
            sign.scale.x, sign.scale.y, sign.scale.z = 0.03, max(l.width, 0.1), max(l.width / 2, 0.05)
            arrow = marker(1, Marker.ARROW)
            arrow.points = [Point(x=c[0], y=c[1], z=c[2]), Point(x=c[0] + 0.6 * n[0], y=c[1] + 0.6 * n[1], z=c[2])]
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.04, 0.08, 0.1
            text = marker(2, Marker.TEXT_VIEW_FACING)
            text.pose.position = Point(x=c[0] + 0.3 * n[0], y=c[1] + 0.3 * n[1], z=c[2] + 0.4)
            text.scale.z = 0.25
            text.text = f"{l.id} {l.cls} ({len(l.sightings)})"
            out.markers += [sign, arrow, text]
            if l.group == "door":
                f = l.door_floor_center
                w, h = self.params.door_width / 2, self.params.door_height
                corners = [f - w * along, f - w * along + [0, 0, h], f + w * along + [0, 0, h], f + w * along]
                door = marker(3, Marker.LINE_STRIP)
                door.points = [Point(x=p[0], y=p[1], z=p[2]) for p in corners]
                door.scale.x = 0.04
                out.markers.append(door)
        self.marker_pub.publish(out)

    def log_summary(self):
        lms = self.landmarks.landmarks()
        doors = [l for l in lms if l.group == "door"]
        self.get_logger().info(
            f"{sum(l.confirmed for l in doors)} confirmed door exits, {len(doors)} door landmarks, "
            f"{len(lms) - len(doors)} route signs; rejected boxes: {self.rejects or 'none'}")

    def destroy_node(self):
        if self.writer is not None:
            self.writer.close()
        elif self.pending:
            self.get_logger().error(f"{len(self.pending)} sightings lost: SC-PGO's save_directory never known")
        super().destroy_node()


def main():
    rclpy.init()
    node = ExitLandmarkMapper()
    # TF on its own thread of the executor, so the lookup at the image stamp can wait for it.
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
