#!/usr/bin/env python3
"""
exit_zone_checker.py (ROS 2 Python)

Verdict of the floor zone in front of every confirmed door exit of exits.yaml (sub-project 2 of the emergency exit
inspection, docs/superpowers/specs/2026-11-11-exit-zone-checker-design.md): CLEAR, WARNING, BLOCKED or UNVERIFIED,
people excluded (exit_inspection.zone).

  in:  exits_file          exits.yaml of finalize_exits.py (empty: <SC-PGO save_directory>/exit_inspection/exits.yaml,
                           asked to alaserPGO); polled every 2 s and reloaded when it changes
       depth_topic         sensor_msgs/Image 32FC1 [m] or 16UC1 [mm] (default /camera/depth/image_rect_raw, the raw
                           D435i depth: wider than the color image) + depth_info_topic
       cloud_topic         sensor_msgs/PointCloud2 in map_frame (default /registered_scan_static)
       persons_topic       vision_msgs/Detection3DArray (default /person_detector/detections: Yolo26RGBD, center on
                           the person's front surface in the color optical frame)
       TF  map_frame <- lio_base_frame at the stamp times base_frame <- sensor frame (URDF), as the landmark mapper;
           map_frame <- lidar_frame for the scans' origin
  out: /exit_inspection/zone_verdicts  std_msgs/String (transient local): YAML, latest verdict of every exit
       /exit_inspection/zone_markers   visualization_msgs/MarkerArray: zone outline colored by verdict, object cells
                                       (red), unseen cells (grey), people and what they hide (blue), text
       <exits.yaml folder>/zone_checks.yaml  the same YAML, rewritten at every evaluation
       <exits.yaml folder>/zone_debug/exit_<id>.npz  with debug_dump: the inputs of the latest evaluation, for
                                       test/replay_zone.py

Each exit keeps the frames (one per depth image or scan with points in its zone) of the last `window` s, taken while
its door is within max_range of the sensor, and is evaluated at `rate` Hz while it gets new frames. The node keeps the
latest verdict; choosing when it counts (robot stopped in front of the exit) is the inspection pass's job.

Usage: ros2 launch exit_inspection exit_zone_check_launch.xml exits_file:=/path/exit_inspection/exits.yaml
"""

import os
import threading
from collections import deque
from typing import Dict, Optional

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import ColorRGBA, String
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

from exit_inspection import geometry as g
from exit_inspection import zone as zn
from exit_inspection.pgo_dir import PgoSaveDirectory

VERDICT_COLORS = {"CLEAR": (0.0, 0.8, 0.2), "WARNING": (1.0, 0.75, 0.0), "BLOCKED": (0.9, 0.0, 0.0),
                  "UNVERIFIED": (0.55, 0.55, 0.55)}
CELL_COLORS = {zn.OBJECT: (0.9, 0.1, 0.1, 0.9), zn.HIDDEN: (0.6, 0.1, 0.1, 0.4), zn.UNKNOWN: (0.5, 0.5, 0.5, 0.35),
               zn.PERSON: (0.2, 0.4, 1.0, 0.6), zn.BEHIND: (0.2, 0.4, 1.0, 0.3)}


def transform_matrix(tf) -> np.ndarray:
    t, q = tf.transform.translation, tf.transform.rotation
    return g.pose_matrix([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])


def depth_image(msg: Image) -> Optional[np.ndarray]:
    """Depth in meters (NaN where invalid), or None for an unknown encoding."""
    if msg.encoding == "32FC1":
        dtype, scale = np.dtype(np.float32), 1.0
    elif msg.encoding in ("16UC1", "mono16"):
        dtype, scale = np.dtype(np.uint16), 0.001
    else:
        return None
    dtype = dtype.newbyteorder(">" if msg.is_bigendian else "<")
    raw = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.step // dtype.itemsize)[:, :msg.width]
    d = raw.astype(np.float32) * scale
    if dtype.kind == "u":
        d[raw == 0] = np.nan
    return d


def write_atomic(path: str, text: str):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


class ExitZoneChecker(Node):
    def __init__(self):
        super().__init__("exit_zone_checker")
        param = lambda name, default: self.declare_parameter(name, default).value
        defaults = zn.ZoneParams()
        self.zparams = zn.ZoneParams(**{k: type(v)(param(k, v)) for k, v in vars(defaults).items()})
        self.exits_file_param = os.path.expanduser(param("exits_file", ""))
        self.pgo_node = param("pgo_node", "alaserPGO")
        self.map_frame = param("map_frame", "camera_init")
        self.lio_base_frame = param("lio_base_frame", "lio_base")
        self.base_frame = param("base_frame", "base")
        self.lidar_frame = param("lidar_frame", "rslidar_lio")
        self.person_classes = set(param("person_classes", ["person"]))
        self.person_min_conf = param("person_min_conf", 0.4)
        self.person_offset = param("person_offset", 0.15)  # front surface -> person center, along the view
        self.depth_every_n = max(1, param("depth_every_n", 3))
        self.depth_stride = max(1, param("depth_stride", 4))
        self.depth_min = param("depth_min", 0.2)
        self.depth_max = param("depth_max", 6.0)
        self.window = param("window", 3.0)
        self.max_range = param("max_range", 7.5)
        self.min_frame_points = param("min_frame_points", 10)
        self.tf_timeout = param("tf_timeout", 0.2)
        use_depth = param("use_depth", True)
        use_lidar = param("use_lidar", True)
        reliable_depth = param("depth_reliable", True)
        depth_topic = param("depth_topic", "/camera/depth/image_rect_raw")
        depth_info_topic = param("depth_info_topic", "/camera/depth/camera_info")
        cloud_topic = param("cloud_topic", "/registered_scan_static")
        persons_topic = param("persons_topic", "/person_detector/detections")
        rate = param("rate", 2.0)
        self.debug_dump = param("debug_dump", False)
        self.last_dump: Dict[int, float] = {}

        self.lock = threading.Lock()
        self.zones: Dict[int, zn.Zone] = {}
        self.exit_info: Dict[int, dict] = {}
        self.frames: Dict[int, deque] = {}
        self.dirty: set = set()
        self.persons: deque = deque()  # (stamp, xy)
        self.results: Dict[int, dict] = {}
        self.last_states: Dict[int, zn.ZoneResult] = {}
        self.exits_path: Optional[str] = self.exits_file_param or None
        self.exits_mtime = None
        self.depth_K = None
        self.depth_count = 0
        self.rays = None
        self.rays_key = None
        self.pgo_dir = None if self.exits_path else PgoSaveDirectory(self, self.pgo_node)

        self.tf_buffer = Buffer()
        # TF on its own thread: in the executor it starved behind the evaluations (~0.1-0.2 s of numpy each) and the
        # lookups at the depth stamps failed, so the zone lost its depth frames
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        if use_depth:
            qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE if reliable_depth
                             else ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(CameraInfo, depth_info_topic, self.on_depth_info, qos_profile_sensor_data)
            self.create_subscription(Image, depth_topic, self.on_depth, qos,
                                     callback_group=MutuallyExclusiveCallbackGroup())
        if use_lidar:
            self.create_subscription(PointCloud2, cloud_topic, self.on_cloud, 5,
                                     callback_group=MutuallyExclusiveCallbackGroup())
        self.create_subscription(Detection3DArray, persons_topic, self.on_persons, 10,
                                 callback_group=MutuallyExclusiveCallbackGroup())
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.verdict_pub = self.create_publisher(String, "/exit_inspection/zone_verdicts", latched)
        self.marker_pub = self.create_publisher(MarkerArray, "/exit_inspection/zone_markers", latched)
        self.create_timer(2.0, self.load_exits, callback_group=MutuallyExclusiveCallbackGroup())
        self.create_timer(1.0 / rate, self.evaluate_all, callback_group=MutuallyExclusiveCallbackGroup())
        self.get_logger().info(f"exits from {self.exits_path or f'{self.pgo_node} save_directory'}; "
                               f"depth {depth_topic if use_depth else 'off'}, lidar {cloud_topic if use_lidar else 'off'}, "
                               f"people {persons_topic}")

    # ---------------------------------------------------------------- exits
    def load_exits(self):
        if self.exits_path is None:
            directory = self.pgo_dir.poll()
            if directory is None:
                return
            self.exits_path = os.path.join(directory, "exit_inspection", "exits.yaml")
            self.get_logger().info(f"exits file {self.exits_path}")
        try:
            mtime = os.path.getmtime(self.exits_path)
        except OSError:
            self.get_logger().warn(f"no {self.exits_path} yet", throttle_duration_sec=60.0)
            return
        if mtime == self.exits_mtime:
            return
        self.exits_mtime = mtime
        try:
            with open(self.exits_path) as f:
                doc = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            self.get_logger().error(f"cannot read {self.exits_path}: {e}")
            return
        doors = [e for e in doc.get("exits") or [] if e.get("type") == "door" and e.get("confirmed")]
        with self.lock:
            self.zones = {int(e["id"]): zn.Zone.from_exit(e, self.zparams) for e in doors}
            self.exit_info = {int(e["id"]): e for e in doors}
            self.frames = {i: self.frames.get(i, deque()) for i in self.zones}
            self.results = {i: r for i, r in self.results.items() if i in self.zones}
        self.get_logger().info(f"{len(doors)} confirmed door exits to check: {sorted(self.zones)}"
                               if doors else f"{self.exits_path} has no confirmed door exit")

    # ---------------------------------------------------------------- inputs
    def sensor_pose(self, frame_id: str, stamp: Time) -> np.ndarray:
        """T_map_sensor at the stamp (raises TransformException)."""
        t_map_base = self.tf_buffer.lookup_transform(self.map_frame, self.lio_base_frame, stamp,
                                                     Duration(seconds=self.tf_timeout))
        t_base_sensor = self.tf_buffer.lookup_transform(self.base_frame, frame_id, Time())
        return transform_matrix(t_map_base) @ transform_matrix(t_base_sensor)

    def zones_in_range(self, sensor: np.ndarray):
        with self.lock:
            zones = list(self.zones.values())
        return [z for z in zones if np.hypot(*(z.center[:2] - sensor[:2])) <= self.max_range]

    def floor_hint(self, z: zn.Zone) -> float:
        last = self.last_states.get(z.exit_id)
        return last.floor_z if last is not None and np.isfinite(last.floor_z) else float(z.center[2])

    def add_frames(self, zones, stamp: float, sensor: np.ndarray, pts: np.ndarray, kind: str):
        for z in zones:
            crop = z.crop(pts, self.floor_hint(z))
            if len(crop) < self.min_frame_points:
                continue
            with self.lock:
                if z.exit_id in self.frames:
                    self.frames[z.exit_id].append(zn.Frame(stamp, sensor.copy(), crop, kind))
                    self.dirty.add(z.exit_id)

    def on_depth_info(self, msg: CameraInfo):
        self.depth_K = np.array(msg.k, dtype=float).reshape(3, 3)

    def depth_rays(self, h: int, w: int):
        K, s = self.depth_K, self.depth_stride
        key = (h, w, s, tuple(K.ravel()))
        if key != self.rays_key:
            v, u = np.mgrid[0:h:s, 0:w:s].astype(float)
            self.rays = ((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1])
            self.rays_key = key
        return self.rays

    def on_depth(self, msg: Image):
        self.depth_count += 1
        if self.depth_count % self.depth_every_n or self.depth_K is None or not self.zones:
            return
        stamp = Time.from_msg(msg.header.stamp)
        try:
            T = self.sensor_pose(msg.header.frame_id, stamp)
        except TransformException as e:
            self.get_logger().warn(f"no depth camera pose: {e}", throttle_duration_sec=5.0)
            return
        zones = self.zones_in_range(T[:3, 3])
        if not zones:
            return
        d = depth_image(msg)
        if d is None:
            self.get_logger().error(f"depth encoding {msg.encoding} not supported", throttle_duration_sec=30.0)
            return
        d = d[::self.depth_stride, ::self.depth_stride]
        rx, ry = self.depth_rays(msg.height, msg.width)
        ok = np.isfinite(d) & (d >= self.depth_min) & (d <= self.depth_max)
        dz = d[ok].astype(float)
        cam = np.column_stack([rx[ok] * dz, ry[ok] * dz, dz])
        pts = cam @ T[:3, :3].T + T[:3, 3]
        self.add_frames(zones, stamp.nanoseconds * 1e-9, T[:3, 3], pts, "depth")

    def on_cloud(self, msg: PointCloud2):
        if msg.header.frame_id != self.map_frame or not self.zones:
            return
        stamp = Time.from_msg(msg.header.stamp)
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.lidar_frame, stamp,
                                                 Duration(seconds=self.tf_timeout))
        except TransformException as e:
            self.get_logger().warn(f"no lidar pose: {e}", throttle_duration_sec=5.0)
            return
        sensor = transform_matrix(tf)[:3, 3]
        zones = self.zones_in_range(sensor)
        if not zones:
            return
        pts = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True).astype(float)
        self.add_frames(zones, stamp.nanoseconds * 1e-9, sensor, pts, "lidar")

    def on_persons(self, msg: Detection3DArray):
        dets = [d for d in msg.detections if d.results and d.results[0].hypothesis.class_id in self.person_classes
                and d.results[0].hypothesis.score >= self.person_min_conf]
        if not dets:
            return
        stamp = Time.from_msg(msg.header.stamp)
        try:
            T = self.sensor_pose(msg.header.frame_id, stamp)
        except TransformException as e:
            self.get_logger().warn(f"no camera pose for the people: {e}", throttle_duration_sec=5.0)
            return
        cam = T[:3, 3]
        with self.lock:
            for d in dets:
                c = d.bbox.center.position
                p = T[:3, :3] @ [c.x, c.y, c.z] + cam
                ray = p[:2] - cam[:2]
                xy = p[:2] + self.person_offset * ray / max(np.linalg.norm(ray), 1e-6)
                self.persons.append((stamp.nanoseconds * 1e-9, xy))

    # ---------------------------------------------------------------- verdicts
    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def evaluate_all(self):
        # The window ends at the newest frame of each exit, not at the clock: under load scans and depth images arrive
        # a second or two late, and a window cut at "now" emptied itself at a stop (UNVERIFIED, no depth frames).
        now = self.now()
        with self.lock:
            while self.persons and self.persons[0][0] < now - self.window - 10.0:
                self.persons.popleft()
            people = list(self.persons)
            todo = []
            for i, q in self.frames.items():
                if q:
                    newest = max(f.stamp for f in q)
                    if q[0].stamp < newest - self.window:
                        kept = [f for f in q if f.stamp >= newest - self.window]
                        q.clear()
                        q.extend(kept)
                if i in self.dirty and q:
                    todo.append((self.zones[i], list(q), newest))
            self.dirty.clear()
        if not todo:
            return
        for z, frames, newest in todo:
            persons = np.array([xy for t, xy in people if newest - self.window <= t <= newest + 1.0]).reshape(-1, 2)
            r = zn.evaluate(z, frames, persons)
            self.last_states[z.exit_id] = r
            e = self.exit_info[z.exit_id]
            entry = {"id": z.exit_id, "cls": e.get("cls", ""),
                     "door_floor_center": [round(float(x), 3) for x in z.center], **r.summary(),
                     "stamp": round(frames[-1].stamp, 2), "sensor": [round(float(x), 3) for x in frames[-1].sensor]}
            old = self.results.get(z.exit_id)
            if old is None or old["verdict"] != entry["verdict"] or old["reason"] != entry["reason"]:
                self.get_logger().info(f"exit {z.exit_id}: {r.verdict} ({r.reason}; {r.frames} frames, "
                                       f"{r.people} people)")
            self.results[z.exit_id] = entry
            if self.debug_dump:
                self.dump(z, frames, persons)
        self.publish_markers()
        self.publish_verdicts()

    def dump(self, z: zn.Zone, frames, persons: np.ndarray):
        now = self.now()
        if now - self.last_dump.get(z.exit_id, -1e9) < 0.4 or not self.exits_path:
            return
        self.last_dump[z.exit_id] = now
        folder = os.path.join(os.path.dirname(self.exits_path), "zone_debug")
        os.makedirs(folder, exist_ok=True)
        np.savez(os.path.join(folder, f"exit_{z.exit_id}.npz"),
                 points=np.vstack([f.points for f in frames]).astype(np.float32),
                 fid=np.concatenate([np.full(len(f.points), i, np.int32) for i, f in enumerate(frames)]),
                 sensors=np.array([f.sensor for f in frames]), stamps=np.array([f.stamp for f in frames]),
                 kinds=np.array([f.kind for f in frames]),
                 persons=persons, exit=yaml.safe_dump(self.exit_info[z.exit_id]))

    def document(self) -> dict:
        p = self.zparams
        return {"exits_file": self.exits_path, "frame": self.map_frame,
                "params": {"w_min": p.w_min, "margin": p.margin, "depth": p.depth, "wall_gap": p.wall_gap,
                           "object_min_height": p.object_min_height, "object_max_height": p.object_max_height,
                           "clear_coverage": p.clear_coverage, "window": self.window},
                "exits": [self.results[i] for i in sorted(self.results)]}

    def publish_verdicts(self):
        text = yaml.safe_dump(self.document(), sort_keys=False, default_flow_style=None)
        self.verdict_pub.publish(String(data=text))
        if self.exits_path:
            try:
                write_atomic(os.path.join(os.path.dirname(self.exits_path), "zone_checks.yaml"), text)
            except OSError as e:
                self.get_logger().error(f"cannot write zone_checks.yaml: {e}", throttle_duration_sec=30.0)

    def publish_markers(self):
        out = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for i, r in sorted(self.last_states.items()):
            z = self.zones.get(i)
            if z is None:
                continue
            rgb = VERDICT_COLORS[r.verdict]

            def marker(k, mtype, ns, color):
                m = Marker(type=mtype, action=Marker.ADD, id=10 * i + k, ns=ns)
                m.header.frame_id, m.header.stamp = self.map_frame, stamp
                m.pose.orientation.w = 1.0
                m.color.r, m.color.g, m.color.b, m.color.a = color
                return m

            fz = r.floor_z if np.isfinite(r.floor_z) else float(z.center[2])
            p = z.params
            uh, v0, v1 = z.u_half, p.wall_gap + r.wall_offset, p.depth + r.wall_offset
            corners = z.to_map([-uh, -uh, uh, uh, -uh], [v0, v1, v1, v0, v0], [fz + 0.02] * 5)
            outline = marker(0, Marker.LINE_STRIP, "zone", (*rgb, 1.0))
            outline.scale.x = 0.05
            outline.points = [Point(x=c[0], y=c[1], z=c[2]) for c in corners]
            text = marker(1, Marker.TEXT_VIEW_FACING, "zone", (*rgb, 1.0))
            top = z.to_map([0.0], [v1 / 2], [fz + 1.2])[0]
            text.pose.position = Point(x=top[0], y=top[1], z=top[2])
            text.scale.z = 0.3
            text.text = f"exit {i}: {r.verdict}"
            out.markers += [outline, text]
            cells = marker(2, Marker.CUBE_LIST, "zone_cells", (1.0, 1.0, 1.0, 1.0))
            cells.scale.x = cells.scale.y = p.cell * 0.9
            cells.scale.z = 0.02
            cu, cv = z.cell_centers()
            for s, color in CELL_COLORS.items():
                mask = r.state == s
                if not mask.any():
                    continue
                for c in z.to_map(cu[mask], cv[mask] + r.wall_offset, np.full(mask.sum(), fz + 0.01)):
                    cells.points.append(Point(x=c[0], y=c[1], z=c[2]))
                    cells.colors.append(ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3]))
            out.markers.append(cells)
        self.marker_pub.publish(out)


def main():
    rclpy.init()
    node = ExitZoneChecker()
    executor = MultiThreadedExecutor(num_threads=4)
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
