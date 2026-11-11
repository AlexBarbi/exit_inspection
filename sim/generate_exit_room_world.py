#!/usr/bin/env python3
"""
generate_exit_room_world.py

Writes the exit test world of the emergency exit inspection (never edit the outputs by hand, run this again):
  worlds/exit_room.world         20 x 12 m room centered on the origin, walls 3 m, ceiling
  worlds/exit_room_gt.yaml       ground truth: every sign, its door, the expected verdict
  models/exit_sign_<name>/       one model per sign, texture drawn by exit_inspection.sign_art

Five cases along the walls (the robot spawns at the origin, facing +x):
  clear    north wall, x = -5: door + sign (E001 + arrow down), nothing in front
  warning  north wall, x = +4: door + sign (E002, text EXIT), a box beside the door, corridor left free
  blocked  south wall, x = -4: door + sign (E001 + arrow down), pallet with cartons right in front
  person   south wall, x = +5: door + sign (E002), a person standing in front (must count as clear)
  route    east wall,  y = +1: route sign (E001 + arrow left), no door (must be ignored)

Doors: 1.2 m x 2.1 m openings in the wall, door panel recessed 5 cm behind the wall face, frame protruding 2 cm.
Signs: 2 cm thick, bottom 5 cm above the door, lit (emissive), on the wall face.

Usage: python3 sim/generate_exit_room_world.py   (from the package folder, or with the path to it)
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

import yaml

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from exit_inspection.sign_art import SignSpec, render  # noqa: E402

ROOM_X, ROOM_Y, WALL_H, WALL_T = 20.0, 12.0, 3.0, 0.2
DOOR_W, DOOR_H, RECESS, FRAME_W, FRAME_D = 1.2, 2.1, 0.05, 0.06, 0.02
SIGN_T, SIGN_GAP = 0.02, 0.05
PICTO = 0.2  # m, pictogram side
TEXTURE_PX = 256  # pictogram side in the texture


@dataclass
class Case:
    name: str
    wall: str  # north, south, east, west
    pos: float  # along the wall (x on north/south, y on east/west)
    sign: SignSpec
    door: bool
    expected: str  # verdict of the zone check (sub-project 2); route signs: ignored


CASES = [
    Case("clear", "north", -5.0, SignSpec("E001", "down"), True, "CLEAR"),
    Case("warning", "north", 4.0, SignSpec("E002", None, text="EXIT"), True, "WARNING"),
    Case("blocked", "south", -4.0, SignSpec("E001", "down"), True, "BLOCKED"),
    Case("person", "south", 5.0, SignSpec("E002", None), True, "CLEAR"),
    Case("route", "east", 1.0, SignSpec("E001", "left"), False, "IGNORED"),
]


def wall_frame(wall: str):
    """(inner face point at along=0, unit along-wall vector, unit normal into the room, yaw of the normal)."""
    hx, hy = ROOM_X / 2, ROOM_Y / 2
    return {
        "north": ((0.0, hy), (1.0, 0.0), (0.0, -1.0)),
        "south": ((0.0, -hy), (1.0, 0.0), (0.0, 1.0)),
        "east": ((hx, 0.0), (0.0, 1.0), (-1.0, 0.0)),
        "west": ((-hx, 0.0), (0.0, 1.0), (1.0, 0.0)),
    }[wall]


def to_world(wall: str, along: float, out: float):
    """World xy of a point `along` the wall and `out` metres into the room from its inner face."""
    (fx, fy), (ax, ay), (nx, ny) = wall_frame(wall)
    return fx + along * ax + out * nx, fy + along * ay + out * ny


def yaw_of(wall: str) -> float:
    import math
    _, _, (nx, ny) = wall_frame(wall)
    return math.atan2(ny, nx)


def box_model(name, x, y, z, yaw, sx, sy, sz, rgb, collide=True):
    """Static box; size sx along the box x axis (rotated by yaw)."""
    col = (f"<collision name='collision'><geometry><box><size>{sx:g} {sy:g} {sz:g}</size></box></geometry>"
           f"</collision>") if collide else ""
    r, g, b = rgb
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 {yaw:.6f}</pose>
      <link name="link">
        {col}
        <visual name="visual">
          <geometry><box><size>{sx:g} {sy:g} {sz:g}</size></box></geometry>
          <material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse></material>
        </visual>
      </link>
    </model>"""


def walls(cases):
    """Wall boxes with openings for the doors, as segments along each wall plus lintels over the doors."""
    out = []
    grey = (0.85, 0.85, 0.85)
    for wall in ("north", "south", "east", "west"):
        length = ROOM_X if wall in ("north", "south") else ROOM_Y
        full = length + 2 * WALL_T if wall in ("north", "south") else length
        doors = sorted(c.pos for c in cases if c.wall == wall and c.door)
        edges = [-full / 2] + [e for p in doors for e in (p - DOOR_W / 2, p + DOOR_W / 2)] + [full / 2]
        yaw = yaw_of(wall)
        for i in range(0, len(edges), 2):
            a, b = edges[i], edges[i + 1]
            x, y = to_world(wall, (a + b) / 2, -WALL_T / 2)
            out.append(box_model(f"wall_{wall}_{i // 2}", x, y, WALL_H / 2, yaw, WALL_T, b - a, WALL_H, grey))
        for j, p in enumerate(doors):
            x, y = to_world(wall, p, -WALL_T / 2)
            h = WALL_H - DOOR_H
            out.append(box_model(f"lintel_{wall}_{j}", x, y, DOOR_H + h / 2, yaw, WALL_T, DOOR_W, h, grey))
    return out


def door(case):
    yaw = yaw_of(case.wall)
    out = []
    x, y = to_world(case.wall, case.pos, -RECESS - 0.025)
    out.append(box_model(f"door_{case.name}", x, y, DOOR_H / 2, yaw, 0.05, DOOR_W, DOOR_H, (0.45, 0.32, 0.22)))
    for side in (-1, 1):
        x, y = to_world(case.wall, case.pos + side * (DOOR_W / 2 + FRAME_W / 2), FRAME_D / 2)
        out.append(box_model(f"jamb_{case.name}_{side + 1}", x, y, DOOR_H / 2, yaw, FRAME_D, FRAME_W, DOOR_H,
                             (0.3, 0.3, 0.3)))
    x, y = to_world(case.wall, case.pos, FRAME_D / 2)
    out.append(box_model(f"header_{case.name}", x, y, DOOR_H + FRAME_W / 2, yaw, FRAME_D, DOOR_W + 2 * FRAME_W,
                         FRAME_W, (0.3, 0.3, 0.3)))
    return out


def sign_size(spec: SignSpec):
    img = render(spec, TEXTURE_PX)
    return img, PICTO * img.width / TEXTURE_PX, PICTO * img.height / TEXTURE_PX


def write_sign_model(models_dir, case):
    img, w, h = sign_size(case.sign)
    name = f"exit_sign_{case.name}"
    tex_dir = os.path.join(models_dir, name, "materials", "textures")
    os.makedirs(tex_dir, exist_ok=True)
    img.save(os.path.join(tex_dir, "sign.png"))
    tex = f"model://{name}/materials/textures/sign.png"
    with open(os.path.join(models_dir, name, "model.config"), "w") as f:
        f.write(f"""<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>
  <description>Exit sign ({case.sign.cls}), written by exit_inspection/sim/generate_exit_room_world.py</description>
</model>
""")
    # A box textured on every face; the face looking along +x of the model is the front (towards the room).
    with open(os.path.join(models_dir, name, "model.sdf"), "w") as f:
        f.write(f"""<?xml version="1.0"?>
<!-- Written by exit_inspection/sim/generate_exit_room_world.py: run it again rather than editing this file. -->
<sdf version="1.7">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry><box><size>{SIGN_T} {w:.4f} {h:.4f}</size></box></geometry>
      </collision>
      <visual name="visual">
        <geometry><box><size>{SIGN_T} {w:.4f} {h:.4f}</size></box></geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <emissive>0.5 0.5 0.5 1</emissive>
          <pbr>
            <metal>
              <albedo_map>{tex}</albedo_map>
              <emissive_map>{tex}</emissive_map>
              <metalness>0.0</metalness>
              <roughness>0.9</roughness>
            </metal>
          </pbr>
        </material>
      </visual>
    </link>
  </model>
</sdf>
""")
    return w, h


def sign_include(case, w, h):
    x, y = to_world(case.wall, case.pos, SIGN_T / 2)
    z = DOOR_H + SIGN_GAP + h / 2
    return (x, y, z), f"""
    <include>
      <uri>model://exit_sign_{case.name}</uri>
      <name>exit_sign_{case.name}</name>
      <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 {yaw_of(case.wall):.6f}</pose>
    </include>"""


def obstacles():
    out = []
    # warning: a 0.6 m box beside the north-east door (door spans x 3.4..4.6), corridor in front of the door free
    x, y = 5.3, ROOM_Y / 2 - 0.8
    out.append(box_model("carton_warning", x, y, 0.4, 0.0, 0.6, 0.6, 0.8, (0.7, 0.55, 0.35)))
    # blocked: pallet with a stack of cartons right in front of the south-west door (door spans x -4.6..-3.4)
    x, y = -4.0, -ROOM_Y / 2 + 0.7
    out.append(box_model("pallet_blocked", x, y, 0.075, 0.0, 1.2, 0.8, 0.15, (0.6, 0.45, 0.3)))
    out.append(box_model("cartons_blocked", x, y, 0.15 + 0.5, 0.0, 1.1, 0.7, 1.0, (0.7, 0.55, 0.35)))
    return out


def standing_person(case):
    import math
    x, y = to_world(case.wall, case.pos + 0.2, 0.9)
    yaw = yaw_of(case.wall) + math.pi  # facing the door
    pose = f"{x:.3f} {y:.3f} 1 0 0 {yaw:.4f}"
    return f"""
    <actor name="person_standing">
      <skin>
        <filename>model://walking_actor/meshes/walk.dae</filename>
        <scale>1.0</scale>
      </skin>
      <animation name="walking">
        <filename>model://walking_actor/meshes/walk.dae</filename>
        <scale>1.0</scale>
        <interpolate_x>true</interpolate_x>
      </animation>
      <script>
        <loop>true</loop>
        <delay_start>0</delay_start>
        <auto_start>true</auto_start>
        <trajectory id="0" type="walking">
          <waypoint><time>0</time><pose>{pose}</pose></waypoint>
          <waypoint><time>10000</time><pose>{pose}</pose></waypoint>
        </trajectory>
      </script>
    </actor>"""


HEADER = """<?xml version="1.0" ?>
<!-- Written by exit_inspection/sim/generate_exit_room_world.py: change that script and run it again rather than
     editing this file. Exit test room of the emergency exit inspection: see the script and exit_room_gt.yaml. -->
<sdf version="1.7">
  <world name="exit_room">
    <physics name="default_physics" default="true" type="dart">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <!-- Drives <sensor type="imu"> (the IMU of the lidar); the base IMU is read by LeggedHWSim -->
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <scene>
      <ambient>0.6 0.6 0.6 1</ambient>
      <background>0.3 0.7 0.9 1</background>
      <shadows>0</shadows>
      <grid>0</grid>
    </scene>
    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <surface>
            <friction><ode><mu>100</mu><mu2>50</mu2></ode></friction>
            <contact><ode><soft_cfm>0</soft_cfm><soft_erp>0.2</soft_erp><kp>1e+13</kp><kd>1</kd>
              <max_vel>0.01</max_vel><min_depth>0.001</min_depth></ode></contact>
          </surface>
        </collision>
        <visual name="visual">
          <cast_shadows>false</cast_shadows>
          <geometry><plane><normal>0 0 1</normal><size>{floor_x} {floor_y}</size></plane></geometry>
          <material><ambient>0.7 0.7 0.7 1</ambient><diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular></material>
        </visual>
      </link>
    </model>
"""


def write_camera_rig(models_dir):
    """Free D435i color + depth pair (same intrinsics as the robot's, depth to 30 m) moved by set_pose: the render
    harvester (sim/harvest_renders.py) takes labeled frames with it. The cameras look along +x of the model."""
    os.makedirs(os.path.join(models_dir, "exit_camera_rig"), exist_ok=True)
    with open(os.path.join(models_dir, "exit_camera_rig", "model.config"), "w") as f:
        f.write("""<?xml version="1.0"?>
<model>
  <name>exit_camera_rig</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>
  <description>D435i-like color + depth cameras for sim/harvest_renders.py</description>
</model>
""")
    cam = """
        <sensor name="{name}" type="{type}">
          <always_on>true</always_on>
          <update_rate>10</update_rate>
          <topic>rig/{name}</topic>
          <camera>
            <horizontal_fov>{hfov:.6f}</horizontal_fov>
            <image><width>1920</width><height>1080</height><format>{fmt}</format></image>
            <clip><near>0.1</near><far>30.0</far></clip>
          </camera>
        </sensor>"""
    import math
    hfov = math.radians(69.4)
    with open(os.path.join(models_dir, "exit_camera_rig", "model.sdf"), "w") as f:
        f.write(f"""<?xml version="1.0"?>
<!-- Written by exit_inspection/sim/generate_exit_room_world.py: run it again rather than editing this file. -->
<sdf version="1.7">
  <model name="exit_camera_rig">
    <static>true</static>
    <link name="link">{cam.format(name="color", type="camera", hfov=hfov, fmt="R8G8B8")}{cam.format(
            name="depth", type="depth_camera", hfov=hfov, fmt="R_FLOAT32")}
    </link>
  </model>
</sdf>
""")


def main():
    worlds_dir = os.path.join(PKG, "worlds")
    models_dir = os.path.join(PKG, "models")
    os.makedirs(worlds_dir, exist_ok=True)
    write_camera_rig(models_dir)
    parts = [HEADER.format(floor_x=ROOM_X + 2 * WALL_T, floor_y=ROOM_Y + 2 * WALL_T)]
    parts += walls(CASES)
    parts.append(box_model("ceiling", 0.0, 0.0, WALL_H + 0.1, 0.0, ROOM_X + 2 * WALL_T, ROOM_Y + 2 * WALL_T, 0.2,
                           (0.9, 0.9, 0.9)))
    gt = {"world": "exit_room", "door_width": DOOR_W, "door_height": DOOR_H, "exits": []}
    for c in CASES:
        if c.door:
            parts += door(c)
        w, h = write_sign_model(models_dir, c)
        center, inc = sign_include(c, w, h)
        parts.append(inc)
        _, _, (nx, ny) = wall_frame(c.wall)
        fx, fy = to_world(c.wall, c.pos, 0.0)
        gt["exits"].append({
            "name": c.name,
            "type": "door" if c.door else "route",
            "cls": c.sign.cls,
            "sign_center": [round(center[0], 4), round(center[1], 4), round(center[2], 4)],
            "sign_size": [round(w, 4), round(h, 4)],
            "normal": [nx, ny, 0.0],
            "door_floor_center": [round(fx, 4), round(fy, 4), 0.0] if c.door else None,
            "expected": c.expected,
        })
    parts += obstacles()
    parts.append(standing_person(next(c for c in CASES if c.name == "person")))
    parts.append("\n  </world>\n</sdf>\n")
    with open(os.path.join(worlds_dir, "exit_room.world"), "w") as f:
        f.write("".join(parts))
    with open(os.path.join(worlds_dir, "exit_room_gt.yaml"), "w") as f:
        f.write("# Written by exit_inspection/sim/generate_exit_room_world.py: ground truth of exit_room.world\n")
        yaml.safe_dump(gt, f, sort_keys=False)
    print(f"wrote {worlds_dir}/exit_room.world, exit_room_gt.yaml and {len(CASES)} sign models in {models_dir}")


if __name__ == "__main__":
    main()
