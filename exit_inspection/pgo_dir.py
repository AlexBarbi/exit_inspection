"""SC-PGO's save directory, asked to its node (alaserPGO) the way tare_finish_merge_map.py does."""

import os
from typing import List, Optional

from rcl_interfaces.msg import ParameterType
from rclpy.parameter_client import AsyncParameterClient


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


class PgoSaveDirectory:
    """Non-blocking: call poll() from a timer until it returns the directory (absolute, normalized).

    A relative save_directory is taken from the working directory of the SC-PGO process (/proc/<pid>/cwd).
    """

    def __init__(self, node, pgo_node: str = "alaserPGO"):
        self.node, self.pgo_node = node, pgo_node
        self.client = AsyncParameterClient(node, pgo_node)
        self.future = None
        self.directory: Optional[str] = None

    @property
    def service_ready(self) -> bool:
        return self.client.services_are_ready()

    def poll(self) -> Optional[str]:
        if self.directory is not None:
            return self.directory
        if self.future is None:
            if self.service_ready:
                self.future = self.client.get_parameters(["save_directory"])
            return None
        if not self.future.done():
            return None
        result, self.future = self.future.result(), None
        if result is None or not result.values or result.values[0].type != ParameterType.PARAMETER_STRING:
            self.node.get_logger().error(f"/{self.pgo_node} has no save_directory: set it on this node",
                                         throttle_duration_sec=30.0)
            return None
        directory = result.values[0].string_value
        if not os.path.isabs(directory):
            pids = node_pids(self.pgo_node)
            directory = os.path.join(os.readlink(f"/proc/{pids[0]}/cwd") if pids else os.getcwd(), directory)
        self.directory = os.path.normpath(directory)
        return self.directory
