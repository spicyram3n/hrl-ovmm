"""
build_scene_graph_gazebo.py
----------------------------
Builds the SceneGraph from the ground-truth object list of
ros2_ws/src/tmc_gazebo/tmc_gazebo_worlds/worlds/apartment.world.xacro
(world name 'default').

Everything is read from Gazebo's own files/topics - nothing about an object's
size, shape or current pose is hand-picked:
  - `xacro` expands the .xacro to plain SDF, giving every <include>'s name
    and model:// uri (and, as a fallback, its static spawn pose).
  - <include> names that are known objects (see NAME_TO_LABEL) are kept; the
    rest (walls, doors, ...) are structural and skipped.
  - For each kept model:// uri, gazebo_geometry.model_local_bbox() resolves
    the model via GAZEBO_MODEL_PATH, parses its SDF collision geometry
    (box/cylinder/sphere/mesh) and returns the model's local bounding box
    (dims, center). This never changes at runtime, so it's read once from disk.
  - fetch_live_poses() grabs one live snapshot of every model's current pose
    from the running simulation (Gazebo's own /world/default/pose/info topic,
    bridged to ROS2 - see gz_parameter_bridge_node() in gazebo_bringup.launch.py).
    If Gazebo isn't running, the static spawn pose from the world file is used
    instead so the script still works offline.
  - synthesize_points() samples a small point cloud filling the bbox, rotated
    and translated by the object's pose. That point cloud is fed into
    SceneGraph.add_node() exactly like build_scene_graph.py does for Mask3D
    instances.

Usage:
    python -m core.perception.scene_graph.build_scene_graph_gazebo [--visualize]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

# allow running as a script from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.perception.scene_graph.gazebo_geometry import euler_to_matrix, model_local_bbox, quat_to_euler
from core.perception.scene_graph.scene_graph import SceneGraph
from core.utils.config import GRAPH_DIR

# Furniture labels. Movable objects connect TO these.
IMMOVABLE_LABELS_GZ = [
    "trolley", "couch", "coffee table", "desk", "chair", "cabinet", "table", "shelf",
]

# Where to find apartment.world.xacro inside the mounted ROS 2 workspace.
WORLD_XACRO = (
    Path(os.environ.get("HRL_ROS2_WS", "/home/ws/ros2_ws"))
    / "src/tmc_gazebo/tmc_gazebo_worlds/worlds/apartment.world.xacro"
)

# <include><name> in the world file -> human-readable label. Anything not
# listed here (walls, doors, door stoppers, lawn_garden, ...) is structural
# and skipped.
NAME_TO_LABEL = {
    "wagon": "trolley",
    "sofa01": "couch",
    "sofa02": "couch",
    "kitchen_lowtable": "coffee table",
    "office_desk01": "desk",
    "office_desk02": "desk",
    "office_desk06": "desk",
    "chair00": "chair",
    "chair01": "chair",
    "chair02": "chair",
    "chair05": "chair",
    "living_sideboard": "cabinet",
    "high_table01": "table",
    "high_shelf01": "shelf",
    "high_shelf02": "shelf",
    "high_shelf03": "shelf",
    "nasal_cupsul": "medicine box",
    "wallet": "wallet",
    "hsr_orange_02": "orange",
    "hsr_pringles_01": "pringles can",
    "hsr_pringles_02": "pringles can",
    "apple_01": "apple",
    "banana_01": "banana",
    "pear_01": "pear",
}

def fetch_live_poses(world: str = "default", timeout_sec: float = 2.0) -> dict[str, tuple]:
    """Snapshot every model's current (x, y, z, roll, pitch, yaw) straight from the
    running simulation, via Gazebo's own /world/<world>/pose/info topic (bridged to
    ROS2 as tf2_msgs/TFMessage - see gz_parameter_bridge_node() in
    gazebo_bringup.launch.py). Returns {} if Gazebo/the bridge isn't up within
    `timeout_sec`, so callers can fall back to the world file's static spawn pose."""
    rclpy.init(args=[])
    node = Node("scene_graph_pose_snapshot")
    poses: dict[str, tuple] = {}

    def on_tf(msg: TFMessage) -> None:
        for tf in msg.transforms:
            t, q = tf.transform.translation, tf.transform.rotation
            poses[tf.child_frame_id] = (t.x, t.y, t.z, *quat_to_euler(q.x, q.y, q.z, q.w))

    node.create_subscription(TFMessage, f"/world/{world}/pose/info", on_tf, 10)

    deadline = time.monotonic() + timeout_sec
    while not poses and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)

    node.destroy_node()
    rclpy.shutdown()
    return poses


def load_registry(world_xacro: Path = WORLD_XACRO, live_poses: dict[str, tuple] | None = None) -> list[dict]:
    """Expand `world_xacro` with `xacro` and read every <include> whose name
    is in NAME_TO_LABEL, looking up each model's local bounding box
    (dims, center) from its own SDF via gazebo_geometry.model_local_bbox().
    Pose is taken from `live_poses` (see fetch_live_poses) when available;
    otherwise it falls back to the world file's static spawn pose."""
    expanded = subprocess.run(
        ["xacro", str(world_xacro)], capture_output=True, text=True, check=True
    ).stdout
    world = ET.fromstring(expanded).find("world")
    live_poses = live_poses or {}

    registry = []
    for include in world.findall("include"):
        name = include.findtext("name")
        label = NAME_TO_LABEL.get(name)
        if label is None:
            continue
        static_pose = tuple(float(v) for v in include.findtext("pose").split())
        pose = live_poses.get(name, static_pose)
        dims, center = model_local_bbox(include.findtext("uri"))
        registry.append({"name": name, "label": label, "pose": pose, "dims": dims, "center": center})
    return registry


def synthesize_points(pose, dims, center, n: int = 5) -> np.ndarray:
    """Sample an n x n x n grid of points filling a box of size `dims`,
    centered at `center` in the object's local frame, then rotate by the
    pose's roll/pitch/yaw and translate by its x/y/z."""
    x, y, z, roll, pitch, yaw = pose
    w, d, h = dims
    cx, cy, cz = center

    xs = cx + np.linspace(-w / 2, w / 2, n)
    ys = cy + np.linspace(-d / 2, d / 2, n)
    zs = cz + np.linspace(-h / 2, h / 2, n)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)

    R = euler_to_matrix(roll, pitch, yaw)
    return grid @ R.T + np.array([x, y, z])


def build_from_world(scene_graph: SceneGraph, registry: list[dict]) -> None:
    """Populate scene_graph from `registry` (see load_registry)."""
    for entry in registry:
        points = synthesize_points(entry["pose"], entry["dims"], entry["center"])
        scene_graph.add_node(
            color=np.array([0.5, 0.5, 0.5]),
            sem_label=entry["label"],
            points=points,
            mesh_mask=None,
            confidence=1.0,
            movable=entry["label"] not in IMMOVABLE_LABELS_GZ,
        )

    scene_graph.finalize_and_report("apartment.world ground truth")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-xacro", type=Path, default=WORLD_XACRO)
    parser.add_argument("--graph-dir", type=Path, default=GRAPH_DIR)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--static-pose", action="store_true",
        help="Skip the live pose snapshot and use the world file's spawn pose for every object.",
    )
    args = parser.parse_args()

    live_poses = {} if args.static_pose else fetch_live_poses()
    print(f"Live poses: {len(live_poses)} object(s); the rest use the world file's spawn pose.")

    registry = load_registry(args.world_xacro, live_poses)
    scene_graph = SceneGraph(immovable=IMMOVABLE_LABELS_GZ)
    build_from_world(scene_graph, registry)
    scene_graph.save_all(args.graph_dir)

    if args.visualize:
        scene_graph.visualize(labels=True, connections=True, centroids=True)


if __name__ == "__main__":
    main()
