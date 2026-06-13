"""
build_scene_graph_gazebo.py
----------------------------
Builds the SceneGraph from the ground-truth object list of
ros2_ws/src/tmc_gazebo/tmc_gazebo_worlds/worlds/apartment.world.xacro
(world name 'default').

Names and poses are read directly from the world file: `xacro` expands the
.xacro to plain SDF, then every <include> whose name is a known object (see
NAME_TO_LABEL) is paired with a hand-picked local bounding box (dims,
center_z) from LABEL_GEOMETRY to synthesize a small point cloud. That point
cloud is fed into SceneGraph.add_node() exactly like build_scene_graph.py does
for Mask3D instances.

Usage:
    python -m core.perception.scene_graph.build_scene_graph_gazebo [--visualize]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

# allow running as a script from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.perception.scene_graph.scene_graph import SceneGraph

DATA_DIR = Path(os.environ.get("HRL_DATA_DIR", "/home/ws/data"))
GRAPH_DIR = DATA_DIR / "scene_graph"

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

# label -> (dims=(w, d, h), center_z) in the local (unrotated) frame.
# center_z = height of the geometric center above pose.z. Hand-picked from
# each model's SDF; shared by every instance of a label.
LABEL_GEOMETRY = {
    "trolley": ((0.47, 0.44, 0.68), 0.34),
    "couch": ((0.8, 1.3, 0.75), 0.375),
    "coffee table": ((0.68, 1.13, 0.4), 0.2),
    "desk": ((0.6, 1.2, 0.7), 0.35),
    "chair": ((0.45, 0.45, 0.85), 0.425),
    "cabinet": ((0.46, 1.15, 0.6), 0.3),
    "table": ((0.5, 2.3, 1.05), 0.525),
    "shelf": ((0.3, 0.45, 1.8), 0.9),
    "medicine box": ((0.083, 0.135, 0.026), 0),
    "wallet": ((0.11, 0.09, 0.02), 0),
    "orange": ((0.066, 0.066, 0.1235), 0),
    "pringles can": ((0.066, 0.066, 0.225), 0),
    "apple": ((0.09, 0.09, 0.09), 0),
    "banana": ((0.18, 0.05, 0.05), 0),
    "pear": ((0.08, 0.08, 0.11), 0),
}


def load_registry(world_xacro: Path = WORLD_XACRO) -> list[dict]:
    """Expand `world_xacro` with `xacro` and read every <include> whose name
    is in NAME_TO_LABEL, pairing its world-file pose with the label's
    hand-picked geometry from LABEL_GEOMETRY."""
    expanded = subprocess.run(
        ["xacro", str(world_xacro)], capture_output=True, text=True, check=True
    ).stdout
    world = ET.fromstring(expanded).find("world")

    registry = []
    for include in world.findall("include"):
        name = include.findtext("name")
        label = NAME_TO_LABEL.get(name)
        if label is None:
            continue
        pose = tuple(float(v) for v in include.findtext("pose").split())
        dims, center_z = LABEL_GEOMETRY[label]
        registry.append({"name": name, "label": label, "pose": pose, "dims": dims, "center_z": center_z})
    return registry


def _euler_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix for SDF's <pose> roll/pitch/yaw (R = Rz @ Ry @ Rx)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def synthesize_points(pose, dims, center_z, n: int = 5) -> np.ndarray:
    """Sample an n x n x n grid of points filling a box of size `dims`,
    centered at (0, 0, center_z) in the object's local frame, then rotate by
    the pose's roll/pitch/yaw and translate by its x/y/z."""
    x, y, z, roll, pitch, yaw = pose
    w, d, h = dims

    xs = np.linspace(-w / 2, w / 2, n)
    ys = np.linspace(-d / 2, d / 2, n)
    zs = center_z + np.linspace(-h / 2, h / 2, n)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)

    R = _euler_to_matrix(roll, pitch, yaw)
    return grid @ R.T + np.array([x, y, z])


def build_from_world(scene_graph: SceneGraph, registry: list[dict]) -> None:
    """Populate scene_graph from `registry` (see load_registry)."""
    for entry in registry:
        points = synthesize_points(entry["pose"], entry["dims"], entry["center_z"])
        scene_graph.add_node(
            color=np.array([0.5, 0.5, 0.5]),
            sem_label=entry["label"],
            points=points,
            mesh_mask=None,
            confidence=1.0,
            movable=entry["label"] not in IMMOVABLE_LABELS_GZ,
        )

    for node in scene_graph.nodes.values():
        scene_graph.update_connection(node)
    scene_graph.tree = KDTree(np.array([scene_graph.nodes[i].centroid for i in scene_graph.ids]))
    scene_graph.color_with_ibm_palette()

    print(f"[build] {len(scene_graph.nodes)} nodes from apartment.world ground truth")
    for node in scene_graph.nodes.values():
        kind = "furniture" if not node.movable else "object"
        print(f"  [{node.object_id:3d}] {node.sem_label:<14} {kind:<10} "
              f"centroid={np.round(node.centroid, 2)} dims={np.round(node.dimensions, 2)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-xacro", type=Path, default=WORLD_XACRO)
    parser.add_argument("--graph-dir", type=Path, default=GRAPH_DIR)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    registry = load_registry(args.world_xacro)
    labels = {entry["label"] for entry in registry}
    scene_graph = SceneGraph(
        label_mapping={label: label for label in labels},
        immovable=IMMOVABLE_LABELS_GZ,
    )
    build_from_world(scene_graph, registry)
    scene_graph.save_all(args.graph_dir)

    if args.visualize:
        scene_graph.visualize(labels=True, connections=True, centroids=True)


if __name__ == "__main__":
    main()
