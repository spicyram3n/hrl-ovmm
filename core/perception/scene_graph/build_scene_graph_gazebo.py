"""
build_scene_graph_gazebo.py
----------------------------
Builds the SceneGraph from the ground-truth object list of
ros2_ws/src/tmc_gazebo/tmc_gazebo_worlds/worlds/apartment.world.xacro
(world name 'default').

Everything is read from Gazebo's own files - nothing about an object's size
or shape is hand-picked:
  - `xacro` expands the .xacro to plain SDF, giving every <include>'s name,
    model:// uri and world-frame pose.
  - <include> names that are known objects (see NAME_TO_LABEL) are kept; the
    rest (walls, doors, ...) are structural and skipped.
  - For each kept model:// uri, gazebo_geometry.model_local_bbox() resolves
    the model via GAZEBO_MODEL_PATH, parses its SDF collision geometry
    (box/cylinder/sphere/mesh) and returns the model's local bounding box
    (dims, center).
  - synthesize_points() samples a small point cloud filling that box, rotated
    and translated by the include's world-frame pose. That point cloud is fed
    into SceneGraph.add_node() exactly like build_scene_graph.py does for
    Mask3D instances.

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

# allow running as a script from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.perception.scene_graph.gazebo_geometry import euler_to_matrix, model_local_bbox
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

def load_registry(world_xacro: Path = WORLD_XACRO) -> list[dict]:
    """Expand `world_xacro` with `xacro` and read every <include> whose name
    is in NAME_TO_LABEL, looking up each model's local bounding box
    (dims, center) from its own SDF via gazebo_geometry.model_local_bbox()."""
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
    args = parser.parse_args()

    registry = load_registry(args.world_xacro)
    scene_graph = SceneGraph(immovable=IMMOVABLE_LABELS_GZ)
    build_from_world(scene_graph, registry)
    scene_graph.save_all(args.graph_dir)

    if args.visualize:
        scene_graph.visualize(labels=True, connections=True, centroids=True)


if __name__ == "__main__":
    main()
