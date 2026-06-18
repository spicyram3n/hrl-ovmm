"""
build_scene_graph.py  —  REAL ROBOT PATH (Mask3D perception)
-------------------------------------------------------------
NOT used in the Gazebo simulation demo.  For the demo, use:
    python -m core.perception.scene_graph.build_scene_graph_gazebo

This builder reads Mask3D 3D instance segmentation output (ScanNet200
closed-vocabulary labels from a real RGB-D scan) and populates a SceneGraph.

Workflow on the real robot:
    1. bash docker/mask3d/setup_mask3d.sh          (once, sets up Mask3D)
    2. bash docker/mask3d/run_mask3d.sh [ws]       (per scan, needs GPU)
    3. python -m core.perception.scene_graph.build_scene_graph

--workspace defaults to $HRL_DATA_DIR (or /home/ws/data), where the scan
(scene.ply) and Mask3D predictions.txt live.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

# allow running as a script from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.perception.scene_graph.scene_graph import SceneGraph
from core.perception.segmentation.mask3d_interface import (
    ID_TO_LABEL,
    load_predictions,
    load_scene_pointcloud,
)

DATA_DIR = Path(os.environ.get("HRL_DATA_DIR", "/home/ws/data"))
GRAPH_DIR = DATA_DIR / "scene_graph"

# Furniture / anchors. Movable objects connect TO these (and the LLM later
# clusters them into rooms). Names must be ScanNet200 labels.
IMMOVABLE_LABELS = [
    "table", "desk", "shelf", "cabinet", "chair", "armchair", "office chair",
    "sofa chair", "couch", "bed", "bookshelf", "dresser", "nightstand",
    "counter", "kitchen cabinet", "kitchen counter", "refrigerator", "stove",
    "washing machine", "toilet", "sink", "bathtub", "wardrobe", "tv stand",
    "coffee table", "end table", "dining table", "piano", "bench", "stand",
]

# Structural / clutter classes that should never become nodes
IGNORE_LABELS = [
    "wall", "floor", "ceiling", "door", "doorframe", "window", "curtain",
    "blinds", "radiator", "pipe", "ledge", "rail", "stairs", "stair rail",
    "column", "vent", "power outlet", "light switch",
]


def build_from_mask3d(
    scene_graph: SceneGraph,
    workspace: Path,
    min_confidence: float = 0.3,
    min_points: int = 50,
) -> None:
    """
    Populate scene_graph from a Mask3D workspace.

    Mask3D instances can overlap (one point claimed by several masks).
    Like stretch-compose, each point is assigned to exactly one instance:
    smaller, higher-confidence masks win over large background masks.
    """
    instances = load_predictions(workspace, min_confidence=min_confidence)
    # drop ignored classes early
    instances = [i for i in instances if i.label not in IGNORE_LABELS]
    if not instances:
        raise RuntimeError(f"No usable Mask3D instances in {workspace}")

    pcd = load_scene_pointcloud(workspace)
    scene_graph.pcd = pcd
    np_points = np.asarray(pcd.points)
    np_colors = np.asarray(pcd.colors) if pcd.has_colors() else np.full_like(np_points, 0.5)
    n_pts = np_points.shape[0]

    # owner[i] = index of the instance that owns point i (-1 = unowned).
    # Process masks from largest to smallest so small objects on top of
    # furniture override the furniture mask.
    masks = [inst.load_mask() for inst in instances]
    for m in masks:
        if m.shape[0] != n_pts:
            raise ValueError(
                f"Mask length {m.shape[0]} does not match point cloud size {n_pts}. "
                "Make sure masks and mesh come from the same Mask3D run."
            )
    order = np.argsort([-m.sum() for m in masks])

    owner = np.full(n_pts, -1, dtype=np.int64)
    for inst_idx in order:
        owner[masks[inst_idx]] = inst_idx

    for inst_idx, inst in enumerate(instances):
        mesh_mask = owner == inst_idx
        node_points = np_points[mesh_mask]
        if node_points.shape[0] < min_points:
            continue
        color = np_colors[mesh_mask][0]
        # sem_label is the ScanNet200 integer id; the graph's label_mapping
        # translates it to a string (same convention as stretch-compose)
        scene_graph.add_node(
            color=color,
            sem_label=inst.class_id,
            points=node_points,
            mesh_mask=mesh_mask,
            confidence=inst.confidence,
        )

    for node in scene_graph.nodes.values():
        scene_graph.update_connection(node)
    scene_graph.tree = KDTree(np.array([scene_graph.nodes[i].centroid for i in scene_graph.ids]))
    scene_graph.color_with_ibm_palette()

    print(f"[build] {len(scene_graph.nodes)} nodes from {len(instances)} Mask3D instances")
    for node in scene_graph.nodes.values():
        label = ID_TO_LABEL.get(node.sem_label, str(node.sem_label))
        kind = "furniture" if not node.movable else "object"
        print(f"  [{node.object_id:3d}] {label:<20} {kind:<10} conf={node.confidence:.2f} "
              f"pts={node.points.shape[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DATA_DIR,
                        help="Mask3D workspace folder (contains predictions.txt)")
    parser.add_argument("--graph-dir", type=Path, default=GRAPH_DIR)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    immovable_ids_as_names = IMMOVABLE_LABELS  # add_node checks mapped names
    scene_graph = SceneGraph(
        label_mapping=ID_TO_LABEL,
        min_confidence=args.min_confidence,
        immovable=immovable_ids_as_names,
    )
    build_from_mask3d(scene_graph, args.workspace, min_confidence=args.min_confidence)
    scene_graph.save_all(args.graph_dir)

    if args.visualize:
        scene_graph.visualize(labels=True, connections=True, centroids=True)


if __name__ == "__main__":
    main()