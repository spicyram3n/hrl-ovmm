"""
build_scene_graph.py
--------------------
Builds a SceneGraph for the HSR-C robot from OpenMask3D features.

Replaces stretch-compose's scenegraph_preprocessing.py.
No Mask3D Docker needed — uses OpenMask3D masks already computed.

Usage:
    python core/perception/scene_graph_generation/build_scene_graph.py
"""

from __future__ import annotations

import json
import sys
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial import KDTree

# ensure /home/ws is on the path so `core.*` imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.perception.segmentation.openmask_query import OpenMaskQuery
from scene_graph import SceneGraph

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("/home/ws/data")
FEATURES_DIR = Path("/home/ws/data/openmask_features")
GRAPH_DIR    = Path("/home/ws/data/scene_graph")
SCENE_PLY    = DATA_DIR / "scene.ply"

# Objects the robot can pick up (movable)
MOVABLE_OBJECTS = [
    "cup", "mug", "bottle", "bowl", "plate", "book",
    "remote", "keyboard", "mouse", "phone", "toy",
    "box", "can",
]

# Furniture / anchor objects (immovable — nodes connect TO these)
IMMOVABLE_OBJECTS = [
    "table", "desk", "shelf", "cabinet", "chair",
    "sofa", "couch", "counter", "bed", "drawer",
]

# Objects to ignore entirely
IGNORE_OBJECTS = [
    "wall", "floor", "ceiling", "door", "window",
    "radiator", "curtain", "lamp",
]

ALL_QUERIES = MOVABLE_OBJECTS + IMMOVABLE_OBJECTS
# ─────────────────────────────────────────────────────────────────────────────


def build_from_openmask(
    querier: OpenMaskQuery,
    scene_graph: SceneGraph,
    queries: list[str],
    movable: list[str],
    min_points: int = 50,
) -> None:
    import torch
    import clip

    pcd_full  = o3d.io.read_point_cloud(str(SCENE_PLY))
    pts_full  = np.asarray(pcd_full.points)
    n_pts     = pts_full.shape[0]
    seen_masks = set()

    for label in queries:
        # get best mask index directly (for dedup check)
        tokens = clip.tokenize([label.lower()])
        with torch.no_grad():
            text_feat = querier.model.encode_text(tokens)
        cos_sim = torch.nn.functional.cosine_similarity(
            torch.Tensor(querier.features), text_feat, dim=1
        )
        best_idx = torch.argmax(cos_sim).item()

        if best_idx in seen_masks:
            print(f"[build] '{label}' skipped — mask {best_idx} already used by another label")
            continue
        seen_masks.add(best_idx)

        try:
            pcd_obj, _ = querier.query(label, rank=0, visualize=False)
        except Exception as e:
            print(f"[build] Query '{label}' failed: {e}")
            continue

        obj_pts = np.asarray(pcd_obj.points)
        if obj_pts.shape[0] < min_points:
            print(f"[build] '{label}' skipped — only {obj_pts.shape[0]} points")
            continue

        tree     = KDTree(pts_full)
        _, idxs  = tree.query(obj_pts, k=1)
        mesh_mask = np.zeros(n_pts, dtype=bool)
        mesh_mask[idxs] = True

        is_movable = label in movable
        color      = np.array([0.5, 0.5, 0.5]) if not is_movable else np.array([0.8, 0.4, 0.1])

        scene_graph.add_node(
            color=color,
            sem_label=label,
            points=obj_pts,
            mesh_mask=mesh_mask,
            confidence=1.0,
            movable=is_movable,
        )
        print(f"[build] Added '{label}' — {obj_pts.shape[0]} pts, movable={is_movable}")

    for node in scene_graph.nodes.values():
        scene_graph.update_connection(node)

    scene_graph.tree = KDTree(
        np.array([scene_graph.nodes[i].centroid for i in scene_graph.ids])
    )
    print(f"[build] Scene graph: {len(scene_graph.nodes)} nodes, "
          f"{len(scene_graph.outgoing)} edges")


def save_scene_graph(scene_graph: SceneGraph, graph_dir: Path) -> None:
    """Save scene graph outputs to JSON files."""
    graph_dir.mkdir(parents=True, exist_ok=True)

    # graph.json — full structure
    graph_data = {
        "node_ids": scene_graph.ids,
        "node_labels": [
            scene_graph.nodes[nid].sem_label for nid in scene_graph.ids
        ],
        "connections": {
            str(k): v for k, v in scene_graph.outgoing.items()
        },
        "movable_ids": [
            nid for nid, n in scene_graph.nodes.items() if n.movable
        ],
        "immovable_ids": [
            nid for nid, n in scene_graph.nodes.items() if not n.movable
        ],
    }
    with open(graph_dir / "graph.json", "w") as f:
        json.dump(graph_data, f, indent=4)

    # scene.json — furniture/anchor objects
    scene_data = {
        "furniture": {
            str(nid): {
                "label":      n.sem_label,
                "centroid":   n.centroid.tolist(),
                "dimensions": n.dimensions.tolist(),
            }
            for nid, n in scene_graph.nodes.items() if not n.movable
        }
    }
    with open(graph_dir / "scene.json", "w") as f:
        json.dump(scene_data, f, indent=4)

    # objects/<id>.json — one file per movable object
    obj_dir = graph_dir / "objects"
    obj_dir.mkdir(exist_ok=True)
    for nid, n in scene_graph.nodes.items():
        if not n.movable:
            continue
        obj_data = {
            "id":         nid,
            "label":      n.sem_label,
            "centroid":   n.centroid.tolist(),
            "dimensions": n.dimensions.tolist(),
            "pose":       n.pose.tolist(),
            "confidence": n.confidence,
        }
        with open(obj_dir / f"{nid}.json", "w") as f:
            json.dump(obj_data, f, indent=4)

    print(f"[save] Scene graph saved to {graph_dir}")


if __name__ == "__main__":
    # 1. load OpenMask3D features (no server needed if cache exists)
    querier = OpenMaskQuery(
        scene_ply=SCENE_PLY,
        features_dir=FEATURES_DIR,
        data_dir=DATA_DIR,
    )
    querier.load_features(scene_name="hrl_scene", intrinsic_res="[1440,1920]")
    querier.load_pointcloud()

    # 2. build scene graph
    # build label mapping from your own queries (string → string)
    label_mapping = {label: label for label in ALL_QUERIES}

    scene_graph = SceneGraph(
        label_mapping=label_mapping,
        min_confidence=0.0,
    )
    build_from_openmask(
        querier=querier,
        scene_graph=scene_graph,
        queries=ALL_QUERIES,
        movable=MOVABLE_OBJECTS,
    )

    # 3. save
    save_scene_graph(scene_graph, GRAPH_DIR)

    # 4. visualize
    scene_graph.visualize(labels=True, connections=True, centroids=True)