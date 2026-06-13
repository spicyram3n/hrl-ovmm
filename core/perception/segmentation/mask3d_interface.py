"""
mask3d_interface.py
-------------------
Parses the output of the Mask3D docker (rupalsaxena/mask3d_docker running
behretj/Mask3D's mask3d.py). Replaces the OpenMask3D zip/REST flow for
closed-vocabulary scene graph construction.

Expected workspace layout (produced by docker/mask3d/run_mask3d.sh):

    <workspace>/
    ├── scene.ply             input scan (Mask3D reads this via --pcd)
    ├── mesh_labeled.ply       Mask3D output, instance colors (not used here)
    ├── predictions.txt        "<rel_mask_path> <scannet200_id> <confidence>"
    └── pred_mask/NNN.txt      one 0/1 line per point

All functions are pure Python + numpy/open3d, no ROS.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from core.utils.scannet_200_labels import CLASS_LABELS_200, VALID_CLASS_IDS_200

# ScanNet200 id -> human-readable label
ID_TO_LABEL: dict[int, str] = dict(zip(VALID_CLASS_IDS_200, CLASS_LABELS_200))
LABEL_TO_ID: dict[str, int] = dict(zip(CLASS_LABELS_200, VALID_CLASS_IDS_200))

# Default Mask3D workspace: the scan data already lives here (scene.ply,
# predictions.txt once run_mask3d.sh has been run, etc.)
DEFAULT_WORKSPACE = Path(os.environ.get("HRL_DATA_DIR", "/home/ws/data"))


@dataclass
class Mask3DInstance:
    mask_file: Path
    class_id: int
    label: str
    confidence: float

    def load_mask(self) -> np.ndarray:
        """Boolean per-point mask, aligned with mesh.ply / mesh_labeled.ply."""
        return np.loadtxt(self.mask_file, dtype=np.int64).astype(bool)


def load_predictions(workspace: Path = DEFAULT_WORKSPACE, min_confidence: float = 0.0) -> list[Mask3DInstance]:
    """Read predictions.txt and return one Mask3DInstance per detection."""
    pred_file = workspace / "predictions.txt"
    if not pred_file.exists():
        raise FileNotFoundError(
            f"{pred_file} not found. Run docker/mask3d/run_mask3d.sh on this workspace first."
        )

    instances = []
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue
            rel_path, class_id, conf = parts[0], int(parts[1]), float(parts[2])
            if conf < min_confidence:
                continue
            instances.append(
                Mask3DInstance(
                    mask_file=workspace / rel_path,
                    class_id=class_id,
                    label=ID_TO_LABEL.get(class_id, f"unknown_{class_id}"),
                    confidence=conf,
                )
            )
    return instances


def load_scene_pointcloud(workspace: Path = DEFAULT_WORKSPACE) -> o3d.geometry.PointCloud:
    """Load scene.ply, the point cloud Mask3D segmented and the masks index into."""
    path = workspace / "scene.ply"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return o3d.io.read_point_cloud(str(path))


def get_instance_pointcloud(
    label: str,
    workspace: Path = DEFAULT_WORKSPACE,
    index: int = 0,
    min_confidence: float = 0.0,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """
    Extract the point cloud of the index-th instance with the given label.
    Returns (item_cloud, environment_cloud), mirroring
    stretch-compose's mask3D_interface.get_coordinates_from_item.
    """
    if label not in LABEL_TO_ID:
        raise ValueError(f"'{label}' is not a ScanNet200 label")
    class_id = LABEL_TO_ID[label]

    instances = [i for i in load_predictions(workspace, min_confidence) if i.class_id == class_id]
    if not instances:
        raise LookupError(f"No '{label}' instance found in {workspace}")
    if not (0 <= index < len(instances)):
        index = 0

    mask = instances[index].load_mask()
    pcd = load_scene_pointcloud(workspace)
    idx = np.where(mask)[0]
    return pcd.select_by_index(idx), pcd.select_by_index(idx, invert=True)