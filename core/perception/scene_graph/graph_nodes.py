"""ObjectNode: a single labeled object/furniture instance in a SceneGraph."""

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree


class ObjectNode:
    """
    A single object in the scene graph, built from a set of 3D points.

    Computes and stores:
      - centroid: mean of all points
      - pose: 4x4 transform (PCA orientation + centroid translation)
      - dimensions: (width, depth, height) from the oriented bounding box
      - hull_tree: KDTree over points for nearest-surface distance queries
    """

    def __init__(self, object_id: int, color: tuple, sem_label: str,
                 points: np.ndarray, mesh_mask: np.ndarray,
                 confidence: float = None, movable: bool = True):
        self.object_id = object_id
        self.color = color
        self.sem_label = sem_label
        self.points = points
        self.mesh_mask = mesh_mask
        self.confidence = confidence
        self.movable = movable

        self.centroid = np.mean(points, axis=0)
        self.hull_tree = KDTree(points)
        self._compute_pose()
        self._compute_dimensions()

    def _compute_pose(self) -> None:
        """4×4 pose via PCA on the centred points (largest variance = x-axis)."""
        centered = self.points - self.centroid
        _, vecs = np.linalg.eigh(np.cov(centered, rowvar=False))
        R = vecs[:, ::-1]           # descending eigenvalue order
        if np.linalg.det(R) < 0:
            R[:, -1] *= -1          # ensure right-handed frame
        self.pose = np.eye(4)
        self.pose[:3, :3] = R
        self.pose[:3, 3] = self.centroid

    def _compute_dimensions(self) -> None:
        """Minimal oriented bounding box → (width, depth, height)."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        obb = pcd.get_minimal_oriented_bounding_box()
        height_idx = np.argmax(np.abs(obb.R.T @ [0, 0, 1]))
        wd = sorted([i for i in range(3) if i != height_idx],
                    key=lambda i: obb.extent[i], reverse=True)
        self.bb = obb
        self.dimensions = obb.extent[wd + [height_idx]]

    def transform(self, T: np.ndarray) -> None:
        """Apply a translation (3,), rotation (3,3), or 4×4 transform in place."""
        if T.shape == (3,):
            self.centroid += T
            self.points += T
            self.pose[:3, 3] += T
        elif T.shape == (3, 3):
            self.points = (T @ self.points.T).T
            self.centroid = T @ self.centroid
            self.pose[:3, :3] = T @ self.pose[:3, :3]
        elif T.shape == (4, 4):
            ones = np.ones((len(self.points), 1))
            self.points = (T @ np.hstack([self.points, ones]).T).T[:, :3]
            self.centroid = (T @ np.append(self.centroid, 1))[:3]
            self.pose = T @ self.pose
        else:
            raise ValueError(f"Expected shape (3,), (3,3) or (4,4); got {T.shape}")
        self.hull_tree = KDTree(self.points)
        self._compute_dimensions()
