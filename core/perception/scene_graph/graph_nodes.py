"""ObjectNode: a single labeled object/furniture instance in a SceneGraph."""

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree


class ObjectNode:
    """
    A single object in the scene graph, derived from a set of 3D points.

    Stores the object's centroid, a PCA-based pose, and its minimal oriented
    bounding box (dimensions), and keeps a KD-tree over its points for
    nearest-surface queries.
    """

    def __init__(self, object_id: int, color: tuple, sem_label: str, points: np.ndarray,
                 mesh_mask: np.ndarray, confidence: float = None, movable: bool = True):
        self.object_id = object_id
        self.color = color
        self.sem_label = sem_label
        self.centroid = np.mean(points, axis=0)
        self.points = points
        self.mesh_mask = mesh_mask
        self.confidence = confidence
        self.movable = movable
        self.misplaced = False

        self.update_hull_tree()
        self.compute_pose(self.points, self.centroid)
        self.get_dimensions()

    def update_hull_tree(self) -> None:
        """Rebuild the KD-tree over `points` (used for nearest-surface queries)."""
        self.hull_tree = KDTree(self.points)

    def compute_pose(self, points: np.ndarray, centroid: np.ndarray) -> None:
        """Estimate orientation via PCA on the centered points and store it as a 4x4 pose."""
        points_centered = points - centroid
        covariance_matrix = np.cov(points_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_idx]
        R = eigenvectors
        if np.linalg.det(R) < 0:
            R[:, -1] *= -1
        object_pose = np.eye(4)
        object_pose[:3, :3] = R
        object_pose[:3, 3] = centroid
        self.pose = object_pose

    def get_dimensions(self) -> None:
        """Compute the minimal oriented bounding box and (width, depth, height) dimensions."""
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(self.points)

        obb = point_cloud.get_minimal_oriented_bounding_box()
        height_idx = np.argmax(np.abs(obb.R.T @ [0, 0, 1]))
        width_depth = sorted([i for i in range(3) if i != height_idx], key=lambda i: obb.extent[i], reverse=True)
        order = width_depth + [height_idx]

        self.bb = obb
        self.dimensions = obb.extent[order]

    def transform(self, transformation: np.ndarray) -> None:
        """Apply a translation (3,), rotation (3,3), or homogeneous transform (4,4) in place."""
        if not isinstance(transformation, np.ndarray):
            raise TypeError("Invalid argument type. Expected numpy.ndarray.")

        if transformation.shape == (3,):
            self.centroid += transformation
            self.points += transformation
            self.pose[:3, 3] += transformation
        elif transformation.shape == (3, 3):
            self.points = np.dot(transformation, self.points.T).T
            self.centroid = np.dot(transformation, self.centroid)
            self.pose = np.dot(transformation, self.pose[:3, :3])
        elif transformation.shape == (4, 4):
            self.points = np.dot(transformation, np.vstack((self.points.T, np.ones(self.points.shape[0])))).T[:, :3]
            self.centroid = np.dot(transformation, np.append(self.centroid, 1))[:3]
            self.pose = np.dot(transformation, self.pose)
        else:
            raise ValueError("Invalid argument shape. Expected (3,) for translation, (3,3) for rotation, or (4,4) for homogeneous transformation.")

        self.update_hull_tree()
        self.get_dimensions()
