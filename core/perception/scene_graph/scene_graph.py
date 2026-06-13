import datetime
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui  # type: ignore
import open3d.visualization.rendering as rendering  # type: ignore
from scipy.spatial import KDTree

from core.perception.scene_graph.graph_nodes import ObjectNode


class SceneGraph:
    """
    A graph of labeled 3D object nodes with one-way "connections" from each
    movable object to its nearest immovable neighbor (e.g. a mug -> the
    table it's on). Supports spatial queries, Open3D visualization, and JSON
    export for downstream consumers (room clustering, scene queries).
    """

    def __init__(self, label_mapping: Optional[dict] = None, min_confidence: float = 0.0,
                 k: int = 2, immovable: Optional[list] = None, pose: Optional[np.ndarray] = None):
        self.index = 0
        self.nodes: dict[int, ObjectNode] = {}
        self.labels: dict[str, list[int]] = {}
        self.outgoing: dict[int, int] = {}
        self.ingoing: dict[int, list[int]] = {}
        self.ids: list[int] = []
        self.k = k
        self.label_mapping = label_mapping or {}
        self.min_confidence = min_confidence
        self.immovable = immovable or []
        self.pose = pose
        self.tree: Optional[KDTree] = None
        self.mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.pcd: Optional[o3d.geometry.PointCloud] = None

    def add_node(self, color: tuple, sem_label: str, points: np.ndarray, mesh_mask: np.ndarray,
                  confidence: float, movable: bool = True) -> None:
        """Add a node, marking it immovable if its mapped label is in `self.immovable`."""
        if self.label_mapping.get(sem_label, "ID not found") in self.immovable:
            self.nodes[self.index] = ObjectNode(self.index, np.array([0.5, 0.5, 0.5]), sem_label, points, mesh_mask, confidence, movable=False)
        else:
            self.nodes[self.index] = ObjectNode(self.index, color, sem_label, points, mesh_mask, confidence, movable=movable)
        self.labels.setdefault(sem_label, []).append(self.index)
        self.ids.append(self.index)
        self.index += 1

    def update_connection(self, node: ObjectNode) -> None:
        """Connect a movable `node` to its nearest immovable neighbor (one outgoing edge each)."""
        min_index, min_dist = None, None
        if node.movable:
            for other in self.nodes.values():
                if not other.movable:
                    dist = np.linalg.norm(node.centroid - other.centroid)
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
                        min_index = other.object_id

        # one outgoing connection per node; keep ingoing/outgoing in sync
        tmp = self.outgoing.get(node.object_id, None)
        if min_index is not None and tmp != min_index:
            if tmp is not None:
                self.ingoing[tmp].remove(node.object_id)
            self.outgoing[node.object_id] = min_index
            self.ingoing.setdefault(min_index, []).append(node.object_id)

    def get_centroid_distance(self, point: np.ndarray) -> float:
        """Distance from `point` to the centroid of its nearest node."""
        _, idx = self.tree.query(point)
        return np.linalg.norm(point - self.nodes[self.ids[idx]].centroid)

    def query(self, point: np.ndarray) -> int:
        """Return the id of the node whose centroid is closest to `point`."""
        _, idx = self.tree.query(point)
        return self.ids[idx]

    def nearest_node(self, point: Optional[np.ndarray]) -> tuple[Optional[float], Optional[int]]:
        """Return (distance, id) of the nearest movable node's surface to `point`."""
        if point is None:
            return np.inf, None
        _, neighbor_indices = self.tree.query(point, k=4)
        neighbor_indices = [
            self.ids[n_idx]
            for n_idx in neighbor_indices
            if self.nodes[self.ids[n_idx]].movable
        ]

        if not neighbor_indices:
            return None, None

        nearest_neighbor = np.array([
            self.nodes[neighbor_idx].hull_tree.query(point, k=1)[0]
            for neighbor_idx in neighbor_indices
        ])
        return np.min(nearest_neighbor), neighbor_indices[np.argmin(nearest_neighbor)]

    def remove_node(self, remove_index: int) -> None:
        """Delete a node, patch up connections that pointed to/from it, and rebuild the KD-tree."""
        self.nodes.pop(remove_index, None)
        self.ids.remove(remove_index)
        deleted = self.outgoing.pop(remove_index, None)
        for id in self.ingoing.get(remove_index, []):
            del self.outgoing[id]
            self.update_connection(self.nodes[id])
        self.ingoing.pop(remove_index, None)
        ingoing_list = self.ingoing.get(deleted, [])
        if remove_index in ingoing_list:
            ingoing_list.remove(remove_index)
        self.tree = KDTree(np.array([self.nodes[index].centroid for index in self.ids]))

    def color_with_ibm_palette(self) -> None:
        """Assign each movable node a random color from the 10-color IBM palette."""
        colors = np.array([
            [0.39215686, 0.56078431, 1.], [0.47058824, 0.36862745, 0.94117647],
            [0.8627451, 0.14901961, 0.49803922], [0.99607843, 0.38039216, 0.],
            [1., 0.69019608, 0.], [0.29803922, 0.68627451, 0.31372549],
            [0., 0.6, 0.8], [0.70196078, 0.53333333, 1.],
            [0.89803922, 0.22352941, 0.20784314], [1., 0.25098039, 0.50588235],
        ])
        random.seed(10)
        for node in self.nodes.values():
            if node.movable:
                node.color = colors[random.randint(0, len(colors) - 1)]

    def scene_geometries(self, centroids: bool = True, connections: bool = True) -> list[tuple]:
        """Build (geometry, name, material) tuples for all nodes, plus optional centroids/edges."""
        geometries = []
        material = rendering.MaterialRecord()
        material.shader = "defaultLit"

        line_mat = rendering.MaterialRecord()
        line_mat.shader = "unlitLine"
        line_mat.line_width = 5

        for node in self.nodes.values():
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(node.points)
            pcd.paint_uniform_color(np.array(node.color, dtype=np.float64))
            geometries.append((pcd, "node_" + str(node.object_id), material))
            if node.bb is not None:
                bb = node.bb
                bb.color = (0, 0, 1)  # Blue OBBs
                geometries.append((bb, "bb_" + str(node.object_id), line_mat))

        if centroids:
            centroid_pcd = o3d.geometry.PointCloud()
            centroids_xyz = np.array([node.centroid for node in self.nodes.values()])
            centroids_colors = np.array([node.color for node in self.nodes.values()], dtype=np.float64) / 255.0
            centroid_pcd.points = o3d.utility.Vector3dVector(centroids_xyz)
            centroid_pcd.colors = o3d.utility.Vector3dVector(centroids_colors)
            geometries.append((centroid_pcd, "centroids", material))

        if connections:
            line_points = []
            line_indices = []
            idx = 0
            for start, end in self.outgoing.items():
                line_points.append(self.nodes[start].centroid)
                line_points.append(self.nodes[end].centroid)
                line_indices.append([idx, idx + 1])
                idx += 2
            if line_points:
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(line_points),
                    lines=o3d.utility.Vector2iVector(line_indices),
                )
                line_set.paint_uniform_color([0, 0, 0])
                geometries.append((line_set, "connections", line_mat))

        return geometries

    def visualize(self, centroids: bool = True, connections: bool = True, labels: bool = False, frame_center: bool = False) -> None:
        """Open an Open3D window showing all nodes, optionally with centroids, connection edges, and labels."""
        geometries = self.scene_geometries(centroids, connections)

        gui.Application.instance.initialize()
        window = gui.Application.instance.create_window("Press <S> to capture a screenshot or <ESC> to quit the application.", 1024, 1024)
        scene = gui.SceneWidget()
        scene.scene = rendering.Open3DScene(window.renderer)
        scene.scene.set_background(np.array([255.0, 255.0, 255.0, 1.0], dtype=np.float32))
        window.add_child(scene)

        for geometry, name, mat in geometries:
            scene.scene.add_geometry(name, geometry, mat)

        if frame_center:
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            scene.scene.add_geometry("Coordinate Frame", coord_frame, rendering.MaterialRecord())

        if geometries:
            bounds = geometries[0][0].get_axis_aligned_bounding_box()
            for geometry, _, _ in geometries[1:]:
                bounds += geometry.get_axis_aligned_bounding_box()
            scene.setup_camera(60, bounds, bounds.get_center())

        if labels:
            offset = np.array([0, 0, 0.01])
            for node in self.nodes.values():
                label = self.label_mapping.get(node.sem_label, "ID not found")
                scene.add_3d_label(node.centroid + offset, label)

        def on_key_event(event):
            if event.type == gui.KeyEvent.Type.DOWN:
                if event.key == gui.KeyName.S:
                    current_time = datetime.datetime.now().strftime("%m%d-%H%M%S")
                    image = gui.Application.instance.render_to_image(scene.scene, 1024, 1024)
                    o3d.io.write_image(f"screenshot_{current_time}.png", image)
                    time.sleep(0.5)
                    return True
                if event.key == gui.KeyName.ESCAPE:
                    gui.Application.instance.quit()
                    return True
            return False

        window.set_on_key(on_key_event)
        gui.Application.instance.run()

    def save_full_graph_to_json(self, file_path: str) -> None:
        """Save node ids/labels, connections, and immovable ids/labels to JSON."""
        graph_data = {
            "node_ids": self.ids,
            "node_labels": [
                self.label_mapping.get(label_id, "ID not found")
                for node_id in self.ids
                for label_id, ids in self.labels.items() if node_id in ids
            ],
            "connections": self.outgoing,
            "immovable_ids": [idx for idx, node in self.nodes.items() if not node.movable],
            "immovable_labels": [self.label_mapping.get(node.sem_label, "ID not found") for node in self.nodes.values() if not node.movable],
        }
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(graph_data, f, indent=4)

    def save_furniture_to_json(self, file_path: str) -> None:
        """Save immovable nodes as {id: {label, centroid, dimensions}} JSON."""
        scene = {
            "furniture": {
                idx: {
                    "label": self.label_mapping.get(node.sem_label, "ID not found"),
                    "centroid": node.centroid.tolist() if isinstance(node.centroid, np.ndarray) else node.centroid,
                    "dimensions": node.dimensions.tolist() if isinstance(node.dimensions, np.ndarray) else node.dimensions,
                }
                for idx, node in self.nodes.items() if not node.movable
            },
        }
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(scene, f, indent=4)

    def save_objects_to_json(self, dir_path: str) -> None:
        """Write one {id}.json per movable object node."""
        for node in self.nodes.values():
            if node.movable:
                node_data = {
                    "id": node.object_id,
                    "label": self.label_mapping.get(node.sem_label, "ID not found"),
                    "centroid": node.centroid.tolist() if isinstance(node.centroid, np.ndarray) else node.centroid,
                    "dimensions": node.dimensions.tolist() if isinstance(node.dimensions, np.ndarray) else node.dimensions,
                    "pose": node.pose.tolist() if isinstance(node.pose, np.ndarray) else node.pose,
                    "drawer": -1,
                    "confidence": node.confidence,
                }
                file_path = os.path.join(dir_path, f"{node.object_id}.json")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w') as f:
                    json.dump(node_data, f, indent=4)

    def save_all(self, graph_dir: str | Path) -> None:
        """Write graph.json, scene.json, and objects/*.json under `graph_dir`."""
        graph_dir = Path(graph_dir)
        graph_dir.mkdir(parents=True, exist_ok=True)
        self.save_full_graph_to_json(str(graph_dir / "graph.json"))
        self.save_furniture_to_json(str(graph_dir / "scene.json"))
        self.save_objects_to_json(str(graph_dir / "objects"))
        print(f"[save] Scene graph saved to {graph_dir}")
