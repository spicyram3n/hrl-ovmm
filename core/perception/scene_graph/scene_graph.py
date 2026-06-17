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
    A graph of labeled 3D object nodes with directed edges from each movable
    object to the furniture piece that supports it.

    Node types
    ----------
    - movable   : graspable objects (cup, pringles can, …)
    - immovable : furniture anchors (shelf, desk, …)

    Edges
    -----
    Each movable node has exactly one outgoing edge to an immovable node
    (its supporting furniture), determined by update_connection().

    Build paths
    -----------
    Gazebo simulation (ground-truth geometry):
        build_scene_graph_gazebo.build_from_world(graph, registry) → graph.save_all()

    Real robot (Mask3D 3D instance segmentation):
        build_scene_graph.build_from_mask3d(graph, workspace) → graph.save_all()

    Parameters
    ----------
    label_mapping : optional dict mapping sem_label values → human-readable strings.
        Needed for Mask3D (integer ScanNet200 IDs). Pass None when labels are
        already strings (Gazebo path).
    immovable     : list of resolved label strings that should be treated as furniture.
    """

    def __init__(self, label_mapping: Optional[dict] = None,
                 min_confidence: float = 0.0,
                 immovable: Optional[list] = None,
                 pose: Optional[np.ndarray] = None):
        self.index = 0
        self.nodes: dict[int, ObjectNode] = {}
        self.outgoing: dict[int, int] = {}       # movable_id  → furniture_id
        self.ingoing:  dict[int, list[int]] = {} # furniture_id → [movable_ids]
        self.ids: list[int] = []
        self.label_mapping = label_mapping       # None means labels are already strings
        self.min_confidence = min_confidence
        self.immovable = immovable or []
        self.pose = pose
        self.tree: Optional[KDTree] = None
        self.mesh = None  # set by build_from_mask3d for visualization
        self.pcd  = None

    # ── Label helpers ─────────────────────────────────────────────────────────

    def _resolve(self, sem_label) -> str:
        """Return the human-readable label for sem_label."""
        if self.label_mapping:
            return self.label_mapping.get(sem_label, str(sem_label))
        return str(sem_label)

    # ── Core graph operations ─────────────────────────────────────────────────

    def add_node(self, color: tuple, sem_label, points: np.ndarray,
                 mesh_mask, confidence: float, movable: bool = True) -> None:
        """
        Add an object node.

        Forces immovable (grey) if the resolved label appears in self.immovable.
        """
        if self._resolve(sem_label) in self.immovable:
            movable = False
            color = np.array([0.5, 0.5, 0.5])
        self.nodes[self.index] = ObjectNode(
            self.index, color, sem_label, points, mesh_mask, confidence, movable=movable
        )
        self.ids.append(self.index)
        self.index += 1

    def update_connection(self, node: ObjectNode) -> None:
        """
        Connect a movable node to its supporting furniture.

        Scoring uses XY proximity combined with a Z support heuristic:
        furniture whose centroid is above the object is penalised, so a shelf
        the object rests on scores better than a taller shelf beside it.

        Score = XY_distance + max(0, furniture_z - object_z)
        """
        if not node.movable:
            return

        best_id, best_score = None, np.inf
        for furn in self.nodes.values():
            if furn.movable:
                continue
            xy_dist = np.linalg.norm(node.centroid[:2] - furn.centroid[:2])
            z_penalty = max(0.0, furn.centroid[2] - node.centroid[2])
            score = xy_dist + z_penalty
            if score < best_score:
                best_score = score
                best_id = furn.object_id

        if best_id is None:
            return

        old = self.outgoing.get(node.object_id)
        if old == best_id:
            return
        if old is not None:
            self.ingoing[old].remove(node.object_id)
        self.outgoing[node.object_id] = best_id
        self.ingoing.setdefault(best_id, []).append(node.object_id)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_full_graph_to_json(self, file_path: str) -> None:
        """Write graph.json: node ids/labels, connections, movable/immovable lists."""
        data = {
            "node_ids":         self.ids,
            "node_labels":      [self._resolve(self.nodes[i].sem_label) for i in self.ids],
            "connections":      self.outgoing,
            "movable_ids":      [i for i, n in self.nodes.items() if n.movable],
            "immovable_ids":    [i for i, n in self.nodes.items() if not n.movable],
            "immovable_labels": [self._resolve(n.sem_label)
                                 for n in self.nodes.values() if not n.movable],
        }
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    def save_furniture_to_json(self, file_path: str) -> None:
        """Write scene.json: immovable nodes with label, centroid, dimensions."""
        scene = {
            "furniture": {
                idx: {
                    "label":      self._resolve(node.sem_label),
                    "centroid":   node.centroid.tolist(),
                    "dimensions": node.dimensions.tolist(),
                }
                for idx, node in self.nodes.items() if not node.movable
            }
        }
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(scene, f, indent=4)

    def save_objects_to_json(self, dir_path: str) -> None:
        """Write one {id}.json per movable node."""
        for node in self.nodes.values():
            if not node.movable:
                continue
            data = {
                "id":         node.object_id,
                "label":      self._resolve(node.sem_label),
                "centroid":   node.centroid.tolist(),
                "dimensions": node.dimensions.tolist(),
                "pose":       node.pose.tolist(),
                "drawer":     -1,
                "confidence": node.confidence,
            }
            path = os.path.join(dir_path, f"{node.object_id}.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=4)

    def save_all(self, graph_dir: str | Path) -> None:
        """Write graph.json, scene.json, and objects/*.json under graph_dir."""
        graph_dir = Path(graph_dir)
        graph_dir.mkdir(parents=True, exist_ok=True)
        self.save_full_graph_to_json(str(graph_dir / "graph.json"))
        self.save_furniture_to_json(str(graph_dir / "scene.json"))
        self.save_objects_to_json(str(graph_dir / "objects"))
        print(f"[save] Scene graph saved to {graph_dir}")

    # ── Spatial queries (real-robot stack; not used in the Gazebo demo) ───────

    def get_centroid_distance(self, point: np.ndarray) -> float:
        _, idx = self.tree.query(point)
        return float(np.linalg.norm(point - self.nodes[self.ids[idx]].centroid))

    def nearest_node_id(self, point: np.ndarray) -> int:
        """Return the id of the node whose centroid is closest to point."""
        _, idx = self.tree.query(point)
        return self.ids[idx]

    def nearest_movable(self, point: Optional[np.ndarray]) -> tuple[Optional[float], Optional[int]]:
        """Return (surface_distance, node_id) of the nearest movable node."""
        if point is None or self.tree is None:
            return None, None
        _, idxs = self.tree.query(point, k=min(4, len(self.ids)))
        movable = [self.ids[i] for i in idxs if self.nodes[self.ids[i]].movable]
        if not movable:
            return None, None
        dists = np.array([self.nodes[i].hull_tree.query(point)[0] for i in movable])
        best = int(np.argmin(dists))
        return float(dists[best]), movable[best]

    def remove_node(self, remove_index: int) -> None:
        """Delete a node, re-wire its edges, and rebuild the KD-tree."""
        self.nodes.pop(remove_index, None)
        self.ids.remove(remove_index)
        old_target = self.outgoing.pop(remove_index, None)
        for mid in self.ingoing.pop(remove_index, []):
            self.outgoing.pop(mid, None)
            self.update_connection(self.nodes[mid])
        if old_target is not None:
            lst = self.ingoing.get(old_target, [])
            if remove_index in lst:
                lst.remove(remove_index)
        if self.ids:
            self.tree = KDTree(np.array([self.nodes[i].centroid for i in self.ids]))

    # ── Visualization ─────────────────────────────────────────────────────────

    def color_with_ibm_palette(self) -> None:
        """Assign each movable node a distinct color from the IBM 10-color palette."""
        palette = np.array([
            [0.392, 0.561, 1.],    [0.471, 0.369, 0.941],
            [0.863, 0.149, 0.498], [0.996, 0.380, 0.],
            [1.,   0.690, 0.],     [0.298, 0.686, 0.314],
            [0.,   0.600, 0.800],  [0.702, 0.533, 1.],
            [0.898, 0.224, 0.208], [1.,   0.251, 0.506],
        ])
        random.seed(10)
        for node in self.nodes.values():
            if node.movable:
                node.color = palette[random.randint(0, len(palette) - 1)]

    def scene_geometries(self, centroids: bool = True,
                         connections: bool = True) -> list[tuple]:
        """Build (geometry, name, material) tuples for all nodes."""
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        line_mat = rendering.MaterialRecord()
        line_mat.shader = "unlitLine"
        line_mat.line_width = 5

        geometries = []
        for node in self.nodes.values():
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(node.points)
            pcd.paint_uniform_color(np.array(node.color, dtype=np.float64))
            geometries.append((pcd, f"node_{node.object_id}", mat))
            if node.bb is not None:
                bb = node.bb
                bb.color = (0, 0, 1)
                geometries.append((bb, f"bb_{node.object_id}", line_mat))

        if centroids:
            cpcd = o3d.geometry.PointCloud()
            cpcd.points = o3d.utility.Vector3dVector(
                np.array([n.centroid for n in self.nodes.values()]))
            cpcd.colors = o3d.utility.Vector3dVector(
                np.clip(np.array([n.color for n in self.nodes.values()],
                                 dtype=np.float64), 0, 1))
            geometries.append((cpcd, "centroids", mat))

        if connections and self.outgoing:
            pts, lines = [], []
            for k, (src, dst) in enumerate(self.outgoing.items()):
                pts += [self.nodes[src].centroid, self.nodes[dst].centroid]
                lines.append([2 * k, 2 * k + 1])
            ls = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(pts),
                lines=o3d.utility.Vector2iVector(lines),
            )
            ls.paint_uniform_color([0, 0, 0])
            geometries.append((ls, "connections", line_mat))

        return geometries

    def visualize(self, centroids: bool = True, connections: bool = True,
                  labels: bool = False, frame_center: bool = False) -> None:
        """Open an interactive Open3D window. Press S to screenshot, ESC to quit."""
        geometries = self.scene_geometries(centroids, connections)

        gui.Application.instance.initialize()
        win = gui.Application.instance.create_window(
            "Scene Graph — S: screenshot | ESC: quit", 1024, 1024)
        widget = gui.SceneWidget()
        widget.scene = rendering.Open3DScene(win.renderer)
        widget.scene.set_background(np.array([255., 255., 255., 1.], dtype=np.float32))
        win.add_child(widget)

        for geom, name, m in geometries:
            widget.scene.add_geometry(name, geom, m)

        if frame_center:
            cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            widget.scene.add_geometry("origin", cf, rendering.MaterialRecord())

        if geometries:
            bounds = geometries[0][0].get_axis_aligned_bounding_box()
            for g, _, _ in geometries[1:]:
                bounds += g.get_axis_aligned_bounding_box()
            widget.setup_camera(60, bounds, bounds.get_center())

        if labels:
            for node in self.nodes.values():
                widget.add_3d_label(node.centroid + [0, 0, 0.01],
                                    self._resolve(node.sem_label))

        def on_key(event):
            if event.type == gui.KeyEvent.Type.DOWN:
                if event.key == gui.KeyName.S:
                    ts = datetime.datetime.now().strftime("%m%d-%H%M%S")
                    img = gui.Application.instance.render_to_image(widget.scene, 1024, 1024)
                    o3d.io.write_image(f"screenshot_{ts}.png", img)
                    time.sleep(0.5)
                    return True
                if event.key == gui.KeyName.ESCAPE:
                    gui.Application.instance.quit()
                    return True
            return False

        win.set_on_key(on_key)
        gui.Application.instance.run()
