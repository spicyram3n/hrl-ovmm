"""
visualize_scene_graph.py
------------------------
Standalone Open3D viewer: loads scene.ply as the background point cloud
and overlays the scene graph (boxes, spheres, edges, labels) from JSONs.
No rebuild needed.

Usage:
    python core/perception/scene_graph/visualize_scene_graph.py
    Press ESC to quit.
"""

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

DATA_DIR        = Path("/home/ws/data")
GRAPH_DIR       = DATA_DIR / "scene_graph"
SCENE_PLY       = DATA_DIR / "scene.ply"

FURNITURE_COLOR = [0.4, 0.6, 0.9]   # blue boxes
OBJECT_COLOR    = [1.0, 0.5, 0.1]   # orange spheres
EDGE_COLOR      = [0.2, 0.2, 0.2]   # dark lines


def load():
    with open(GRAPH_DIR / "graph.json") as f:
        graph = json.load(f)
    with open(GRAPH_DIR / "scene.json") as f:
        scene = json.load(f)
    objects = {}
    for p in sorted((GRAPH_DIR / "objects").glob("*.json")):
        with open(p) as f:
            obj = json.load(f)
            objects[obj["id"]] = obj
    return graph, {int(k): v for k, v in scene["furniture"].items()}, objects


def box(centroid, dims, color):
    cx, cy, cz = centroid
    w, d, h = dims
    mesh = o3d.geometry.TriangleMesh.create_box(w, d, h)
    mesh.translate([cx - w / 2, cy - d / 2, cz - h / 2])
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def sphere(centroid, color, r=0.06):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=r)
    mesh.translate(centroid)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def lineset(pts, idxs, color):
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(idxs),
    )
    ls.paint_uniform_color(color)
    return ls


def visualize():
    graph, furniture, objects = load()

    geos = []

    # background point cloud
    print("Loading scene.ply …")
    pcd = o3d.io.read_point_cloud(str(SCENE_PLY))
    geos.append(pcd)

    # furniture boxes
    for fdata in furniture.values():
        geos.append(box(fdata["centroid"], fdata["dimensions"], FURNITURE_COLOR))

    # object spheres
    for odata in objects.values():
        geos.append(sphere(odata["centroid"], OBJECT_COLOR))

    # connection lines
    pts, idxs, i = [], [], 0
    for oid, odata in objects.items():
        conn = graph["connections"].get(str(oid))
        if conn is not None and conn in furniture:
            pts += [odata["centroid"], furniture[conn]["centroid"]]
            idxs.append([i, i + 1])
            i += 2
    if pts:
        geos.append(lineset(pts, idxs, EDGE_COLOR))

    # --- Open3D GUI ---
    gui.Application.instance.initialize()
    win = gui.Application.instance.create_window("Scene Graph  |  ESC to quit", 1200, 900)
    widget = gui.SceneWidget()
    widget.scene = rendering.Open3DScene(win.renderer)
    widget.scene.set_background(np.array([1, 1, 1, 1], dtype=np.float32))
    win.add_child(widget)

    pcd_mat = rendering.MaterialRecord()
    pcd_mat.shader = "defaultUnlit"
    pcd_mat.point_size = 2.0

    mesh_mat = rendering.MaterialRecord()
    mesh_mat.shader = "defaultLit"

    line_mat = rendering.MaterialRecord()
    line_mat.shader = "unlitLine"
    line_mat.line_width = 3

    for idx, geo in enumerate(geos):
        if isinstance(geo, o3d.geometry.PointCloud):
            mat = pcd_mat
        elif isinstance(geo, o3d.geometry.LineSet):
            mat = line_mat
        else:
            mat = mesh_mat
        widget.scene.add_geometry(f"geo_{idx}", geo, mat)

    for fdata in furniture.values():
        widget.add_3d_label(np.array(fdata["centroid"]) + [0, 0, 0.1], fdata["label"])
    for odata in objects.values():
        widget.add_3d_label(np.array(odata["centroid"]) + [0, 0, 0.07], odata["label"])

    bounds = geos[0].get_axis_aligned_bounding_box()
    for geo in geos[1:]:
        bounds += geo.get_axis_aligned_bounding_box()
    widget.setup_camera(60, bounds, bounds.get_center())

    def on_key(event):
        if event.type == gui.KeyEvent.Type.DOWN and event.key == gui.KeyName.ESCAPE:
            gui.Application.instance.quit()
            return True
        return False
    win.set_on_key(on_key)

    gui.Application.instance.run()


if __name__ == "__main__":
    visualize()
