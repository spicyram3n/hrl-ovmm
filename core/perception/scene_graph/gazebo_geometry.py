"""
gazebo_geometry.py
-------------------
Pure geometry helpers that read object size and shape directly from Gazebo
model files (SDF), so build_scene_graph_gazebo.py never has to hardcode
per-object dimensions.

For a model referenced as `model://<name>` in a world file, `model_local_bbox`
returns the model's axis-aligned bounding box (`dims`) and its center offset
relative to the model origin (`center`), both expressed in the model's own
(unrotated) local frame:

    1. Resolve `model://<name>` to a directory via GAZEBO_MODEL_PATH and read
       model.config to find the actual <model>.sdf file.
    2. For every <link>, walk its <collision> shapes (falling back to
       <visual> if a link has no collisions). Each shape is one of
       box / cylinder / sphere / mesh; for each we compute the 8 corners of
       its local bounding box.
    3. Compose the shape's pose with its link's pose (and the model's own
       pose, if any) to bring those corners into the model frame.
    4. The union of all corners over all links/shapes gives the model's
       overall bounding box: dims = max - min, center = (max + min) / 2.

Mesh shapes (.dae / .stl, referenced from <mesh><uri>...</uri>) are loaded
with `trimesh` to get their local bounding box, scaled by the SDF's
<mesh><scale>, and (for COLLADA files) by the <asset><unit meter="..."> the
file declares - some assets are authored in decimetres and would otherwise be
off by 10x.

Everything here is read from files; nothing is hand-picked per object.
"""

from __future__ import annotations

import itertools
import os
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

COLLADA_NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}

# 8 sign combinations (+-1) for turning half-extents into box corners.
_CORNER_SIGNS = np.array(list(itertools.product([-1, 1], repeat=3)))


def euler_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix for SDF's <pose> roll/pitch/yaw (R = Rz @ Ry @ Rx)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def pose_to_matrix(pose_text: Optional[str]) -> np.ndarray:
    """Parse an SDF <pose>x y z roll pitch yaw</pose> into a 4x4 transform."""
    T = np.eye(4)
    if pose_text is None:
        return T
    x, y, z, roll, pitch, yaw = (float(v) for v in pose_text.split())
    T[:3, :3] = euler_to_matrix(roll, pitch, yaw)
    T[:3, 3] = [x, y, z]
    return T


def _gazebo_model_paths() -> list[Path]:
    return [Path(p) for p in os.environ.get("GAZEBO_MODEL_PATH", "").split(":") if p]


def resolve_uri(uri: str) -> Path:
    """Resolve a `model://<name>[/<rest>]` URI via GAZEBO_MODEL_PATH."""
    if not uri.startswith("model://"):
        return Path(uri)
    name, _, sub = uri[len("model://"):].partition("/")
    for base in _gazebo_model_paths():
        candidate = base / name
        if candidate.is_dir():
            return candidate / sub if sub else candidate
    raise FileNotFoundError(f"Could not resolve '{uri}' via GAZEBO_MODEL_PATH")


def _model_sdf_path(model_dir: Path) -> Path:
    config = ET.parse(model_dir / "model.config").getroot()
    return model_dir / config.findtext("sdf")


def _box_corners(size_text: str) -> np.ndarray:
    half_extent = np.array([float(v) for v in size_text.split()]) / 2.0
    return _CORNER_SIGNS * half_extent


def _cylinder_corners(geom: ET.Element) -> np.ndarray:
    r = float(geom.findtext("radius"))
    length = float(geom.findtext("length"))
    return _CORNER_SIGNS * np.array([r, r, length / 2.0])


def _sphere_corners(geom: ET.Element) -> np.ndarray:
    r = float(geom.findtext("radius"))
    return _CORNER_SIGNS * r


def _collada_unit_meters(dae_path: Path) -> float:
    """COLLADA <asset><unit meter="X"/> - some assets are authored in
    decimetres/centimetres; 1.0 if the file doesn't declare a unit."""
    unit = ET.parse(dae_path).getroot().find(".//c:asset/c:unit", COLLADA_NS)
    return float(unit.get("meter", "1.0")) if unit is not None else 1.0


@lru_cache(maxsize=None)
def _mesh_local_bounds(mesh_path: str) -> tuple[np.ndarray, np.ndarray]:
    """(min, max) of a mesh file's own vertices, in metres."""
    path = Path(mesh_path)
    mesh = trimesh.load(str(path), force="mesh")
    if path.suffix.lower() == ".dae":
        mesh.apply_scale(_collada_unit_meters(path))
    return mesh.bounds[0].copy(), mesh.bounds[1].copy()


def _mesh_corners(geom: ET.Element) -> np.ndarray:
    path = resolve_uri(geom.findtext("uri"))
    scale = np.array([float(v) for v in (geom.findtext("scale") or "1 1 1").split()])
    bmin, bmax = _mesh_local_bounds(str(path))
    bmin, bmax = bmin * scale, bmax * scale
    return np.array(list(itertools.product(*zip(bmin, bmax))))


def _shape_corners(geometry: ET.Element) -> Optional[np.ndarray]:
    """8 local-frame corner points of a <geometry>'s box/cylinder/sphere/mesh,
    or None for unsupported shapes (e.g. <plane>, used for floors/walls)."""
    shape = next(iter(geometry), None)
    if shape is None:
        return None
    if shape.tag == "box":
        return _box_corners(shape.findtext("size"))
    if shape.tag == "cylinder":
        return _cylinder_corners(shape)
    if shape.tag == "sphere":
        return _sphere_corners(shape)
    if shape.tag == "mesh":
        return _mesh_corners(shape)
    return None


@lru_cache(maxsize=None)
def model_local_bbox(model_uri: str) -> tuple[np.ndarray, np.ndarray]:
    """
    For a `model://<name>` URI, return (dims, center): the axis-aligned
    bounding box of the model's collision geometry (dims) and its center
    offset from the model origin (center), both in the model's own
    unrotated local frame. Cached per URI - sibling instances (e.g. four
    `model://kitchen_chair` includes) reuse the same result.
    """
    model_dir = resolve_uri(model_uri)
    model = ET.parse(_model_sdf_path(model_dir)).getroot().find("model")
    model_pose = pose_to_matrix(model.findtext("pose"))

    corners = []
    for link in model.findall("link"):
        link_pose = model_pose @ pose_to_matrix(link.findtext("pose"))
        for shape in link.findall("collision") or link.findall("visual"):
            local_corners = _shape_corners(shape.find("geometry"))
            if local_corners is None:
                continue
            shape_pose = link_pose @ pose_to_matrix(shape.findtext("pose"))
            corners.append(local_corners @ shape_pose[:3, :3].T + shape_pose[:3, 3])

    if not corners:
        raise ValueError(f"No usable collision/visual geometry in {model_uri}")

    points = np.concatenate(corners, axis=0)
    bbox_min, bbox_max = points.min(axis=0), points.max(axis=0)
    return bbox_max - bbox_min, (bbox_min + bbox_max) / 2.0
