"""
anygrasp_client.py
-------------------
Client for the AnyGrasp/GraspNet docker server (docker/anygrasp).

Ports the grasp request/filter/top-k logic from the supervisor's
graspnet_interface.py (originally from spot-compose), but talks to the
server via core/utils/rest_client.py's RestClient instead of spot-compose's
utils/docker_communication.py - same pattern as sam3_client.py.

The server's wire protocol is fixed by the prebuilt graspnet:v1.0 image
(docker/anygrasp/run_anygrasp.sh) and can't be changed on our end:

    POST http://<ip>:<port>/graspnet/predict
        files:  points.npy, colors.npy, limits.npy, rotations.npy  (np.save'd arrays)
        params: max_gripper_width, gripper_height, top_n, vis, top_down_grasp
                (each value sent as a ("<type>", <value>) pair - the server
                reads the repeated query param to know how to cast it)
    response 200: zip of tf_matricess.npy, scoress.npy, widthss.npy
                  (shape (num_rotations, top_n[, 4, 4|...])); invalid slots
                  have score == -1
             204: no valid grasp found
             408: request timed out

Pipeline this fits into, mirroring sam3_client.py:

    item_pcd, env_pcd <- mask -> point cloud (e.g. SAM3 + depth, or Mask3D)
    grasps = AnygraspClient().predict(item_pcd, env_pcd, limits)
    if grasps:
        best = grasps[0]   # best.tf_matrix (4x4), best.width, best.score
"""

from __future__ import annotations

import copy
import heapq
import io
import math
import zipfile
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from core.utils.rest_client import RestClient

# AnyGrasp's network was trained at a fixed gripper scale; stretch-compose's
# graspnet_interface.py scales clouds in/out of model space accordingly.
# Real gripper max width is 0.175m, but the network expects ~0.10-0.12m.
SCALE = 0.1 / 0.15
MAX_GRIPPER_WIDTH = 0.12
GRIPPER_HEIGHT = 0.31 * SCALE


@dataclass
class Grasp:
    tf_matrix: np.ndarray  # 4x4, same frame as the input clouds
    width: float
    score: float


def _npy_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array)
    return buf.getvalue()


def _ensure_colors(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Callers like sam3_client.mask_to_pointcloud don't set colors; the
    server still expects a colors.npy aligned with points.npy."""
    if not pcd.has_colors():
        pcd.colors = o3d.utility.Vector3dVector(np.ones((len(pcd.points), 3)))
    return pcd


def _polar_to_cartesian(vector: np.ndarray) -> np.ndarray:
    r, theta, phi = vector[..., 0], vector[..., 1], vector[..., 2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=-1)


def _uniform_sphere_directions(resolution: int = 4) -> np.ndarray:
    """
    Unit-vector directions uniformly distributed on a sphere.
    Ported from the supervisor's utils/coordinates.py get_uniform_sphere_directions
    (their bosdyn-stripped copy - note its theta range is 0..2pi, not the 0..pi in
    the public spot-compose repo this was originally based on).
    """
    assert resolution % 2 == 0, "resolution must be even"
    half = resolution // 2
    r = np.ones((half, half + 1))
    theta = -np.linspace(0, 2 * math.pi, half + 1)
    phi = np.linspace(0, math.pi, half + 1)[:-1]
    thetas, phis = np.meshgrid(theta, phi)
    directions = np.stack((r, thetas, phis), axis=-1)
    directions = np.transpose(directions, (1, 0, 2))

    opposite_directions = np.copy(directions)
    opposite_directions[..., 2] += math.pi

    full_directions = np.concatenate([directions, opposite_directions], axis=1)
    return _polar_to_cartesian(full_directions)


def _remove_duplicate_rows(arr: np.ndarray, tolerance: float = 1e-5) -> np.ndarray:
    """Ported from the supervisor's utils/coordinates.py remove_duplicate_rows."""
    rounded = np.round(arr / tolerance)
    view = rounded.view(dtype=[("f0", rounded.dtype), ("f1", rounded.dtype), ("f2", rounded.dtype)])
    _, unique_idx = np.unique(view, return_index=True)
    return rounded[unique_idx]


def _rotation_from_direction(direction: np.ndarray, invariant_direction=(0.0, 0.0, 1.0)) -> np.ndarray:
    """
    Ported from the supervisor's utils/coordinates.py _rotation_from_direction,
    specialized to the only call shape graspnet_interface.py uses (roll=0,
    degrees=False, invert=True). Returns a rotation matrix R such that
    R @ invariant_direction ~= -direction (R's 3rd column is -direction).
    """
    start_vector = np.asarray(invariant_direction).reshape((1, 3))
    end_vector = -np.asarray(direction).reshape((1, 3))
    rotation, _ = Rotation.align_vectors(end_vector, start_vector)
    return rotation.as_matrix()


def sphere_rotation_matrices(resolution: int = 24) -> np.ndarray:
    """
    Rotation matrices to query AnyGrasp from `resolution` viewing angles
    uniformly distributed on a sphere, instead of a single straight-on view.
    Ported from the supervisor's graspnet_interface._get_rotation_matrices.
    """
    directions = _uniform_sphere_directions(resolution).reshape((-1, 3))
    directions = _remove_duplicate_rows(directions, tolerance=1e-5)
    return np.stack([_rotation_from_direction(d) for d in directions])


def compute_limits(item_cloud: o3d.geometry.PointCloud, height_cap: float | None = None) -> np.ndarray:
    """(2, 3) [mins, maxs] bounding box of item_cloud, as used by predict()'s
    default `limits`. Ported from the supervisor's predict_full_grasp; the
    original hardcoded a 1.15m height_cap tied to Spot's arm reach, so here
    it's opt-in instead of assumed for HSR."""
    points = np.asarray(item_cloud.points)
    mins, maxs = points.min(axis=0), points.max(axis=0)
    limits = np.stack([mins, maxs], axis=0)
    if height_cap is not None:
        limits[1, 2] = min(limits[1, 2], height_cap)
    return limits


def _filter_candidates(
    tf_matricess: np.ndarray,
    scoress: np.ndarray,
    item_points: np.ndarray,
    limits: np.ndarray,
    thresh: float = 0.02,
) -> list[tuple[int, int]]:
    """
    Ported as-is from the supervisor's graspnet_interface._filter (the
    angle-with-vertical semantics aren't documented upstream, so the
    `2 * thresh` reuse is kept rather than reinterpreted). Keeps only
    candidates with a valid score that fall within `limits`, are close to
    the item point cloud, and aren't near-vertical.
    """
    mins, maxs = limits
    center = np.array([0.0, 0.0, 0.0, 1.0])
    keep = []
    for idx_rot, tf_matrices in enumerate(tf_matricess):
        for idx_nr, tf_matrix in enumerate(tf_matrices):
            if scoress[idx_rot, idx_nr] == -1:
                continue
            point = tf_matrix @ center
            point = point[:3] / point[3]
            if np.any((point < mins) | (point > maxs)):
                continue
            distances = np.linalg.norm(item_points - point, axis=1)
            if distances.min() >= thresh:
                continue
            grasp_direction = tf_matrix[:3, 2]
            angle_with_vertical = np.arccos(np.abs(grasp_direction[2]))
            if angle_with_vertical < 2 * thresh:
                continue
            keep.append((idx_rot, idx_nr))
    return keep


class AnygraspClient(RestClient):
    def __init__(self, **kwargs):
        super().__init__(server="anygrasp", **kwargs)

    def _request(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        limits: np.ndarray,
        rotations: np.ndarray,
        max_gripper_width: float,
        gripper_height: float,
        top_n: int,
        vis: bool,
        top_down_grasp: bool,
        timeout: int,
    ) -> dict:
        files = {
            "points": ("points.npy", _npy_bytes(points), "application/octet-stream"),
            "colors": ("colors.npy", _npy_bytes(colors), "application/octet-stream"),
            "limits": ("limits.npy", _npy_bytes(limits), "application/octet-stream"),
            "rotations": ("rotations.npy", _npy_bytes(rotations), "application/octet-stream"),
        }
        params = {
            "max_gripper_width": ("float", max_gripper_width),
            "gripper_height": ("float", gripper_height),
            "top_n": ("int", top_n),
            "vis": ("bool", vis),
            "top_down_grasp": ("bool", top_down_grasp),
        }
        response = self.session.post(self.url, files=files, params=params, timeout=timeout)
        if response.status_code == 204:
            return {}
        if response.status_code == 408:
            raise TimeoutError(f"{self.url} timed out")
        if response.status_code != 200:
            raise RuntimeError(f"{self.url} -> {response.status_code}: {response.text}")

        contents = {}
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            for name in zf.namelist():
                if not name.endswith(".npy"):
                    continue  # .ply meshes (vis=True only) aren't needed here
                contents[name[: -len(".npy")]] = np.load(io.BytesIO(zf.read(name)))
        return contents

    def predict(
        self,
        item_cloud: o3d.geometry.PointCloud,
        env_cloud: o3d.geometry.PointCloud,
        limits: np.ndarray | None = None,
        rotations: np.ndarray | None = None,
        rotation_resolution: int = 24,
        top_n: int = 5,
        n_best: int = 1,
        top_down_grasp: bool = True,
        timeout: int = 90,
    ) -> list[Grasp]:
        """
        limits: (2, 3) array [mins, maxs] in the input clouds' frame -
                restricts where a candidate grasp point may lie. Defaults to
                item_cloud's bounding box (see compute_limits).
        rotations: (R, 3, 3) viewing-angle rotations to query the network
                   from. Defaults to `rotation_resolution` viewpoints sampled
                   uniformly over a sphere (see sphere_rotation_matrices) -
                   pass np.eye(3).reshape(1, 3, 3) for a single straight-on
                   view instead.
        Returns up to n_best Grasp candidates, filtered and rescaled back to
        the input clouds' units, best score first. Empty if the server found
        nothing (204) or nothing passed the filter.
        """
        if limits is None:
            limits = compute_limits(item_cloud)
        assert limits.shape == (2, 3)
        if rotations is None:
            rotations = sphere_rotation_matrices(rotation_resolution)
        assert rotations.shape[-2:] == (3, 3)

        center = np.zeros((3, 1))
        scaled_item = _ensure_colors(copy.deepcopy(item_cloud)).scale(SCALE, center)
        scaled_env = _ensure_colors(copy.deepcopy(env_cloud)).scale(SCALE, center)
        scaled_limits = np.asarray(limits) * SCALE

        merged = scaled_item + scaled_env
        contents = self._request(
            points=np.asarray(merged.points),
            colors=np.asarray(merged.colors),
            limits=scaled_limits,
            rotations=rotations,
            max_gripper_width=MAX_GRIPPER_WIDTH,
            gripper_height=GRIPPER_HEIGHT,
            top_n=top_n,
            vis=False,
            top_down_grasp=top_down_grasp,
            timeout=timeout,
        )
        if not contents:
            print("[anygrasp] server returned no candidates (204)")
            return []

        tf_matricess, scoress, widthss = contents["tf_matricess"], contents["scoress"], contents["widthss"]
        candidates = _filter_candidates(tf_matricess, scoress, np.asarray(scaled_item.points), scaled_limits)
        if not candidates:
            print("[anygrasp] no candidates passed filtering")
            return []

        best = heapq.nlargest(n_best, candidates, key=lambda c: scoress[c])

        grasps = []
        for idx_rot, idx_nr in best:
            tf = tf_matricess[idx_rot, idx_nr].copy()
            tf[:3, 3] /= SCALE
            grasps.append(
                Grasp(
                    tf_matrix=tf,
                    width=float(widthss[idx_rot, idx_nr] / SCALE),
                    score=float(scoress[idx_rot, idx_nr]),
                )
            )
        print(f"[anygrasp] {len(grasps)} grasp(s) (best score {grasps[0].score:.3f})")
        return grasps