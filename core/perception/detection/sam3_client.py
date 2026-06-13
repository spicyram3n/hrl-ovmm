"""
sam3_client.py
--------------
Thin HTTP client for the SAM3 docker server (docker/sam3).

SAM3 needs Python >= 3.12 / PyTorch >= 2.7 / CUDA >= 12.6, so it cannot run
inside the ROS 2 Humble devcontainer (Python 3.10). It runs in its own
container on port 5005 and we talk to it over REST, same pattern as the
OpenMask3D and AnyGrasp servers.

Pipeline this fits into (per video frame, from a ROS 2 node):

    rgb, depth, K  <- camera topics (color + aligned depth + CameraInfo)
    det = client.detect(rgb, "<query from LLM/user>")
    if det.found:
        mask = det.best_mask()                       # (H, W) bool
        pcd  = mask_to_pointcloud(mask, depth, K)     # 3D recon of the object
        # pcd -> grasp pose, e.g. via a separate AnyGrasp-style REST server

A single Sam3Client instance should be reused across frames (it keeps a
persistent HTTP connection to the server), not recreated per call.

Usage:
    from core.perception.detection.sam3_client import Sam3Client
    client = Sam3Client()
    det = client.detect(rgb_image, "red cup")
    if det.found:
        mask = det.best_mask()   # (H, W) bool, lift to 3D with the depth image
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import cv2
import numpy as np
import requests


@dataclass
class Sam3Detection:
    prompt: str
    masks: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0), dtype=bool))
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    scores: np.ndarray = field(default_factory=lambda: np.zeros((0,)))

    @property
    def found(self) -> bool:
        return self.masks.shape[0] > 0

    def best(self) -> int:
        return int(np.argmax(self.scores))

    def best_mask(self) -> np.ndarray:
        return self.masks[self.best()]

    def best_box(self) -> np.ndarray:
        return self.boxes[self.best()]


class Sam3Client:
    def __init__(self, host: str = "localhost", port: int = 5005, timeout: int = 120):
        self.url = f"http://{host}:{port}/sam3/predict"
        self.timeout = timeout
        # Reuse one connection across calls — this client is meant to be
        # called once per video frame.
        self.session = requests.Session()

    def detect(self, image_bgr: np.ndarray, prompt: str, conf: float = 0.5) -> Sam3Detection:
        """image_bgr: OpenCV image (as delivered by cv_bridge)."""
        ok, encoded = cv2.imencode(".jpg", image_bgr)
        if not ok:
            raise ValueError("Could not encode image")

        response = self.session.post(
            self.url,
            files={"image": ("frame.jpg", encoded.tobytes(), "image/jpeg")},
            params={"prompt": prompt, "conf": conf},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"SAM3 server error {response.status_code}: {response.text}")

        meta = json.loads(response.headers["X-Sam3-Meta"])
        n, h, w = meta["num_instances"], meta["height"], meta["width"]
        body = response.content
        mask_bytes, box_bytes = n * h * w, n * 4 * 4  # bool=1B, float32=4B

        det = Sam3Detection(
            prompt=prompt,
            masks=np.frombuffer(body[:mask_bytes], dtype=bool).reshape(n, h, w),
            boxes=np.frombuffer(body[mask_bytes:mask_bytes + box_bytes], dtype=np.float32).reshape(n, 4),
            scores=np.frombuffer(body[mask_bytes + box_bytes:], dtype=np.float32).reshape(n),
        )
        print(f"[sam3] '{prompt}': {n} instance(s)")
        return det


def mask_to_pointcloud(
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    depth_scale: float = 1000.0,
):
    """
    Lift a SAM3 mask + aligned depth image to a 3D point cloud in the
    camera frame (step 6: build the RGBD point cloud of the object).

    mask:       (H, W) bool
    depth:      (H, W) uint16/float, aligned to the RGB image
    intrinsics: 3x3 camera matrix K
    """
    import open3d as o3d

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    v, u = np.where(mask)
    z = depth[v, u].astype(np.float64) / depth_scale
    valid = z > 0
    u, v, z = u[valid], v[valid], z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd
