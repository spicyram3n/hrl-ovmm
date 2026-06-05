"""
Open-vocabulary 3D segmentation using OpenMask3D CLIP features.
Query a scene pointcloud with text and get the matching point cluster.

Usage:
    q = OpenMaskQuery(scene_ply=Path("scene.ply"), features_dir=Path("openmask_features"))
    pcd_obj, pcd_bg = q.query("cup", visualize=True)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import clip
from sklearn.cluster import DBSCAN

from core.perception.segmentation.openmask_server import fetch_features, load_cached_features


class OpenMaskQuery:
    def __init__(
        self,
        scene_ply: Path,
        features_dir: Path,
        data_dir: Path | None = None,
        clip_model: str = "ViT-L/14@336px",
        device: str = "cpu",
    ) -> None:
        self.scene_ply = scene_ply
        self.features_dir = features_dir
        self.data_dir = data_dir
        self.device = device

        print(f"[OpenMaskQuery] Loading CLIP model '{clip_model}' on {device} ...")
        self.model, _ = clip.load(clip_model, device=device)

        self.features: np.ndarray | None = None
        self.masks: np.ndarray | None = None
        self.pcd: o3d.geometry.PointCloud | None = None

    def load_features(
        self, scene_name: str = "hrl_scene", intrinsic_res: str = "[1440,1920]"
    ) -> None:
        feat_path = self.features_dir / "clip_features_comp.npy"
        if feat_path.exists():
            self.features, self.masks = load_cached_features(self.features_dir)
        else:
            if self.data_dir is None:
                raise ValueError("data_dir must be set to fetch features from server.")
            self.features, self.masks = fetch_features(
                data_dir=self.data_dir,
                features_dir=self.features_dir,
                scene_name=scene_name,
                intrinsic_res=intrinsic_res,
            )

    def load_pointcloud(self) -> None:
        print(f"[OpenMaskQuery] Loading pointcloud from {self.scene_ply} ...")
        self.pcd = o3d.io.read_point_cloud(str(self.scene_ply))
        print(f"[OpenMaskQuery] {len(np.asarray(self.pcd.points))} points loaded.")

    def _encode_text(self, text: str) -> torch.Tensor:
        tokens = clip.tokenize([text.lower()]).to(self.device)
        with torch.no_grad():
            return self.model.encode_text(tokens)

    def _get_best_mask(self, text: str, rank: int = 0) -> tuple[np.ndarray, float]:
        text_feat = self._encode_text(text)
        cos_sim = torch.nn.functional.cosine_similarity(
            torch.Tensor(self.features).to(self.device), text_feat, dim=1
        )
        idx = torch.argsort(cos_sim, descending=True)[rank].item()
        score = cos_sim[idx].item()
        print(f"[OpenMaskQuery] '{text}': mask_idx={idx}, score={score:.3f}")
        return self.masks[:, idx].astype(bool), score

    def _largest_cluster(
        self, point_indices: np.ndarray, eps: float = 0.05, min_samples: int = 10
    ) -> np.ndarray:
        pts = np.asarray(self.pcd.points)[point_indices]
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit(pts).labels_
        # exclude noise points (label == -1) before picking largest cluster
        valid = labels[labels >= 0]
        if len(valid) == 0:
            return point_indices
        best_label = np.bincount(valid).argmax()
        return point_indices[labels == best_label]

    def query(
        self,
        text: str,
        rank: int = 0,
        dbscan_eps: float = 0.05,
        dbscan_min_samples: int = 10,
        visualize: bool = False,
    ) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        if self.features is None or self.masks is None:
            raise RuntimeError("Call load_features() first.")
        if self.pcd is None:
            raise RuntimeError("Call load_pointcloud() first.")

        mask, _ = self._get_best_mask(text, rank=rank)
        sel_idx = np.where(mask)[0]
        final_idx = self._largest_cluster(sel_idx, dbscan_eps, dbscan_min_samples)

        print(f"[OpenMaskQuery] '{text}': {len(final_idx)} points in cluster")

        pcd_obj = self.pcd.select_by_index(final_idx)
        pcd_bg = self.pcd.select_by_index(final_idx, invert=True)

        if visualize:
            pcd_obj.paint_uniform_color([1, 0, 1])
            o3d.visualization.draw_geometries(
                [pcd_obj, pcd_bg], window_name=f"OpenMask3D — {text}"
            )

        return pcd_obj, pcd_bg

    def query_multi(
        self, queries: list[str], visualize: bool = False
    ) -> dict[str, o3d.geometry.PointCloud]:
        return {q: self.query(q, visualize=visualize)[0] for q in queries}


if __name__ == "__main__":
    DATA_DIR = Path("/home/ws/data")
    FEATURES_DIR = DATA_DIR / "openmask_features"
    FEATURES_DIR.mkdir(exist_ok=True)

    querier = OpenMaskQuery(
        scene_ply=DATA_DIR / "scene.ply", features_dir=FEATURES_DIR, data_dir=DATA_DIR
    )
    querier.load_features(scene_name="hrl_scene", intrinsic_res="[1440,1920]")
    querier.load_pointcloud()

    results = querier.query_multi(
        queries=["chair", "cup", "table", "door"], visualize=True
    )
    for name, pcd in results.items():
        print(f"{name}: {len(np.asarray(pcd.points))} points")
