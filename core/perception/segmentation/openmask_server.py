"""
Handles communication with the OpenMask3D Docker server:
  - Zip the scene directory and POST to /openmask/save_and_predict
  - Unpack the zip response into numpy arrays
  - Save raw and deduplicated features to disk
"""

import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import requests

PORT = 5001
SERVER_URL = f"http://localhost:{PORT}/openmask/save_and_predict"
TIMEOUT = 7200  # seconds — openmask is slow on large scenes


def zip_scene(data_dir: Path, output_path: Path) -> Path:
    """Zip the scene directory, skipping .zip files and the openmask_features cache."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[openmask_server] Zipping {data_dir} ...")
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(data_dir.rglob("*")):
            if (
                f.is_file()
                and f.suffix != ".zip"
                and "openmask_features" not in f.parts
            ):
                zf.write(f, f.relative_to(data_dir))
    size_mb = output_path.stat().st_size / 1e6
    print(f"[openmask_server] Zip ready: {output_path.name} ({size_mb:.1f} MB)")
    return output_path


def _unpack_response(response: requests.Response, save_dir: Path) -> dict:
    """Unpack the server's zip response into a dict of numpy arrays."""
    save_dir.mkdir(parents=True, exist_ok=True)
    contents = {}
    with zipfile.ZipFile(io.BytesIO(response.content), "r") as zf:
        for entry in sorted(zf.filelist, key=lambda x: x.filename):
            name, ext = os.path.splitext(entry.filename)
            extracted = zf.extract(entry.filename, save_dir)
            if ext == ".npy":
                contents[name] = np.load(extracted)
            elif ext == ".json":
                with open(extracted) as f:
                    contents[name] = json.load(f)
    return contents


def fetch_features(
    data_dir: Path,
    features_dir: Path,
    scene_name: str = "hrl_scene",
    intrinsic_res: str = "[1440,1920]",
) -> tuple[np.ndarray, np.ndarray]:
    """Upload scene to OpenMask3D server and save CLIP features + masks to disk."""
    zip_path = zip_scene(data_dir, data_dir / "scene_upload.zip")

    # server reads each param as [type_str, value_str] via args.getlist(key)
    params = [
        ("name", "str"),
        ("name", scene_name),
        ("overwrite", "bool"),
        ("overwrite", "False"),
        ("scene_intrinsic_resolution", "str"),
        ("scene_intrinsic_resolution", intrinsic_res),
    ]

    print(f"[openmask_server] POSTing to {SERVER_URL} (may take several minutes) ...")
    with open(zip_path, "rb") as f:
        response = requests.post(
            SERVER_URL, files={"scene": f}, params=params, timeout=TIMEOUT
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"[openmask_server] Server error {response.status_code}:\n{response.text}"
        )

    contents = _unpack_response(response, features_dir / "raw")
    features = contents["clip_features"]  # (N, 768)
    masks = contents["scene_MASKS"]  # (N_pts, N)

    np.save(features_dir / "clip_features.npy", features)
    np.save(features_dir / "scene_MASKS.npy", masks)

    # deduplicate: remove identical feature vectors, then identical mask columns
    features, feat_idx = np.unique(features, axis=0, return_index=True)
    masks = masks[:, feat_idx]
    masks, mask_idx = np.unique(masks, axis=1, return_index=True)
    features = features[mask_idx]

    np.save(features_dir / "clip_features_comp.npy", features)
    np.save(features_dir / "scene_MASKS_comp.npy", masks)

    print(f"[openmask_server] Done. features={features.shape}, masks={masks.shape}")
    return features, masks


def load_cached_features(features_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load deduplicated features from disk."""
    feat_path = features_dir / "clip_features_comp.npy"
    mask_path = features_dir / "scene_MASKS_comp.npy"

    if not feat_path.exists() or not mask_path.exists():
        raise FileNotFoundError(
            f"[openmask_server] Cached features not found in {features_dir}. Run fetch_features() first."
        )

    features = np.load(feat_path)
    masks = np.load(mask_path)
    print(
        f"[openmask_server] Loaded cache: features={features.shape}, masks={masks.shape}"
    )
    return features, masks
