#!/usr/bin/env python3
"""
scripts/project_to_2d_map.py
-----------------------------
Slice a 3D point-cloud scan into a 2D occupancy map (.pgm + .yaml,
nav2_map_server format) by voxelising it and projecting the voxels
at a given height band onto the ground plane.

Assumes the scan is gravity-aligned (+z up), as iPad/ARKit LiDAR scans are.

Usage:
    python scripts/project_to_2d_map.py data/scene.ply
    python scripts/project_to_2d_map.py data/scene.ply --out ipad_map --resolution 0.05
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

FREE, OCCUPIED = 254, 0


def voxelise(pcd, voxel_size):
    """Discretise a point cloud into a voxel grid; return (voxel indices, grid origin)."""
    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    idx = np.array([v.grid_index for v in vg.get_voxels()])
    return idx, np.asarray(vg.origin)


def detect_floor_z(pcd, distance_threshold=0.02, ransac_n=3, num_iterations=1000):
    """RANSAC-fit the dominant plane and return its z height. Raises if it isn't ~horizontal."""
    plane, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
    normal = np.array(plane[:3])
    normal /= np.linalg.norm(normal)
    if abs(normal[2]) < 0.9:  # dominant plane is a wall, not the floor
        raise RuntimeError(
            f"Dominant RANSAC plane isn't horizontal (normal={normal}); "
            "scan may not be z-up."
        )
    return np.asarray(pcd.points)[inliers][:, 2].mean()


def height_band_mask(idx, origin, voxel_size, sensor_height, band, floor_z):
    """Mask voxels within [sensor_height - band, sensor_height + band] above the floor."""
    z = origin[2] + (idx[:, 2] + 0.5) * voxel_size  # voxel z-index -> metres
    h = z - floor_z                                  # height above floor
    return (h >= sensor_height - band) & (h <= sensor_height + band)


def to_occupancy_image(ij):
    """2D voxel columns [N,2] -> (uint8 image, (i_min, j_min) offset)."""
    offset = ij.min(axis=0)
    ij = ij - offset
    w, h = ij[:, 0].max() + 1, ij[:, 1].max() + 1

    occ = np.full((h, w), FREE, dtype=np.uint8)
    rows = (h - 1) - ij[:, 1]   # image row 0 is top; world +y is up
    occ[rows, ij[:, 0]] = OCCUPIED
    return occ, offset


def save_map(occ, origin_xy, resolution, out_stem):
    """Write <stem>.pgm (binary occupancy image) and <stem>.yaml (nav2_map_server metadata)."""
    out_stem = Path(out_stem)
    pgm = out_stem.with_suffix(".pgm")
    h, w = occ.shape

    with open(pgm, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode())
        f.write(occ.tobytes())

    with open(out_stem.with_suffix(".yaml"), "w") as f:
        yaml.safe_dump({
            "image": pgm.name,
            "mode": "trinary",
            "resolution": float(resolution),
            "origin": [float(origin_xy[0]), float(origin_xy[1]), 0.0],
            "negate": 0,
            "occupied_thresh": 0.65,
            "free_thresh": 0.25,
        }, f, sort_keys=False)

    return pgm


def project(scan_path, out_stem, resolution=0.05, sensor_height=0.20, band=0.05,
            floor_threshold=0.02):
    """Load a point cloud and write its 2D occupancy map at out_stem.{pgm,yaml}."""
    pcd = o3d.io.read_point_cloud(str(scan_path))
    floor_z = detect_floor_z(pcd, distance_threshold=floor_threshold)
    idx, origin = voxelise(pcd, resolution)

    mask = height_band_mask(idx, origin, resolution, sensor_height, band, floor_z)
    if not mask.any():
        raise RuntimeError(
            f"No voxels near {sensor_height} m above floor (floor z={floor_z:.2f}). "
            "Try adjusting --sensor-height or --band."
        )

    occ, (i_min, j_min) = to_occupancy_image(idx[mask, :2])
    origin_xy = (origin[0] + i_min * resolution, origin[1] + j_min * resolution)
    pgm = save_map(occ, origin_xy, resolution, out_stem)

    print(f"floor_z={floor_z:.2f}  voxels={mask.sum()}  "
          f"size={occ.shape[1]}x{occ.shape[0]}px  -> {pgm}")


def main():
    ap = argparse.ArgumentParser(
        description="Project a 3D point cloud to a Nav2 2D occupancy map."
    )
    ap.add_argument("scan", type=Path, help="Input .ply / .pcd file (z-up)")
    ap.add_argument("--out", default="ipad_map", help="Output file stem (default: ipad_map)")
    ap.add_argument("--resolution", type=float, default=0.05, help="Metres per pixel (default: 0.05)")
    ap.add_argument("--sensor-height", type=float, default=0.20,
                    help="2D LiDAR height above floor in metres (default: 0.20)")
    ap.add_argument("--band", type=float, default=0.05,
                    help="Slab half-thickness in metres (default: 0.05)")
    ap.add_argument("--floor-threshold", type=float, default=0.02,
                    help="RANSAC inlier distance for floor detection, in metres (default: 0.02)")
    args = ap.parse_args()
    project(args.scan, args.out, args.resolution, args.sensor_height, args.band,
            args.floor_threshold)


if __name__ == "__main__":
    main()
