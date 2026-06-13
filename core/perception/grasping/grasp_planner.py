"""
grasp_planner.py
-----------------
Simple geometric grasp planner: a top-down grasp centred on the object
point cloud's centroid. No model, no docker server — just numpy.

Returns a dict-shaped grasp pose so this can be swapped for a learned
planner (e.g. AnyGrasp) later without changing the caller
(ros2_ws/src/fetcher/fetcher/robot_adapter.py).
"""

import numpy as np

# Hand pointing straight down (HSR convention; matches the low/tabletop
# example pose in hsrb_ik_solver_node/example/solve_ik.py).
TOP_DOWN_QUAT = (1.0, 0.0, 0.0, 0.0)

# Metres above the centroid for the pre-grasp approach height.
PREGRASP_HEIGHT = 0.05


def compute_grasp_pose(object_pcd) -> dict:
    """object_pcd: open3d PointCloud in the head camera frame."""
    centroid = np.asarray(object_pcd.points).mean(axis=0)
    return {
        "position_camera": centroid.tolist(),
        "orientation": TOP_DOWN_QUAT,
        "pregrasp_height": PREGRASP_HEIGHT,
    }
