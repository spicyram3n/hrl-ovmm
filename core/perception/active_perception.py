"""
active_perception.py
--------------------
Geometry helpers for navigating around a target location.

Used by search_and_fetch.py:
  - reachable_approach_pose(centroid, costmap) → nav goal within arm reach
    of the target, picked from the live costmap instead of a guessed offset
  - approach_pose(centroid) → fixed-standoff fallback, used when no costmap
    is available (dry runs, or the costmap service being unreachable)
  - viewpoints(centroid, n) → orbit of n poses around the target for when
    the object is occluded from the approach angle

Nothing here talks to ROS — pure geometry (plus CostmapGrid, a plain data
holder) that returns Nav2-compatible pose dicts {x, y, qz, qw}. The ROS
adapter converts nav2_msgs/Costmap into a CostmapGrid before calling
reachable_approach_pose().
"""

import math
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

# nav2_costmap_2d cost values (Costmap.msg, the raw 0-255 layer, not the
# 0-100 normalized OccupancyGrid view): 253 = inscribed-inflated obstacle,
# 254 = lethal, 255 = unknown. Anything below that is traversable, just
# increasingly discouraged the closer it is to an obstacle.
COSTMAP_BLOCKED_THRESHOLD = 253


def _facing_pose(cx: float, cy: float, vx: float, vy: float) -> dict:
    """Pose dict at (vx, vy) facing back toward (cx, cy). Shared by every
    function below that places the robot on a circle around a target."""
    yaw = math.atan2(cy - vy, cx - vx)
    return {"x": vx, "y": vy, "qz": math.sin(yaw / 2), "qw": math.cos(yaw / 2)}


@dataclass
class CostmapGrid:
    """Minimal costmap representation, decoupled from any ROS message type.

    data: (size_y, size_x) uint8, row-major, nav2_costmap_2d raw cost values
    (see COSTMAP_BLOCKED_THRESHOLD above).
    origin_x/origin_y: world-frame coordinates of cell (0, 0) - i.e.
    Costmap.metadata.origin.position, same frame the centroid must be in.
    """
    data: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float

    def cost_at(self, x: float, y: float) -> Optional[int]:
        """Cost at a world-frame point, or None if outside the grid."""
        col = int((x - self.origin_x) / self.resolution)
        row = int((y - self.origin_y) / self.resolution)
        if 0 <= row < self.data.shape[0] and 0 <= col < self.data.shape[1]:
            return int(self.data[row, col])
        return None


def reachable_approach_pose(centroid: list[float], costmap: CostmapGrid,
                            reach_min: float = 0.4, reach_max: float = 0.65,
                            n_radii: int = 4, n_angles: int = 36) -> Optional[dict]:
    """
    Nav goal within arm reach of centroid, instead of a guessed standoff:
    scans the annulus [reach_min, reach_max] around centroid for the
    nearest unblocked costmap cell, checking radii closest-first so the
    tightest comfortable spot wins over a farther one. reach_min/reach_max
    should bracket the working range of the grasp arm-joint formula
    (robot_adapter._grasp_pose_to_joints) - they're tunable, not a hard
    robot spec, since that formula is an analytic approximation, not real
    IK.

    Returns a pose dict {x, y, qz, qw} (same frame as centroid/costmap)
    facing the centroid, or None if every cell in the band is blocked or
    off the grid - the caller should fall back to approach_pose() or try
    the next candidate location.
    """
    cx, cy = centroid[0], centroid[1]
    for radius in np.linspace(reach_min, reach_max, n_radii):
        for i in range(n_angles):
            angle = 2 * math.pi * i / n_angles
            vx = cx + radius * math.cos(angle)
            vy = cy + radius * math.sin(angle)
            cost = costmap.cost_at(vx, vy)
            if cost is not None and cost < COSTMAP_BLOCKED_THRESHOLD:
                return _facing_pose(cx, cy, vx, vy)
    return None


def approach_pose(centroid: list[float], standoff: float = 0.8) -> dict:
    """
    Fallback nav goal: a pose standoff metres in front of the target, facing
    it, used when reachable_approach_pose() has no costmap to work with
    (dry runs) or found no free cell in its reach band.

    Approaches from the -X direction by default, facing back along +X
    (yaw=0) - with no costmap, there's no way to know if that's actually
    clear, so this is only a last resort.
    """
    x, y = centroid[0], centroid[1]
    ax, ay = x - standoff, y
    return {"x": ax, "y": ay, "qz": 0.0, "qw": 1.0}


def viewpoints(centroid: list[float], radius: float = 0.9,
               n: int = 6, start_angle_deg: float = 0.0) -> Iterator[dict]:
    """
    Yield n evenly-spaced poses on a circle of `radius` metres around the
    centroid, each facing inward toward it.

    Used when the approach pose doesn't reveal the object (occlusion).
    n=6 gives 60° gaps, which is enough to cover most household furniture.
    """
    cx, cy = centroid[0], centroid[1]
    for i in range(n):
        angle = math.radians(start_angle_deg + i * 360.0 / n)
        vx = cx + radius * math.cos(angle)
        vy = cy + radius * math.sin(angle)
        yield _facing_pose(cx, cy, vx, vy)
