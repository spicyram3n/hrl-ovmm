"""
active_perception.py
--------------------
Geometry helpers for navigating around a target location.

Used by search_and_fetch.py:
  - approach_pose(centroid) → primary nav goal in front of the target
  - viewpoints(centroid, n) → orbit of n poses around the target for when
    the object is occluded from the approach angle

Nothing here talks to ROS — pure geometry that returns Nav2-compatible
pose dicts {x, y, qz, qw}.
"""

import math
from typing import Iterator


def approach_pose(centroid: list[float], standoff: float = 0.8) -> dict:
    """
    Primary nav goal: a pose standoff metres in front of the target, facing it.

    Approaches from the -X direction by default, facing back along +X (yaw=0).
    A costmap-aware version would pick the nearest unoccupied approach
    direction instead, computing yaw from the actual approach vector.
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
        yaw = math.atan2(cy - vy, cx - vx)
        yield {"x": vx, "y": vy, "qz": math.sin(yaw / 2), "qw": math.cos(yaw / 2)}
