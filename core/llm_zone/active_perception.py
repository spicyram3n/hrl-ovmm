"""
active_perception.py
--------------------
Given a target centroid from scene_query, generates candidate viewpoints
around the target so the robot can actively search for occluded objects.

The caller flow:
    1. scene_query.query(...)  → result with centroid + object_id
    2. navigate to approach_pose(centroid)
    3. call is_visible() — your camera/detection callback
    4. if not visible: for pose in viewpoints(centroid): navigate, check again
    5. if still not found: report failure

Nothing here talks to ROS directly — it only does geometry.
Wire it into good_boy.py or your behaviour tree.
"""

import math
from typing import Callable, Iterator


def approach_pose(
    centroid: list[float],
    standoff: float = 0.8,
    face_target: bool = True,
) -> dict:
    """
    Returns the primary nav goal: a pose directly in front of the target.

    standoff: metres to stay away from the centroid (robot footprint + margin)
    face_target: if True, the pose yaw points toward the centroid
    """
    x, y = centroid[0], centroid[1]
    # Approach from the +X side by default; real code should pick the
    # nearest reachable direction using a costmap query.
    approach_x = x - standoff
    approach_y = y
    yaw = math.atan2(y - approach_y, x - approach_x)  # pointing at target
    qz = math.sin(yaw / 2)
    qw = math.cos(yaw / 2)
    return {"x": approach_x, "y": approach_y, "qz": qz, "qw": qw}


def viewpoints(
    centroid: list[float],
    radius: float = 0.9,
    n: int = 6,
    start_angle_deg: float = 0.0,
) -> Iterator[dict]:
    """
    Yields n evenly-spaced viewpoints on a circle of `radius` metres around
    the centroid, each oriented to face the centroid.

    Use these one by one as Nav2 goals when the approach pose doesn't reveal
    the object (it may be occluded from that angle).

    radius: orbit radius in metres
    n:      number of candidate viewpoints (6 covers 60° gaps)
    """
    cx, cy = centroid[0], centroid[1]
    for i in range(n):
        angle = math.radians(start_angle_deg + i * 360.0 / n)
        vx = cx + radius * math.cos(angle)
        vy = cy + radius * math.sin(angle)
        # Face toward centroid
        yaw = math.atan2(cy - vy, cx - vx)
        qz = math.sin(yaw / 2)
        qw = math.cos(yaw / 2)
        yield {"x": vx, "y": vy, "qz": qz, "qw": qw}


def active_search(
    centroid: list[float],
    is_visible: Callable[[], bool],
    navigate_to: Callable[[dict], bool],
    radius: float = 0.9,
    n: int = 6,
) -> bool:
    """
    High-level active perception loop.

    centroid:    [x, y, z] from scene_query result
    is_visible:  callback → True if the object is currently detected by camera
    navigate_to: callback(pose_dict) → True if nav succeeded, False if failed/cancelled
    radius:      orbit radius around target
    n:           number of viewpoints to try

    Returns True if the object was found, False after exhausting all viewpoints.

    Example (pseudocode in good_boy.py):

        from core.llm_zone.scene_query import query
        from core.llm_zone.active_perception import approach_pose, active_search

        result = query("a cup")
        if result["present"]:
            navigate_to(approach_pose(result["centroid"]))
            if not is_visible():
                found = active_search(result["centroid"], is_visible, navigate_to)
    """
    for pose in viewpoints(centroid, radius=radius, n=n):
        ok = navigate_to(pose)
        if not ok:
            continue  # nav failed (obstacle etc.), try next
        if is_visible():
            return True
    return False
