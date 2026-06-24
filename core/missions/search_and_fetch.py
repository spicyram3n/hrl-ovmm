"""
search_and_fetch.py
-------------------
Single-command mission runner for demo steps 3 to 7:

    3. Scenario A: target not in scene graph -> LLM predicts top-k locations
       Scenario B: target in scene graph -> use its location
    4. Navigate to the first location
    5. Observe with the head camera, detect with SAM3
    6. If detected: lift mask + depth to an RGBD point cloud, plan grasp, move arm
    7. If not detected: next predicted location; in scenario B, fall back to
       LLM prediction after the known location fails

Pure-Python core: all robot I/O goes through the RobotAdapter protocol below.
The ROS 2 implementation (Nav2 BasicNavigator, head trajectory controller,
camera topics, grasp service) lives in ros2_ws and is injected at runtime.

Run (inside the devcontainer, with sim + Nav2 + SAM3 docker up):

    DEEPSEEK_API_KEY=<key> python -m core.missions.search_and_fetch "cup"
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from core.reasoning import scene_query
from core.perception.active_perception import viewpoints
from core.perception.detection.sam3_client import Sam3Client, mask_to_pointcloud

# Clearance *beyond* a candidate's own approach_radius (see scene_query.
# _approach_radius) - not a standoff distance by itself. A single fixed
# standoff measured from a furniture's centroid doesn't generalize: it's too
# close for furniture with real depth (confirmed in practice - the robot
# ended up right against a table's edge) and would be needlessly far for
# small objects. So the actual reach band passed to navigate_near() is
# CLEARANCE_MIN/MAX + the candidate's own approach_radius, computed per
# candidate in _reach_band() below - these two constants are the only part
# that's still a tunable guess (roughly the robot's own footprint radius,
# nav2_params.yaml, plus a small margin), and they apply uniformly regardless
# of which furniture is involved.
CLEARANCE_MIN = 0.3
CLEARANCE_MAX = 0.55


def _reach_band(cand: dict) -> tuple[float, float]:
    """(reach_min, reach_max) for navigate_near(), scaled to this
    candidate's actual furniture size instead of one global constant."""
    radius = cand.get("approach_radius", 0.0)
    return CLEARANCE_MIN + radius, CLEARANCE_MAX + radius


class RobotAdapter(Protocol):
    """Thin interface the ROS 2 layer must implement."""

    def navigate_to(self, pose: dict) -> bool:
        """Send a Nav2 goal {'x','y','qz','qw'} in Gazebo world-frame ground-truth
        coordinates (as produced by the scene graph); the adapter converts to the
        Nav2 map frame internally. Blocks until done."""
        ...

    def navigate_near(self, centroid: list[float], reach_min: float, reach_max: float) -> bool:
        """Navigate to within [reach_min, reach_max] metres of a world-frame
        ground-truth point, picking a free, reachable standoff from the live
        costmap instead of a fixed offset (falls back to a fixed standoff if
        no costmap is available). Blocks until done."""
        ...

    def look_at(self, point_world: list[float]) -> None:
        """Point the head camera at a 3D world-frame ground-truth point (active
        perception); the adapter converts to the map frame internally."""
        ...

    def localize_point(self, point_camera: list[float]) -> Optional[list[float]]:
        """Convert a 3D point in the head camera's own frame to a world-frame
        point, via a live TF lookup of the robot's actual current pose (not
        the nav goal it was aiming for). Used to re-aim at where the target
        actually is after a first detection, instead of trusting that
        navigation landed exactly on the planned approach pose. Returns None
        if the lookup fails."""
        ...

    def get_pose(self) -> Optional[list[float]]:
        """Robot's current position in world-frame ground truth, via a live
        TF lookup, or None if unavailable. Used to pick the nearest match
        when a query matches multiple identically-labeled objects (e.g. two
        pringles cans) - the LLM has no spatial awareness of where the robot
        currently is, so without this it picks between them arbitrarily."""
        ...

    def get_rgbd(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (rgb_bgr, depth_aligned, K) from the head camera."""
        ...

    def grasp(self, object_pcd) -> bool:
        """Plan and execute a grasp on the object point cloud (camera frame)."""
        ...


@dataclass
class MissionResult:
    found: bool
    grasped: bool
    location_label: Optional[str] = None
    tried: int = 0


def _observe_and_detect(robot: RobotAdapter, sam3: Sam3Client, target: str,
                        centroid: list[float]) -> Optional[tuple]:
    """Look at the target location, run SAM3; return (mask, depth, K) or None.

    The first look_at() aims at the a-priori centroid (from the scene graph or
    the LLM), not at the object itself - if navigation stopped a bit off from
    the planned approach pose, that first detection can be off-center. Rather
    than trust it directly, re-aim at the detection's own centroid and detect
    once more, so the point cloud handed to grasp() comes from a centered,
    just-captured view of the real object instead of the original estimate.
    """
    robot.look_at(centroid)
    rgb, depth, K = robot.get_rgbd()
    det = sam3.detect(rgb, target)
    if not det.found:
        return None

    refined = _recenter_and_redetect(robot, sam3, target, det.best_mask(), depth, K)
    if refined:
        return refined
    return det.best_mask(), depth, K


def _recenter_and_redetect(robot: RobotAdapter, sam3: Sam3Client, target: str,
                           mask: np.ndarray, depth: np.ndarray, K: np.ndarray) -> Optional[tuple]:
    """Re-aim the head at the actual centroid of `mask` (lifted to a 3D point
    and converted to world frame) and detect again. Returns None - falling
    back to the original detection - if the centroid can't be localized or
    the object isn't found in the re-centered view (e.g. it was at the edge
    of the frame and panned out of view; rare, since we're aiming *more*
    precisely at it, but the original detection is still valid either way)."""
    points = np.asarray(mask_to_pointcloud(mask, depth, K).points)
    if len(points) == 0:
        return None
    center_world = robot.localize_point(points.mean(axis=0).tolist())
    if center_world is None:
        return None

    robot.look_at(center_world)
    rgb2, depth2, K2 = robot.get_rgbd()
    det2 = sam3.detect(rgb2, target)
    if not det2.found:
        return None
    return det2.best_mask(), depth2, K2


def _search_location(robot: RobotAdapter, sam3: Sam3Client, target: str,
                     centroid: list[float], reach_min: float, reach_max: float,
                     n_viewpoints: int = 3) -> Optional[tuple]:
    """Navigate to a location and try the approach pose plus a few viewpoints."""
    if not robot.navigate_near(centroid, reach_min, reach_max):
        return None
    hit = _observe_and_detect(robot, sam3, target, centroid)
    if hit:
        return hit
    for pose in viewpoints(centroid, n=n_viewpoints):
        if not robot.navigate_to(pose):
            continue
        hit = _observe_and_detect(robot, sam3, target, centroid)
        if hit:
            return hit
    return None


def _print_candidates(candidates: list[dict]) -> None:
    """Show every candidate location and the LLM's reasoning for it, not
    just the first one we're about to try - otherwise candidates 2..k are
    invisible unless candidate 1 fails."""
    for rank, cand in enumerate(candidates, start=1):
        label = cand.get("furniture_label", "?")
        reasoning = cand.get("reasoning", "")
        print(f"[mission]   #{rank} {label}" + (f" - {reasoning}" if reasoning else ""))


def run(target: str, robot: RobotAdapter, k: int = 3) -> MissionResult:
    sam3 = Sam3Client()
    result = MissionResult(found=False, grasped=False)

    # Step 3: scenario selection
    answer = scene_query.query(target, robot_position=robot.get_pose())
    candidates: list[dict] = []
    if answer.get("present") and answer.get("centroid"):
        # Scenario B: the object is already in the scene graph, so this is a
        # grounded lookup of its one known location - not a top-k guess, by
        # design. Print the LLM's reasoning for *which* item it matched, since
        # the query can be open-vocabulary (e.g. "something to drink" -> mug).
        print(f"[mission] Scenario B: '{target}' is in the scene graph "
              f"({answer['object_label']}) - {answer.get('reasoning', '')}")
        candidates.append({
            "centroid": answer["centroid"],
            "furniture_label": answer.get("object_label", "known location"),
            "approach_radius": answer.get("approach_radius", 0.0),
        })
    else:
        print(f"[mission] Scenario A: '{target}' not in scene graph, asking LLM "
              f"for top-{k} locations")
        candidates = scene_query.predict_locations(target, k=k)
        _print_candidates(candidates)

    visited = set()
    while candidates:
        cand = candidates.pop(0)
        centroid = cand.get("centroid")
        label = cand.get("furniture_label", "?")
        if centroid is None or tuple(np.round(centroid, 2)) in visited:
            continue
        visited.add(tuple(np.round(centroid, 2)))
        result.tried += 1
        print(f"[mission] Step 4: navigating to '{label}' at {centroid}")

        reach_min, reach_max = _reach_band(cand)
        hit = _search_location(robot, sam3, target, centroid, reach_min, reach_max)
        if hit:
            mask, depth, K = hit
            print(f"[mission] Step 6: '{target}' detected at '{label}', grasping")
            object_pcd = mask_to_pointcloud(mask, depth, K)
            result.found = True
            result.location_label = label
            result.grasped = robot.grasp(object_pcd)
            return result

        print(f"[mission] Step 7: '{target}' not at '{label}'")
        if not candidates:
            # Scenario B fallback (and scenario A exhaustion): ask the LLM,
            # skipping locations we already visited
            more = scene_query.predict_locations(target, k=k)
            candidates = [
                m for m in more
                if m.get("centroid")
                and tuple(np.round(m["centroid"], 2)) not in visited
            ]
            if candidates:
                print(f"[mission] LLM proposed {len(candidates)} new location(s)")
                _print_candidates(candidates)

    print(f"[mission] '{target}' not found after {result.tried} location(s)")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m core.missions.search_and_fetch <target object>")
        sys.exit(1)

    # The real adapter lives in ros2_ws (Nav2 BasicNavigator + head controller
    # + camera subscribers + grasp client). Import it here when running live:
    #
    #   from hsr_adapter.robot import HsrRobotAdapter
    #   robot = HsrRobotAdapter()
    #
    # For a dry run without a robot, this mock just logs the calls.
    class _DryRunAdapter:
        def navigate_to(self, pose):
            print(f"  [dry-run] navigate_to {pose}")
            return True
        def navigate_near(self, centroid, reach_min, reach_max):
            print(f"  [dry-run] navigate_near {centroid} reach=[{reach_min}, {reach_max}]")
            return True
        def look_at(self, point):
            print(f"  [dry-run] look_at {point}")
        def localize_point(self, point_camera):
            print(f"  [dry-run] localize_point {point_camera}")
            return point_camera
        def get_pose(self):
            return [0.0, 0.0, 0.0]
        def get_rgbd(self):
            return (np.zeros((480, 640, 3), np.uint8),
                    np.zeros((480, 640), np.uint16),
                    np.array([[525.0, 0, 320], [0, 525.0, 240], [0, 0, 1]]))
        def grasp(self, pcd):
            print(f"  [dry-run] grasp on {len(pcd.points)} points")
            return True

    run(" ".join(sys.argv[1:]), _DryRunAdapter())
