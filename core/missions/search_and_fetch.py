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

from core.llm_zone import scene_query
from core.llm_zone.active_perception import approach_pose, viewpoints
from core.perception.detection.sam3_client import Sam3Client, mask_to_pointcloud


class RobotAdapter(Protocol):
    """Thin interface the ROS 2 layer must implement."""

    def navigate_to(self, pose: dict) -> bool:
        """Send a Nav2 goal {'x','y','qz','qw'} in the map frame, block until done."""
        ...

    def look_at(self, point_map: list[float]) -> None:
        """Point the head camera at a 3D map-frame point (active perception)."""
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
    """Look at the target location, run SAM3; return (mask, depth, K) or None."""
    robot.look_at(centroid)
    rgb, depth, K = robot.get_rgbd()
    det = sam3.detect(rgb, target)
    if det.found:
        return det.best_mask(), depth, K
    return None


def _search_location(robot: RobotAdapter, sam3: Sam3Client, target: str,
                     centroid: list[float], n_viewpoints: int = 3) -> Optional[tuple]:
    """Navigate to a location and try the approach pose plus a few viewpoints."""
    if not robot.navigate_to(approach_pose(centroid)):
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


def run(target: str, robot: RobotAdapter, k: int = 3) -> MissionResult:
    sam3 = Sam3Client()
    result = MissionResult(found=False, grasped=False)

    # Step 3: scenario selection
    answer = scene_query.query(target)
    candidates: list[dict] = []
    if answer.get("present") and answer.get("centroid"):
        print(f"[mission] Scenario B: '{target}' is in the scene graph "
              f"({answer['object_label']})")
        candidates.append({
            "centroid": answer["centroid"],
            "furniture_label": answer.get("object_label", "known location"),
        })
    else:
        print(f"[mission] Scenario A: '{target}' not in scene graph, asking LLM")
        candidates = scene_query.predict_locations(target, k=k)

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

        hit = _search_location(robot, sam3, target, centroid)
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
        def look_at(self, point):
            print(f"  [dry-run] look_at {point}")
        def get_rgbd(self):
            return (np.zeros((480, 640, 3), np.uint8),
                    np.zeros((480, 640), np.uint16),
                    np.array([[525.0, 0, 320], [0, 525.0, 240], [0, 0, 1]]))
        def grasp(self, pcd):
            print(f"  [dry-run] grasp on {len(pcd.points)} points")
            return True

    run(" ".join(sys.argv[1:]), _DryRunAdapter())
