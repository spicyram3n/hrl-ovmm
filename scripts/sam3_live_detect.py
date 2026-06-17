#!/usr/bin/env python3
"""
scripts/sam3_live_detect.py
----------------------------
Sanity-check that SAM3 is working against the live Gazebo camera.

Grabs ONE frame from the HSR head camera, sends it to the SAM3 docker
server, and saves an annotated image (sam3_output.jpg) to the current
directory so you can inspect the result.

Usage:
    python scripts/sam3_live_detect.py "pringles can"
    python scripts/sam3_live_detect.py "orange"

Make sure the SAM3 docker container is running before you call this.
"""

import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

from core.perception.detection.sam3_client import Sam3Client

CAMERA_TOPIC = '/head_rgbd_sensor/rgb/image_raw'


# ── ROS2-agnostic: pure image + SAM3 ─────────────────────────────────────────

def detect(image_bgr: np.ndarray, prompt: str) -> dict:
    """
    Run SAM3 on a single BGR image.  No ROS2 dependency.

    Returns a plain dict so callers don't need to know about Sam3Detection:
        {
            "found":  bool,
            "masks":  (N, H, W) bool,
            "boxes":  (N, 4)    float32  [x1, y1, x2, y2],
            "scores": (N,)      float32,
        }
    """
    client = Sam3Client()
    det = client.detect(image_bgr, prompt)
    return {
        "found":  det.found,
        "masks":  det.masks,
        "boxes":  det.boxes,
        "scores": det.scores,
    }


def annotate(image_bgr: np.ndarray, result: dict, prompt: str) -> np.ndarray:
    """Draw masks and bounding boxes on a copy of image_bgr."""
    out = image_bgr.copy()
    colors = [
        (0, 220, 0), (0, 100, 255), (0, 0, 220),
        (220, 220, 0), (0, 220, 220), (220, 0, 220),
    ]

    for i, (mask, box, score) in enumerate(
            zip(result["masks"], result["boxes"], result["scores"])):
        color = colors[i % len(colors)]

        # semi-transparent mask fill
        overlay = out.copy()
        overlay[mask] = color
        out = cv2.addWeighted(out, 0.55, overlay, 0.45, 0)

        # bounding box + label
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{prompt}  {score:.2f}",
                    (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if not result["found"]:
        cv2.putText(out, f"NOT FOUND: {prompt}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 220), 2)

    return out


# ── Thin ROS2 wrapper: grab one frame ─────────────────────────────────────────

class _FrameGrabber(Node):
    """Subscribes to the camera topic and captures the first frame it receives."""

    def __init__(self):
        super().__init__('sam3_live_detect')
        self.bridge = CvBridge()
        self.frame: np.ndarray | None = None
        self.create_subscription(Image, CAMERA_TOPIC, self._cb, 1)

    def _cb(self, msg: Image):
        if self.frame is None:
            self.frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')


def grab_frame(timeout_sec: float = 10.0) -> np.ndarray:
    """Spin ROS2 until one camera frame arrives, then return it as BGR."""
    node = _FrameGrabber()
    deadline = node.get_clock().now().nanoseconds / 1e9 + timeout_sec

    print(f"[sam3_live_detect] Waiting for frame on {CAMERA_TOPIC} ...")
    while rclpy.ok() and node.frame is None:
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.get_clock().now().nanoseconds / 1e9 > deadline:
            node.destroy_node()
            raise TimeoutError(
                f"No frame received on {CAMERA_TOPIC} after {timeout_sec}s. "
                "Is Gazebo running?"
            )

    frame = node.frame
    node.destroy_node()
    return frame


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "pringles can"

    rclpy.init()
    frame = grab_frame()
    rclpy.shutdown()

    print(f"[sam3_live_detect] Frame received: {frame.shape}  "
          f"→ querying SAM3 for '{prompt}' ...")

    result = detect(frame, prompt)

    if result["found"]:
        n = len(result["scores"])
        print(f"[sam3_live_detect] Found {n} instance(s).")
        for i, (box, score) in enumerate(zip(result["boxes"], result["scores"])):
            print(f"  [{i}]  score={score:.3f}  box={box.astype(int).tolist()}")
    else:
        print(f"[sam3_live_detect] Nothing found for '{prompt}'.")

    out = annotate(frame, result, prompt)
    cv2.imwrite("sam3_output.jpg", out)
    print("[sam3_live_detect] Annotated image saved to sam3_output.jpg")


if __name__ == '__main__':
    main()
