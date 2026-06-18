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

from core.perception.detection.sam3_client import Sam3Client, draw_detections

CAMERA_TOPIC = '/head_rgbd_sensor/rgb/image_raw'


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

    det = Sam3Client().detect(frame, prompt)

    if det.found:
        print(f"[sam3_live_detect] Found {len(det.scores)} instance(s).")
        for i, (box, score) in enumerate(zip(det.boxes, det.scores)):
            print(f"  [{i}]  score={score:.3f}  box={box.astype(int).tolist()}")
    else:
        print(f"[sam3_live_detect] Nothing found for '{prompt}'.")

    cv2.imwrite("sam3_output.jpg", draw_detections(frame, det, prompt))
    print("[sam3_live_detect] Annotated image saved to sam3_output.jpg")


if __name__ == '__main__':
    main()
