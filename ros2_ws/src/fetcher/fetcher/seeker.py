#!/usr/bin/env python3
"""
seeker.py
---------
Query an object in natural language -> scan with the HSR head camera
using YOLO → approach with Nav2 once found.

Usage:
    ros2 run fetcher seeker "a bottle"
    DEEPSEEK_API_KEY=<key> ros2 run fetcher seeker "something to sit on"

The LLM (DeepSeek chat) is called ONCE at startup to translate the
natural-language query into YOLO COCO class names. All real-time
detection is local YOLO (no API latency in the perception loop).
"""

import json
import math
import os
import sys
import time

import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
import tf2_geometry_msgs  # noqa: F401 — registers PointStamped transform support
import tf2_ros
from builtin_interfaces.msg import Duration as DurationMsg
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from openai import OpenAI
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from ultralytics import YOLO

# ── Topics & frames ──────────────────────────────────────────────────────────
COLOR_TOPIC  = '/rgbd_sensor/rgb/image_rect_color'
DEPTH_TOPIC  = '/rgbd_sensor/depth_registered/image_rect_raw'
INFO_TOPIC   = '/rgbd_sensor/rgb/camera_info'
HEAD_TOPIC   = '/head_trajectory_controller/joint_trajectory'
CAMERA_FRAME = 'head_rgbd_sensor_rgb_frame'

# ── Scan parameters ───────────────────────────────────────────────────────────
PAN_SWEEP   = [-1.0, -0.5, 0.0, 0.5, 1.0]   # radians, left → right
HEAD_TILT   = -0.3                            # slightly down
SETTLE_SECS = 1.2                             # wait after each head move
CONF_THRESH = 0.45
STANDOFF    = 1.0                             # approach distance from object (m)


# ── LLM query → YOLO class names ─────────────────────────────────────────────

def resolve_classes(user_query: str) -> list[str]:
    """Map natural language to COCO class names via DeepSeek chat (one call)."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[warn] DEEPSEEK_API_KEY not set — using query words as class names")
        return [user_query.strip().lower()]

    client = OpenAI(base_url="https://api.deepseek.com/v1", api_key=api_key)
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": (
                "You map a natural language query to YOLO COCO class names. "
                "Return ONLY valid JSON: {\"classes\": [\"name\", ...]}. "
                "Use only names from the 80 COCO classes. "
                "Examples: "
                "'a bottle' → {\"classes\": [\"bottle\"]}  "
                "'something to drink' → {\"classes\": [\"bottle\", \"cup\"]}  "
                "'a place to sit' → {\"classes\": [\"chair\", \"couch\"]}"
            )},
            {"role": "user", "content": user_query},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=64,
    )
    data = json.loads(resp.choices[0].message.content)
    classes = data.get("classes", [])
    return classes if classes else [user_query.strip().lower()]


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class Seeker(Node):
    def __init__(self, target_classes: list[str]):
        super().__init__('seeker')
        self.target_classes = set(target_classes)
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')

        # Camera intrinsics — populated on first CameraInfo message
        self.fx = self.fy = self.cx = self.cy = None
        self.latest_depth: np.ndarray | None = None

        # (u, v, label) when a detection is confirmed; None while searching
        self.detection: tuple | None = None

        self.create_subscription(CameraInfo, INFO_TOPIC,  self._info_cb,  1)
        self.create_subscription(Image,      DEPTH_TOPIC, self._depth_cb, 1)
        self.create_subscription(Image,      COLOR_TOPIC, self._rgb_cb,   1)

        self.head_pub = self.create_publisher(JointTrajectory, HEAD_TOPIC, 1)

        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf, self)

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _info_cb(self, msg: CameraInfo):
        if self.fx is None:
            self.fx, self.fy = msg.k[0], msg.k[4]
            self.cx, self.cy = msg.k[2], msg.k[5]

    def _depth_cb(self, msg: Image):
        self.latest_depth = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='passthrough')

    def _rgb_cb(self, msg: Image):
        if self.fx is None or self.detection is not None:
            return
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        for box in self.model(img, verbose=False)[0].boxes:
            label = self.model.names[int(box.cls)]
            if label in self.target_classes and float(box.conf) > CONF_THRESH:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)
                self.detection = (u, v, label)
                self.get_logger().info(
                    f'Detected "{label}"  pixel=({u},{v})  conf={float(box.conf):.2f}')
                return

    # ── head control ──────────────────────────────────────────────────────────

    def pan_head(self, pan: float):
        traj = JointTrajectory()
        traj.joint_names = ['head_pan_joint', 'head_tilt_joint']
        pt = JointTrajectoryPoint()
        pt.positions = [pan, HEAD_TILT]
        pt.time_from_start = DurationMsg(sec=1, nanosec=0)
        traj.points = [pt]
        self.head_pub.publish(traj)

    # ── 3-D localisation ──────────────────────────────────────────────────────

    def pixel_to_map(self, u: int, v: int) -> tuple[float, float, float] | None:
        depth = self.latest_depth
        if depth is None or self.fx is None:
            return None

        z = float(depth[v, u])
        if z > 100.0:        # 16UC1 millimetres → metres
            z /= 1000.0
        if not (0.2 < z < 8.0):
            self.get_logger().warn(f'Depth out of range: {z:.2f} m')
            return None

        pt = PointStamped()
        pt.header.frame_id = CAMERA_FRAME
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.point.x = (u - self.cx) * z / self.fx
        pt.point.y = (v - self.cy) * z / self.fy
        pt.point.z = z

        try:
            pt_map = self.tf_buf.transform(
                pt, 'map', timeout=rclpy.duration.Duration(seconds=1.0))
            return pt_map.point.x, pt_map.point.y, pt_map.point.z
        except Exception as e:
            self.get_logger().warn(f'TF transform failed: {e}')
            return None

    # ── approach goal ─────────────────────────────────────────────────────────

    def approach_goal(self, obj_xyz: tuple) -> PoseStamped:
        """Nav2 goal at STANDOFF distance from the object, robot facing it."""
        ox, oy = obj_xyz[0], obj_xyz[1]

        try:
            t = self.tf_buf.lookup_transform(
                'map', 'base_footprint',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
            rx, ry = t.transform.translation.x, t.transform.translation.y
        except Exception:
            rx, ry = 0.0, 0.0

        dx, dy = ox - rx, oy - ry
        dist = max(math.hypot(dx, dy), 1e-3)
        ux, uy = dx / dist, dy / dist           # unit vector toward object

        gx = ox - ux * STANDOFF
        gy = oy - uy * STANDOFF
        yaw = math.atan2(uy, ux)

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = gx
        goal.pose.position.y = gy
        goal.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.orientation.w = math.cos(yaw / 2)
        return goal


# ── helpers ───────────────────────────────────────────────────────────────────

def spin_for(node: Node, seconds: float):
    end = time.time() + seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.05)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: ros2 run fetcher seeker '<query>'")
        sys.exit(1)

    user_query = " ".join(sys.argv[1:])

    print(f'\nQuery: "{user_query}"')
    print("Resolving YOLO classes via LLM...")
    target_classes = resolve_classes(user_query)
    print(f"Target classes: {target_classes}\n")

    rclpy.init()
    node = Seeker(target_classes)

    nav = BasicNavigator()
    nav.waitUntilNav2Active()

    # ── scan head sweep ───────────────────────────────────────────────────────
    print("Scanning...")
    for pan in PAN_SWEEP:
        node.pan_head(pan)
        spin_for(node, SETTLE_SECS)
        if node.detection:
            break

    node.pan_head(0.0)  # return head to centre

    if node.detection is None:
        print("Object not found in head sweep.")
        rclpy.shutdown()
        return

    # ── localise in 3-D ──────────────────────────────────────────────────────
    u, v, label = node.detection
    print(f'Localising "{label}" in 3-D...')
    spin_for(node, 0.3)   # let a fresh depth frame arrive
    xyz = node.pixel_to_map(u, v)

    if xyz is None:
        print("Could not get valid depth. Aborting.")
        rclpy.shutdown()
        return

    print(f'Object at map ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f} m). Approaching...')

    # ── navigate ──────────────────────────────────────────────────────────────
    goal = node.approach_goal(xyz)
    nav.goToPose(goal)

    while not nav.isTaskComplete():
        spin_for(node, 0.1)

    result = nav.getResult()
    if result == TaskResult.SUCCEEDED:
        print(f'Arrived at "{label}".')
    else:
        print(f'Navigation ended: {result}')

    rclpy.shutdown()


if __name__ == '__main__':
    main()
