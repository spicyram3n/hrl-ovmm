"""
grasp_pipeline.py
-----------------
Full grasping pipeline for HSR in simulation:
1. Capture RGBD point cloud from /head_rgbd_sensor
2. Send to AnyGrasp server (port 5000)
3. Get best grasp pose (4x4 transform in camera frame)
4. Transform to base_footprint frame via TF2
5. Execute grasp with HSR arm

Usage (inside dev container):
    python3 scripts/grasp_pipeline.py
"""
from __future__ import annotations

import io
import threading
import time
import zipfile
from typing import Optional

import numpy as np
import requests
import rclpy

from core.perception.grasping.anygrasp_client import (
    SCALE, MAX_GRIPPER_WIDTH, GRIPPER_HEIGHT, sphere_rotation_matrices,
)
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import tf2_ros

# ── Config ────────────────────────────────────────────────────────────────
ANYGRASP_URL = "http://localhost:5000/graspnet/predict"
ARM_TOPIC     = "/arm_trajectory_controller/joint_trajectory"
GRIPPER_TOPIC = "/gripper_controller/joint_trajectory"
PC_TOPIC      = "/head_rgbd_sensor/depth_registered/rectified_points"
RGB_TOPIC     = "/head_rgbd_sensor/rgb/image_rect_color"

# HSR gripper open/close positions
GRIPPER_OPEN  = 1.2
GRIPPER_CLOSE = 0.1

# Pre-grasp arm pose
PREGRASP = {
    "arm_lift_joint":   0.30,
    "arm_flex_joint":  -1.50,
    "arm_roll_joint":   0.0,
    "wrist_flex_joint": -0.5,
    "wrist_roll_joint":  0.0,
}
CARRY = {
    "arm_lift_joint":   0.25,
    "arm_flex_joint":  -0.5,
    "arm_roll_joint":   0.0,
    "wrist_flex_joint": -0.5,
    "wrist_roll_joint":  0.0,
}
ARM_JOINTS = ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint",
              "wrist_flex_joint", "wrist_roll_joint"]


# ── ROS2 Node ────────────────────────────────────────────────────────────

class GraspPipeline(Node):
    def __init__(self):
        super().__init__("grasp_pipeline")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self._pc: Optional[PointCloud2] = None
        self._rgb: Optional[np.ndarray] = None
        self._lock = threading.Lock()

        self.create_subscription(PointCloud2, PC_TOPIC, self._pc_cb, qos)
        self.create_subscription(Image, RGB_TOPIC, self._rgb_cb, qos)

        self._arm_pub = self.create_publisher(JointTrajectory, ARM_TOPIC, 10)
        self._grip_pub = self.create_publisher(JointTrajectory, GRIPPER_TOPIC, 10)

        self._tf_buf = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)

        self.get_logger().info("GraspPipeline ready.")

    # ── Callbacks ────────────────────────────────────────────────────────

    def _pc_cb(self, msg: PointCloud2):
        with self._lock:
            self._pc = msg

    def _rgb_cb(self, msg: Image):
        with self._lock:
            h, w = msg.height, msg.width
            data = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
            self._rgb = data.astype(np.float32) / 255.0

    # ── Point cloud extraction ──────────────────────────────────────────

    def get_pointcloud(self, timeout=5.0):
        """Wait for point cloud and return (points Nx3, colors Nx3).

        Callbacks are driven by the background spin thread started in
        main() - this only polls the lock-protected state, it must not also
        call rclpy.spin_once() on the same node (two threads pumping the
        same executor concurrently corrupts rclpy's internal state and
        aborts on shutdown).
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if self._pc is not None and self._rgb is not None:
                    pc = self._pc
                    rgb = self._rgb
                    break
            time.sleep(0.1)
        else:
            raise TimeoutError("No point cloud received.")

        # Parse PointCloud2
        points = self._parse_pc2(pc)
        h, w = pc.height, pc.width

        # Sample colors from RGB image
        if rgb.shape[:2] == (h, w):
            colors = rgb.reshape(-1, 3)
        else:
            colors = np.ones((points.shape[0], 3), dtype=np.float32) * 0.5

        # Filter invalid points
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.1) & (points[:, 2] < 3.0)
        points = points[valid]
        colors = colors[valid]

        self.get_logger().info(f"Point cloud: {points.shape[0]} valid points")
        return points, colors

    def _parse_pc2(self, msg: PointCloud2) -> np.ndarray:
        """Extract XYZ from PointCloud2 message."""
        import struct
        fields = {f.name: f for f in msg.fields}
        ox = fields['x'].offset
        oy = fields['y'].offset
        oz = fields['z'].offset
        step = msg.point_step
        data = bytes(msg.data)
        n = msg.width * msg.height
        points = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            base = i * step
            points[i, 0] = struct.unpack_from('f', data, base + ox)[0]
            points[i, 1] = struct.unpack_from('f', data, base + oy)[0]
            points[i, 2] = struct.unpack_from('f', data, base + oz)[0]
        return points

    # ── AnyGrasp ─────────────────────────────────────────────────────────

    def query_anygrasp(self, points: np.ndarray, colors: np.ndarray, top_n: int = 5):
        """Send point cloud to AnyGrasp and return grasp poses (camera frame, metres).

        The server's network was trained at a fixed gripper scale, so points
        and limits are scaled in by SCALE and grasp positions/widths scaled
        back out - see core/perception/grasping/anygrasp_client.py, whose
        AnygraspClient.predict() this mirrors. It also requires `limits` and
        `rotations` files plus several params; omitting them (as an earlier
        version of this method did) makes the server fall back to a bad
        default and fail with a 500.
        """
        self.get_logger().info(f"Sending {len(points)} points to AnyGrasp...")

        scaled_points = points * SCALE
        mins, maxs = scaled_points.min(axis=0), scaled_points.max(axis=0)
        limits = np.stack([mins, maxs], axis=0)
        rotations = sphere_rotation_matrices(resolution=24)

        # Save as npy buffers
        pts_buf = io.BytesIO(); np.save(pts_buf, scaled_points); pts_buf.seek(0)
        col_buf = io.BytesIO(); np.save(col_buf, colors); col_buf.seek(0)
        lim_buf = io.BytesIO(); np.save(lim_buf, limits); lim_buf.seek(0)
        rot_buf = io.BytesIO(); np.save(rot_buf, rotations); rot_buf.seek(0)

        response = requests.post(
            ANYGRASP_URL,
            files={
                "points": pts_buf,
                "colors": col_buf,
                "limits": lim_buf,
                "rotations": rot_buf,
            },
            params={
                "max_gripper_width": ("float", MAX_GRIPPER_WIDTH),
                "gripper_height": ("float", GRIPPER_HEIGHT),
                "top_n": ("int", top_n),
                "vis": ("bool", False),
                "top_down_grasp": ("bool", True),
            },
            timeout=60,
        )

        if response.status_code == 204:
            raise RuntimeError("AnyGrasp found no valid grasp candidates.")
        if response.status_code != 200:
            raise RuntimeError(f"AnyGrasp error: {response.status_code} {response.text[:200]}")

        # Unpack zip response
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            with zf.open("tf_matricess.npy") as f:
                tf_matricess = np.load(io.BytesIO(f.read()))
            with zf.open("scoress.npy") as f:
                scoress = np.load(io.BytesIO(f.read()))
            with zf.open("widthss.npy") as f:
                widthss = np.load(io.BytesIO(f.read()))

        self.get_logger().info(f"AnyGrasp returned {tf_matricess.shape} poses")

        # Flatten across rotation views and undo the SCALE applied above.
        tf_matrices = tf_matricess.reshape(-1, 4, 4).copy()
        tf_matrices[:, :3, 3] /= SCALE
        scores = scoress.reshape(-1)
        widths = widthss.reshape(-1) / SCALE
        return tf_matrices, scores, widths

    # ── Arm control ──────────────────────────────────────────────────────

    def send_arm(self, pose: dict, duration_sec: float = 3.0):
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = [pose[j] for j in ARM_JOINTS]
        pt.time_from_start = Duration(sec=int(duration_sec))
        traj.points = [pt]
        self._arm_pub.publish(traj)
        time.sleep(duration_sec + 0.3)

    def send_gripper(self, position: float, duration_sec: float = 2.0):
        traj = JointTrajectory()
        traj.joint_names = ["hand_motor_joint"]
        pt = JointTrajectoryPoint()
        pt.positions = [position]
        pt.time_from_start = Duration(sec=int(duration_sec))
        traj.points = [pt]
        self._grip_pub.publish(traj)
        time.sleep(duration_sec + 0.2)

    # ── TF transform ─────────────────────────────────────────────────────

    def camera_to_base(self, tf_matrix_cam: np.ndarray) -> Optional[np.ndarray]:
        """Transform 4x4 grasp pose from camera frame to base_footprint."""
        try:
            transform = self._tf_buf.lookup_transform(
                "base_footprint",
                "head_rgbd_sensor_rgb_frame",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0),
            )
            t = transform.transform.translation
            q = transform.transform.rotation
            x, y, z, w = q.x, q.y, q.z, q.w
            R = np.array([
                [1-2*(y**2+z**2), 2*(x*y-z*w),   2*(x*z+y*w)],
                [2*(x*y+z*w),   1-2*(x**2+z**2), 2*(y*z-x*w)],
                [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x**2+y**2)],
            ])
            T_cam_base = np.eye(4)
            T_cam_base[:3, :3] = R
            T_cam_base[:3, 3] = [t.x, t.y, t.z]
            return T_cam_base @ tf_matrix_cam
        except Exception as e:
            self.get_logger().error(f"TF error: {e}")
            return None

    # ── Compute arm joints from grasp pose ──────────────────────────────

    def grasp_pose_to_joints(self, tf_base: np.ndarray) -> dict:
        """Simple IK: compute arm joints from grasp position in base frame."""
        x, y, z = tf_base[:3, 3]
        self.get_logger().info(f"Grasp in base frame: ({x:.3f}, {y:.3f}, {z:.3f})")

        arm_lift = float(np.clip(z - 0.15, 0.0, 0.69))
        reach = float(np.sqrt(x**2 + y**2))
        arm_flex = float(np.clip(-1.0 - (reach - 0.3) * 0.5, -2.62, 0.0))

        return {
            "arm_lift_joint":   arm_lift,
            "arm_flex_joint":   arm_flex,
            "arm_roll_joint":   0.0,
            "wrist_flex_joint": -1.0,
            "wrist_roll_joint":  0.0,
        }

    # ── Full pipeline ────────────────────────────────────────────────────

    def run(self):
        self.get_logger().info("=== Starting grasp pipeline ===")

        # Step 1: Capture point cloud
        self.get_logger().info("Step 1: Capturing point cloud...")
        points, colors = self.get_pointcloud(timeout=10.0)

        # Step 2: Query AnyGrasp
        self.get_logger().info("Step 2: Querying AnyGrasp...")
        tf_matrices, scores, widths = self.query_anygrasp(points, colors, top_n=5)

        # Pick best grasp
        best_idx = np.argmax(scores)
        best_tf = tf_matrices[best_idx]
        best_score = scores[best_idx]
        best_width = widths[best_idx]
        if best_score < 0:
            self.get_logger().error("No valid AnyGrasp candidate (all slots invalid).")
            return False
        self.get_logger().info(f"Best grasp: score={best_score:.3f}, width={best_width:.3f}")
        self.get_logger().info(f"Grasp pose (camera frame):\n{best_tf}")

        # Step 3: Transform to base frame
        self.get_logger().info("Step 3: Transforming to base frame...")
        tf_base = self.camera_to_base(best_tf)
        if tf_base is None:
            self.get_logger().error("TF transform failed!")
            return False

        # Step 4: Compute arm joints
        grasp_joints = self.grasp_pose_to_joints(tf_base)

        # Step 5: Execute grasp sequence
        self.get_logger().info("Step 4: Opening gripper...")
        self.send_gripper(GRIPPER_OPEN)

        self.get_logger().info("Step 5: Pre-grasp pose...")
        self.send_arm(PREGRASP, duration_sec=3.0)

        self.get_logger().info("Step 6: Moving to grasp pose...")
        self.send_arm(grasp_joints, duration_sec=4.0)

        self.get_logger().info("Step 7: Closing gripper...")
        self.send_gripper(GRIPPER_CLOSE)

        self.get_logger().info("Step 8: Lifting to carry pose...")
        self.send_arm(CARRY, duration_sec=3.0)

        self.get_logger().info("=== Grasp pipeline complete ===")
        return True


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = GraspPipeline()

    # Spin via an explicit executor so it can be shut down cleanly before
    # destroy_node()/rclpy.shutdown() - calling those while another thread
    # is still inside rclpy.spin() on the same node aborts on teardown.
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(2.0)

    try:
        node.run()
    except Exception as e:
        node.get_logger().error(f"Pipeline failed: {e}")
        raise
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
