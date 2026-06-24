#!/usr/bin/env python3
"""
robot_adapter.py
-----------------
ROS 2 implementation of the `RobotAdapter` protocol from
core/missions/search_and_fetch.py for the HSRC robot.

navigate_to -> nav2_simple_commander.BasicNavigator
look_at     -> TF + core/perception/head_geometry.py, publishes to
               /head_trajectory_controller/joint_trajectory
get_rgbd    -> cached head camera topics
grasp       -> core/grasping/anygrasp_client.py (AnyGrasp server)
               for a collision-aware grasp pose, then an analytic arm-joint
               formula - no separate IK service, since AnyGrasp's own grasp
               selection already accounts for collisions

navigate_to() and look_at() receive poses/points sourced from the Gazebo
scene graph (core/scene_graph/build_scene_graph_gazebo.py),
which are ground truth read straight from apartment.world.xacro - i.e. in
Gazebo's "world" frame, not the Nav2 "map" frame (the SLAM map is anchored
at the robot's spawn pose, not world (0,0)). hsrb_apartment_world.launch.py
publishes a static "world" -> "map" transform for exactly this conversion,
so both methods stamp incoming poses as frame_id="world" and let tf2 do the
map-frame conversion instead of assuming they already are map-frame.

The node must be spun continuously (e.g. on a background thread) while the
mission in core/missions/search_and_fetch.py drives it from the main thread
- see search_and_fetch_node.py.
"""

import time

import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
import tf2_geometry_msgs  # noqa: F401 - registers PointStamped transform support
import tf2_ros
from builtin_interfaces.msg import Duration as DurationMsg
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from core.perception import head_geometry
from core.perception.active_perception import CostmapGrid, approach_pose, reachable_approach_pose
from core.grasping.anygrasp_client import AnygraspClient

# ── Topics & frames ──────────────────────────────────────────────────────────
COLOR_TOPIC   = '/head_rgbd_sensor/rgb/image_raw'
DEPTH_TOPIC   = '/head_rgbd_sensor/depth_registered/image_rect_raw'
INFO_TOPIC    = '/head_rgbd_sensor/rgb/camera_info'
HEAD_TOPIC    = '/head_trajectory_controller/joint_trajectory'
ARM_TOPIC     = '/arm_trajectory_controller/joint_trajectory'
GRIPPER_TOPIC = '/gripper_controller/command'

CAMERA_FRAME = 'head_rgbd_sensor_rgb_frame'
PAN_LINK     = 'head_pan_link'
BASE_FRAME   = 'base_footprint'
WORLD_FRAME  = 'world'
MAP_FRAME    = 'map'

ARM_JOINTS = ['arm_lift_joint', 'arm_flex_joint', 'arm_roll_joint',
              'wrist_flex_joint', 'wrist_roll_joint']

# ── Timing ──────────────────────────────────────────────────────────────────
SETTLE_SECS         = 1.2  # wait after a head move for a fresh camera frame
ARM_SETTLE_SECS     = 2.5  # wait after an arm move
GRIPPER_SETTLE_SECS = 1.0  # wait after a gripper move
ARM_MOVE_SECS       = 2.0
GRIPPER_MOVE_SECS   = 1.0

# ── Gripper (hand_motor_joint, limits [-0.798, 1.24]) ─────────────────────────
# Positive = open, negative = squeeze closed (HSR convention).
GRIPPER_OPEN  = 1.2
GRIPPER_CLOSE = -0.6
LIFT_DELTA    = 0.05  # m, extra arm_lift_joint after grasp to show it lifted


class HsrRobotAdapter(Node):
    def __init__(self):
        # Gazebo's TF tree is stamped with sim time (/clock); without this,
        # this node's own clock defaults to wall time and every tf_buf
        # lookup/transform stamped with self.get_clock().now() ends up
        # "in the future" relative to the sim-time TF buffer.
        super().__init__('hsr_robot_adapter', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.bridge = CvBridge()

        # Camera intrinsics - populated on first CameraInfo message
        self.fx = self.fy = self.cx = self.cy = None
        self.latest_rgb: np.ndarray | None = None
        self.latest_depth: np.ndarray | None = None

        # Current head pan angle, from /joint_states
        self.head_pan = 0.0
        self._have_joint_state = False

        self.create_subscription(CameraInfo, INFO_TOPIC, self._info_cb, 1)
        self.create_subscription(Image, DEPTH_TOPIC, self._depth_cb, 1)
        self.create_subscription(Image, COLOR_TOPIC, self._rgb_cb, 1)
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)

        self.head_pub = self.create_publisher(JointTrajectory, HEAD_TOPIC, 1)
        self.arm_pub = self.create_publisher(JointTrajectory, ARM_TOPIC, 1)
        self.gripper_pub = self.create_publisher(JointTrajectory, GRIPPER_TOPIC, 1)

        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf, self)

        self.nav = BasicNavigator()

        self.anygrasp = AnygraspClient()

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _info_cb(self, msg: CameraInfo):
        if self.fx is None:
            self.fx, self.fy = msg.k[0], msg.k[4]
            self.cx, self.cy = msg.k[2], msg.k[5]

    def _depth_cb(self, msg: Image):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _rgb_cb(self, msg: Image):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def _joint_cb(self, msg: JointState):
        if 'head_pan_joint' in msg.name:
            self.head_pan = msg.position[msg.name.index('head_pan_joint')]
            self._have_joint_state = True

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def spin_for(self, seconds: float):
        """The node is spun on a background thread (see search_and_fetch_node.py);
        just give it time to process callbacks."""
        time.sleep(seconds)

    def wait_until_ready(self, timeout_sec: float = 30.0):
        self.get_logger().info('Waiting for camera, joint states, TF and Nav2...')
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if (self.fx is not None and self.latest_rgb is not None
                    and self.latest_depth is not None and self._have_joint_state
                    and self.tf_buf.can_transform(BASE_FRAME, PAN_LINK, rclpy.time.Time())):
                break
            time.sleep(0.2)
        else:
            self.get_logger().warn('Timed out waiting for sensors/TF - continuing anyway')

        # waitUntilNav2Active(localizer='amcl') (the default) publishes a
        # (0, 0, 0) /initialpose before checking AMCL is active - fine on a
        # cold start, but AMCL keeps running across mission reruns and is
        # already tracking the robot's real pose, so that would snap its
        # belief back to the map origin (the spawn point) every run. Passing
        # a non-'amcl' localizer skips that publish; bt_navigator being
        # active already implies amcl (started earlier in the same bringup)
        # is too.
        self.nav.waitUntilNav2Active(localizer='bt_navigator')
        self.get_logger().info('Robot adapter ready.')

    # ── navigation ────────────────────────────────────────────────────────────

    def navigate_to(self, pose: dict) -> bool:
        world_pose = PoseStamped()
        world_pose.header.frame_id = WORLD_FRAME
        world_pose.header.stamp = self.get_clock().now().to_msg()
        world_pose.pose.position.x = pose['x']
        world_pose.pose.position.y = pose['y']
        world_pose.pose.orientation.z = pose['qz']
        world_pose.pose.orientation.w = pose['qw']

        try:
            goal = self.tf_buf.transform(
                world_pose, MAP_FRAME, timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception as e:
            self.get_logger().error(f'navigate_to: TF {WORLD_FRAME}->{MAP_FRAME} failed ({e})')
            return False

        return self._goto_map_pose(goal)

    def _goto_map_pose(self, goal: PoseStamped) -> bool:
        self.nav.goToPose(goal)
        while not self.nav.isTaskComplete():
            time.sleep(0.1)
        return self.nav.getResult() == TaskResult.SUCCEEDED

    def _global_costmap_grid(self) -> CostmapGrid | None:
        """Fetch and convert Nav2's global costmap, or None if unavailable."""
        try:
            costmap = self.nav.getGlobalCostmap()
        except Exception as e:
            self.get_logger().warn(f'navigate_near: getGlobalCostmap failed ({e})')
            return None
        data = np.asarray(costmap.data, dtype=np.uint8).reshape(
            costmap.metadata.size_y, costmap.metadata.size_x)
        return CostmapGrid(
            data=data,
            resolution=costmap.metadata.resolution,
            origin_x=costmap.metadata.origin.position.x,
            origin_y=costmap.metadata.origin.position.y,
        )

    def navigate_near(self, centroid_world: list[float], reach_min: float, reach_max: float) -> bool:
        """Navigate within [reach_min, reach_max] of a world-frame point,
        using the global costmap to pick a free, reachable standoff instead
        of approach_pose()'s fixed offset. Falls back to that fixed offset
        if the costmap or TF isn't available, or if every cell in the reach
        band turns out to be blocked."""
        pt = PointStamped()
        pt.header.frame_id = WORLD_FRAME
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.point.x, pt.point.y, pt.point.z = float(centroid_world[0]), float(centroid_world[1]), 0.0

        try:
            pt_map = self.tf_buf.transform(pt, MAP_FRAME, timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception as e:
            self.get_logger().warn(f'navigate_near: TF {WORLD_FRAME}->{MAP_FRAME} failed ({e}), '
                                    'falling back to fixed standoff')
            return self.navigate_to(approach_pose(centroid_world))

        grid = self._global_costmap_grid()
        pose_map = (reachable_approach_pose([pt_map.point.x, pt_map.point.y], grid, reach_min, reach_max)
                    if grid is not None else None)
        if pose_map is None:
            self.get_logger().warn('navigate_near: no free cell in reach band, '
                                    'falling back to fixed standoff')
            return self.navigate_to(approach_pose(centroid_world))

        goal = PoseStamped()
        goal.header.frame_id = MAP_FRAME
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = pose_map['x']
        goal.pose.position.y = pose_map['y']
        goal.pose.orientation.z = pose_map['qz']
        goal.pose.orientation.w = pose_map['qw']
        return self._goto_map_pose(goal)

    # ── active perception: head control ───────────────────────────────────────

    def look_at(self, point_world: list[float]) -> None:
        pt = PointStamped()
        pt.header.frame_id = WORLD_FRAME
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.point.x, pt.point.y, pt.point.z = (
            float(point_world[0]), float(point_world[1]), float(point_world[2]))

        try:
            # Pan: bearing to the target in the (currently-rotated) pan-link frame.
            pt_pan = self.tf_buf.transform(pt, PAN_LINK, timeout=rclpy.duration.Duration(seconds=1.0))
            delta_pan = head_geometry.pan_delta(pt_pan.point.x, pt_pan.point.y)
            new_pan = head_geometry.clamp(self.head_pan + delta_pan, *head_geometry.PAN_LIMITS)

            # Tilt: elevation to the target relative to the head's height above the base.
            head_tf = self.tf_buf.lookup_transform(
                BASE_FRAME, PAN_LINK, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
            head_height = head_tf.transform.translation.z

            pt_base = self.tf_buf.transform(pt, BASE_FRAME, timeout=rclpy.duration.Duration(seconds=1.0))
            tilt = head_geometry.tilt_for_target(pt_base.point.x, pt_base.point.y, pt_base.point.z, head_height)
            tilt = head_geometry.clamp(tilt, *head_geometry.TILT_LIMITS)
        except Exception as e:
            self.get_logger().warn(f'look_at: TF lookup failed ({e}); skipping head move')
            return

        traj = JointTrajectory()
        traj.joint_names = ['head_pan_joint', 'head_tilt_joint']
        traj_pt = JointTrajectoryPoint()
        traj_pt.positions = [new_pan, tilt]
        traj_pt.time_from_start = DurationMsg(sec=1, nanosec=0)
        traj.points = [traj_pt]
        self.head_pub.publish(traj)
        self.get_logger().info(f'[look_at] pan={new_pan:.2f} tilt={tilt:.2f}')

        self.spin_for(SETTLE_SECS)

    def localize_point(self, point_camera: list[float]) -> list[float] | None:
        """A point in the camera's own frame -> world frame, via a live TF
        lookup of the robot's actual current pose. Used to re-aim at where a
        just-detected object really is, instead of trusting that navigation
        landed exactly on the planned approach pose."""
        pt = PointStamped()
        pt.header.frame_id = CAMERA_FRAME
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.point.x, pt.point.y, pt.point.z = (
            float(point_camera[0]), float(point_camera[1]), float(point_camera[2]))

        try:
            pt_world = self.tf_buf.transform(pt, WORLD_FRAME, timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception as e:
            self.get_logger().warn(f'localize_point: TF {CAMERA_FRAME}->{WORLD_FRAME} failed ({e})')
            return None
        return [pt_world.point.x, pt_world.point.y, pt_world.point.z]

    def get_pose(self) -> list[float] | None:
        """Robot's current position (base_footprint origin) in world-frame
        ground truth, via a live TF lookup. Used to pick the nearest match
        when a query matches multiple identically-labeled objects - the LLM
        has no spatial awareness of where the robot currently is."""
        pt = PointStamped()
        pt.header.frame_id = BASE_FRAME
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.point.x = pt.point.y = pt.point.z = 0.0

        try:
            pt_world = self.tf_buf.transform(pt, WORLD_FRAME, timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception as e:
            self.get_logger().warn(f'get_pose: TF {BASE_FRAME}->{WORLD_FRAME} failed ({e})')
            return None
        return [pt_world.point.x, pt_world.point.y, pt_world.point.z]

    # ── camera ────────────────────────────────────────────────────────────────

    def get_rgbd(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        K = np.array([[self.fx, 0.0, self.cx],
                       [0.0, self.fy, self.cy],
                       [0.0, 0.0, 1.0]], dtype=np.float32)
        return self.latest_rgb, self.latest_depth, K

    # ── grasping ──────────────────────────────────────────────────────────────

    def _send_arm(self, joint_names: list[str], positions: list[float],
                   duration_sec: float = ARM_MOVE_SECS):
        traj = JointTrajectory()
        traj.joint_names = list(joint_names)
        traj_pt = JointTrajectoryPoint()
        traj_pt.positions = [float(p) for p in positions]
        traj_pt.time_from_start = DurationMsg(sec=int(duration_sec))
        traj.points = [traj_pt]
        self.arm_pub.publish(traj)

    def _send_gripper(self, position: float, duration_sec: float = GRIPPER_MOVE_SECS):
        traj = JointTrajectory()
        traj.joint_names = ['hand_motor_joint']
        traj_pt = JointTrajectoryPoint()
        traj_pt.positions = [position]
        traj_pt.time_from_start = DurationMsg(sec=int(duration_sec))
        traj.points = [traj_pt]
        self.gripper_pub.publish(traj)

    def _camera_to_base(self, tf_matrix_cam: np.ndarray) -> np.ndarray | None:
        """4x4 grasp pose, camera frame -> base_footprint."""
        try:
            transform = self.tf_buf.lookup_transform(
                BASE_FRAME, CAMERA_FRAME, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0))
        except Exception as e:
            self.get_logger().error(f'grasp: TF {CAMERA_FRAME}->{BASE_FRAME} failed ({e})')
            return None

        t = transform.transform.translation
        q = transform.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w),     2 * (x * z + y * w)],
            [2 * (x * y + z * w),   1 - 2 * (x**2 + z**2),   2 * (y * z - x * w)],
            [2 * (x * z - y * w),   2 * (y * z + x * w),     1 - 2 * (x**2 + y**2)],
        ])
        T_cam_base = np.eye(4)
        T_cam_base[:3, :3] = R
        T_cam_base[:3, 3] = [t.x, t.y, t.z]
        return T_cam_base @ tf_matrix_cam

    def _grasp_pose_to_joints(self, tf_base: np.ndarray) -> dict:
        """Analytic arm joints from a grasp position in base frame. AnyGrasp's
        own candidate selection already accounts for collisions, so this
        only needs to reach the chosen pose - no separate IK service."""
        x, y, z = tf_base[:3, 3]
        self.get_logger().info(f'[grasp] target in base frame: ({x:.3f}, {y:.3f}, {z:.3f})')

        arm_lift = float(np.clip(z - 0.15, 0.0, 0.69))
        reach = float(np.sqrt(x**2 + y**2))
        arm_flex = float(np.clip(-1.0 - (reach - 0.3) * 0.5, -2.62, 0.0))

        return {
            'arm_lift_joint':   arm_lift,
            'arm_flex_joint':   arm_flex,
            'arm_roll_joint':   0.0,
            'wrist_flex_joint': -1.0,
            'wrist_roll_joint': 0.0,
        }

    def grasp(self, object_pcd) -> bool:
        # AnyGrasp wants an environment cloud too (for collision filtering);
        # we only have the object's own masked cloud here, so pass it for
        # both - same as before the GCNGrasp-VP detour.
        #
        # top_down_grasp=False (was the AnyGrasp client's default True):
        # _grasp_pose_to_joints below only ever reads the grasp's *position*
        # (tf_base[:3, 3]) - it discards AnyGrasp's chosen orientation
        # entirely and always commands the same fixed wrist_flex/arm_roll,
        # so whatever approach direction AnyGrasp actually picked is never
        # reflected in what the arm executes. This toggle doesn't fix that -
        # it just changes which (x, y, z) AnyGrasp returns, which may or may
        # not land somewhere the fixed wrist angle handles better. The real
        # fix is deriving wrist_flex/arm_roll from tf_base[:3, :3] instead of
        # hardcoding them, which needs live testing against the real arm to
        # get the axis conventions right rather than guessing it blind.
        grasps = self.anygrasp.predict(object_pcd, object_pcd, top_down_grasp=False)
        if not grasps:
            self.get_logger().error('grasp: AnyGrasp found no valid candidates')
            return False

        best = grasps[0]
        self.get_logger().info(f'[grasp] AnyGrasp best score={best.score:.3f} width={best.width:.3f}')

        tf_base = self._camera_to_base(best.tf_matrix)
        if tf_base is None:
            return False

        joints = self._grasp_pose_to_joints(tf_base)

        self._send_gripper(GRIPPER_OPEN)
        self.spin_for(GRIPPER_SETTLE_SECS)

        # Two-phase move instead of sending every joint at once: lift first
        # with the arm still tucked in (arm_flex at its retracted limit,
        # 0.0), *then* extend forward to the target. Commanding lift and
        # forward-reach together lets the controller interpolate them
        # simultaneously, so the gripper can sweep forward while still
        # rising - exactly what risks clipping the edge of whatever surface
        # the object is resting on. Wrist/roll can move during the lift too
        # (they only orient the gripper, they don't extend the reach).
        lift_then_reach = dict(joints, arm_flex_joint=0.0)
        self._send_arm(ARM_JOINTS, [lift_then_reach[j] for j in ARM_JOINTS])
        self.spin_for(ARM_SETTLE_SECS)

        self._send_arm(ARM_JOINTS, [joints[j] for j in ARM_JOINTS])
        self.spin_for(ARM_SETTLE_SECS)

        self._send_gripper(GRIPPER_CLOSE)
        self.spin_for(GRIPPER_SETTLE_SECS)

        # Lift flourish: raise arm_lift_joint a bit to show the grasp.
        joints['arm_lift_joint'] = min(joints['arm_lift_joint'] + LIFT_DELTA, 0.69)
        self._send_arm(ARM_JOINTS, [joints[j] for j in ARM_JOINTS])
        self.spin_for(ARM_SETTLE_SECS)

        return True
