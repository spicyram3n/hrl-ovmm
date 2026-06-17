import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class ArmMover(Node):
    def __init__(self):
        super().__init__('arm_mover')
        self.publisher = self.create_publisher(
            JointTrajectory,
            '/arm_trajectory_controller/joint_trajectory',
            10
        )

    def move_arm_up(self):
        msg = JointTrajectory()
        msg.joint_names = [
            'arm_lift_joint',
            'arm_flex_joint',
            'arm_roll_joint',
            'wrist_flex_joint',
            'wrist_roll_joint'
        ]
        point = JointTrajectoryPoint()
        point.positions = [0.0, 0.0, 0.0, 0.0, 0.0]  # lift arm up
        point.time_from_start = Duration(sec=2)
        msg.points = [point]
        self.publisher.publish(msg)
        self.get_logger().info('Arm moving!')

def main():
    rclpy.init()
    node = ArmMover()
    import time
    time.sleep(1)  # wait for publisher to connect
    node.move_arm_up()
    time.sleep(3)
    rclpy.shutdown()

if __name__ == '__main__':
    main()