#!usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped

def make_pose(nav, x, y, yaw_qz=0.0, yaw_qw=1.0):
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.header.stamp = nav.get_clock().now().to_msg()
    p.pose.position.x = x
    p.pose.position.y = y
    p.pose.orientation.z = yaw_qz
    p.pose.orientation.w = yaw_qw
    
    return p

def main():
    rclpy.init()
    nav = BasicNavigator()

    # rough init pose
    init = make_pose(nav, 0.0, 0.0)
    nav.setInitialPose(init)

    nav.waitUntilNav2Active()  # waiter


    goal = make_pose(nav, 3.0, -2.0)
    nav.goToPose(goal)

    while not nav.isTaskComplete():
        feedback = nav.getFeedback()
        # feedback.distance_remaining, feedback.navigation_time, etc.
        pass

    result = nav.getResult()
    if result == TaskResult.SUCCEEDED:
        nav.get_logger().info('Arrived at destination!!')
    elif result == TaskResult.CANCELED:
        nav.get_logger().info('Canceled')
    else:
        nav.get_logger().info('Failed — pick a fallback pose')

    rclpy.shutdown()


if __name__ == '__main__':
    main()