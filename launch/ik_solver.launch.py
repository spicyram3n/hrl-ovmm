#!/usr/bin/env python3
"""
ros2 launch fetcher ik_solver.launch.py

Brings up the HSR IK solver service (ik_solver_node/solve_ik_with_collision)
used by HsrRobotAdapter.grasp(). Gazebo already provides
robot_state_publisher/joint_state_publisher for the sim robot, so this
launch file starts only the solver itself, configured for HSRC.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from tmc_launch_ros_utils.tmc_launch_ros_utils import (
    load_collision_description,
    load_robot_description,
)


def declare_arguments():
    return [
        DeclareLaunchArgument('description_package', default_value='hsrc_description',
                               description="Description package with the robot's URDF/xacro files."),
        DeclareLaunchArgument('description_file', default_value='hsrc1s.urdf.xacro',
                               description='URDF/XACRO description file for the robot.'),
        DeclareLaunchArgument('collision_file', default_value='collision_pair_hsrc.xml',
                               description='Collision pair config for the robot.'),
    ]


def generate_launch_description():
    robot_description = load_robot_description()
    collision_description = load_collision_description()

    ik_solver = Node(
        package='tmc_ik_solver_node',
        executable='ik_solver_node',
        parameters=[robot_description, collision_description,
                     {'ik_plugin_type': 'hsrb_ik_solver_node::HsrbIkSolverPluginRobustToBasePositionError',
                      'map_convolution_type': 'tmc_ik_solver_node::EuclideanDistanceMapConvolution',
                      'convolution': {'grid_distance_threhsold': 1.5}}])

    return LaunchDescription(declare_arguments() + [ik_solver])
