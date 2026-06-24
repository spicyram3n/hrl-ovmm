#!/usr/bin/env python3
"""
ros2 launch fetcher ovmm_demo.launch.py [target_object:="pringles can"] [run_mission:=true]

Single launch file for the whole OVMM pipeline, in dependency order:

    1. Gazebo (hsrc_apartment_world.launch.py) - spawns the world + robot.
    2. Nav2 (hsrb_rosnav_config navigation_launch.py) + RViz, once Gazebo has
       had time to come up.
    3. search_and_fetch (this package), once Nav2 has had time to come up -
       only if run_mission:=true (default). It self-waits for sensors/TF/Nav2
       internally (HsrRobotAdapter.wait_until_ready()), so the delay below
       just avoids starting it pointlessly early; it isn't load-bearing.

No manual "2D Pose Estimate" click needed: AMCL's initial pose is already
fixed in hsrb_rosnav_config/config/nav2_params.yaml (set_initial_pose: true,
initial_pose: {x: 0, y: 0, z: 0, yaw: 0}) - that's not a placeholder, it's
exact, because the world->map static transform (hsrb_apartment_world.launch.py)
anchors the map frame's origin AT the robot's fixed spawn point
(robot_pos_x/robot_pos_y), so the robot deterministically starts at map
(0, 0, 0) every run. Don't add a second /initialpose publisher here - on a
mission rerun (AMCL still running, already tracking the robot's real pose
since the *previous* run), that would incorrectly snap its belief back to
the spawn point even though the robot has since moved (see
HsrRobotAdapter.wait_until_ready()'s use of localizer='bt_navigator' for the
same reason).

DEEPSEEK_API_KEY is read from whatever's already in your shell - export it
(or prefix this command with it) before launching; it is deliberately not a
launch argument so it never ends up with a default value sitting in this
tracked file:

    DEEPSEEK_API_KEY=<key> ros2 launch fetcher ovmm_demo.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Seconds to wait before starting the next stage - Gazebo and Nav2 both take
# a while to actually come up, and there's no cheap launch-level "is Gazebo
# ready" signal to wait on instead. search_and_fetch's own internal wait
# makes its delay non-critical; Nav2's map_server is the one worth being
# generous with: it's a *lifecycle* node, so "the Nav2 include started" is
# well before "map_server actually configured, activated, and published the
# map" - and map_server only publishes that map once (latched via
# transient-local QoS), so a subscriber whose discovery hasn't finished
# matching by then can miss it entirely. Don't start RViz in the same
# TimerAction as the Nav2 include for this reason - give it its own, later
# slot instead of racing map_server's lifecycle bringup.
GAZEBO_SETTLE_SECS = 20.0
NAV2_SETTLE_SECS = 6.0
RVIZ_DELAY_AFTER_NAV2_SECS = 5.0
# ^ GAZEBO_SETTLE_SECS was 10.0 - bumped after a real bringup failure: the
# log showed Gazebo's own controller spawners (omni_base_controller) still
# finishing at almost the exact moment Nav2's lifecycle nodes started
# (map_server failed with "yaml_filename is not initialized",
# controller_server with "No critics defined for FollowPath" - both
# immediately on configure(), both verified NOT a content bug in
# nav2_params.yaml or the map argument's path through the 3-level
# IncludeLaunchDescription chain - this is parameter-passing infra reused
# unchanged from the manual command that already works). That timing
# overlap, under the CPU/IO load of Gazebo still loading meshes and
# controllers, is the most likely cause. Not verified against a live rerun -
# if this still fails at 20s, the timing theory is wrong and the actual
# cause needs a different investigation (e.g. logging the resolved `map`
# value inside navigation_launch.py right before the failure).


def generate_launch_description():
    target_object = LaunchConfiguration('target_object')
    run_mission = LaunchConfiguration('run_mission')
    map_yaml = LaunchConfiguration('map')
    rviz_config = LaunchConfiguration('rviz_config')
    use_sim_time = LaunchConfiguration('use_sim_time')

    declared_arguments = [
        DeclareLaunchArgument('target_object', default_value='pringles can',
                              description='Object for search_and_fetch to fetch'),
        DeclareLaunchArgument('run_mission', default_value='true',
                              description='Launch search_and_fetch automatically'),
        DeclareLaunchArgument('map', default_value='/home/ws/apartment_world_map.yaml',
                              description='Full path to the Nav2 map yaml'),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=os.path.join(
                get_package_share_directory('hsrb_rosnav_config'),
                'rviz', 'hsr_navigation2.rviz'),
            description='Full path to the RViz config'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
    ]

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('hsrb_gazebo_launch'),
                        'launch', 'hsrc_apartment_world.launch.py')))

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('hsrb_rosnav_config'),
                        'launch', 'navigation_launch.py')),
        launch_arguments={
            'map': map_yaml,
            'use_sim_time': use_sim_time,
        }.items())

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        output='screen')

    mission = Node(
        package='fetcher',
        executable='search_and_fetch',
        arguments=[target_object],
        output='screen',
        condition=IfCondition(run_mission))

    return LaunchDescription(declared_arguments + [
        gazebo,
        TimerAction(period=GAZEBO_SETTLE_SECS, actions=[navigation]),
        # TimerAction(period=GAZEBO_SETTLE_SECS + RVIZ_DELAY_AFTER_NAV2_SECS, actions=[rviz]),
        TimerAction(period=GAZEBO_SETTLE_SECS + NAV2_SETTLE_SECS, actions=[mission]),
    ])
