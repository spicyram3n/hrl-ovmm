#!/usr/bin/env python3
"""
ros2 run fetcher search_and_fetch "<object>"

Single command for demo steps 3-7. Assumes Gazebo + Nav2, the SAM3 docker
server (docker/sam3), and the IK solver (ros2 launch fetcher ik_solver.launch.py)
are already running.
"""
import sys
import threading

import rclpy
from rclpy.executors import SingleThreadedExecutor

from core.missions.search_and_fetch import run
from fetcher.robot_adapter import HsrRobotAdapter


def main():
    if len(sys.argv) < 2:
        print('Usage: ros2 run fetcher search_and_fetch "<object>"')
        sys.exit(1)
    target = " ".join(sys.argv[1:])

    rclpy.init()
    adapter = HsrRobotAdapter()
    executor = SingleThreadedExecutor()
    executor.add_node(adapter)
    threading.Thread(target=executor.spin, daemon=True).start()
    adapter.wait_until_ready()

    result = run(target, adapter)
    print(f"\n[search_and_fetch] {result}")

    rclpy.shutdown()


if __name__ == '__main__':
    main()
