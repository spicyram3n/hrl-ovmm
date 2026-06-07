#!/bin/bash
sudo chown -R $(whoami) /home/ws
echo 'export PYTHONPATH=$PYTHONPATH:/home/ws/source' >> ~/.bashrc

# Install all ROS 2 workspace dependencies
rosdep update
rosdep install --from-paths /home/ws/ros2_ws/src --ignore-src -r -y
