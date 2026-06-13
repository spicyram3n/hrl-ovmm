#!/bin/bash
sudo chown -R $(whoami) /home/ws
echo 'export PYTHONPATH=$PYTHONPATH:/home/ws/source' >> ~/.bashrc

# The pure-Python core is imported as `core.*` from the repo root
grep -qxF 'export PYTHONPATH=$PYTHONPATH:/home/ws' ~/.bashrc || \
    echo 'export PYTHONPATH=$PYTHONPATH:/home/ws' >> ~/.bashrc

# Install all ROS 2 workspace dependencies
# (apt-get update is required here: the Dockerfile clears /var/lib/apt/lists
# to shrink the image, so apt has no package index until refreshed)
sudo apt-get update
rosdep update
rosdep install --from-paths /home/ws/ros2_ws/src --ignore-src -r -y
