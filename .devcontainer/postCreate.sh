#!/bin/bash
set -e
sudo chown -R $(whoami) /home/ws
echo 'export PYTHONPATH=$PYTHONPATH:/home/ws/source' >> ~/.bashrc

# The pure-Python core is imported as `core.*` from the repo root
grep -qxF 'export PYTHONPATH=$PYTHONPATH:/home/ws' ~/.bashrc || \
    echo 'export PYTHONPATH=$PYTHONPATH:/home/ws' >> ~/.bashrc

ROS2_WS=/home/ws/ros2_ws
mkdir -p "$ROS2_WS/src"

# First run on this host: ros2_ws is freshly bind-mounted and empty, so
# fetch the hsr_ros2 packages per https://github.com/hsr-project/hsr_ros2_doc
# (humble branch, setup_sim_en.md) instead of relying on a pre-built workspace.
if [ -z "$(ls -A "$ROS2_WS/src")" ]; then
    echo "ros2_ws/src is empty - cloning hsr_ros2 packages"
    cd "$ROS2_WS/src"
    for repo in \
        hsrb_controllers hsrb_common hsrb_drivers hsrb_launch hsrb_manipulation \
        hsrb_rosnav hsrb_simulator hsr_common hsrb_teleop tmc_gazebo tmc_teleop \
        tmc_common tmc_common_msgs tmc_drivers tmc_database tmc_manipulation \
        tmc_manipulation_base tmc_manipulation_planner tmc_point_cloud \
        tmc_realtime_control tmc_voice tmc_navigation; do
        git clone -b humble "https://github.com/hsr-project/${repo}.git"
    done
    rm -rf hsrb_launch/hsrb_robot_launch hsrb_simulator/hsrb_rviz_simulator tmc_drivers/tmc_pgr_camera
fi

# Install all ROS 2 workspace dependencies
# (apt-get update is required here: the Dockerfile clears /var/lib/apt/lists
# to shrink the image, so apt has no package index until refreshed)
sudo apt-get update
rosdep update
rosdep install --from-paths "$ROS2_WS/src" --ignore-src -r -y

# -DCMAKE_POLICY_VERSION_MINIMUM=3.5 works around packages whose
# cmake_minimum_required() predates the CMake shipped on Ubuntu 24.04.
cd "$ROS2_WS"
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
