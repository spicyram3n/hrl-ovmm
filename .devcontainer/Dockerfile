ARG DOCKER_REPO
ARG ROS_DISTRO
ARG IMAGE_SUFFIX
FROM $DOCKER_REPO:$ROS_DISTRO$IMAGE_SUFFIX

RUN set -eux; \
    apt-get update || true;                                           \
    apt-get install -y --no-install-recommends curl gnupg;            \
    rm -f /usr/share/keyrings/ros-archive-keyring.gpg || true;        \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
         -o /usr/share/keyrings/ros-archive-keyring.gpg

ARG USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# System packages: ROS nav stack, sim tools, dev tools
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    python3-pip python-is-python3 ssh neovim git wget unzip \
    ros-humble-control-msgs \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-nav2-simple-commander \
    ros-humble-tf-transformations \
    ros-humble-gazebo-ros-pkgs \
    && rm -rf /var/lib/apt/lists/*

ENV SHELL=/bin/bash

# Python deps for the pure-Python core (NO heavy model deps here:
# Mask3D and SAM3 run in their own containers, see docker/).
# numpy<1.25 is required by Humble's cv_bridge / open3d combo.
RUN pip install --no-cache-dir \
    "numpy<1.25" \
    open3d \
    scipy \
    scikit-learn \
    pandas \
    openai \
    graphviz \
    requests \
    opencv-python \
    transforms3d \
    pyyaml

# Source ROS environment automatically
RUN echo "source /opt/ros/humble/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "[ -f /home/ws/ros2_ws/install/setup.bash ] && source /home/ws/ros2_ws/install/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "export PYTHONPATH=\$PYTHONPATH:/home/ws" >> /home/$USERNAME/.bashrc

USER $USERNAME
CMD ["/bin/bash"]
