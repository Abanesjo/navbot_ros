#!/bin/bash
source /opt/ros/humble/setup.bash
cd /workspace/ros2_ws
colcon build --symlink-install
source install/setup.bash
echo echo "source /workspace/ros2_ws/install/setup.bash" >> ~/.bashrc
exec bash