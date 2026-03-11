# navbot_ros

This package contains ROS drivers and solutions for EKF and Particle Filtered, built on the two-wheel drive from the ROB-GY 6213 course. More information is on the class website in 

<p align="center">
    <a href="https://youtu.be/C_pzKH91Ji0">https://sites.google.com/nyu.edu/rob-gy6213</a>
</p>

## Installation
This is a standard ROS2 package. You may install it as follows. 

```
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/Abanesjo/navbot_ros
cd ~/ros2_ws
rosdep install --from-path src --ignore-src -r -y
colcon build --symlink-install && source install/setup.bash
```

Also make sure the necessary python packages are installed
```
pip install -r requirements.txt
```

Then modify the following files as per the setup (IP addresses, covariances, etc.)
- `navbot_ros/parameters.py`
- `robot_arduino_code/robot_arduino_code.ino`

The `.ino` file is what is compiled on the robot's arduino board.

## Extended Kalman Filter
The EKF implementation fuses wheel odometry and camera odometry, as shown in the demonstration below. 


<p align="center">
    <img src="docs/ekf.gif" alt="EKF sample">
    <a href="https://youtu.be/C_pzKH91Ji0">https://youtu.be/C_pzKH91Ji0</a>
</p>

You can launch the EKF via

```
source install/setup.bash
ros2 launch navbot_ros bringup_ekf.launch.xml rviz:=true
```

## Particle Filter