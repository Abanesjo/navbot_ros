#!/usr/bin/env python3
"""Headless robot control with ROS odometry publishing."""

import time

import cv2
import cv2.aruco as aruco
import numpy as np
import pygame
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from cv_bridge import CvBridge

from navbot_ros import parameters
from navbot_ros import robot_python_code
from navbot_ros.extended_kalman_filter import ExtendedKalmanFilter
from navbot_ros.robot import Robot


CONTROL_PERIOD_S = 0.1
STEER_AXIS = 2
SPEED_AXIS = 1
SPEED_AXIS_SIGN = -1.0
STEER_SCALE = 20
MAX_SPEED = 100
LOG_TOGGLE_BUTTON = 3


def clamp(value, low, high):
    return max(low, min(high, value))


def connect_controller():
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        return None

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    return joystick


def disconnect_controller(joystick):
    if joystick is not None:
        joystick.quit()
    pygame.joystick.quit()
    pygame.quit()


def connect_robot():
    robot = Robot()
    udp = None
    udp, udp_success = robot_python_code.create_udp_communication(
        parameters.arduinoIP,
        parameters.localIP,
        parameters.arduinoPort,
        parameters.localPort,
        parameters.bufferSize,
    )
    if udp_success:
        robot.setup_udp_connection(udp)
        robot.connected_to_hardware = True
    else:
        print("Failed to create UDP connection", flush=True)
        udp = None
    return robot, udp


def disconnect_robot(robot, udp):
    if robot is not None:
        if robot.connected_to_hardware:
            robot.eliminate_udp_connection()
            robot.connected_to_hardware = False
        if hasattr(robot, "camera_sensor") and robot.camera_sensor is not None:
            robot.camera_sensor.close()
    if udp is not None:
        udp.UDPServerSocket.close()


class RobotRosNode(Node):
    def __init__(self):
        super().__init__("robot_ros_node")
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.camera_odom_pub = self.create_publisher(Odometry, "/odom/camera", 10)
        self.filtered_odom_pub = self.create_publisher(Odometry, "/odom/filtered", 10)
        self.wheel_odom_pub = self.create_publisher(Odometry, "/odom/wheel", 10)
        self.image_pub = self.create_publisher(Image, "/camera/image", 10)
        self.cv_bridge = CvBridge()
        self.marker_length = float(parameters.marker_length)
        self.camera_matrix = np.asarray(parameters.camera_matrix)
        self.dist_coeffs = np.asarray(parameters.dist_coeffs)
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        detector_params = aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.detector = aruco.ArucoDetector(dictionary, detector_params)
        self.publish_static_transforms()

    def publish_marker_tf(self, planar_pose: np.ndarray) -> None:
        x = float(planar_pose[0])
        y = float(planar_pose[1])
        yaw = float(planar_pose[2])
        qx, qy, qz, qw = Rotation.from_euler("xyz", [0.0, 0.0, yaw]).as_quat(
            canonical=False
        )
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "odom"
        transform.child_frame_id = "aruco_frame"
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = float(parameters.marker_height)
        transform.transform.rotation.x = float(qx)
        transform.transform.rotation.y = float(qy)
        transform.transform.rotation.z = float(qz)
        transform.transform.rotation.w = float(qw)

        self.tf_broadcaster.sendTransform(transform)

    def publish_static_transforms(self) -> None:
        stamp = self.get_clock().now().to_msg()

        odom_to_tripod = TransformStamped()
        odom_to_tripod.header.stamp = stamp
        odom_to_tripod.header.frame_id = "odom"
        odom_to_tripod.child_frame_id = "tripod"
        odom_to_tripod.transform.translation.x = float(parameters.tripod_x)
        odom_to_tripod.transform.translation.y = float(parameters.tripod_y)
        odom_to_tripod.transform.translation.z = float(parameters.tripod_z)
        tripod_quat = Rotation.from_euler(
            "xyz",
            [parameters.tripod_roll, parameters.tripod_pitch, parameters.tripod_yaw],
        ).as_quat(canonical=False)
        odom_to_tripod.transform.rotation.x = float(tripod_quat[0])
        odom_to_tripod.transform.rotation.y = float(tripod_quat[1])
        odom_to_tripod.transform.rotation.z = float(tripod_quat[2])
        odom_to_tripod.transform.rotation.w = float(tripod_quat[3])

        tripod_to_camera = TransformStamped()
        tripod_to_camera.header.stamp = stamp
        tripod_to_camera.header.frame_id = "tripod"
        tripod_to_camera.child_frame_id = "camera_frame"
        tripod_to_camera.transform.translation.x = 0.0
        tripod_to_camera.transform.translation.y = 0.0
        tripod_to_camera.transform.translation.z = 0.0
        camera_quat = Rotation.from_euler(
            "xyz",
            [parameters.camera_roll, parameters.camera_pitch, parameters.camera_yaw],
        ).as_quat(canonical=False)
        tripod_to_camera.transform.rotation.x = float(camera_quat[0])
        tripod_to_camera.transform.rotation.y = float(camera_quat[1])
        tripod_to_camera.transform.rotation.z = float(camera_quat[2])
        tripod_to_camera.transform.rotation.w = float(camera_quat[3])

        self.static_tf_broadcaster.sendTransform([odom_to_tripod, tripod_to_camera])

    def publish_camera_odom(self, planar_pose: np.ndarray) -> None:
        x = float(planar_pose[0])
        y = float(planar_pose[1])
        yaw = float(planar_pose[2])
        qx, qy, qz, qw = Rotation.from_euler("xyz", [0.0, 0.0, yaw]).as_quat(
            canonical=False
        )

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "aruco_frame"
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = float(parameters.marker_height)
        odom.pose.pose.orientation.x = float(qx)
        odom.pose.pose.orientation.y = float(qy)
        odom.pose.pose.orientation.z = float(qz)
        odom.pose.pose.orientation.w = float(qw)
        cov = self.build_pose_covariance(
            parameters.Q3,
            parameters.z_var,
            parameters.roll_var,
            parameters.pitch_var,
        )
        odom.pose.covariance = cov.flatten().tolist()

        self.camera_odom_pub.publish(odom)
        self.publish_marker_tf(planar_pose)

    #This is necessary since most covariances are only for 3dof. We need 6dof for visualization. This also ensures that the matrix is still positive semidefinite. (This is purely for visualization purposes in RViz)
    def build_pose_covariance(
        self, state_covariance, z_var: float, roll_var: float, pitch_var: float
    ) -> np.ndarray:
        state_cov = np.asarray(state_covariance, dtype=float).reshape(3, 3)
        state_cov = 0.5 * (state_cov + state_cov.T)

        #This checks for positive semidefiniteness
        try:
            eigvals, eigvecs = np.linalg.eigh(state_cov)
            eigvals = np.maximum(eigvals, 1e-9)
            state_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
        except np.linalg.LinAlgError:
            diag = np.maximum(np.diag(state_cov), 1e-9)
            state_cov = np.diag(diag)
        
        #This just appends the existing 3-state covariance with variances for other positions. 
        cov = np.zeros((6, 6), dtype=float)
        cov[0:2, 0:2] = state_cov[0:2, 0:2]
        cov[0, 5] = state_cov[0, 2]
        cov[1, 5] = state_cov[1, 2]
        cov[5, 0] = state_cov[2, 0]
        cov[5, 1] = state_cov[2, 1]
        cov[5, 5] = state_cov[2, 2]
        cov[2, 2] = z_var
        cov[3, 3] = roll_var
        cov[4, 4] = pitch_var
        return cov

    def publish_filtered_odom(self, state_mean, state_covariance) -> None:
        if state_mean is None or len(state_mean) < 3:
            return

        x = float(state_mean[0])
        y = float(state_mean[1])
        yaw = float(state_mean[2])
        qx, qy, qz, qw = Rotation.from_euler("xyz", [0.0, 0.0, yaw]).as_quat(
            canonical=False
        )

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = float(parameters.marker_height)
        odom.pose.pose.orientation.x = float(qx)
        odom.pose.pose.orientation.y = float(qy)
        odom.pose.pose.orientation.z = float(qz)
        odom.pose.pose.orientation.w = float(qw)

        cov = self.build_pose_covariance(
            state_covariance,
            parameters.z_var,
            parameters.roll_var,
            parameters.pitch_var,
        )
        odom.pose.covariance = cov.flatten().tolist()

        self.filtered_odom_pub.publish(odom)

    def publish_wheel_odom(self, state_mean, state_covariance) -> None:
        if state_mean is None or len(state_mean) < 3:
            return

        x = float(state_mean[0])
        y = float(state_mean[1])
        yaw = float(state_mean[2])
        qx, qy, qz, qw = Rotation.from_euler("xyz", [0.0, 0.0, yaw]).as_quat(
            canonical=False
        )

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = float(parameters.marker_height)
        odom.pose.pose.orientation.x = float(qx)
        odom.pose.pose.orientation.y = float(qy)
        odom.pose.pose.orientation.z = float(qz)
        odom.pose.pose.orientation.w = float(qw)

        cov = self.build_pose_covariance(
            state_covariance,
            parameters.z_var,
            parameters.roll_var,
            parameters.pitch_var,
        )
        odom.pose.covariance = cov.flatten().tolist()

        self.wheel_odom_pub.publish(odom)

    def publish_image(self, frame: np.ndarray) -> None:
        msg = self.cv_bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_frame"
        self.image_pub.publish(msg)

    def process_frame(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        vis = frame.copy()
        markers = []

        if ids is not None:
            aruco.drawDetectedMarkers(vis, corners, ids)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners,
                self.marker_length,
                self.camera_matrix,
                self.dist_coeffs,
            )

            for i in range(len(ids)):
                marker_id = int(ids[i])
                tvec = tvecs[i].ravel()
                rvec = rvecs[i].ravel()
                markers.append((marker_id, tvec, rvec))
                cv2.drawFrameAxes(
                    vis,
                    self.camera_matrix,
                    self.dist_coeffs,
                    rvecs[i],
                    tvecs[i],
                    0.05,
                )
        return vis, markers


def main() -> None:
    joystick = connect_controller()
    if joystick is None:
        print("No controller detected. Exiting.", flush=True)
        return

    rclpy.init()
    ros_node = RobotRosNode()
    robot, udp = connect_robot()
    wheel_ekf = ExtendedKalmanFilter(
        x_0=[0.0, 0.0, 0.0],
        Sigma_0=parameters.I3,
        encoder_counts_0=robot.robot_sensor_signal.encoder_counts,
    )
    wheel_ekf.correct = False
    wheel_last_encoder_count = robot.robot_sensor_signal.encoder_counts
    wheel_initialized = False

    logging_active = False
    log_toggle_button_prev = False
    last_loop_time = time.perf_counter()

    try:
        while rclpy.ok():
            loop_start = time.perf_counter()

            pygame.event.pump()

            steer_axis = clamp(joystick.get_axis(STEER_AXIS), -1.0, 1.0)
            cmd_steering = int(round(steer_axis * STEER_SCALE))

            speed_axis = clamp(
                SPEED_AXIS_SIGN * joystick.get_axis(SPEED_AXIS), 0.0, 1.0
            )
            cmd_speed = int(round(speed_axis * MAX_SPEED))

            log_toggle_button_pressed = bool(joystick.get_button(LOG_TOGGLE_BUTTON))
            if log_toggle_button_pressed and not log_toggle_button_prev:
                logging_active = not logging_active
                if logging_active:
                    print("Recording started", flush=True)
                else:
                    print("Recording stopped", flush=True)
            log_toggle_button_prev = log_toggle_button_pressed

            robot.control_loop(cmd_speed, cmd_steering, logging_active)

            encoder_counts = robot.robot_sensor_signal.encoder_counts
            now = time.perf_counter()
            if not wheel_initialized:
                wheel_last_encoder_count = encoder_counts
                last_loop_time = now
                wheel_initialized = True
            else:
                delta_t = now - last_loop_time
                last_loop_time = now
                if delta_t <= 0.0:
                    delta_t = CONTROL_PERIOD_S

                delta_counts = encoder_counts - wheel_last_encoder_count
                wheel_last_encoder_count = encoder_counts
                v = wheel_ekf.motion_model.get_linear_velocity(
                    delta_counts, delta_t
                )
                phi = wheel_ekf.motion_model.get_steering_angle(
                    robot.robot_sensor_signal.steering
                )
                wheel_ekf.update([v, phi], None, delta_t)

            ros_node.publish_filtered_odom(
                robot.extended_kalman_filter.state_mean,
                robot.extended_kalman_filter.state_covariance,
            )
            ros_node.publish_wheel_odom(
                wheel_ekf.state_mean,
                wheel_ekf.state_covariance,
            )
            frame = getattr(robot.camera_sensor, "last_frame", None)
            if frame is not None:
                vis, markers = ros_node.process_frame(frame)
                ros_node.publish_image(vis)
                if markers:
                    _, tvec, rvec = markers[0]
                    planar_pose = robot.extended_kalman_filter.camera_pose_to_odom_planar(
                        tvec, rvec
                    )
                    ros_node.publish_camera_odom(planar_pose)
            rclpy.spin_once(ros_node, timeout_sec=0.0)

            elapsed = time.perf_counter() - loop_start
            if elapsed < CONTROL_PERIOD_S:
                time.sleep(CONTROL_PERIOD_S - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        disconnect_robot(robot, udp)
        disconnect_controller(joystick)
        ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
