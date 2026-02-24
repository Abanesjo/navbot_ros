# External libraries
import serial
import time
import pickle
import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
import socket
import math
from time import strftime

# Local libraries
from . import parameters
from . import extended_kalman_filter
from . import robot_python_code
from .motion_models import MyMotionModel


# The core robot class
class Robot:
    def __init__(self):
        self.connected_to_hardware = False
        self.running_trial = False
        self.extra_logging = False
        self.trial_start_time = 0
        self.msg_sender = None
        self.msg_receiver = None
        self.camera_sensor = robot_python_code.CameraSensor(parameters.camera_id)
        self.data_logger = robot_python_code.DataLogger(
            parameters.filename_start, parameters.data_name_list
        )
        self.robot_sensor_signal = robot_python_code.RobotSensorSignal([0, 0, 0])
        self.camera_sensor_signal = [0, 0, 0, 0, 0, 0]
        self.camera_signal_new = False
        self.extended_kalman_filter = extended_kalman_filter.ExtendedKalmanFilter(
            x_0=[0, 0, 0], Sigma_0=parameters.I3 * 10e12, encoder_counts_0=0
        )

        self.motion_model = MyMotionModel([0, 0, 0], 0)
        self.last_encoder_count = self.robot_sensor_signal.encoder_counts
        self.last_update_time = time.perf_counter()

    # Create udp senders and receiver instances with the udp communication
    def setup_udp_connection(self, udp_communication):
        self.msg_sender = robot_python_code.MsgSender(
            time.perf_counter(), parameters.num_robot_control_signals, udp_communication
        )
        self.msg_receiver = robot_python_code.MsgReceiver(
            time.perf_counter(), parameters.num_robot_sensors, udp_communication
        )
        print("Reset msg_senders and receivers!")

    # Stop udp senders and receiver instances with the udp communication
    def eliminate_udp_connection(self):
        self.msg_sender = None
        self.msg_receiver = None
        print("Eliminate UDP !!!")

    def update_state_estimate(self):
        # u_t = np.array([self.robot_sensor_signal.encoder_counts, self.robot_sensor_signal.steering]) # robot_sensor_signal
        # z_t = np.array([self.camera_sensor_signal[0],self.camera_sensor_signal[1],self.camera_sensor_signal[5]]) # camera_sensor_signal

        now = time.perf_counter()
        delta_t = now - self.last_update_time
        if delta_t <= 0.0:
            delta_t = 0.1
        elif delta_t > 0.2:
            delta_t = 0.2
        self.last_update_time = now

        v = self.motion_model.get_linear_velocity(
            self.robot_sensor_signal.encoder_counts - self.last_encoder_count,
            delta_t=delta_t,
        )
        phi = self.motion_model.get_steering_angle(self.robot_sensor_signal.steering)

        u_t = np.array([v, phi])

        z_t = None
        if self.camera_signal_new:
            [z_x, z_y, z_z] = self.camera_sensor_signal[0:3]
            rvec = np.array(self.camera_sensor_signal[3:6], dtype=float).reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)
            sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            singular = sy < 1e-6
            if not singular:
                roll = math.atan2(R[2, 1], R[2, 2])
                pitch = math.atan2(-R[2, 0], sy)
                yaw = math.atan2(R[1, 0], R[0, 0])
            else:
                roll = math.atan2(-R[1, 2], R[1, 1])
                pitch = math.atan2(-R[2, 0], sy)
                yaw = 0.0
            z_t = np.array([z_x, z_y, z_z, roll, pitch, yaw])

        self.last_encoder_count = self.robot_sensor_signal.encoder_counts

        self.extended_kalman_filter.update(u_t, z_t, delta_t)
        self.camera_signal_new = False

    # One iteration of the control loop to be called repeatedly
    def control_loop(self, cmd_speed=0, cmd_steering_angle=0, logging_switch_on=False):
        # Get camera signal
        self.camera_sensor_signal, self.camera_signal_new = self.camera_sensor.get_signal(
            self.camera_sensor_signal
        )
        # print(
        #     "Camera signal: ",
        #     int(100 * self.camera_sensor_signal[0]),
        #     int(100 * self.camera_sensor_signal[1]),
        #     int(100 * self.camera_sensor_signal[2]),
        # )

        # Receive msg
        if self.msg_sender != None:
            self.robot_sensor_signal = self.msg_receiver.receive_robot_sensor_signal(
                self.robot_sensor_signal
            )

        # Update the state estimates
        self.update_state_estimate()

        # Update control signals
        control_signal = [cmd_speed, cmd_steering_angle]

        # Send msg
        if self.msg_receiver != None:
            self.msg_sender.send_control_signal(control_signal)

        # Log the data
        self.data_logger.log(
            logging_switch_on,
            time.perf_counter(),
            control_signal,
            self.robot_sensor_signal,
            self.camera_sensor_signal,
            self.extended_kalman_filter.state_mean,
            self.extended_kalman_filter.state_covariance,
        )
