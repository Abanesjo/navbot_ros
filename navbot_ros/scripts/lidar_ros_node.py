#!/usr/bin/env python3
import argparse
import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from navbot_ros import parameters
from navbot_ros import robot_python_code


RANGE_MIN_M = 0.02
RANGE_MAX_M = 12.0
POLL_PERIOD_S = 0.02
LIDAR_ANGLE_RES_DEG = 2
NUM_BINS = int(360 / LIDAR_ANGLE_RES_DEG)
ANGLE_INCREMENT_RAD = math.radians(LIDAR_ANGLE_RES_DEG)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish LiDAR data as ROS2 LaserScan messages.")
    parser.add_argument("--local-ip", default=parameters.localIP)
    parser.add_argument("--arduino-ip", default=parameters.arduinoIP)
    parser.add_argument("--local-port", type=int, default=parameters.localPort)
    parser.add_argument("--arduino-port", type=int, default=parameters.arduinoPort)
    parser.add_argument("--buffer-size", type=int, default=parameters.bufferSize)
    return parser


class LidarRosNode(Node):
    def __init__(self, args):
        super().__init__("lidar_ros_node")
        self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)

        udp, ok = robot_python_code.create_udp_communication(
            args.arduino_ip,
            args.local_ip,
            args.arduino_port,
            args.local_port,
            args.buffer_size,
        )
        if not ok:
            raise RuntimeError("Failed to create UDP communication.")

        self.udp = udp
        self.receiver = robot_python_code.MsgReceiver(
            time.perf_counter(), parameters.num_robot_sensors, udp
        )
        self.last_scan = robot_python_code.RobotSensorSignal([0, 0, 0])
        self.lidar_distance_list = [float('inf')] * NUM_BINS

        self.create_timer(POLL_PERIOD_S, self._poll)

    def _update_bins(self, scan):
        for i in range(scan.num_lidar_rays):
            distance_in_mm = scan.distances[i]
            angle = 360 - scan.angles[i]
            if distance_in_mm > 20 and abs(angle) < 360:
                index = max(
                    0,
                    min(
                        NUM_BINS - 1,
                        int((angle - (LIDAR_ANGLE_RES_DEG / 2)) / LIDAR_ANGLE_RES_DEG),
                    ),
                )
                self.lidar_distance_list[index] = distance_in_mm / 1000.0

    def _poll(self):
        scan = self.receiver.receive_robot_sensor_signal(self.last_scan)
        if scan is not self.last_scan:
            self._update_bins(scan)
            self.last_scan = scan

        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "laser_frame"
        msg.angle_min = 0.0
        msg.angle_max = 2 * math.pi - ANGLE_INCREMENT_RAD
        msg.angle_increment = ANGLE_INCREMENT_RAD
        msg.time_increment = 0.0
        msg.scan_time = POLL_PERIOD_S
        msg.range_min = RANGE_MIN_M
        msg.range_max = RANGE_MAX_M
        msg.ranges = list(self.lidar_distance_list)

        self.scan_pub.publish(msg)

    def destroy_node(self):
        if self.udp is not None:
            self.udp.UDPServerSocket.close()
        super().destroy_node()


def main() -> None:
    args = build_arg_parser().parse_args()
    rclpy.init()
    node = LidarRosNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
