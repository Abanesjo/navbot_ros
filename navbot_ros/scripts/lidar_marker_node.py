#!/usr/bin/env python3
import argparse
import math
import time

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from navbot_ros import parameters
from navbot_ros import robot_python_code


MAX_LIDAR_RANGE_M = 12.0
LIDAR_ANGLE_RES_DEG = 2
NUM_BINS = int(360 / LIDAR_ANGLE_RES_DEG)
POLL_PERIOD_S = 0.1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish LiDAR data as RViz LINE_LIST markers using GUI-style 2-degree binning."
    )
    parser.add_argument("--local-ip", default=parameters.localIP)
    parser.add_argument("--arduino-ip", default=parameters.arduinoIP)
    parser.add_argument("--local-port", type=int, default=parameters.localPort)
    parser.add_argument("--arduino-port", type=int, default=parameters.arduinoPort)
    parser.add_argument("--buffer-size", type=int, default=parameters.bufferSize)
    return parser


class LidarMarkerNode(Node):
    def __init__(self, args):
        super().__init__("lidar_marker_node")

        self.marker_pub = self.create_publisher(MarkerArray, "/lidar_markers", 10)

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

        # Persistent 2-degree bins — same as GUI
        self.lidar_distance_list = [MAX_LIDAR_RANGE_M] * NUM_BINS

        # Precompute cos/sin per bin — same as GUI
        self.cos_list = [
            math.cos(i * LIDAR_ANGLE_RES_DEG / 180.0 * math.pi) for i in range(NUM_BINS)
        ]
        self.sin_list = [
            math.sin(i * LIDAR_ANGLE_RES_DEG / 180.0 * math.pi) for i in range(NUM_BINS)
        ]

        self.create_timer(POLL_PERIOD_S, self._poll)

    def _update_bins(self, scan):
        """Update persistent bins using the same logic as the GUI's update_lidar_data."""
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

    def _build_marker(self) -> Marker:
        """Build a LINE_LIST marker using the same visualization as the GUI's show_lidar_plot."""
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "laser_frame"
        marker.ns = "lidar"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.02
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for i in range(NUM_BINS):
            dist = self.lidar_distance_list[i]
            cos_ang = self.cos_list[i]
            sin_ang = self.sin_list[i]

            p_origin = Point()
            p_origin.x = 0.0
            p_origin.y = 0.0
            p_origin.z = 0.0

            p_hit = Point()
            p_hit.x = dist * cos_ang
            p_hit.y = dist * sin_ang
            p_hit.z = 0.0

            marker.points.append(p_origin)
            marker.points.append(p_hit)

        return marker

    def _poll(self):
        scan = self.receiver.receive_robot_sensor_signal(self.last_scan)
        if scan is not self.last_scan:
            self._update_bins(scan)
            self.last_scan = scan

        marker_array = MarkerArray()
        marker_array.markers.append(self._build_marker())
        self.marker_pub.publish(marker_array)

    def destroy_node(self):
        if self.udp is not None:
            self.udp.UDPServerSocket.close()
        super().destroy_node()


def main() -> None:
    args = build_arg_parser().parse_args()
    rclpy.init()
    node = LidarMarkerNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
