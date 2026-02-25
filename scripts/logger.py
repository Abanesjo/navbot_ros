#!/usr/bin/env python3
"""Log odometry estimates from multiple sources to a CSV file."""

import csv
import math
import os
from datetime import datetime
from functools import partial
from typing import Dict, Optional

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


TOPICS = {
    "wheel": "/odom/wheel",
    "camera": "/odom/camera",
    "filtered": "/odom/filtered",
}
LOG_PERIOD_S = 0.1


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Return planar yaw (radians) from a quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class OdometryLogger(Node):
    def __init__(self) -> None:
        super().__init__("odometry_logger")

        default_output_dir = os.path.expanduser("/home/nav/ros2_ws/src/navbot_ros/data")
        self.declare_parameter("output_dir", default_output_dir)
        output_dir = os.path.expanduser(
            str(self.get_parameter("output_dir").get_parameter_value().string_value)
        )
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = os.path.join(output_dir, f"{timestamp}.csv")
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames())
        self._writer.writeheader()
        self._csv_file.flush()

        self._latest_msgs: Dict[str, Optional[Odometry]] = {name: None for name in TOPICS}
        self._start_time = None
        self._announced_waiting = False

        self._subscriptions = []
        for name, topic in TOPICS.items():
            sub = self.create_subscription(
                Odometry,
                topic,
                partial(self._odom_callback, name),
                10,
            )
            self._subscriptions.append(sub)

        self._timer = self.create_timer(LOG_PERIOD_S, self._on_timer)
        self.get_logger().info(f"Logging odometry to CSV: {self._csv_path}")

    def _fieldnames(self):
        fields = ["time_from_start_s"]
        for prefix in TOPICS:
            fields.extend(
                [
                    f"{prefix}_x",
                    f"{prefix}_y",
                    f"{prefix}_yaw",
                    f"{prefix}_x_var",
                    f"{prefix}_y_var",
                    f"{prefix}_yaw_var",
                    f"{prefix}_xy_cov",
                ]
            )
        return fields

    def _odom_callback(self, source: str, msg: Odometry) -> None:
        self._latest_msgs[source] = msg
        if all(self._latest_msgs.values()) and self._announced_waiting:
            self.get_logger().info("Received all odometry topics. Starting CSV writes at 10 Hz.")
            self._announced_waiting = False

    def _on_timer(self) -> None:
        if not all(self._latest_msgs.values()):
            if not self._announced_waiting:
                missing = [name for name, msg in self._latest_msgs.items() if msg is None]
                self.get_logger().info(
                    "Waiting for first messages on: " + ", ".join(missing)
                )
                self._announced_waiting = True
            return

        now = self.get_clock().now()
        if self._start_time is None:
            self._start_time = now

        elapsed_s = (now - self._start_time).nanoseconds * 1e-9
        row = {"time_from_start_s": elapsed_s}
        for prefix, msg in self._latest_msgs.items():
            row.update(self._extract_row_fields(prefix, msg))

        self._writer.writerow(row)
        self._csv_file.flush()

    def _extract_row_fields(self, prefix: str, msg: Odometry):
        pose = msg.pose.pose
        cov = list(msg.pose.covariance)
        yaw = quaternion_to_yaw(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )

        xy_cov = 0.5 * (float(cov[1]) + float(cov[6]))
        return {
            f"{prefix}_x": float(pose.position.x),
            f"{prefix}_y": float(pose.position.y),
            f"{prefix}_yaw": float(yaw),
            f"{prefix}_x_var": float(cov[0]),
            f"{prefix}_y_var": float(cov[7]),
            f"{prefix}_yaw_var": float(cov[35]),
            f"{prefix}_xy_cov": float(xy_cov),
        }

    def destroy_node(self) -> bool:
        try:
            if hasattr(self, "_csv_file") and not self._csv_file.closed:
                self._csv_file.flush()
                self._csv_file.close()
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OdometryLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
