#!/usr/bin/env python3
"""Publish cumulative nav_msgs/Path topics from odometry and sparse ground truth poses."""

from functools import partial
from typing import Dict

import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile


ODOM_TOPICS = {
    "wheel": "/odom/wheel",
    "camera": "/odom/camera",
    "filtered": "/odom/filtered",
}
GROUND_TRUTH_NAME = "ground_truth"
GROUND_TRUTH_TOPIC = "/ground_truth"


class PathPublisher(Node):
    def __init__(self) -> None:
        super().__init__("path_publisher")

        path_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._paths: Dict[str, Path] = {}
        self._path_publishers = {}
        self._path_subscriptions = []

        for name in ODOM_TOPICS:
            self._paths[name] = Path()
            self._path_publishers[name] = self.create_publisher(
                Path, f"/path/{name}", path_qos
            )

        self._ground_truth_pose_array = PoseArray()
        self._ground_truth_pose_array_publisher = self.create_publisher(
            PoseArray, "/path/ground_truth", path_qos
        )

        for name, topic in ODOM_TOPICS.items():
            sub = self.create_subscription(
                Odometry,
                topic,
                partial(self._odom_callback, name),
                10,
            )
            self._path_subscriptions.append(sub)

        self._ground_truth_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            GROUND_TRUTH_TOPIC,
            self._ground_truth_callback,
            10,
        )

        self.get_logger().info(
            "Publishing paths on /path/{wheel,camera,filtered} and ground truth PoseArray on /path/ground_truth"
        )

    def _odom_callback(self, source: str, msg: Odometry) -> None:
        self._append_and_publish(source, msg.header, msg.pose.pose)

    def _ground_truth_callback(self, msg: PoseWithCovarianceStamped) -> None:
        self._ground_truth_pose_array.header = msg.header
        self._ground_truth_pose_array.poses.append(msg.pose.pose)
        self._ground_truth_pose_array_publisher.publish(self._ground_truth_pose_array)

    def _append_and_publish(self, source: str, header, pose) -> None:
        pose_stamped = PoseStamped()
        pose_stamped.header = header
        pose_stamped.pose = pose

        path = self._paths[source]
        path.header = header
        path.poses.append(pose_stamped)
        self._path_publishers[source].publish(path)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PathPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
