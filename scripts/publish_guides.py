#!/usr/bin/env python3
"""Publish static guide markers in the odom frame."""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


class GuidePublisher(Node):
    def __init__(self):
        super().__init__("publish_guides")
        self.publisher = self.create_publisher(MarkerArray, "/guide_markers", 10)
        self.spacing = 0.3
        self.max_x = 0.9
        self.max_y = 1.8
        self.marker_scale = (0.05, 0.05, 0.01)
        self.markers = self._build_markers()
        self.timer = self.create_timer(1.0, self.publish_markers)

    def _build_markers(self) -> MarkerArray:
        markers = MarkerArray()
        x_count = int(round(self.max_x / self.spacing))
        y_count = int(round(self.max_y / self.spacing))
        marker_id = 0

        for ix in range(x_count + 1):
            x = ix * self.spacing
            for iy in range(y_count + 1):
                y = iy * self.spacing
                marker = Marker()
                marker.header.frame_id = "odom"
                marker.ns = "guide"
                marker.id = marker_id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = float(x)
                marker.pose.position.y = float(y)
                marker.pose.position.z = self.marker_scale[2] / 2.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = self.marker_scale[0]
                marker.scale.y = self.marker_scale[1]
                marker.scale.z = self.marker_scale[2]
                marker.color.r = 0.2
                marker.color.g = 0.8
                marker.color.b = 0.2
                marker.color.a = 1.0
                markers.markers.append(marker)
                marker_id += 1

        return markers

    def publish_markers(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for marker in self.markers.markers:
            marker.header.stamp = stamp
        self.publisher.publish(self.markers)


def main() -> None:
    rclpy.init()
    node = GuidePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
