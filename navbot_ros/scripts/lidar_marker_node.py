#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class LidarMarkerNode(Node):
    def __init__(self):
        super().__init__("lidar_marker_node")

        self.marker_pub = self.create_publisher(MarkerArray, "/lidar_markers", 10)
        self.create_subscription(LaserScan, "/scan", self._scan_callback, 10)

    def _scan_callback(self, msg: LaserScan):
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

        origin = Point()

        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r):
                continue

            angle = msg.angle_min + i * msg.angle_increment

            p_hit = Point()
            p_hit.x = r * math.cos(angle)
            p_hit.y = r * math.sin(angle)

            marker.points.append(origin)
            marker.points.append(p_hit)

        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)


def main() -> None:
    rclpy.init()
    node = LidarMarkerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
