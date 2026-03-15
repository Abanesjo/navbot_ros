#!/usr/bin/env python3
"""Diagnose PF rosbag data health for dual-PF localization runs.

Reports per-topic:
- message count
- first/last timestamp
- average message rate
- monotonic timestamp check
- maximum timestamp gap
- long-gap count

For odometry topics, also reports pose-stream sanity:
- repeated identical pose count
- max per-step translation jump
- max per-step yaw jump
- jump counts above thresholds

Usage:
  python3 check_pf_bag_diagnostics.py /path/to/bag_folder
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import rosbag2_py
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


TRACKED_TOPICS = [
    "/odom/camera",
    "/odom/pf_baseline",
    "/odom/pf_variant",
    "/pf/particles_baseline",
    "/pf/particles_variant",
    "/path/pf_baseline",
    "/path/pf_variant",
]

ODOM_TOPICS = {
    "/odom/camera",
    "/odom/pf_baseline",
    "/odom/pf_variant",
}


@dataclass
class TopicDiag:
    count: int = 0
    first_t: Optional[float] = None
    last_t: Optional[float] = None
    prev_t: Optional[float] = None
    monotonic: bool = True
    max_gap_s: float = 0.0
    long_gap_count: int = 0


@dataclass
class OdomDiag:
    repeated_pose_count: int = 0
    max_step_translation_m: float = 0.0
    max_step_yaw_deg: float = 0.0
    jump_translation_count: int = 0
    jump_yaw_count: int = 0
    prev_pose: Optional[Tuple[float, float, float]] = None


def angle_wrap(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def format_ts(ts: Optional[float]) -> str:
    return "n/a" if ts is None else f"{ts:.6f}"


def format_rate(count: int, first_t: Optional[float], last_t: Optional[float]) -> str:
    if count <= 1 or first_t is None or last_t is None or last_t <= first_t:
        return "n/a"
    rate = (count - 1) / (last_t - first_t)
    return f"{rate:.3f} Hz"


def compute_pf_continuity(
    topic_diag: TopicDiag,
    run_start: float,
    run_end: float,
    long_gap_s: float,
    start_end_tol_s: float,
) -> Tuple[bool, str]:
    if topic_diag.count == 0 or topic_diag.first_t is None or topic_diag.last_t is None:
        return False, "no messages"

    start_lag = topic_diag.first_t - run_start
    end_lag = run_end - topic_diag.last_t
    reasons = []
    if start_lag > start_end_tol_s:
        reasons.append(f"late-start {start_lag:.3f}s")
    if end_lag > start_end_tol_s:
        reasons.append(f"early-stop {end_lag:.3f}s")
    if topic_diag.max_gap_s > long_gap_s:
        reasons.append(f"max-gap {topic_diag.max_gap_s:.3f}s")

    if reasons:
        return False, ", ".join(reasons)
    return True, "continuous enough"


def diagnose_bag(
    bag_path: str,
    long_gap_s: float,
    jump_translation_threshold_m: float,
    jump_yaw_threshold_deg: float,
    repeat_translation_epsilon_m: float,
    repeat_yaw_epsilon_deg: float,
    start_end_tol_s: float,
) -> int:
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    topic_diags: Dict[str, TopicDiag] = {topic: TopicDiag() for topic in TRACKED_TOPICS}
    odom_diags: Dict[str, OdomDiag] = {topic: OdomDiag() for topic in ODOM_TOPICS}

    jump_yaw_threshold_rad = math.radians(jump_yaw_threshold_deg)
    repeat_yaw_epsilon_rad = math.radians(repeat_yaw_epsilon_deg)

    while reader.has_next():
        topic_name, data, timestamp_ns = reader.read_next()
        if topic_name not in topic_diags:
            continue

        t_sec = timestamp_ns * 1e-9
        diag = topic_diags[topic_name]
        diag.count += 1

        if diag.first_t is None:
            diag.first_t = t_sec
        if diag.prev_t is not None:
            if t_sec < diag.prev_t:
                diag.monotonic = False
            gap = t_sec - diag.prev_t
            if gap > diag.max_gap_s:
                diag.max_gap_s = gap
            if gap > long_gap_s:
                diag.long_gap_count += 1
        diag.prev_t = t_sec
        diag.last_t = t_sec

        if topic_name not in ODOM_TOPICS:
            continue

        msg_type_name = type_map.get(topic_name, "")
        if msg_type_name != "nav_msgs/msg/Odometry":
            continue

        msg_type = get_message(msg_type_name)
        msg = deserialize_message(data, msg_type)

        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        qx = float(msg.pose.pose.orientation.x)
        qy = float(msg.pose.pose.orientation.y)
        qz = float(msg.pose.pose.orientation.z)
        qw = float(msg.pose.pose.orientation.w)
        yaw = yaw_from_quaternion(qx, qy, qz, qw)

        odom_diag = odom_diags[topic_name]
        if odom_diag.prev_pose is not None:
            prev_x, prev_y, prev_yaw = odom_diag.prev_pose
            dx = x - prev_x
            dy = y - prev_y
            step_translation = math.hypot(dx, dy)
            step_yaw = abs(angle_wrap(yaw - prev_yaw))

            if step_translation > odom_diag.max_step_translation_m:
                odom_diag.max_step_translation_m = step_translation
            step_yaw_deg = math.degrees(step_yaw)
            if step_yaw_deg > odom_diag.max_step_yaw_deg:
                odom_diag.max_step_yaw_deg = step_yaw_deg

            if step_translation > jump_translation_threshold_m:
                odom_diag.jump_translation_count += 1
            if step_yaw > jump_yaw_threshold_rad:
                odom_diag.jump_yaw_count += 1

            if (
                step_translation <= repeat_translation_epsilon_m
                and step_yaw <= repeat_yaw_epsilon_rad
            ):
                odom_diag.repeated_pose_count += 1

        odom_diag.prev_pose = (x, y, yaw)

    nonempty_topics = [d for d in topic_diags.values() if d.count > 0 and d.first_t is not None and d.last_t is not None]
    if nonempty_topics:
        run_start = min(d.first_t for d in nonempty_topics if d.first_t is not None)
        run_end = max(d.last_t for d in nonempty_topics if d.last_t is not None)
        run_duration = max(0.0, run_end - run_start)
    else:
        run_start, run_end, run_duration = 0.0, 0.0, 0.0

    print("\n=== Topic Diagnostics ===")
    print(
        "topic,count,first_s,last_s,avg_rate,monotonic,max_gap_s,long_gap_count"
    )
    for topic in TRACKED_TOPICS:
        d = topic_diags[topic]
        print(
            f"{topic},{d.count},{format_ts(d.first_t)},{format_ts(d.last_t)},"
            f"{format_rate(d.count, d.first_t, d.last_t)},{d.monotonic},{d.max_gap_s:.3f},{d.long_gap_count}"
        )

    print("\n=== PF Continuity Check ===")
    print(f"Run span from tracked topics: {run_duration:.3f}s")
    for topic in ["/odom/pf_baseline", "/odom/pf_variant"]:
        ok, reason = compute_pf_continuity(
            topic_diags[topic],
            run_start,
            run_end,
            long_gap_s,
            start_end_tol_s,
        )
        print(f"{topic}: {'OK' if ok else 'WARN'} ({reason})")

    print("\n=== Odom Pose Sanity ===")
    print(
        "topic,repeated_pose_count,max_step_translation_m,max_step_yaw_deg,"
        "jump_translation_count,jump_yaw_count"
    )
    for topic in ["/odom/camera", "/odom/pf_baseline", "/odom/pf_variant"]:
        d = odom_diags[topic]
        print(
            f"{topic},{d.repeated_pose_count},{d.max_step_translation_m:.4f},"
            f"{d.max_step_yaw_deg:.2f},{d.jump_translation_count},{d.jump_yaw_count}"
        )

    missing = [topic for topic in TRACKED_TOPICS if topic_diags[topic].count == 0]
    if missing:
        print("\nMissing tracked topics in bag:")
        for topic in missing:
            print(f"- {topic}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_path", help="Path to rosbag2 folder")
    parser.add_argument(
        "--long-gap-s",
        type=float,
        default=0.5,
        help="Gap above this is counted as a long no-message stretch",
    )
    parser.add_argument(
        "--jump-translation-threshold-m",
        type=float,
        default=0.35,
        help="Count odom translation jumps above this per-step threshold",
    )
    parser.add_argument(
        "--jump-yaw-threshold-deg",
        type=float,
        default=30.0,
        help="Count odom yaw jumps above this per-step threshold",
    )
    parser.add_argument(
        "--repeat-translation-epsilon-m",
        type=float,
        default=1e-6,
        help="Per-step translation below this counts as repeated pose",
    )
    parser.add_argument(
        "--repeat-yaw-epsilon-deg",
        type=float,
        default=1e-4,
        help="Per-step yaw below this counts as repeated pose",
    )
    parser.add_argument(
        "--start-end-tol-s",
        type=float,
        default=2.0,
        help="Tolerance for late-start/early-stop continuity warning",
    )
    args = parser.parse_args()

    bag_path = Path(args.bag_path).expanduser().resolve()
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    rclpy.init()
    try:
        return diagnose_bag(
            str(bag_path),
            long_gap_s=args.long_gap_s,
            jump_translation_threshold_m=args.jump_translation_threshold_m,
            jump_yaw_threshold_deg=args.jump_yaw_threshold_deg,
            repeat_translation_epsilon_m=args.repeat_translation_epsilon_m,
            repeat_yaw_epsilon_deg=args.repeat_yaw_epsilon_deg,
            start_end_tol_s=args.start_end_tol_s,
        )
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
