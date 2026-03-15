#!/usr/bin/env python3
"""Plot PF comparison directly from a rosbag2 folder.

Reads:
  - /odom/camera       (used as ground-truth/reference)
  - /odom/pf_baseline
  - /odom/pf_variant

Outputs:
  1) *_pf_compare_xy.png
  2) *_pf_compare_error.png

Also prints RMSE against /odom/camera using nearest-timestamp matching.

Usage:
  python3 plot_odom_l4.py /path/to/bag_folder
  python3 plot_odom_l4.py /path/to/bag_folder --output-dir /tmp/plots
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


TOPICS = {
    "gt": "/odom/camera",
    "baseline": "/odom/pf_baseline",
    "variant": "/odom/pf_variant",
}

COLORS = {
    "gt": "black",
    "baseline": "#1f77b4",
    "variant": "#d62728",
}

# Break plotted trajectory if consecutive points jump by more than this.
MAX_STEP_M = 0.35

# Break plotted trajectory if consecutive timestamps are too far apart.
MAX_TIME_GAP_S = 0.50

# Visualization-only smoothing window for PF trajectories.
# Ground truth is left unsmoothed.
PLOT_SMOOTH_WINDOW = 7


def read_bag_odom_series(
    bag_path: str,
    use_header_stamp: bool = False,
) -> Dict[str, List[Tuple[float, float, float, float]]]:
    """Return dict: key -> list of (t_sec, x, y, yaw)."""
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    series = {key: [] for key in TOPICS.keys()}

    while reader.has_next():
        topic_name, data, timestamp_ns = reader.read_next()

        matching_key = None
        for key, wanted_topic in TOPICS.items():
            if topic_name == wanted_topic:
                matching_key = key
                break
        if matching_key is None:
            continue

        msg_type = get_message(type_map[topic_name])
        msg = deserialize_message(data, msg_type)

        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)

        qx = float(msg.pose.pose.orientation.x)
        qy = float(msg.pose.pose.orientation.y)
        qz = float(msg.pose.pose.orientation.z)
        qw = float(msg.pose.pose.orientation.w)

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        bag_t_sec = timestamp_ns * 1e-9
        if use_header_stamp:
            header_stamp = msg.header.stamp
            header_t_sec = float(header_stamp.sec) + float(header_stamp.nanosec) * 1e-9
            # Fall back to bag timestamp if header stamp is missing/invalid.
            t_sec = header_t_sec if header_t_sec > 0.0 else bag_t_sec
        else:
            t_sec = bag_t_sec
        series[matching_key].append((t_sec, x, y, yaw))

    # Make absolutely sure each series is time-sorted
    for key in series:
        series[key].sort(key=lambda item: item[0])

    return series


def extract_xy(series: List[Tuple[float, float, float, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not series:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )
    arr = np.asarray(series, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def nearest_time_match(
    ref_t: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray,
    est_t: np.ndarray, est_x: np.ndarray, est_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match estimate samples to nearest reference timestamps."""
    if len(ref_t) == 0 or len(est_t) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    matched_dx = []
    matched_dy = []
    for i in range(len(est_t)):
        idx = int(np.argmin(np.abs(ref_t - est_t[i])))
        matched_dx.append(est_x[i] - ref_x[idx])
        matched_dy.append(est_y[i] - ref_y[idx])
    return np.asarray(matched_dx), np.asarray(matched_dy)


def compute_rmse(
    ref_t: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray,
    est_t: np.ndarray, est_x: np.ndarray, est_y: np.ndarray,
) -> float:
    dx, dy = nearest_time_match(ref_t, ref_x, ref_y, est_t, est_x, est_y)
    if len(dx) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(dx * dx + dy * dy)))


def compute_time_error_series(
    ref_t: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray,
    est_t: np.ndarray, est_x: np.ndarray, est_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(ref_t) == 0 or len(est_t) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    errs = []
    times = []
    t0 = est_t[0]
    for i in range(len(est_t)):
        idx = int(np.argmin(np.abs(ref_t - est_t[i])))
        dx = est_x[i] - ref_x[idx]
        dy = est_y[i] - ref_y[idx]
        errs.append(math.hypot(dx, dy))
        times.append(est_t[i] - t0)
    return np.asarray(times), np.asarray(errs)


def segmented_xy(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    max_step_m: float = MAX_STEP_M,
    max_time_gap_s: float = MAX_TIME_GAP_S,
) -> Tuple[np.ndarray, np.ndarray]:
    """Insert NaNs where trajectory jumps too far, so lines are broken."""
    if len(t) == 0:
        return x, y

    xs = [x[0]]
    ys = [y[0]]

    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        ds = math.hypot(x[i] - x[i - 1], y[i] - y[i - 1])

        if ds > max_step_m or dt > max_time_gap_s:
            xs.append(np.nan)
            ys.append(np.nan)

        xs.append(x[i])
        ys.append(y[i])

    return np.asarray(xs), np.asarray(ys)


def count_segments(x: np.ndarray, y: np.ndarray) -> int:
    if len(x) == 0:
        return 0
    valid = ~(np.isnan(x) | np.isnan(y))
    if not np.any(valid):
        return 0
    segments = 0
    in_segment = False
    for is_valid in valid:
        if is_valid and not in_segment:
            segments += 1
            in_segment = True
        elif not is_valid:
            in_segment = False
    return segments


def smooth_with_nans(data: np.ndarray, window: int) -> np.ndarray:
    """Apply moving-average smoothing separately on each non-NaN segment."""
    if window <= 1 or len(data) == 0:
        return data.copy()

    out = data.copy()
    isnan = np.isnan(data)
    n = len(data)

    start = 0
    while start < n:
        while start < n and isnan[start]:
            start += 1
        if start >= n:
            break

        end = start
        while end < n and not isnan[end]:
            end += 1

        segment = data[start:end]
        if len(segment) >= window:
            kernel = np.ones(window, dtype=float) / float(window)
            out[start:end] = np.convolve(segment, kernel, mode="same")

        start = end

    return out


def plot_xy(
    series_dict,
    output_path: Path,
    baseline_rmse: float,
    variant_rmse: float,
    max_step_m: float = MAX_STEP_M,
    max_gap_s: float = MAX_TIME_GAP_S,
    segmentation_enabled: bool = True,
    smoothing_enabled: bool = True,
    points_only: bool = False,
) -> Dict[str, int]:
    fig, ax = plt.subplots(figsize=(10, 8))
    segment_counts: Dict[str, int] = {}

    for key in ["gt", "baseline", "variant"]:
        t, x, y, yaw = extract_xy(series_dict[key])
        if len(x) == 0:
            segment_counts[key] = 0
            continue

        if segmentation_enabled:
            x_plot, y_plot = segmented_xy(
                t, x, y, max_step_m=max_step_m, max_time_gap_s=max_gap_s
            )
        else:
            x_plot, y_plot = x.copy(), y.copy()

        # Smooth only the PF trajectories for visualization.
        if smoothing_enabled and key in ("baseline", "variant"):
            x_plot = smooth_with_nans(x_plot, PLOT_SMOOTH_WINDOW)
            y_plot = smooth_with_nans(y_plot, PLOT_SMOOTH_WINDOW)

        if segmentation_enabled:
            segment_counts[key] = count_segments(x_plot, y_plot)
        else:
            segment_counts[key] = 1 if len(x_plot) > 0 else 0

        if points_only:
            label = "ground truth / camera" if key == "gt" else key
            marker_size = 8 if key == "gt" else 6
            ax.scatter(x_plot, y_plot, color=COLORS[key], s=marker_size, label=label, zorder=4)
        elif key == "gt":
            ax.plot(
                x_plot,
                y_plot,
                color=COLORS[key],
                linewidth=2.8,
                label="ground truth / camera",
                zorder=5,
            )
        else:
            ax.plot(
                x_plot,
                y_plot,
                color=COLORS[key],
                linewidth=2.0,
                label=key,
                zorder=3,
            )

    title = "PF comparison against camera ground truth"
    metrics = (
        f"RMSE vs camera: baseline: {baseline_rmse:.3f} m | "
        f"variant: {variant_rmse:.3f} m"
    )
    ax.set_title(f"{title}\n{metrics}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.4)
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return segment_counts


def plot_error(series_dict, output_path: Path) -> None:
    gt_t, gt_x, gt_y, gt_yaw = extract_xy(series_dict["gt"])
    fig, ax = plt.subplots(figsize=(10, 6))

    for key in ["baseline", "variant"]:
        est_t, est_x, est_y, est_yaw = extract_xy(series_dict[key])
        t_rel, err = compute_time_error_series(gt_t, gt_x, gt_y, est_t, est_x, est_y)
        if len(t_rel) == 0:
            continue
        ax.plot(t_rel, err, color=COLORS[key], linewidth=2.0, label=f"{key} position error")

    ax.set_title("Position error vs camera ground truth")
    ax.set_xlabel("time from start [s]")
    ax.set_ylabel("position error [m]")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_path", help="Path to rosbag2 folder")
    parser.add_argument("--output-dir", default="", help="Optional output directory")
    parser.add_argument(
        "--max-step-m",
        type=float,
        default=MAX_STEP_M,
        help=f"Segmentation threshold on per-step position jump [m] (default: {MAX_STEP_M})",
    )
    parser.add_argument(
        "--max-gap-s",
        type=float,
        default=MAX_TIME_GAP_S,
        help=f"Segmentation threshold on timestamp gap [s] (default: {MAX_TIME_GAP_S})",
    )
    parser.add_argument(
        "--no-segmentation",
        action="store_true",
        help="Disable NaN-based trajectory segmentation.",
    )
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable PF trajectory smoothing.",
    )
    parser.add_argument(
        "--points-only",
        action="store_true",
        help="Render XY trajectories as points (scatter) instead of lines.",
    )
    parser.add_argument(
        "--use-header-stamp",
        action="store_true",
        help="Use Odometry header stamp instead of rosbag timestamp (fallback to bag stamp if invalid).",
    )
    args = parser.parse_args()

    bag_path = Path(args.bag_path).expanduser().resolve()
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else bag_path
    output_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    try:
        series_dict = read_bag_odom_series(
            str(bag_path), use_header_stamp=args.use_header_stamp
        )
    finally:
        rclpy.shutdown()

    gt_t, gt_x, gt_y, gt_yaw = extract_xy(series_dict["gt"])
    base_t, base_x, base_y, base_yaw = extract_xy(series_dict["baseline"])
    var_t, var_x, var_y, var_yaw = extract_xy(series_dict["variant"])

    baseline_rmse = compute_rmse(gt_t, gt_x, gt_y, base_t, base_x, base_y)
    variant_rmse = compute_rmse(gt_t, gt_x, gt_y, var_t, var_x, var_y)

    print(f"Samples /odom/camera:      {len(gt_t)}")
    print(f"Samples /odom/pf_baseline: {len(base_t)}")
    print(f"Samples /odom/pf_variant:  {len(var_t)}")

    print(f"Baseline RMSE vs /odom/camera: {baseline_rmse:.4f} m")
    print(f"Variant  RMSE vs /odom/camera: {variant_rmse:.4f} m")

    stem = bag_path.name
    xy_path = output_dir / f"{stem}_pf_compare_xy.png"
    err_path = output_dir / f"{stem}_pf_compare_error.png"

    segment_counts = plot_xy(
        series_dict,
        xy_path,
        baseline_rmse=baseline_rmse,
        variant_rmse=variant_rmse,
        max_step_m=args.max_step_m,
        max_gap_s=args.max_gap_s,
        segmentation_enabled=not args.no_segmentation,
        smoothing_enabled=not args.no_smoothing,
        points_only=args.points_only,
    )
    plot_error(series_dict, err_path)

    print(
        "XY mode: "
        f"points_only={args.points_only}, "
        f"segmentation={'off' if args.no_segmentation else 'on'}, "
        f"smoothing={'off' if args.no_smoothing else 'on'}"
    )
    print(
        "Timestamp source: "
        f"{'odom header stamp (fallback to bag stamp)' if args.use_header_stamp else 'rosbag record timestamp'}"
    )
    print(
        f"Segmentation thresholds: max_step_m={args.max_step_m:.3f}, max_gap_s={args.max_gap_s:.3f}"
    )
    print("Segment counts (gt, baseline, variant): "
          f"{segment_counts.get('gt', 0)}, {segment_counts.get('baseline', 0)}, {segment_counts.get('variant', 0)}")

    print(f"Saved: {xy_path}")
    print(f"Saved: {err_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
