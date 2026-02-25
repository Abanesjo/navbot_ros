#!/usr/bin/env python3
"""Plot odometry trajectories and covariance ellipses from a CSV log."""

import csv
import math
import os
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from matplotlib.patches import Ellipse
from rclpy.node import Node


DEFAULT_CSV_PATH = "/home/nav/ros2_ws/src/navbot_ros/data/data.csv"
DEFAULT_ELLIPSE_INTERVAL_S = 0.1
SIGMA_SCALE = 1.0
SOURCES = ("wheel", "camera", "filtered")
COLORS = {
    "wheel": "#1f77b4",
    "camera": "#ff7f0e",
    "filtered": "#2ca02c",
}
GROUND_TRUTH_SOURCE = "ground_truth"
WHEEL_START_RADIUS_M = 0.01


class OdometryPlotter(Node):
    def __init__(self) -> None:
        super().__init__("odometry_plotter")
        self.declare_parameter("csv_path", DEFAULT_CSV_PATH)
        self.declare_parameter("ellipse_interval_s", DEFAULT_ELLIPSE_INTERVAL_S)

        self.csv_path = os.path.expanduser(
            str(self.get_parameter("csv_path").get_parameter_value().string_value)
        )
        self.ellipse_interval_s = float(
            self.get_parameter("ellipse_interval_s")
            .get_parameter_value()
            .double_value
        )


def _parse_csv(csv_path: str) -> List[Dict[str, float]]:
    if os.path.isdir(csv_path):
        raise IsADirectoryError(
            f"csv_path must be a CSV file, but a directory was provided: {csv_path}"
        )
    if not csv_path.lower().endswith(".csv"):
        raise ValueError(f"csv_path must point to a .csv file: {csv_path}")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows: List[Dict[str, float]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for raw_row in reader:
            parsed: Dict[str, float] = {}
            try:
                for key, value in raw_row.items():
                    parsed[key] = float(value) if value is not None and value != "" else math.nan
            except ValueError:
                # Skip malformed rows and continue plotting whatever is usable.
                continue
            rows.append(parsed)

    if not rows:
        raise ValueError(f"No valid rows found in CSV: {csv_path}")
    return rows


def _select_ellipse_rows(rows: List[Dict[str, float]], interval_s: float) -> List[Dict[str, float]]:
    if interval_s <= 0.0:
        return rows

    selected: List[Dict[str, float]] = []
    next_t = None
    for row in rows:
        t = row.get("time_from_start_s", math.nan)
        if not math.isfinite(t):
            continue
        if next_t is None or t >= next_t:
            selected.append(row)
            next_t = t + interval_s
    return selected


def _find_plot_start_index(rows: List[Dict[str, float]]) -> int:
    for idx, row in enumerate(rows):
        x = row.get("wheel_x", math.nan)
        y = row.get("wheel_y", math.nan)
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if math.hypot(x, y) > WHEEL_START_RADIUS_M:
            return idx
    return 0


def _add_covariance_ellipse(ax, x: float, y: float, cov_2x2: np.ndarray, color: str) -> None:
    if cov_2x2.shape != (2, 2):
        return
    if not np.all(np.isfinite(cov_2x2)):
        return

    cov_2x2 = 0.5 * (cov_2x2 + cov_2x2.T)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov_2x2)
    except np.linalg.LinAlgError:
        return

    if np.any(eigvals <= 0.0):
        return

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    major_axis = eigvecs[:, 0]
    angle_deg = math.degrees(math.atan2(major_axis[1], major_axis[0]))
    width = 2.0 * SIGMA_SCALE * math.sqrt(float(eigvals[0]))
    height = 2.0 * SIGMA_SCALE * math.sqrt(float(eigvals[1]))

    ellipse = Ellipse(
        (x, y),
        width=width,
        height=height,
        angle=angle_deg,
        fill=False,
        edgecolor=color,
        linewidth=0.9,
        alpha=1.0,
    )
    ax.add_patch(ellipse)


def _compute_ground_truth_rmse(rows: List[Dict[str, float]], source: str) -> tuple[float, int]:
    squared_errors: List[float] = []
    for row in rows:
        odom_x = row.get(f"{source}_x", math.nan)
        odom_y = row.get(f"{source}_y", math.nan)
        ground_truth_x = row.get("ground_truth_x", math.nan)
        ground_truth_y = row.get("ground_truth_y", math.nan)
        if not all(
            math.isfinite(v)
            for v in (odom_x, odom_y, ground_truth_x, ground_truth_y)
        ):
            continue
        dx = odom_x - ground_truth_x
        dy = odom_y - ground_truth_y
        squared_errors.append(dx * dx + dy * dy)

    if not squared_errors:
        return math.nan, 0
    return math.sqrt(sum(squared_errors) / len(squared_errors)), len(squared_errors)


def _unwrap_angle_series(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(arr))
    if finite_idx.size == 0:
        return out
    out[finite_idx] = np.unwrap(arr[finite_idx])
    return out


def _comparison_marker_points(
    rows: List[Dict[str, float]], source: str
) -> tuple[List[float], List[float]]:
    x_points: List[float] = []
    y_points: List[float] = []
    for row in rows:
        odom_x = row.get(f"{source}_x", math.nan)
        odom_y = row.get(f"{source}_y", math.nan)
        ground_truth_x = row.get("ground_truth_x", math.nan)
        ground_truth_y = row.get("ground_truth_y", math.nan)
        if not all(
            math.isfinite(v)
            for v in (odom_x, odom_y, ground_truth_x, ground_truth_y)
        ):
            continue
        # Plot in odom-frame convention: horizontal=y, vertical=x.
        x_points.append(odom_y)
        y_points.append(odom_x)
    return x_points, y_points


def _plot(rows: List[Dict[str, float]], csv_path: str, ellipse_interval_s: float) -> str:
    start_idx = _find_plot_start_index(rows)
    rows = rows[start_idx:]
    rmse_by_source = {
        source: _compute_ground_truth_rmse(rows, source) for source in SOURCES
    }
    fig, ax = plt.subplots(figsize=(10, 8))

    for source in SOURCES:
        x_points: List[float] = []
        y_points: List[float] = []
        for row in rows:
            x = row.get(f"{source}_x", math.nan)
            y = row.get(f"{source}_y", math.nan)
            if math.isfinite(x) and math.isfinite(y):
                # Plot in odom-frame convention: horizontal=y, vertical=x.
                x_points.append(y)
                y_points.append(x)
        if x_points and y_points:
            ax.plot(
                x_points,
                y_points,
                label=source,
                color=COLORS[source],
                linewidth=3.0,
                alpha=1.0,
            )

    ground_truth_x_points: List[float] = []
    ground_truth_y_points: List[float] = []
    for row in rows:
        x = row.get("ground_truth_x", math.nan)
        y = row.get("ground_truth_y", math.nan)
        if math.isfinite(x) and math.isfinite(y):
            # Plot in odom-frame convention: horizontal=y, vertical=x.
            ground_truth_x_points.append(y)
            ground_truth_y_points.append(x)
    if ground_truth_x_points and ground_truth_y_points:
        ax.plot(
            ground_truth_x_points,
            ground_truth_y_points,
            label=GROUND_TRUTH_SOURCE,
            color="black",
            linestyle="None",
            marker="s",
            markersize=7.0,
            markeredgewidth=1.0,
            alpha=1.0,
            zorder=4,
        )

    for source in SOURCES:
        marker_x_points, marker_y_points = _comparison_marker_points(rows, source)
        if not marker_x_points or not marker_y_points:
            continue
        ax.plot(
            marker_x_points,
            marker_y_points,
            linestyle="None",
            marker="o",
            markersize=4.0,
            markerfacecolor="none",
            markeredgecolor="red",
            markeredgewidth=1.0,
            alpha=1.0,
        )

    ellipse_rows = _select_ellipse_rows(rows, ellipse_interval_s)
    for row in ellipse_rows:
        for source in SOURCES:
            x = row.get(f"{source}_x", math.nan)
            y = row.get(f"{source}_y", math.nan)
            x_var = row.get(f"{source}_x_var", math.nan)
            y_var = row.get(f"{source}_y_var", math.nan)
            xy_cov = row.get(f"{source}_xy_cov", math.nan)
            if not all(math.isfinite(v) for v in (x, y, x_var, y_var, xy_cov)):
                continue
            # Plot coordinates are [y, x], so transform covariance into that basis.
            cov_plot = np.array([[y_var, xy_cov], [xy_cov, x_var]], dtype=float)
            _add_covariance_ellipse(ax, y, x, cov_plot, COLORS[source])

    rmse_parts = []
    for source in SOURCES:
        rmse_m, n_samples = rmse_by_source[source]
        if math.isfinite(rmse_m):
            rmse_parts.append(f"{source}: {rmse_m:.3f} m (n={n_samples})")
        else:
            rmse_parts.append(f"{source}: N/A")
    rmse_text = "RMSE vs ground truth: " + " | ".join(rmse_parts)
    ax.set_title(f"Odometry Trajectories with 1-Sigma Covariance Ellipses\n{rmse_text}")
    ax.set_xlabel("y [m] (positive left)")
    ax.set_ylabel("x [m] (positive up)")
    ax.grid(True, alpha=1.0)
    ax.axis("equal")
    ax.invert_xaxis()
    ax.legend()
    fig.tight_layout()

    input_path = Path(csv_path)
    output_path = input_path.with_name(f"{input_path.stem}_odometry_plot.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def _plot_yaw_uncertainty(rows: List[Dict[str, float]], csv_path: str) -> str:
    start_idx = _find_plot_start_index(rows)
    rows = rows[start_idx:]

    times = np.asarray([row.get("time_from_start_s", math.nan) for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(12, 6))

    for source in SOURCES:
        yaw_values = [row.get(f"{source}_yaw", math.nan) for row in rows]
        yaw_var_values = np.asarray(
            [row.get(f"{source}_yaw_var", math.nan) for row in rows], dtype=float
        )
        yaw_mean = _unwrap_angle_series(yaw_values)

        yaw_sigma = np.full(yaw_var_values.shape, np.nan, dtype=float)
        finite_var = np.isfinite(yaw_var_values)
        yaw_sigma[finite_var] = np.sqrt(np.maximum(yaw_var_values[finite_var], 0.0))

        yaw_upper = yaw_mean + yaw_sigma
        yaw_lower = yaw_mean - yaw_sigma

        band_mask = np.isfinite(times) & np.isfinite(yaw_upper) & np.isfinite(yaw_lower)
        mean_mask = np.isfinite(times) & np.isfinite(yaw_mean)

        ax.fill_between(
            times,
            yaw_lower,
            yaw_upper,
            where=band_mask,
            color=COLORS[source],
            alpha=0.15,
            interpolate=False,
        )
        ax.plot(times, yaw_upper, color=COLORS[source], linewidth=1.0, alpha=0.8)
        ax.plot(times, yaw_lower, color=COLORS[source], linewidth=1.0, alpha=0.8)
        ax.plot(
            times,
            yaw_mean,
            label=f"{source} yaw",
            color=COLORS[source],
            linewidth=2.5,
            alpha=1.0,
        )

        # Mark the exact odom yaw points used in RMSE comparisons (rows with ground truth present).
        comparison_mask = (
            mean_mask
            & np.isfinite(
                np.asarray([row.get("ground_truth_yaw", math.nan) for row in rows], dtype=float)
            )
        )
        ax.plot(
            times[comparison_mask],
            yaw_mean[comparison_mask],
            linestyle="None",
            marker="o",
            markersize=4.0,
            markerfacecolor="none",
            markeredgecolor="red",
            markeredgewidth=1.0,
            alpha=1.0,
            zorder=5,
        )

    ground_truth_yaw = _unwrap_angle_series([row.get("ground_truth_yaw", math.nan) for row in rows])
    gt_mask = np.isfinite(times) & np.isfinite(ground_truth_yaw)
    ax.plot(
        times[gt_mask],
        ground_truth_yaw[gt_mask],
        label="ground_truth yaw",
        color="black",
        linestyle="None",
        marker="s",
        markersize=7.0,
        markeredgewidth=1.0,
        alpha=1.0,
        zorder=6,
    )

    ax.set_title("Yaw vs Time with ±1σ Uncertainty Bands (Unwrapped Angles)")
    ax.set_xlabel("time_from_start_s [s]")
    ax.set_ylabel("yaw [rad] (continuous / unwrapped)")
    ax.grid(True, alpha=1.0)
    ax.legend()
    fig.tight_layout()

    input_path = Path(csv_path)
    output_path = input_path.with_name(f"{input_path.stem}_yaw_plot.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def main(args=None) -> int:
    rclpy.init(args=args)
    node = OdometryPlotter()
    exit_code = 0
    try:
        node.get_logger().info(
            f"Reading odometry CSV: {node.csv_path} (ellipse_interval_s={node.ellipse_interval_s})"
        )
        rows = _parse_csv(node.csv_path)
        trajectory_output_path = _plot(rows, node.csv_path, node.ellipse_interval_s)
        yaw_output_path = _plot_yaw_uncertainty(rows, node.csv_path)
        node.get_logger().info(f"Saved odometry plot: {trajectory_output_path}")
        node.get_logger().info(f"Saved yaw uncertainty plot: {yaw_output_path}")
    except Exception as exc:  # noqa: BLE001
        node.get_logger().error(str(exc))
        exit_code = 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
