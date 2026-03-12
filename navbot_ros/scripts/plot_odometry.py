#!/usr/bin/env python3
"""Plot odometry trajectories and covariance ellipses from CSV logs.

Outputs, for each CSV:
  1) *_odometry_plot.png            : wheel + camera + filtered + GT + ellipses
  2) *_filtered_gt_plot.png         : filtered + GT + scaled filtered ellipses
  3) *_filtered_covariance_plot.png : focus on filtered covariance (scaled), with GT context

Coordinate convention (as requested):
  - floor +x is LEFT  (so we invert matplotlib x-axis)
  - floor +y is DOWN  (so we invert matplotlib y-axis)

No yaw plots are generated.
"""

import csv
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from matplotlib.patches import Ellipse
from rclpy.node import Node


DEFAULT_CSV_PATH = ""
DEFAULT_CSV_PATHS = (
    "/home/nav/ros2_ws/src/navbot_ros/data/baseline.csv",
    "/home/nav/ros2_ws/src/navbot_ros/data/noview.csv",
    "/home/nav/ros2_ws/src/navbot_ros/data/online.csv",
    "/home/nav/ros2_ws/src/navbot_ros/data/unknown_start.csv",
)

DEFAULT_ELLIPSE_INTERVAL_S = 0.10

# Global 1-sigma scale for wheel/camera ellipses
DEFAULT_SIGMA_SCALE = 1.0

# Separate scale multiplier for filtered ellipses in the extra plots
DEFAULT_FILTERED_COVARIANCE_SCALE = 10.0

SOURCES = ("wheel", "camera", "filtered")
COLORS = {
    "wheel": "#1f77b4",
    "camera": "red",
    "filtered": "#2ca02c",
}

GROUND_TRUTH_SOURCE = "ground_truth"
WHEEL_START_RADIUS_M = 0.01


class OdometryPlotter(Node):
    def __init__(self) -> None:
        super().__init__("odometry_plotter")

        self.declare_parameter("csv_path", DEFAULT_CSV_PATH)
        self.declare_parameter("csv_paths", list(DEFAULT_CSV_PATHS))
        self.declare_parameter("ellipse_interval_s", DEFAULT_ELLIPSE_INTERVAL_S)
        self.declare_parameter("sigma_scale", DEFAULT_SIGMA_SCALE)
        self.declare_parameter("filtered_covariance_scale", DEFAULT_FILTERED_COVARIANCE_SCALE)

        self.csv_path = os.path.expanduser(
            str(self.get_parameter("csv_path").get_parameter_value().string_value)
        ).strip()

        self.csv_paths = [
            os.path.expanduser(str(path)).strip()
            for path in self.get_parameter("csv_paths").get_parameter_value().string_array_value
            if str(path).strip()
        ]

        self.ellipse_interval_s = float(
            self.get_parameter("ellipse_interval_s").get_parameter_value().double_value
        )
        self.sigma_scale = float(
            self.get_parameter("sigma_scale").get_parameter_value().double_value
        )
        self.filtered_covariance_scale = float(
            self.get_parameter("filtered_covariance_scale").get_parameter_value().double_value
        )


def _resolve_input_csv_paths(csv_path: str, csv_paths: Sequence[str]) -> List[str]:
    raw_paths = [csv_path] if csv_path else list(csv_paths)
    resolved: List[str] = []
    seen = set()
    for raw_path in raw_paths:
        path = os.path.expanduser(raw_path).strip()
        if not path or path in seen:
            continue
        seen.add(path)
        resolved.append(path)
    if not resolved:
        raise ValueError("No CSV paths configured. Provide csv_path or populate csv_paths.")
    return resolved


def _parse_csv(csv_path: str) -> List[Dict[str, float]]:
    if os.path.isdir(csv_path):
        raise IsADirectoryError(f"csv_path must be a CSV file, but got directory: {csv_path}")
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
                    parsed[key] = float(value) if value not in (None, "") else math.nan
            except ValueError:
                continue
            rows.append(parsed)

    if not rows:
        raise ValueError(f"No valid rows found in CSV: {csv_path}")
    return rows


def _find_plot_start_index(rows: List[Dict[str, float]]) -> int:
    # start once the wheel odom moves a tiny bit (so plots aren't dominated by initial)
    for idx, row in enumerate(rows):
        x = row.get("wheel_x", math.nan)
        y = row.get("wheel_y", math.nan)
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if math.hypot(x, y) > WHEEL_START_RADIUS_M:
            return idx
    return 0


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


def _add_covariance_ellipse(
    ax,
    x: float,
    y: float,
    cov_2x2: np.ndarray,
    color: str,
    sigma_scale: float,
) -> None:
    if cov_2x2.shape != (2, 2):
        return
    if not np.all(np.isfinite(cov_2x2)):
        return
    if sigma_scale <= 0.0:
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
    width = 2.0 * sigma_scale * math.sqrt(float(eigvals[0]))
    height = 2.0 * sigma_scale * math.sqrt(float(eigvals[1]))

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
        gt_x = row.get("ground_truth_x", math.nan)
        gt_y = row.get("ground_truth_y", math.nan)
        if not all(math.isfinite(v) for v in (odom_x, odom_y, gt_x, gt_y)):
            continue
        dx = odom_x - gt_x
        dy = odom_y - gt_y
        squared_errors.append(dx * dx + dy * dy)

    if not squared_errors:
        return math.nan, 0
    return math.sqrt(sum(squared_errors) / len(squared_errors)), len(squared_errors)


def _extract_xy_points(rows: List[Dict[str, float]], x_key: str, y_key: str) -> tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        x = row.get(x_key, math.nan)
        y = row.get(y_key, math.nan)
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return xs, ys


def _plot_ground_truth(ax, rows: List[Dict[str, float]]) -> None:
    gt_x, gt_y = _extract_xy_points(rows, "ground_truth_x", "ground_truth_y")
    if not gt_x or not gt_y:
        return

    # Connect with black line
    ax.plot(
        gt_x,
        gt_y,
        label=GROUND_TRUTH_SOURCE,
        color="black",
        linewidth=2.2,
        alpha=1.0,
        zorder=4,
    )
    # Also keep black square markers (as you had)
    ax.plot(
        gt_x,
        gt_y,
        linestyle="None",
        marker="s",
        markersize=6.5,
        markeredgewidth=1.0,
        color="black",
        alpha=1.0,
        zorder=5,
        label="_nolegend_",
    )


def _plot_source_trajectory(ax, rows: List[Dict[str, float]], source: str, linewidth: float = 3.0) -> None:
    xs, ys = _extract_xy_points(rows, f"{source}_x", f"{source}_y")
    if not xs or not ys:
        return
    ax.plot(xs, ys, label=source, color=COLORS[source], linewidth=linewidth, alpha=1.0)


def _comparison_marker_points(rows: List[Dict[str, float]], source: str) -> tuple[List[float], List[float]]:
    # red hollow markers: show samples where ground truth exists
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        odom_x = row.get(f"{source}_x", math.nan)
        odom_y = row.get(f"{source}_y", math.nan)
        gt_x = row.get("ground_truth_x", math.nan)
        gt_y = row.get("ground_truth_y", math.nan)
        if not all(math.isfinite(v) for v in (odom_x, odom_y, gt_x, gt_y)):
            continue
        xs.append(odom_x)
        ys.append(odom_y)
    return xs, ys

def _plot_covariance_ellipses(
    ax,
    rows: List[Dict[str, float]],
    sources: Iterable[str],
    ellipse_interval_s: float,  # now treated as distance in meters
    sigma_scale_default: float,
    sigma_scales: Dict[str, float] | None = None,
) -> None:
    sigma_scales = sigma_scales or {}

    # Distance-based selection (interval interpreted as meters)
    interval_m = float(ellipse_interval_s)
    ellipse_rows = _select_ellipse_rows_by_distance(
        rows=rows,
        interval_m=interval_m,
        source_for_distance="filtered",
    )

    for row in ellipse_rows:
        for source in sources:
            x = row.get(f"{source}_x", math.nan)
            y = row.get(f"{source}_y", math.nan)
            x_var = row.get(f"{source}_x_var", math.nan)
            y_var = row.get(f"{source}_y_var", math.nan)
            xy_cov = row.get(f"{source}_xy_cov", math.nan)

            if not all(math.isfinite(v) for v in (x, y, x_var, y_var, xy_cov)):
                continue

            cov = np.array([[x_var, xy_cov], [xy_cov, y_var]], dtype=float)
            _add_covariance_ellipse(
                ax=ax,
                x=x,
                y=y,
                cov_2x2=cov,
                color=COLORS[source],
                sigma_scale=sigma_scales.get(source, sigma_scale_default),
            )


def _style_axes_floor_convention(ax, title: str) -> None:
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("floor +x [m] (positive left)", fontsize=14)
    ax.set_ylabel("floor +y [m] (positive down)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)

    ax.grid(True, alpha=1.0)
    ax.axis("equal")

    ax.invert_xaxis()
    ax.invert_yaxis()

    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(fontsize=12)


def _save_plot(fig, csv_path: str, suffix: str) -> str:
    input_path = Path(csv_path)
    output_path = input_path.with_name(f"{input_path.stem}_{suffix}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def _plot_full_odometry(rows: List[Dict[str, float]], csv_path: str, ellipse_interval_s: float, sigma_scale: float) -> str:
    rmse_by_source = {source: _compute_ground_truth_rmse(rows, source) for source in SOURCES}
    fig, ax = plt.subplots(figsize=(10, 8))

    for source in SOURCES:
        _plot_source_trajectory(ax, rows, source, linewidth=3.0)

    _plot_ground_truth(ax, rows)

    # Red hollow markers at times where ground truth exists (per-source)
    for source in SOURCES:
        mx, my = _comparison_marker_points(rows, source)
        if mx and my:
            ax.plot(
                mx, my,
                linestyle="None",
                marker="o",
                markersize=4.0,
                markerfacecolor="none",
                markeredgecolor="red",
                markeredgewidth=1.0,
                alpha=1.0,
            )

    _plot_covariance_ellipses(
        ax=ax,
        rows=rows,
        sources=SOURCES,
        ellipse_interval_s=ellipse_interval_s,
        sigma_scale_default=sigma_scale,
    )

    rmse_parts = []
    for source in SOURCES:
        rmse_m, n = rmse_by_source[source]
        rmse_parts.append(f"{source}: {rmse_m:.3f} m (n={n})" if math.isfinite(rmse_m) else f"{source}: N/A")
    rmse_text = "RMSE vs GT: " + " | ".join(rmse_parts)

    _style_axes_floor_convention(ax, f"Odometry Trajectories with 1σ Covariance Ellipses\n{rmse_text}")
    fig.tight_layout()
    return _save_plot(fig, csv_path, "odometry_plot")


def _plot_filtered_gt(rows: List[Dict[str, float]], csv_path: str, ellipse_interval_s: float, sigma_scale: float, filtered_cov_scale: float) -> str:
    fig, ax = plt.subplots(figsize=(10, 8))
    _plot_source_trajectory(ax, rows, "filtered", linewidth=3.0)
    _plot_ground_truth(ax, rows)

    _plot_covariance_ellipses(
        ax=ax,
        rows=rows,
        sources=("filtered",),
        ellipse_interval_s=ellipse_interval_s,
        sigma_scale_default=sigma_scale,
        sigma_scales={"filtered": sigma_scale * filtered_cov_scale},
    )

    _style_axes_floor_convention(
        ax,
        f"Filtered vs Ground Truth (Filtered covariance scaled ×{filtered_cov_scale:.1f})",
    )
    fig.tight_layout()
    return _save_plot(fig, csv_path, "filtered_gt_plot")


def _plot_filtered_covariance_focus(rows: List[Dict[str, float]], csv_path: str, ellipse_interval_s: float, sigma_scale: float, filtered_cov_scale: float) -> str:
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw ellipses first, then paths (so ellipses are visible)
    _plot_covariance_ellipses(
        ax=ax,
        rows=rows,
        sources=("filtered",),
        ellipse_interval_s=ellipse_interval_s,
        sigma_scale_default=sigma_scale,
        sigma_scales={"filtered": sigma_scale * filtered_cov_scale},
    )
    _plot_source_trajectory(ax, rows, "filtered", linewidth=2.4)
    _plot_ground_truth(ax, rows)

    _style_axes_floor_convention(
        ax,
        "Filtered covariance focus (with GT + filtered context)\n"
        f"Filtered covariance scaled ×{filtered_cov_scale:.1f}",
    )
    fig.tight_layout()
    return _save_plot(fig, csv_path, "filtered_covariance_plot")


def _generate_plots_for_csv(
    rows: List[Dict[str, float]],
    csv_path: str,
    ellipse_interval_s: float,
    sigma_scale: float,
    filtered_cov_scale: float,
) -> List[str]:
    start_idx = _find_plot_start_index(rows)
    plot_rows = rows[start_idx:]
    return [
        _plot_full_odometry(plot_rows, csv_path, ellipse_interval_s, sigma_scale),
        _plot_filtered_gt(plot_rows, csv_path, ellipse_interval_s, sigma_scale, filtered_cov_scale),
        _plot_filtered_covariance_focus(plot_rows, csv_path, ellipse_interval_s, sigma_scale, filtered_cov_scale),
    ]


def _select_ellipse_rows_by_distance(
    rows: List[Dict[str, float]],
    interval_m: float,
    source_for_distance: str = "filtered",
) -> List[Dict[str, float]]:
    """Select rows such that consecutive selected rows are ~interval_m apart in traveled distance.

    Uses the specified source (default: filtered) to compute distance along the path.
    Falls back to wheel if filtered is unavailable for a row.
    """
    if interval_m <= 0.0:
        return rows

    selected: List[Dict[str, float]] = []

    last_x = None
    last_y = None
    dist_since_last_pick = 0.0

    for row in rows:
        # Prefer filtered pose for spacing; fallback to wheel if missing
        x = row.get(f"{source_for_distance}_x", math.nan)
        y = row.get(f"{source_for_distance}_y", math.nan)

        if not (math.isfinite(x) and math.isfinite(y)):
            x = row.get("wheel_x", math.nan)
            y = row.get("wheel_y", math.nan)

        if not (math.isfinite(x) and math.isfinite(y)):
            continue

        if last_x is None:
            # Always take the first valid point
            selected.append(row)
            last_x, last_y = x, y
            dist_since_last_pick = 0.0
            continue

        step = math.hypot(x - last_x, y - last_y)
        last_x, last_y = x, y
        dist_since_last_pick += step

        if dist_since_last_pick >= interval_m:
            selected.append(row)
            dist_since_last_pick = 0.0

    return selected



def main(args=None) -> int:
    rclpy.init(args=args)
    node = OdometryPlotter()
    exit_code = 0
    try:
        csv_paths = _resolve_input_csv_paths(node.csv_path, node.csv_paths)
        node.get_logger().info(
            f"Processing {len(csv_paths)} CSV(s); "
            f"ellipse_interval_s={node.ellipse_interval_s}, "
            f"sigma_scale={node.sigma_scale}, "
            f"filtered_covariance_scale={node.filtered_covariance_scale}"
        )

        for csv_path in csv_paths:
            try:
                node.get_logger().info(f"Reading: {csv_path}")
                rows = _parse_csv(csv_path)
                outs = _generate_plots_for_csv(
                    rows=rows,
                    csv_path=csv_path,
                    ellipse_interval_s=node.ellipse_interval_s,
                    sigma_scale=node.sigma_scale,
                    filtered_cov_scale=node.filtered_covariance_scale,
                )
                for out in outs:
                    node.get_logger().info(f"Saved plot: {out}")
            except Exception as exc:  # noqa: BLE001
                node.get_logger().error(f"{csv_path}: {exc}")
                exit_code = 1

    except Exception as exc:  # noqa: BLE001
        node.get_logger().error(str(exc))
        exit_code = 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
