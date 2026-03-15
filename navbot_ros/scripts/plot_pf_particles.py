#!/usr/bin/env python3
"""Offline particle-cloud snapshot plotting from rosbag2.

Reads:
  - /pf/particles_baseline (PoseArray)
  - /pf/particles_variant  (PoseArray)
  - /odom/camera           (Odometry)
  - /odom/pf_baseline      (Odometry)
  - /odom/pf_variant       (Odometry)

Generates snapshot XY plots for lab-report visualization.
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


TOPICS = {
    "particles_baseline": "/pf/particles_baseline",
    "particles_variant": "/pf/particles_variant",
    "odom_camera": "/odom/camera",
    "odom_baseline": "/odom/pf_baseline",
    "odom_variant": "/odom/pf_variant",
}

# Report-friendly typography defaults.
SUPTITLE_FONT_SIZE = 18
PLOT_TITLE_FONT_SIZE = 14
AXIS_LABEL_FONT_SIZE = 13
LEGEND_FONT_SIZE = 11

# ── odom marker appearance ─────────────────────────────────────────────────────
ODOM_MARKER_SIZE    = 260    # was 100 — large enough to see at a glance
ODOM_MARKER_LW      = 2.5   # linewidth for 'x' cross
ODOM_ZORDER         = 6     # always on top of particle scatter
GT_MARKER_SIZE      = 110   # camera GT dot
GT_ZORDER           = 7

ParticleSample = Tuple[float, np.ndarray]
OdomSample = Tuple[float, float, float, float]


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def read_bag_data(
    bag_path: str,
) -> Tuple[Dict[str, List[ParticleSample]], Dict[str, List[OdomSample]], float, float]:
    """Read requested particle and odom streams from a rosbag2 folder."""
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    particle_series: Dict[str, List[ParticleSample]] = {
        "particles_baseline": [],
        "particles_variant": [],
    }
    odom_series: Dict[str, List[OdomSample]] = {
        "odom_camera": [],
        "odom_baseline": [],
        "odom_variant": [],
    }

    bag_t0: Optional[float] = None
    bag_t1: Optional[float] = None

    while reader.has_next():
        topic_name, data, timestamp_ns = reader.read_next()

        if topic_name not in type_map:
            continue

        bag_t = float(timestamp_ns) * 1e-9
        if bag_t0 is None:
            bag_t0 = bag_t
        bag_t1 = bag_t

        key: Optional[str] = None
        for k, wanted in TOPICS.items():
            if topic_name == wanted:
                key = k
                break
        if key is None:
            continue

        msg_type = get_message(type_map[topic_name])
        msg = deserialize_message(data, msg_type)

        if key in ("particles_baseline", "particles_variant"):
            if not msg.poses:
                pts = np.empty((0, 2), dtype=float)
            else:
                pts = np.array(
                    [[float(p.position.x), float(p.position.y)] for p in msg.poses],
                    dtype=float,
                )
            particle_series[key].append((bag_t, pts))
        else:
            x = float(msg.pose.pose.position.x)
            y = float(msg.pose.pose.position.y)
            q = msg.pose.pose.orientation
            yaw = yaw_from_quaternion(
                float(q.x), float(q.y), float(q.z), float(q.w)
            )
            odom_series[key].append((bag_t, x, y, yaw))

    for key in particle_series:
        particle_series[key].sort(key=lambda s: s[0])
    for key in odom_series:
        odom_series[key].sort(key=lambda s: s[0])

    if bag_t0 is None or bag_t1 is None:
        raise RuntimeError(f"No readable messages found in bag: {bag_path}")

    return particle_series, odom_series, bag_t0, bag_t1


def nearest_index(times: np.ndarray, target_t: float) -> Optional[int]:
    if len(times) == 0:
        return None
    return int(np.argmin(np.abs(times - target_t)))


def nearest_particle_match(
    series: List[ParticleSample],
    target_t: float,
    max_gap_s: float,
) -> Optional[Tuple[float, np.ndarray, float]]:
    if not series:
        return None
    times = np.asarray([s[0] for s in series], dtype=float)
    idx = nearest_index(times, target_t)
    if idx is None:
        return None
    t_match, pts = series[idx]
    dt = float(t_match - target_t)
    if abs(dt) > max_gap_s:
        return None
    return t_match, pts, dt


def nearest_odom_match(
    series: List[OdomSample],
    target_t: float,
    max_gap_s: float,
) -> Optional[Tuple[float, float, float, float, float]]:
    if not series:
        return None
    times = np.asarray([s[0] for s in series], dtype=float)
    idx = nearest_index(times, target_t)
    if idx is None:
        return None
    t_match, x, y, yaw = series[idx]
    dt = float(t_match - target_t)
    if abs(dt) > max_gap_s:
        return None
    return t_match, x, y, yaw, dt


def select_variant_t_star(
    variant_series: List[OdomSample],
    gt_x: float,
    gt_y: float,
    t_start: float,
    t_end: float,
) -> Optional[Tuple[float, float]]:
    """Pick variant-odom time in [t_start, t_end] with minimum XY error to GT."""
    best_t: Optional[float] = None
    best_err: float = float("inf")

    for t, x, y, _ in variant_series:
        if t < t_start or t > t_end:
            continue
        err = math.hypot(x - gt_x, y - gt_y)
        if err < best_err:
            best_err = err
            best_t = t

    if best_t is None:
        return None
    return best_t, best_err


def trajectory_xy(series: List[OdomSample]) -> Tuple[np.ndarray, np.ndarray]:
    if not series:
        return np.array([], dtype=float), np.array([], dtype=float)
    arr = np.asarray(series, dtype=float)
    return arr[:, 1], arr[:, 2]


def format_snapshot_tag(t_rel_s: float) -> str:
    return f"t{t_rel_s:05.1f}".replace(".", "p")


def resolve_snapshot_times(
    requested_times: Optional[List[float]],
    num_snapshots: int,
    duration_s: float,
) -> List[float]:
    if requested_times is not None and len(requested_times) > 0:
        return sorted(requested_times)

    if num_snapshots <= 0:
        raise ValueError("--num-snapshots must be > 0 when --times is not provided")

    if duration_s <= 0.0:
        return [0.0]

    ts = np.linspace(0.0, duration_s, num_snapshots + 2, dtype=float)[1:-1]
    return [float(t) for t in ts]


def draw_common_context(
    ax,
    show_trajectories: bool,
    odom_series: Dict[str, List[OdomSample]],
    xlim: Optional[Tuple[float, float]],
    ylim: Optional[Tuple[float, float]],
) -> None:
    if show_trajectories:
        cam_x, cam_y = trajectory_xy(odom_series["odom_camera"])
        base_x, base_y = trajectory_xy(odom_series["odom_baseline"])
        var_x, var_y = trajectory_xy(odom_series["odom_variant"])

        if len(cam_x):
            ax.plot(cam_x, cam_y, color="black", linewidth=1.2, alpha=0.20, label="camera traj")
        if len(base_x):
            ax.plot(base_x, base_y, color="#1f77b4", linewidth=1.0, alpha=0.18, label="baseline traj")
        if len(var_x):
            ax.plot(var_x, var_y, color="#d62728", linewidth=1.0, alpha=0.18, label="variant traj")

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel("x [m]", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("y [m]", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.55, linewidth=0.6)
    ax.set_aspect("equal", adjustable="box")


def _draw_markers(
    ax,
    cam_match,
    odom_base_match,
    odom_var_match,
) -> None:
    """Draw GT and odom position markers with high visibility."""
    if cam_match is not None:
        _, x, y, _, _ = cam_match
        ax.scatter(
            [x], [y],
            s=GT_MARKER_SIZE, c="black", marker="o",
            edgecolors="white", linewidths=1.2,
            zorder=GT_ZORDER, label="camera GT",
        )

    if odom_base_match is not None:
        _, x, y, _, _ = odom_base_match
        ax.scatter(
            [x], [y],
            s=ODOM_MARKER_SIZE, c="#1f77b4", marker="X",
            edgecolors="white", linewidths=0.8,
            zorder=ODOM_ZORDER, label="baseline odom",
        )

    if odom_var_match is not None:
        _, x, y, _, _ = odom_var_match
        ax.scatter(
            [x], [y],
            s=ODOM_MARKER_SIZE, c="#d62728", marker="X",
            edgecolors="white", linewidths=0.8,
            zorder=ODOM_ZORDER, label="variant odom",
        )


def plot_snapshot_overlay(
    out_path: Path,
    gt_snapshot_t_rel: float,
    cloud_selection_lag_s: float,
    baseline_match: Optional[Tuple[float, np.ndarray, float]],
    variant_match: Optional[Tuple[float, np.ndarray, float]],
    cam_match: Optional[Tuple[float, float, float, float, float]],
    odom_base_match: Optional[Tuple[float, float, float, float, float]],
    odom_var_match: Optional[Tuple[float, float, float, float, float]],
    show_trajectories: bool,
    odom_series: Dict[str, List[OdomSample]],
    marker_size: float,
    particle_alpha: float,
    xlim: Optional[Tuple[float, float]],
    ylim: Optional[Tuple[float, float]],
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))

    if xlim is None or ylim is None:
        auto_xlim, auto_ylim = compute_snapshot_plot_limits(
            baseline_match=baseline_match,
            variant_match=variant_match,
            cam_match=cam_match,
            odom_base_match=odom_base_match,
            odom_var_match=odom_var_match,
        )
        xlim_to_use = xlim if xlim is not None else auto_xlim
        ylim_to_use = ylim if ylim is not None else auto_ylim
    else:
        xlim_to_use, ylim_to_use = xlim, ylim

    draw_common_context(ax, show_trajectories, odom_series, xlim_to_use, ylim_to_use)

    if baseline_match is not None:
        _, pts, _ = baseline_match
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=marker_size, c="#1f77b4",
                       alpha=particle_alpha, label="baseline particles", zorder=2)

    if variant_match is not None:
        _, pts, _ = variant_match
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=marker_size, c="#d62728",
                       alpha=particle_alpha, label="variant particles", zorder=2)

    _draw_markers(ax, cam_match, odom_base_match, odom_var_match)

    ax.set_title(f"t = {gt_snapshot_t_rel:.1f} s", fontsize=PLOT_TITLE_FONT_SIZE)
    ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_snapshot_triptych_overlay(
    out_path: Path,
    snapshots: List[Dict[str, object]],
    show_trajectories: bool,
    odom_series: Dict[str, List[OdomSample]],
    marker_size: float,
    particle_alpha: float,
    xlim: Optional[Tuple[float, float]],
    ylim: Optional[Tuple[float, float]],
) -> None:
    """Render 3 selected snapshots in a single 1×3 figure.

    KEY CHANGE vs original: each panel gets its own per-snapshot axis limits
    (no sharex/sharey). This eliminates the whitespace caused by the shared
    global bounding box that spans all three robot positions.
    """
    # ── No sharex/sharey — per-panel limits are set inside the loop ────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))

    # ── Pass 1: compute per-panel limits, then equalize so every panel shares
    # the same span (= same scale, same physical size with equal-aspect axes). ──
    raw_panel_limits = []
    for snap in snapshots[:3]:
        if xlim is not None and ylim is not None:
            raw_panel_limits.append((xlim, ylim))
        else:
            auto_xlim, auto_ylim = compute_snapshot_plot_limits(
                baseline_match=snap["baseline_match"],
                variant_match=snap["variant_match"],
                cam_match=snap["cam_match"],
                odom_base_match=snap["odom_base_match"],
                odom_var_match=snap["odom_var_match"],
            )
            raw_panel_limits.append((
                xlim if xlim is not None else auto_xlim,
                ylim if ylim is not None else auto_ylim,
            ))

    equalized_limits = equalize_panel_limits(raw_panel_limits)

    # ── Pass 2: draw each panel with its equalized limits ─────────────────────
    for i, snap in enumerate(snapshots[:3]):
        ax = axes[i]

        baseline_match  = snap["baseline_match"]
        variant_match   = snap["variant_match"]
        cam_match       = snap["cam_match"]
        odom_base_match = snap["odom_base_match"]
        odom_var_match  = snap["odom_var_match"]
        gt_t_rel        = float(snap["gt_t_rel"])

        xlim_panel, ylim_panel = equalized_limits[i]

        draw_common_context(ax, show_trajectories, odom_series, xlim_panel, ylim_panel)

        if baseline_match is not None:
            _, pts, _ = baseline_match
            if len(pts):
                ax.scatter(
                    pts[:, 0], pts[:, 1],
                    s=marker_size, c="#1f77b4", alpha=particle_alpha,
                    label="baseline particles", zorder=2,
                )

        if variant_match is not None:
            _, pts, _ = variant_match
            if len(pts):
                ax.scatter(
                    pts[:, 0], pts[:, 1],
                    s=marker_size, c="#d62728", alpha=particle_alpha,
                    label="variant particles", zorder=2,
                )

        _draw_markers(ax, cam_match, odom_base_match, odom_var_match)

        ax.set_title(f"t = {gt_t_rel:.1f} s", fontsize=PLOT_TITLE_FONT_SIZE)

    # ── Shared legend at the bottom ────────────────────────────────────────────
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        # Collect unique handles across all axes (different panels may have
        # different subsets of artists).
        all_handles: Dict[str, object] = {}
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            for lbl, hdl in zip(l, h):
                all_handles.setdefault(lbl, hdl)
        fig.legend(
            all_handles.values(), all_handles.keys(),
            loc="lower center", ncol=5,
            bbox_to_anchor=(0.5, 0.01),
            prop={"size": LEGEND_FONT_SIZE},
            framealpha=0.92,
            edgecolor="#cccccc",
        )

    plot_area_cx = 0.5 * (0.06 + 0.99)  # midpoint of left+right margins
    fig.suptitle("Particle cloud snapshots", x=plot_area_cx, y=0.96, fontsize=SUPTITLE_FONT_SIZE)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.85, bottom=0.16, wspace=0.28)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_snapshot_single(
    out_path: Path,
    gt_snapshot_t_rel: float,
    cloud_selection_lag_s: float,
    cloud_name: str,
    cloud_color: str,
    cloud_match: Optional[Tuple[float, np.ndarray, float]],
    cam_match: Optional[Tuple[float, float, float, float, float]],
    odom_match: Optional[Tuple[float, float, float, float, float]],
    show_trajectories: bool,
    odom_series: Dict[str, List[OdomSample]],
    marker_size: float,
    particle_alpha: float,
    xlim: Optional[Tuple[float, float]],
    ylim: Optional[Tuple[float, float]],
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))

    baseline_for_limits = cloud_match if cloud_name == "baseline" else None
    variant_for_limits = cloud_match if cloud_name == "variant" else None
    if xlim is None or ylim is None:
        auto_xlim, auto_ylim = compute_snapshot_plot_limits(
            baseline_match=baseline_for_limits,
            variant_match=variant_for_limits,
            cam_match=cam_match,
            odom_base_match=odom_match if cloud_name == "baseline" else None,
            odom_var_match=odom_match if cloud_name == "variant" else None,
        )
        xlim_to_use = xlim if xlim is not None else auto_xlim
        ylim_to_use = ylim if ylim is not None else auto_ylim
    else:
        xlim_to_use, ylim_to_use = xlim, ylim

    draw_common_context(ax, show_trajectories, odom_series, xlim_to_use, ylim_to_use)

    if cloud_match is not None:
        _, pts, _ = cloud_match
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=marker_size, c=cloud_color,
                       alpha=particle_alpha, label=f"{cloud_name} particles", zorder=2)

    odom_base = odom_match if cloud_name == "baseline" else None
    odom_var  = odom_match if cloud_name == "variant"  else None
    _draw_markers(ax, cam_match, odom_base, odom_var)

    ax.set_title(f"t = {gt_snapshot_t_rel:.1f} s", fontsize=PLOT_TITLE_FONT_SIZE)
    ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_limits(values: Optional[List[float]]) -> Optional[Tuple[float, float]]:
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError("Axis limits require exactly two values")
    lo, hi = float(values[0]), float(values[1])
    if lo >= hi:
        raise ValueError("Axis limits must satisfy min < max")
    return lo, hi


def _robust_limits_from_collected_points(
    particle_xs: List[float],
    particle_ys: List[float],
    marker_xs: List[float],
    marker_ys: List[float],
    pad_ratio: float = 0.12,
    percentile_lo: float = 1.0,
    percentile_hi: float = 99.0,
    min_span: float = 0.6,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Compute robust x/y limits from particle percentiles + marker points."""
    xs: List[float] = []
    ys: List[float] = []

    if particle_xs and particle_ys:
        x_lo = float(np.percentile(particle_xs, percentile_lo))
        x_hi = float(np.percentile(particle_xs, percentile_hi))
        y_lo = float(np.percentile(particle_ys, percentile_lo))
        y_hi = float(np.percentile(particle_ys, percentile_hi))
        xs.extend([x_lo, x_hi])
        ys.extend([y_lo, y_hi])

    xs.extend(marker_xs)
    ys.extend(marker_ys)

    if not xs or not ys:
        return None, None

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    x_span = max(min_span, x_max - x_min)
    y_span = max(min_span, y_max - y_min)
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    x_half = 0.5 * x_span * (1.0 + pad_ratio)
    y_half = 0.5 * y_span * (1.0 + pad_ratio)

    return (x_center - x_half, x_center + x_half), (y_center - y_half, y_center + y_half)


def compute_snapshot_plot_limits(
    baseline_match: Optional[Tuple[float, np.ndarray, float]],
    variant_match: Optional[Tuple[float, np.ndarray, float]],
    cam_match: Optional[Tuple[float, float, float, float, float]],
    odom_base_match: Optional[Tuple[float, float, float, float, float]],
    odom_var_match: Optional[Tuple[float, float, float, float, float]],
    pad_ratio: float = 0.12,
    percentile_lo: float = 1.0,
    percentile_hi: float = 99.0,
    min_span: float = 0.6,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    particle_xs: List[float] = []
    particle_ys: List[float] = []
    marker_xs: List[float] = []
    marker_ys: List[float] = []

    if baseline_match is not None:
        _, pts, _ = baseline_match
        if pts.size:
            particle_xs.extend(pts[:, 0].tolist())
            particle_ys.extend(pts[:, 1].tolist())

    if variant_match is not None:
        _, pts, _ = variant_match
        if pts.size:
            particle_xs.extend(pts[:, 0].tolist())
            particle_ys.extend(pts[:, 1].tolist())

    for match in (cam_match, odom_base_match, odom_var_match):
        if match is not None:
            _, x, y, _, _ = match
            marker_xs.append(float(x))
            marker_ys.append(float(y))

    return _robust_limits_from_collected_points(
        particle_xs=particle_xs,
        particle_ys=particle_ys,
        marker_xs=marker_xs,
        marker_ys=marker_ys,
        pad_ratio=pad_ratio,
        percentile_lo=percentile_lo,
        percentile_hi=percentile_hi,
        min_span=min_span,
    )


def equalize_panel_limits(
    panel_limits: List[Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]],
) -> List[Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]]:
    """Given per-panel (xlim, ylim) pairs, return new limits that share the same
    half-span (= the largest half-span seen across all panels in either axis),
    each still centered on its original data center.

    This guarantees:
    - Identical physical axis size for every panel (no more shrunken panels)
    - Identical map scale (1 m looks the same in every panel)
    - Each panel stays centered on its own robot position
    """
    # Collect per-panel centers and half-spans.
    centers_x: List[float] = []
    centers_y: List[float] = []
    half_spans: List[float] = []

    for xlim_p, ylim_p in panel_limits:
        if xlim_p is None or ylim_p is None:
            # Can't equalize a panel with no data — leave as-is.
            centers_x.append(float("nan"))
            centers_y.append(float("nan"))
            half_spans.append(float("nan"))
            continue
        cx = 0.5 * (xlim_p[0] + xlim_p[1])
        cy = 0.5 * (ylim_p[0] + ylim_p[1])
        hx = 0.5 * (xlim_p[1] - xlim_p[0])
        hy = 0.5 * (ylim_p[1] - ylim_p[0])
        centers_x.append(cx)
        centers_y.append(cy)
        # Use the larger of x/y half-spans so the square window fits all data.
        half_spans.append(max(hx, hy))

    valid = [h for h in half_spans if not math.isnan(h)]
    if not valid:
        return panel_limits

    global_half = max(valid)

    result = []
    for (xlim_p, ylim_p), cx, cy, _ in zip(panel_limits, centers_x, centers_y, half_spans):
        if xlim_p is None or ylim_p is None or math.isnan(cx):
            result.append((xlim_p, ylim_p))
        else:
            result.append(
                ((cx - global_half, cx + global_half),
                 (cy - global_half, cy + global_half))
            )
    return result


# compute_triptych_plot_limits is kept for API compatibility but no longer
# used by plot_snapshot_triptych_overlay (which now uses per-panel limits).
def compute_triptych_plot_limits(
    snapshots: List[Dict[str, object]],
    pad_ratio: float = 0.12,
    percentile_lo: float = 1.0,
    percentile_hi: float = 99.0,
    min_span: float = 0.6,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Shared limit set across triptych snapshots (kept for external callers)."""
    particle_xs: List[float] = []
    particle_ys: List[float] = []
    marker_xs: List[float] = []
    marker_ys: List[float] = []

    for snap in snapshots:
        for key in ("baseline_match", "variant_match"):
            match = snap.get(key)
            if match is None:
                continue
            _, pts, _ = match
            if pts.size:
                particle_xs.extend(pts[:, 0].tolist())
                particle_ys.extend(pts[:, 1].tolist())

        for key in ("cam_match", "odom_base_match", "odom_var_match"):
            match = snap.get(key)
            if match is None:
                continue
            _, x, y, _, _ = match
            marker_xs.append(float(x))
            marker_ys.append(float(y))

    return _robust_limits_from_collected_points(
        particle_xs=particle_xs,
        particle_ys=particle_ys,
        marker_xs=marker_xs,
        marker_ys=marker_ys,
        pad_ratio=pad_ratio,
        percentile_lo=percentile_lo,
        percentile_hi=percentile_hi,
        min_span=min_span,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot PF particle-cloud snapshots from rosbag2")
    parser.add_argument("bag_path", help="Path to rosbag2 folder")
    parser.add_argument("--output-dir", default="", help="Optional output directory (default: bag folder)")
    parser.add_argument(
        "--times",
        nargs="+",
        type=float,
        default=None,
        help="Snapshot times in seconds from bag start (e.g. --times 5 10 15)",
    )
    parser.add_argument(
        "--num-snapshots",
        type=int,
        default=3,
        help="Number of evenly spaced snapshots when --times is not provided (default: 3)",
    )
    parser.add_argument(
        "--snapshot-mode",
        choices=("fixed_time", "variant_closest_to_gt"),
        default="fixed_time",
        help="Snapshot selection mode (default: fixed_time)",
    )
    parser.add_argument(
        "--search-window-s",
        type=float,
        default=1.5,
        help="Forward search window for variant_closest_to_gt mode (default: 1.5)",
    )
    parser.add_argument(
        "--overlay-clouds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay baseline+variant clouds in one image (default: true).",
    )
    parser.add_argument(
        "--show-trajectories",
        action="store_true",
        help="Plot faint trajectory context from /odom/{camera,pf_baseline,pf_variant}",
    )
    parser.add_argument(
        "--max-match-gap-s",
        type=float,
        default=0.25,
        help="Maximum allowed nearest-time match gap in seconds (default: 0.25)",
    )
    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Optional x-axis limits",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Optional y-axis limits",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=8.0,
        help="Particle point size (default: 8.0)",
    )
    parser.add_argument(
        "--particle-alpha",
        type=float,
        default=0.45,
        help="Particle alpha in [0,1] (default: 0.45)",
    )

    args = parser.parse_args()

    if args.times is not None and len(args.times) > 0 and args.num_snapshots != 3:
        print("[WARN] --times provided; --num-snapshots is ignored.")

    if args.max_match_gap_s <= 0.0:
        raise ValueError("--max-match-gap-s must be > 0")
    if args.search_window_s <= 0.0:
        raise ValueError("--search-window-s must be > 0")
    if args.marker_size <= 0.0:
        raise ValueError("--marker-size must be > 0")
    if not (0.0 <= args.particle_alpha <= 1.0):
        raise ValueError("--particle-alpha must be in [0, 1]")

    xlim = parse_limits(args.xlim)
    ylim = parse_limits(args.ylim)

    bag_path = Path(args.bag_path).expanduser().resolve()
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else bag_path
    output_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    try:
        particle_series, odom_series, bag_t0, bag_t1 = read_bag_data(str(bag_path))
    finally:
        rclpy.shutdown()

    duration_s = float(max(0.0, bag_t1 - bag_t0))
    selected_t_rel = resolve_snapshot_times(args.times, args.num_snapshots, duration_s)

    print(f"Bag: {bag_path}")
    print(f"Bag duration: {duration_s:.3f} s")
    print(f"Messages {TOPICS['particles_baseline']}: {len(particle_series['particles_baseline'])}")
    print(f"Messages {TOPICS['particles_variant']}:  {len(particle_series['particles_variant'])}")
    print(f"Messages {TOPICS['odom_camera']}:         {len(odom_series['odom_camera'])}")
    print(f"Messages {TOPICS['odom_baseline']}:      {len(odom_series['odom_baseline'])}")
    print(f"Messages {TOPICS['odom_variant']}:       {len(odom_series['odom_variant'])}")
    print(f"Snapshot mode: {args.snapshot_mode}")
    print("Selected snapshot times [s from start]: " + ", ".join(f"{t:.3f}" for t in selected_t_rel))

    use_triptych = (
        args.snapshot_mode == "variant_closest_to_gt"
        and args.overlay_clouds
        and len(selected_t_rel) == 3
    )
    triptych_snapshots: List[Dict[str, object]] = []

    valid_baseline_count = 0
    valid_variant_count = 0

    for gt_t_rel in selected_t_rel:
        gt_target_t = bag_t0 + gt_t_rel
        cloud_target_t = gt_target_t
        cloud_selection_lag_s = 0.0
        gt_used_t: Optional[float] = None

        if args.snapshot_mode == "variant_closest_to_gt":
            gt_pose_match = nearest_odom_match(
                odom_series["odom_camera"], gt_target_t, args.max_match_gap_s
            )
            if gt_pose_match is None:
                print(
                    f"[WARN] GT t={gt_t_rel:.3f}s: no /odom/camera within {args.max_match_gap_s:.3f}s, skipping snapshot"
                )
                continue

            gt_used_t, gt_x, gt_y, _, _ = gt_pose_match
            t_star_info = select_variant_t_star(
                odom_series["odom_variant"],
                gt_x=gt_x,
                gt_y=gt_y,
                t_start=gt_target_t,
                t_end=gt_target_t + args.search_window_s,
            )
            if t_star_info is None:
                print(
                    f"[WARN] GT t={gt_t_rel:.3f}s: no /odom/pf_variant in [{gt_t_rel:.3f}, {gt_t_rel + args.search_window_s:.3f}]s, skipping snapshot"
                )
                continue

            cloud_target_t, _ = t_star_info
            cloud_selection_lag_s = cloud_target_t - gt_target_t

        baseline_match = nearest_particle_match(
            particle_series["particles_baseline"], cloud_target_t, args.max_match_gap_s
        )
        variant_match = nearest_particle_match(
            particle_series["particles_variant"], cloud_target_t, args.max_match_gap_s
        )

        cam_match = nearest_odom_match(
            odom_series["odom_camera"], gt_target_t, args.max_match_gap_s
        )
        if cam_match is not None:
            gt_used_t = cam_match[0]

        odom_base_match = nearest_odom_match(
            odom_series["odom_baseline"], cloud_target_t, args.max_match_gap_s
        )
        odom_var_match = nearest_odom_match(
            odom_series["odom_variant"], cloud_target_t, args.max_match_gap_s
        )

        tag = format_snapshot_tag(gt_t_rel)

        if baseline_match is None:
            print(
                f"[WARN] GT t={gt_t_rel:.3f}s: no baseline particle cloud near selected cloud time within {args.max_match_gap_s:.3f}s"
            )
        else:
            valid_baseline_count += 1

        if variant_match is None:
            print(
                f"[WARN] GT t={gt_t_rel:.3f}s: no variant particle cloud near selected cloud time within {args.max_match_gap_s:.3f}s"
            )
        else:
            valid_variant_count += 1

        if cam_match is None:
            print(f"[WARN] GT t={gt_t_rel:.3f}s: no /odom/camera match within {args.max_match_gap_s:.3f}s")
        if odom_base_match is None:
            print(f"[WARN] GT t={gt_t_rel:.3f}s: no /odom/pf_baseline match within {args.max_match_gap_s:.3f}s")
        if odom_var_match is None:
            print(f"[WARN] GT t={gt_t_rel:.3f}s: no /odom/pf_variant match within {args.max_match_gap_s:.3f}s")

        if baseline_match is None and variant_match is None:
            print(
                f"[WARN] GT t={gt_t_rel:.3f}s: both clouds missing at selected time, skipping figure"
            )
            if not use_triptych:
                continue

        gt_used_rel = (gt_used_t - bag_t0) if gt_used_t is not None else float("nan")
        t_star_rel = cloud_target_t - bag_t0
        base_ok = baseline_match is not None
        var_ok = variant_match is not None
        print(
            f"[SNAPSHOT] GT t={gt_t_rel:.3f}s | gt_pose_t={gt_used_rel:.3f}s | "
            f"t_star={t_star_rel:.3f}s | lag={cloud_selection_lag_s:+.3f}s | "
            f"baseline_cloud={'yes' if base_ok else 'no'} | variant_cloud={'yes' if var_ok else 'no'}"
        )

        if use_triptych:
            triptych_snapshots.append(
                {
                    "gt_t_rel": gt_t_rel,
                    "lag_s": cloud_selection_lag_s,
                    "baseline_match": baseline_match,
                    "variant_match": variant_match,
                    "cam_match": cam_match,
                    "odom_base_match": odom_base_match,
                    "odom_var_match": odom_var_match,
                }
            )
            continue

        if args.overlay_clouds:
            out_path = output_dir / f"pf_particles_{tag}_overlay.png"
            plot_snapshot_overlay(
                out_path=out_path,
                gt_snapshot_t_rel=gt_t_rel,
                cloud_selection_lag_s=cloud_selection_lag_s,
                baseline_match=baseline_match,
                variant_match=variant_match,
                cam_match=cam_match,
                odom_base_match=odom_base_match,
                odom_var_match=odom_var_match,
                show_trajectories=args.show_trajectories,
                odom_series=odom_series,
                marker_size=args.marker_size,
                particle_alpha=args.particle_alpha,
                xlim=xlim,
                ylim=ylim,
            )
            print(f"Saved: {out_path}")
        else:
            out_base = output_dir / f"pf_particles_{tag}_baseline.png"
            out_var  = output_dir / f"pf_particles_{tag}_variant.png"

            plot_snapshot_single(
                out_path=out_base,
                gt_snapshot_t_rel=gt_t_rel,
                cloud_selection_lag_s=cloud_selection_lag_s,
                cloud_name="baseline",
                cloud_color="#1f77b4",
                cloud_match=baseline_match,
                cam_match=cam_match,
                odom_match=odom_base_match,
                show_trajectories=args.show_trajectories,
                odom_series=odom_series,
                marker_size=args.marker_size,
                particle_alpha=args.particle_alpha,
                xlim=xlim,
                ylim=ylim,
            )
            plot_snapshot_single(
                out_path=out_var,
                gt_snapshot_t_rel=gt_t_rel,
                cloud_selection_lag_s=cloud_selection_lag_s,
                cloud_name="variant",
                cloud_color="#d62728",
                cloud_match=variant_match,
                cam_match=cam_match,
                odom_match=odom_var_match,
                show_trajectories=args.show_trajectories,
                odom_series=odom_series,
                marker_size=args.marker_size,
                particle_alpha=args.particle_alpha,
                xlim=xlim,
                ylim=ylim,
            )
            print(f"Saved: {out_base}")
            print(f"Saved: {out_var}")

    if use_triptych and len(triptych_snapshots) == 3:
        out_path = output_dir / "pf_particles_triptych_overlay.png"
        plot_snapshot_triptych_overlay(
            out_path=out_path,
            snapshots=triptych_snapshots,
            show_trajectories=args.show_trajectories,
            odom_series=odom_series,
            marker_size=args.marker_size,
            particle_alpha=args.particle_alpha,
            xlim=xlim,
            ylim=ylim,
        )
        print(f"Saved: {out_path}")
    elif use_triptych:
        print("[WARN] Triptych generation requested, but fewer than 3 snapshots were available.")

    print(
        "Valid particle matches: "
        f"baseline={valid_baseline_count}/{len(selected_t_rel)}, "
        f"variant={valid_variant_count}/{len(selected_t_rel)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())