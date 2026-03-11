#!/usr/bin/env python3
"""
Estimate camera measurement noise covariance Q for EKF from static-pose logs.

Given CSV files with columns:
  camera_x, camera_y, camera_yaw
and optionally:
  ground_truth_x, ground_truth_y, ground_truth_yaw

We compute residuals r = [x - x_gt, y - y_gt, wrap(yaw - yaw_gt)]
and estimate Q = Cov[r] (3x3).

Usage examples:
  python estimate_Q_from_static_logs.py \
    --csv pose1.csv --gt 0 0 0 \
    --csv pose2.csv --gt 0.6 0.15 45 \
    --csv pose3.csv --gt 0.9 -0.45 -90 \
    --csv pose4.csv --gt 1.2 -0.15 -45

If your CSV already has ground_truth_* columns, you can omit --gt and pass:
  python estimate_Q_from_static_logs.py --csv pose1.csv --use_csv_gt ...

Options:
  --skip_s 0.5         : skip first 0.5 seconds of each log
  --mad_k 4.0          : outlier rejection threshold in MAD units (0 disables)
  --min_n 30           : minimum samples required after filtering
"""

import argparse
import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


def wrap_pi(angle_rad: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]. Works on numpy arrays."""
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def yaw_deg_to_rad(yaw_deg: float) -> float:
    return float(yaw_deg) * math.pi / 180.0


def mad_filter(residuals: np.ndarray, k: float) -> np.ndarray:
    """
    MAD-based outlier filtering.
    residuals: (N,3)
    Returns mask of inliers.
    """
    if k <= 0:
        return np.ones(residuals.shape[0], dtype=bool)

    mask = np.ones(residuals.shape[0], dtype=bool)
    for d in range(residuals.shape[1]):
        x = residuals[:, d]
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        # If MAD is 0, fall back to no filtering for that dim
        if mad < 1e-12:
            continue
        z = 0.6745 * (x - med) / mad  # approx. z-score under normality
        mask &= (np.abs(z) <= k)
    return mask


def estimate_cov(residuals: np.ndarray) -> np.ndarray:
    """
    Unbiased sample covariance (N-1).
    residuals: (N,3)
    """
    if residuals.shape[0] < 2:
        raise ValueError("Need at least 2 samples to compute covariance.")
    return np.cov(residuals.T, ddof=1)


def summarize_residuals(residuals: np.ndarray) -> str:
    mu = residuals.mean(axis=0)
    sd = residuals.std(axis=0, ddof=1) if residuals.shape[0] >= 2 else np.zeros(3)
    return (
        f"mean [dx, dy, dtheta(rad)] = {mu}\n"
        f"std  [dx, dy, dtheta(rad)] = {sd}\n"
        f"std  [dx, dy, dtheta(deg)] = {[sd[0], sd[1], sd[2]*180/np.pi]}"
    )


def load_and_compute_residuals(
    csv_path: Path,
    gt_pose: Optional[Tuple[float, float, float]],
    use_csv_gt: bool,
    skip_s: float,
) -> np.ndarray:
    df = pd.read_csv(csv_path)

    required = {"time_from_start_s", "camera_x", "camera_y", "camera_yaw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")

    # Skip initial seconds if requested
    if skip_s > 0:
        df = df[df["time_from_start_s"] >= skip_s].copy()

    if len(df) == 0:
        raise ValueError(f"{csv_path}: no rows after skip_s={skip_s}")

    cam_x = df["camera_x"].astype(float).to_numpy()
    cam_y = df["camera_y"].astype(float).to_numpy()
    cam_yaw = df["camera_yaw"].astype(float).to_numpy()  # assume radians in your logs

    if use_csv_gt:
        gt_required = {"ground_truth_x", "ground_truth_y", "ground_truth_yaw"}
        gt_missing = gt_required - set(df.columns)
        if gt_missing:
            raise ValueError(
                f"{csv_path}: --use_csv_gt set but missing columns {gt_missing}"
            )
        gt_x = df["ground_truth_x"].astype(float).to_numpy()
        gt_y = df["ground_truth_y"].astype(float).to_numpy()
        gt_yaw = df["ground_truth_yaw"].astype(float).to_numpy()

        # If GT columns are empty (as in your snippet), this will create NaNs.
        if np.any(np.isnan(gt_x)) or np.any(np.isnan(gt_y)) or np.any(np.isnan(gt_yaw)):
            raise ValueError(
                f"{csv_path}: ground_truth_* contains NaNs; provide --gt instead."
            )
    else:
        if gt_pose is None:
            raise ValueError(f"{csv_path}: must provide --gt x y yaw_deg unless --use_csv_gt.")
        x0, y0, yaw_deg = gt_pose
        gt_x = np.full_like(cam_x, float(x0))
        gt_y = np.full_like(cam_y, float(y0))
        gt_yaw = np.full_like(cam_yaw, yaw_deg_to_rad(yaw_deg))

    dx = cam_x - gt_x
    dy = cam_y - gt_y
    dtheta = wrap_pi(cam_yaw - gt_yaw)

    residuals = np.stack([dx, dy, dtheta], axis=1)
    return residuals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        action="append",
        required=True,
        help="Path to a CSV log file. Provide multiple --csv entries.",
    )
    ap.add_argument(
        "--gt",
        action="append",
        nargs=3,
        metavar=("X", "Y", "YAW_DEG"),
        help="Ground truth pose for the preceding --csv: x y yaw_deg. Provide one per CSV.",
    )
    ap.add_argument(
        "--use_csv_gt",
        action="store_true",
        help="Use ground_truth_x/y/yaw columns from CSV instead of --gt.",
    )
    ap.add_argument("--skip_s", type=float, default=0.5, help="Skip first N seconds of each log.")
    ap.add_argument("--mad_k", type=float, default=4.0, help="MAD outlier threshold (0 disables).")
    ap.add_argument("--min_n", type=int, default=30, help="Minimum samples per pose after filtering.")
    args = ap.parse_args()

    csv_paths = [Path(p) for p in args.csv]

    gt_list: List[Optional[Tuple[float, float, float]]] = [None] * len(csv_paths)
    if not args.use_csv_gt:
        if args.gt is None or len(args.gt) != len(csv_paths):
            raise SystemExit(
                "You must provide exactly one --gt x y yaw_deg for each --csv, "
                "unless you pass --use_csv_gt."
            )
        gt_list = [(float(x), float(y), float(yaw_deg)) for x, y, yaw_deg in args.gt]

    per_pose_results = []
    pooled_residuals = []

    print("\n=== Estimating Q from static poses ===")
    for i, (csv_path, gt_pose) in enumerate(zip(csv_paths, gt_list), start=1):
        residuals = load_and_compute_residuals(
            csv_path=csv_path,
            gt_pose=gt_pose,
            use_csv_gt=args.use_csv_gt,
            skip_s=args.skip_s,
        )

        inlier_mask = mad_filter(residuals, k=args.mad_k)
        residuals_f = residuals[inlier_mask]

        if residuals_f.shape[0] < args.min_n:
            raise ValueError(
                f"{csv_path}: only {residuals_f.shape[0]} samples after filtering; "
                f"increase data length, reduce --skip_s, or reduce --mad_k."
            )

        Q_pose = estimate_cov(residuals_f)
        pooled_residuals.append(residuals_f)

        print(f"\n--- Pose {i}: {csv_path.name} ---")
        if not args.use_csv_gt:
            x0, y0, yaw_deg = gt_pose  # type: ignore
            print(f"GT = [{x0}, {y0}, {yaw_deg} deg]")
        print(f"Samples: raw={residuals.shape[0]}, inliers={residuals_f.shape[0]}")
        print(summarize_residuals(residuals_f))
        print("Q_pose (3x3):\n", Q_pose)

        per_pose_results.append((csv_path.name, residuals_f, Q_pose))

    pooled = np.concatenate(pooled_residuals, axis=0)
    Q_pooled = estimate_cov(pooled)

    print("\n=== Pooled (all poses) ===")
    print(f"Total inlier samples pooled: {pooled.shape[0]}")
    print(summarize_residuals(pooled))
    print("Q_pooled (3x3):\n", Q_pooled)

    # Convenience: print diag as (sigma_x^2, sigma_y^2, sigma_theta^2)
    diag = np.diag(Q_pooled)
    print("\nDiag(Q_pooled) = ", diag)
    print("Std dev: [sigma_x, sigma_y, sigma_theta(rad)] = ", np.sqrt(diag))
    print("Std dev: [sigma_x, sigma_y, sigma_theta(deg)] = ",
          [float(np.sqrt(diag[0])),
           float(np.sqrt(diag[1])),
           float(np.sqrt(diag[2]) * 180/np.pi)])

    # Also output a ROS/params-friendly line
    print("\nSuggested parameters.py line:")
    print(f"Q3 = np.diag([{diag[0]:.6g}, {diag[1]:.6g}, {diag[2]:.6g}])")


if __name__ == "__main__":
    main()
