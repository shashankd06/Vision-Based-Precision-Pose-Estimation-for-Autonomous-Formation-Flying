"""
Evaluation Metrics for Formation Flight Pose Estimation.
Computes Position RMSE, Attitude Error, and Processing Latency.
"""
from typing import Dict, List
import numpy as np


def compute_trajectory_metrics(true_poses: np.ndarray,
                               est_poses: np.ndarray,
                               latencies: List[float] = None) -> Dict[str, float]:
    """
    Compute comprehensive tracking metrics over a flight trajectory.
    true_poses: (T, 6) -> [x, y, z, roll, pitch, yaw]
    est_poses: (T, 6) -> [x, y, z, roll, pitch, yaw]
    """
    pos_errors = np.linalg.norm(est_poses[:, :3] - true_poses[:, :3], axis=1)  # meters
    roll_errors = np.abs(est_poses[:, 3] - true_poses[:, 3])  # degrees

    metrics = {
        "pos_rmse_m": float(np.sqrt(np.mean(pos_errors ** 2))),
        "pos_mean_m": float(np.mean(pos_errors)),
        "pos_median_m": float(np.median(pos_errors)),
        "pos_max_m": float(np.max(pos_errors)),
        "roll_rmse_deg": float(np.sqrt(np.mean(roll_errors ** 2))),
        "roll_mean_deg": float(np.mean(roll_errors)),
        "roll_median_deg": float(np.median(roll_errors)),
        "roll_max_deg": float(np.max(roll_errors)),
    }

    if latencies:
        metrics["latency_mean_ms"] = float(np.mean(latencies) * 1000.0)
        metrics["fps"] = float(1.0 / np.mean(latencies)) if np.mean(latencies) > 0 else 0.0

    return metrics
