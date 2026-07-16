from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.quaternion import quat_geodesic_distance, normalize_quaternion


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def compute_metrics(truth: np.ndarray, est: np.ndarray, nees: np.ndarray | None = None, nis: np.ndarray | None = None) -> dict[str, float]:
    qerr = np.array([quat_geodesic_distance(truth[i, :4], est[i, :4]) for i in range(len(truth))])
    return {
        'attitude_geodesic_rmse': float(np.sqrt(np.mean(qerr**2))),
        'quaternion_error_mean': float(np.mean(qerr)),
        'position_rmse': rmse(truth[:, 7:10], est[:, 7:10]),
        'velocity_rmse': rmse(truth[:, 10:13], est[:, 10:13]),
        'angular_rate_rmse': rmse(truth[:, 4:7], est[:, 4:7]),
        'gyro_bias_rmse': rmse(truth[:, 13:16], est[:, 13:16]),
        'accel_bias_rmse': rmse(truth[:, 16:19], est[:, 16:19]),
        'disturbance_torque_rmse': rmse(truth[:, 19:22], est[:, 19:22]),
        'disturbance_accel_rmse': rmse(truth[:, 22:25], est[:, 22:25]),
        'nees_mean': float(np.nanmean(nees)) if nees is not None and np.size(nees) > 0 and not np.all(np.isnan(nees)) else np.nan,
        'nis_mean': float(np.nanmean(nis)) if nis is not None and np.size(nis) > 0 and not np.all(np.isnan(nis)) else np.nan,
    }


def summarize_results(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    agg = df.groupby(group_cols).agg(['mean', 'std'])
    return agg
