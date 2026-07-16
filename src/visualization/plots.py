from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_STYLE = {
    "EKF": {"label": "EKF"},
    "UKF": {"label": "UKF"},
    "MEKF": {"label": "MEKF"},
    "PINN-only": {"label": "PINN-only"},
    "Transformer-only": {"label": "Transformer-only"},
    "PINN+EKF": {"label": "PINN+EKF"},
    "PINN+UKF": {"label": "PINN+UKF"},
    "PINN+MEKF": {"label": "PINN+MEKF"},
    "Transformer+MEKF": {"label": "Transformer+MEKF"},
    "PINN+UKF+MEKF": {"label": "PINN+UKF+MEKF"},
}


def _load_npz_if_exists(path: Path) -> Optional[np.lib.npyio.NpzFile]:
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def _safe_save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _quat_geodesic_error(q_true: np.ndarray, q_est: np.ndarray) -> np.ndarray:
    q_true = np.asarray(q_true, dtype=float)
    q_est = np.asarray(q_est, dtype=float)
    dot = np.abs(np.sum(q_true * q_est, axis=1))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def _load_prediction(pred_dir: Path, name: str) -> Optional[dict]:
    f = _load_npz_if_exists(pred_dir / f"{name}.npz")
    if f is None:
        return None
    out = {}
    for k in f.files:
        out[k] = f[k]
    return out


def _plot_quaternion_norm(t: np.ndarray, truth: np.ndarray, methods: Dict[str, np.ndarray], fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, np.linalg.norm(truth[:, :4], axis=1), label="Ground truth", linewidth=2)
    for name in ["EKF", "UKF", "MEKF", "PINN+UKF+MEKF"]:
        if name in methods:
            ax.plot(t, np.linalg.norm(methods[name][:, :4], axis=1), label=name)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Quaternion norm")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _safe_save(fig, fig_dir / "quaternion_norm.pdf")


def _plot_position_error_time(t: np.ndarray, truth: np.ndarray, methods: Dict[str, np.ndarray], fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ["EKF", "UKF", "MEKF", "PINN+MEKF", "Transformer+MEKF", "PINN+UKF+MEKF"]:
        if name in methods:
            err = np.linalg.norm(methods[name][:, 7:10] - truth[:, 7:10], axis=1)
            ax.plot(t, err, label=name)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position error [km]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _safe_save(fig, fig_dir / "position_error_time.pdf")


def _plot_velocity_error_time(t: np.ndarray, truth: np.ndarray, methods: Dict[str, np.ndarray], fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ["EKF", "UKF", "MEKF", "PINN+MEKF", "Transformer+MEKF", "PINN+UKF+MEKF"]:
        if name in methods:
            err = np.linalg.norm(methods[name][:, 10:13] - truth[:, 10:13], axis=1)
            ax.plot(t, err, label=name)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity error [km/s]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _safe_save(fig, fig_dir / "velocity_error_time.pdf")


def _plot_bias_error_time(t: np.ndarray, truth: np.ndarray, methods: Dict[str, np.ndarray], fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if "EKF" in methods:
        ax.plot(t, np.linalg.norm(methods["EKF"][:, 13:16] - truth[:, 13:16], axis=1), label="Gyro bias")
        ax.plot(t, np.linalg.norm(methods["EKF"][:, 16:19] - truth[:, 16:19], axis=1), label="Accel bias")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Bias error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _safe_save(fig, fig_dir / "bias_error_time.pdf")


def _plot_disturbance_error_time(t: np.ndarray, truth: np.ndarray, methods: Dict[str, np.ndarray], fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if "EKF" in methods:
        ax.plot(t, np.linalg.norm(methods["EKF"][:, 19:22] - truth[:, 19:22], axis=1), label="Torque disturbance")
        ax.plot(t, np.linalg.norm(methods["EKF"][:, 22:25] - truth[:, 22:25], axis=1), label="Accel disturbance")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Disturbance error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _safe_save(fig, fig_dir / "disturbance_error_time.pdf")


def _plot_attitude_error_time(t: np.ndarray, truth: np.ndarray, methods: Dict[str, np.ndarray], fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ["EKF", "UKF", "MEKF", "PINN+EKF", "PINN+UKF", "PINN+MEKF", "Transformer+MEKF", "PINN+UKF+MEKF"]:
        if name in methods:
            e = _quat_geodesic_error(truth[:, :4], methods[name][:, :4])
            ax.plot(t, e, label=name)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Attitude geodesic error [rad]")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    _safe_save(fig, fig_dir / "attitude_error_time.pdf")


def _plot_3d_trajectory_comparison(truth: np.ndarray, methods: Dict[str, np.ndarray], fig_dir: Path) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(truth[:, 7], truth[:, 8], truth[:, 9], label="Ground truth", linewidth=2)

    for name in ["EKF", "MEKF", "PINN+MEKF", "Transformer+MEKF"]:
        if name in methods:
            est = methods[name]
            ax.plot(est[:, 7], est[:, 8], est[:, 9], label=name, alpha=0.9)

    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")
    ax.legend()
    _safe_save(fig, fig_dir / "trajectory_3d_comparison.pdf")


def _plot_boxplots(per_trial: pd.DataFrame, fig_dir: Path) -> None:
    """[MC-STATS FIX] Box plots over the 100 Monte Carlo trials per method.

    v1 passed the one-row-per-method summary table, so each "box" was a
    single point. This version consumes metrics_per_trial.csv.
    """
    box_metrics = [
        ("position_rmse", "Position RMSE [km]", "position_rmse_boxplot.pdf"),
        ("attitude_geodesic_rmse", "Attitude geodesic RMSE [rad]", "attitude_rmse_boxplot.pdf"),
        ("velocity_rmse", "Velocity RMSE [km/s]", "velocity_rmse_boxplot.pdf"),
        ("angular_rate_rmse", "Angular-rate RMSE [rad/s]", "angular_rate_rmse_boxplot.pdf"),
        ("runtime_per_step_s", "Runtime per step [s]", "runtime_boxplot.pdf"),
    ]
    methods = list(per_trial["method"].unique())
    for col, ylabel, fname in box_metrics:
        if col not in per_trial.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        data = [per_trial.loc[per_trial["method"] == m, col].dropna().values for m in methods]
        ax.boxplot(data, labels=methods, showfliers=True)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        _safe_save(fig, fig_dir / fname)


def _plot_ci_bars(per_trial: pd.DataFrame, fig_dir: Path) -> None:
    """[MC-STATS FIX] Mean with 95% confidence interval error bars."""
    from scipy import stats as _st

    for col, ylabel, fname in [
        ("position_rmse", "Position RMSE [km]", "position_rmse_ci.pdf"),
        ("attitude_geodesic_rmse", "Attitude geodesic RMSE [rad]", "attitude_rmse_ci.pdf"),
        ("runtime_per_step_s", "Runtime per step [s]", "runtime_ci.pdf"),
    ]:
        if col not in per_trial.columns:
            continue
        methods, means, errs = [], [], []
        for m, g in per_trial.groupby("method", sort=False):
            v = g[col].dropna().values
            if v.size < 2:
                continue
            methods.append(m)
            means.append(np.mean(v))
            errs.append(_st.sem(v) * _st.t.ppf(0.975, len(v) - 1))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(methods, means, yerr=errs, capsize=5)
        ax.set_ylabel(ylabel + " (mean ± 95% CI)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        _safe_save(fig, fig_dir / fname)


def _plot_cdf_from_summary(metrics: pd.DataFrame, column: str, fig_dir: Path, fname: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for method, group in metrics.groupby("method"):
        if column not in group:
            continue
        vals = np.asarray(group[column].dropna(), dtype=float)
        if vals.size == 0:
            continue
        vals = np.sort(vals)
        cdf = np.arange(1, len(vals) + 1, dtype=float) / len(vals)
        ax.plot(vals, cdf, label=method)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("P(error ≤ x)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _safe_save(fig, fig_dir / fname)


def _plot_bar(metrics: pd.DataFrame, column: str, fig_dir: Path, fname: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(metrics["method"], metrics[column])
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    _safe_save(fig, fig_dir / fname)


def _plot_radar(metrics: pd.DataFrame, fig_dir: Path) -> None:
    top = metrics.sort_values("position_rmse").head(5).copy()
    cols = [
        "attitude_geodesic_rmse",
        "position_rmse",
        "velocity_rmse",
        "runtime_per_step_s",
        "memory_usage_mb",
    ]
    vals = top[cols].to_numpy(dtype=float)

    # Normalize per column so metrics share a comparable scale.
    denom = np.maximum(np.nanmax(vals, axis=0) - np.nanmin(vals, axis=0), 1e-12)
    norm = (vals - np.nanmin(vals, axis=0)) / denom

    labels = ["Attitude", "Position", "Velocity", "Runtime", "Memory"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(111, polar=True)

    for i, row in enumerate(norm):
        data = row.tolist()
        data += data[:1]
        ax.plot(angles, data, label=top.iloc[i]["method"])
        ax.fill(angles, data, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("Radar comparison of top methods")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    _safe_save(fig, fig_dir / "radar_comparison.pdf")


def _plot_pareto(metrics: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(metrics["runtime_per_step_s"], metrics["position_rmse"])
    for _, row in metrics.iterrows():
        ax.annotate(row["method"], (row["runtime_per_step_s"], row["position_rmse"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Runtime per step [s]")
    ax.set_ylabel("Position RMSE")
    ax.grid(True, alpha=0.3)
    _safe_save(fig, fig_dir / "pareto_front.pdf")


def _plot_innovation_traces(pred_dir: Path, fig_dir: Path) -> None:
    # If innovations are saved as .npz arrays, plot their norms and mean traces.
    for name in ["EKF", "UKF", "MEKF", "PINN_UKF_MEKF", "PINN_EKF", "PINN_MEKF", "Transformer_MEKF"]:
        f = _load_npz_if_exists(pred_dir / f"{name}.npz")
        if f is None or "innovations" not in f.files:
            continue

        innov = np.asarray(f["innovations"], dtype=float)
        if innov.ndim == 1:
            innov = innov[:, None]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(np.mean(innov, axis=1), label="Mean innovation")
        ax.set_xlabel("Step")
        ax.set_ylabel("Innovation mean")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _safe_save(fig, fig_dir / f"{name.lower()}_innovation_mean.pdf")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(np.var(innov, axis=1), label="Innovation variance")
        ax.set_xlabel("Step")
        ax.set_ylabel("Innovation variance")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _safe_save(fig, fig_dir / f"{name.lower()}_innovation_var.pdf")


def make_all_plots(project) -> None:
    fig_dir = project.figure_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = project.table_dir / "metrics_summary.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics table: {metrics_path}")

    metrics = pd.read_csv(metrics_path)
    pred_dir = project.output_dir / "predictions"

    ekf = _load_prediction(pred_dir, "EKF")
    ukf = _load_prediction(pred_dir, "UKF")
    mekf = _load_prediction(pred_dir, "MEKF")
    hybrid = _load_prediction(pred_dir, "PINN_UKF_MEKF")
    pinn_mekf = _load_prediction(pred_dir, "PINN_MEKF")
    trans_mekf = _load_prediction(pred_dir, "Transformer_MEKF")
    pinn_ekf = _load_prediction(pred_dir, "PINN_EKF")
    pinn_ukf = _load_prediction(pred_dir, "PINN_UKF")

    available = {
        "EKF": ekf["est"] if ekf is not None and "est" in ekf else None,
	"UKF": ukf["est"] if ukf is not None and "est" in ukf else None,
	"MEKF": mekf["est"] if mekf is not None and "est" in mekf else None,
	"PINN+UKF+MEKF": hybrid["est"] if hybrid is not None and "est" in hybrid else None,
	"PINN+MEKF": pinn_mekf["est"] if pinn_mekf is not None and "est" in pinn_mekf else None,
	"Transformer+MEKF": trans_mekf["est"] if trans_mekf is not None and "est" in trans_mekf else None,
	"PINN+EKF": pinn_ekf["est"] if pinn_ekf is not None and "est" in pinn_ekf else None,
	"PINN+UKF": pinn_ukf["est"] if pinn_ukf is not None and "est" in pinn_ukf else None,
    }

    # Use EKF as the representative trajectory source.
    if ekf is None or "time" not in ekf or "truth" not in ekf:
        raise FileNotFoundError("EKF prediction file must contain 'time' and 'truth' arrays.")

    t = np.asarray(ekf["time"], dtype=float)
    truth = np.asarray(ekf["truth"], dtype=float)

    # Existing core plots
    _plot_quaternion_norm(t, truth, available, fig_dir)
    _plot_position_error_time(t, truth, available, fig_dir)
    _plot_velocity_error_time(t, truth, available, fig_dir)
    _plot_bias_error_time(t, truth, available, fig_dir)
    _plot_disturbance_error_time(t, truth, available, fig_dir)
    _plot_attitude_error_time(t, truth, available, fig_dir)

    _plot_bar(metrics, "position_rmse", fig_dir, "comparison_bar_position.pdf", "Position RMSE")
    _plot_bar(metrics, "attitude_geodesic_rmse", fig_dir, "comparison_bar_attitude.pdf", "Attitude geodesic RMSE")
    _plot_bar(metrics, "runtime_per_step_s", fig_dir, "runtime_comparison.pdf", "Runtime per step [s]")

    # Requested publication-style plots
    _plot_3d_trajectory_comparison(truth, available, fig_dir)
    per_trial_path = project.table_dir / "metrics_per_trial.csv"
    if per_trial_path.exists():
        per_trial = pd.read_csv(per_trial_path)
        _plot_boxplots(per_trial, fig_dir)
        _plot_ci_bars(per_trial, fig_dir)
    cdf_source = per_trial if per_trial_path.exists() else metrics
    _plot_cdf_from_summary(cdf_source, "position_rmse", fig_dir, "position_cdf.pdf", "Position RMSE")
    _plot_cdf_from_summary(cdf_source, "attitude_geodesic_rmse", fig_dir, "attitude_cdf.pdf", "Attitude geodesic RMSE")
    _plot_radar(metrics, fig_dir)
    _plot_pareto(metrics, fig_dir)

    # Innovation traces if available
    _plot_innovation_traces(pred_dir, fig_dir)

    # NEES/NIS
    if ekf is not None and "nees" in ekf:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, ekf["nees"], label="NEES")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("NEES")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _safe_save(fig, fig_dir / "nees_plot.pdf")

    if ekf is not None and "nis" in ekf:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, ekf["nis"], label="NIS")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("NIS")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _safe_save(fig, fig_dir / "nis_plot.pdf")