from __future__ import annotations

"""Benchmark experiments.

Revision 2 (reviewer-response version). Changes vs. v1:

  [TRUTH-LEAK FIX] Removed all uses of ground-truth states inside
      estimators. v1 nudged the estimated quaternion toward
      traj.states[k, :4] (simulator truth) with gain 0.5-0.7. That is
      truth leakage and invalidated the reported attitude RMSE. The
      proposed method now runs a genuine parallel UKF (translational) +
      MEKF (rotational) architecture as described in the paper.

  [RUNTIME FIX] Runtime is now measured over the FULL estimation cycle
      (PINN inference + residual injection + filter step) using
      time.perf_counter, with a per-component breakdown
      (pinn_time_per_step_s, filter_time_per_step_s). v1 timed only
      filt.step and called psutil twice per step inside the timed region,
      which produced the non-physical "hybrid faster than UKF" numbers.

  [MC-STATS FIX] Per-trial metrics are saved (metrics_per_trial.csv) and
      the summary table now includes std, variance, 95% confidence
      intervals, and Wilcoxon signed-rank + paired t-test p-values.

  [COUPLING FIX] The PINN->filter coupling coefficient is read from
      config['fusion']['coupling'] instead of being hard-coded.

  [NEES FIX] v1 compared a 24-dim error vector against a 25-dim
      covariance, so every NEES was NaN. A linear map G now reduces the
      quaternion-state covariance to the 24-dim error-state covariance.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from src.dynamics.simulator import Trajectory
from src.dynamics.spacecraft import SpacecraftParams, rk4_step
from src.evaluation.metrics import compute_metrics
from src.evaluation.stats import confidence_interval, paired_ttest, wilcoxon_test, improvement
from src.filters.ekf import EKF
from src.filters.adaptive_ekf import AdaptiveEKF
from src.filters.ukf import UKF
from src.filters.mekf import MEKF
from src.pinn.train import load_pinn
from src.models.train import load_transformer
from src.sensors.measurement_models import measurement_noise_cov
from src.utils.reproducibility import ensure_dir
from src.utils.quaternion import (
    normalize_quaternion,
    quat_conjugate,
    quat_multiply,
    rotvec_from_quat,
)

METHODS = [
    "EKF",
    "Adaptive-EKF",
    "UKF",
    "MEKF",
    "PINN-only",
    "Transformer-only",
    "PINN+EKF",
    "PINN+UKF",
    "PINN+MEKF",
    "Transformer+MEKF",
    "PINN+UKF+MEKF",
]

# Default PINN->filter coupling coefficients; overridden by config['fusion']['coupling'].
DEFAULT_COUPLING = {
    "PINN+EKF": 0.65,
    "PINN+UKF": 0.75,
    "PINN+MEKF": 0.90,
    "Transformer+MEKF": 0.85,
    "PINN+UKF+MEKF": 0.98,
    "PINN-only": 1.00,
    "Transformer-only": 1.00,
}

# State layout: [0:4 quat, 4:7 omega, 7:10 pos, 10:13 vel,
#                13:16 bg, 16:19 ba, 19:22 tau_d, 22:25 a_d]
ROT_IDX = np.r_[4:7, 13:16, 19:22]
TRANS_IDX = np.r_[7:10, 10:13, 16:19, 22:25]
# Error-vector layout (24): [att 0:3, omega 3:6, pos 6:9, vel 9:12,
#                            bg 12:15, ba 15:18, tau 18:21, ad 21:24]
ROT_ERR_IDX = np.r_[0:6, 12:15, 18:21]
TRANS_ERR_IDX = np.r_[6:12, 15:18, 21:24]


def _load_split(path: Path) -> list[Trajectory]:
    data = np.load(path, allow_pickle=True)
    return list(data["trajectories"])


def _init_state(traj: Trajectory) -> np.ndarray:
    x0 = traj.states[0].copy()
    x0[:4] = normalize_quaternion(x0[:4])
    return x0


def _state_error_vec(truth: np.ndarray, est: np.ndarray) -> np.ndarray:
    dq = quat_multiply(truth[:4], quat_conjugate(est[:4]))
    att = rotvec_from_quat(dq)
    return np.hstack([att, truth[4:] - est[4:]])


def _error_covariance(P: np.ndarray) -> np.ndarray:
    """Reduce a covariance to the 24-dim error-state space.

    For 25-state (quaternion) filters the attitude error rotvec satisfies
    delta_theta ~= 2 * delta_q_v, so G maps quaternion covariance rows 1:4
    with factor 2. MEKF covariance is already 24-dim error-state.
    """
    P = np.atleast_2d(np.asarray(P, dtype=float))
    if P.shape[0] == 24:
        return P
    if P.shape[0] == 25:
        G = np.zeros((24, 25))
        G[0:3, 1:4] = 2.0 * np.eye(3)
        G[3:, 4:] = np.eye(21)
        return G @ P @ G.T
    return P


def _process_noise(cfg: dict) -> np.ndarray:
    q = cfg["process_noise"]
    diag = np.array(
        [
            1e-8, 1e-8, 1e-8, 1e-8,
            5e-6, 5e-6, 5e-6,
            1e-4, 1e-4, 1e-4,
            1e-5, 1e-5, 1e-5,
            q["bg"], q["bg"], q["bg"],
            q["ba"], q["ba"], q["ba"],
            q["torque"], q["torque"], q["torque"],
            q["accel"], q["accel"], q["accel"],
        ],
        dtype=float,
    )
    return np.diag(diag)


def _coupling(project, method: str) -> float:
    cfg = project.config.get("fusion", {}).get("coupling", {})
    return float(cfg.get(method, DEFAULT_COUPLING.get(method, 0.0)))


def _build_feature(x: np.ndarray, traj: Trajectory, k: int, hist: np.ndarray) -> np.ndarray:
    return np.hstack(
        [
            x,
            traj.controls[k],
            traj.env["sun_vec_i"][k],
            traj.env["earth_vec_i"][k],
            traj.env["magnetic_field_i"][k],
            [traj.env["weather_index"][k]],
            hist.reshape(-1),
        ]
    )


def _predict_residual(method: str, feat: np.ndarray, hist: np.ndarray, pinn, trans):
    """Return (disturbance prediction[6], predictive variance[6] or None).

    [FUSION UPGRADE] v1 discarded the PINN's heteroscedastic logvar head.
    The variance is now returned so the prior can be fused as a
    pseudo-measurement with calibrated uncertainty.
    """
    import torch

    if method in {"PINN-only", "PINN+EKF", "PINN+UKF", "PINN+MEKF", "PINN+UKF+MEKF"} and pinn is not None:
        inp = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred, logvar = pinn.model(inp)
        pred = pred.squeeze(0).cpu().numpy()
        var = np.exp(logvar.squeeze(0).cpu().numpy())
        if pred.shape[0] < 6:
            pred = np.pad(pred, (0, 6 - pred.shape[0]))
            var = np.pad(var, (0, 6 - var.shape[0]), constant_values=1.0)
        return pred[:6], var[:6]

    if method in {"Transformer-only", "Transformer+MEKF"} and trans is not None:
        seq = torch.tensor(hist.reshape(1, hist.shape[0], -1), dtype=torch.float32)
        with torch.no_grad():
            pred, _state_hat = trans.model(seq)
        pred = pred.squeeze(0).cpu().numpy()
        if pred.shape[0] < 6:
            pred = np.pad(pred, (0, 6 - pred.shape[0]))
        return pred[:6], np.full(6, 1e-2)  # transformer has no uncertainty head

    return np.zeros(6, dtype=float), None


def _make_filter(filter_kind: str, x0: np.ndarray, P0: np.ndarray, params, Q, R, include_rd: bool, project):
    if filter_kind == "UKF":
        return UKF(
            x0,
            P0,
            params,
            Q,
            R,
            include_range_doppler=include_rd,
            alpha=project.config["filter"]["ukf_alpha"],
            beta=project.config["filter"]["ukf_beta"],
            kappa=project.config["filter"]["ukf_kappa"],
        )
    if filter_kind == "EKF":
        return EKF(x0, P0, params, Q, R, include_range_doppler=include_rd)
    if filter_kind == "Adaptive-EKF":
        return AdaptiveEKF(
            x0,
            P0,
            params,
            Q,
            R,
            include_range_doppler=include_rd,
        )
    if filter_kind == "MEKF":
        return MEKF(x0, P0, params, Q, R, include_range_doppler=include_rd)
    raise ValueError(filter_kind)


def _inject_residual(x: np.ndarray, residual: np.ndarray, gamma: float) -> np.ndarray:
    """Blend PINN disturbance prediction into disturbance sub-states.

    gamma is the coupling coefficient (config: fusion.coupling).
    """
    x = x.copy()
    res = np.asarray(residual, dtype=float)
    if res.shape[0] < 6:
        res = np.pad(res, (0, 6 - res.shape[0]))
    if gamma > 0.0:
        x[19:22] = (1.0 - gamma) * x[19:22] + gamma * res[:3]
        x[22:25] = (1.0 - gamma) * x[22:25] + gamma * res[3:6]
    return x


def _fuse_pinn_prior(filt, mean: np.ndarray, var: np.ndarray | None, r_scale: float) -> None:
    """[FUSION UPGRADE] Fuse the PINN disturbance prior as a pseudo-measurement.

    Replaces v1's convex blending (which ignored both the filter covariance
    and the PINN uncertainty). A standard linear Kalman update on the
    disturbance sub-states [tau_d, a_d] with R_pinn = diag(var) * r_scale:
    cross-covariances propagate the correction to correlated states, and the
    covariance contraction lets the filter benefit at consistency level too.
    """
    if var is None:
        return
    n = filt.P.shape[0]
    if n == 24:        # MEKF error-state covariance
        idx = np.arange(18, 24)
    elif n == 25:      # quaternion-state covariance (EKF/UKF)
        idx = np.arange(19, 25)
    else:
        return
    Rp = np.diag(np.maximum(np.asarray(var, dtype=float), 1e-10) * max(r_scale, 1e-6))
    H = np.zeros((6, n))
    H[np.arange(6), idx] = 1.0
    S = H @ filt.P @ H.T + Rp
    try:
        K = filt.P @ H.T @ np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return
    z = np.asarray(mean, dtype=float) - filt.x[19:25]
    dx = K @ z
    if n == 24:
        filt.x = filt.inject(filt.x, dx)   # MEKF multiplicative injection
    else:
        filt.x = filt.x + dx
        filt.x[:4] = normalize_quaternion(filt.x[:4])
    filt.P = (np.eye(n) - K @ H) @ filt.P
    filt.P = 0.5 * (filt.P + filt.P.T)


def _env_at(traj: Trajectory, k: int):
    return type(
        "Env",
        (),
        {
            "sun_vec_i": traj.env["sun_vec_i"][k],
            "earth_vec_i": traj.env["earth_vec_i"][k],
            "magnetic_field_i": traj.env["magnetic_field_i"][k],
            "weather_index": float(traj.env["weather_index"][k]),
        },
    )


def _ctrl_at(traj: Trajectory, k: int):
    return type(
        "Ctrl",
        (),
        {"torque_cmd": traj.controls[k, :3], "accel_cmd": traj.controls[k, 3:]},
    )


def _hist_window(traj: Trajectory, k: int, window: int) -> np.ndarray:
    hist = np.nan_to_num(traj.measurements[max(0, k - window):k], nan=0.0)
    if len(hist) < window:
        pad = np.zeros((window - len(hist), traj.measurements.shape[1]))
        hist = np.vstack([pad, hist])
    return hist


def _finalize_metrics(truth, est, nees, nis, cycle_times, pinn_times, filter_times):
    metrics = compute_metrics(truth, est, nees, nis)
    metrics["runtime_per_step_s"] = float(np.mean(cycle_times))
    metrics["runtime_per_step_s_median"] = float(np.median(cycle_times))
    metrics["pinn_time_per_step_s"] = float(np.mean(pinn_times)) if pinn_times else 0.0
    metrics["filter_time_per_step_s"] = float(np.mean(filter_times)) if filter_times else 0.0
    metrics["total_inference_latency_s"] = float(np.sum(cycle_times))
    metrics["memory_usage_mb"] = float(psutil.Process().memory_info().rss / (1024 ** 2))
    return metrics


def _run_filter(method: str, traj: Trajectory, project, pinn=None, trans=None):
    params = SpacecraftParams(**project.config["simulation"])
    include_rd = bool(project.config["synthetic"]["include_range_doppler"])
    R = measurement_noise_cov(project.config["measurement_noise"], include_rd) * float(
        project.config["filter"]["r_scale"]
    )
    Q = _process_noise(project.config) * float(project.config["filter"]["q_scale"])

    x0 = _init_state(traj)
    P0 = np.diag([1e-3] * 4 + [1e-2] * 21)
    if method in {"MEKF", "PINN+MEKF", "Transformer+MEKF"}:
        P0 = np.diag([1e-2] * 25)

    if method == "EKF":
        filter_kind = "EKF"

    elif method == "Adaptive-EKF":
        filter_kind = "Adaptive-EKF"

    elif method == "UKF":
        filter_kind = "UKF"

    elif method == "MEKF":
        filter_kind = "MEKF"

    elif method == "PINN+EKF":
        filter_kind = "EKF"

    elif method == "PINN+UKF":
        filter_kind = "UKF"

    elif method in {"PINN+MEKF", "Transformer+MEKF"}:
        filter_kind = "MEKF"

    else:
        raise ValueError(method)

    filt = _make_filter(filter_kind, x0, P0, params, Q, R, include_rd, project)
    use_pinn = method in {"PINN+EKF", "PINN+UKF", "PINN+MEKF", "Transformer+MEKF"}
    r_scale_pinn = float(project.config.get("fusion", {}).get("pinn_r_scale", 1.0))
    window = int(project.config["training"]["window"])

    est = np.zeros_like(traj.states)
    n = len(traj.time)
    nees = np.full(n, np.nan)
    nis = np.full(n, np.nan)
    innovations: list[np.ndarray] = []
    cycle_times: list[float] = []
    pinn_times: list[float] = []
    filter_times: list[float] = []

    for k in range(n):
        env = _env_at(traj, k)
        ctrl = _ctrl_at(traj, k)
        hist = _hist_window(traj, k, window)

        # ---- full estimation cycle is timed (NN inference + injection + filter) ----
        t_cycle = time.perf_counter()

        t_nn = time.perf_counter()
        if use_pinn:
            feat = _build_feature(filt.x, traj, k, hist)
            residual, res_var = _predict_residual(method, feat, hist, pinn, trans)
            _fuse_pinn_prior(filt, residual, res_var, r_scale_pinn)
        pinn_times.append(time.perf_counter() - t_nn)

        t_f = time.perf_counter()
        step = filt.step(
            traj.time[k],
            project.config["synthetic"]["dt"],
            env,
            ctrl,
            traj.measurements[k],
            traj.measurement_mask[k],
        )
        filter_times.append(time.perf_counter() - t_f)
        cycle_times.append(time.perf_counter() - t_cycle)
        # NOTE [TRUTH-LEAK FIX]: v1 applied a correction toward the true
        # quaternion traj.states[k, :4] here for the proposed method.
        # That was ground-truth leakage and has been removed.

        est[k] = filt.x

        if getattr(step, "S", None) is not None and np.size(step.S) > 0:
            innov = np.asarray(step.innovation).reshape(-1, 1)
            innovations.append(innov.squeeze().copy())
            try:
                nis[k] = float(innov.T @ np.linalg.solve(step.S, innov))
            except np.linalg.LinAlgError:
                nis[k] = np.nan

        e = _state_error_vec(traj.states[k], est[k])
        P24 = _error_covariance(filt.P)
        if P24.shape[0] == e.size:
            try:
                nees[k] = float(e.T @ np.linalg.solve(P24, e))
            except np.linalg.LinAlgError:
                nees[k] = np.nan

    metrics = _finalize_metrics(traj.states, est, nees, nis, cycle_times, pinn_times, filter_times)
    return est, metrics, nees, nis, np.asarray(innovations, dtype=object)


def _run_learned(method: str, traj: Trajectory, project, pinn=None, trans=None):
    params = SpacecraftParams(**project.config["simulation"])
    x = traj.states[0].copy()
    x[:4] = normalize_quaternion(x[:4])

    est = np.zeros_like(traj.states)
    n = len(traj.time)
    window = int(project.config["training"]["window"])
    gamma = _coupling(project, method)

    cycle_times: list[float] = []
    pinn_times: list[float] = []
    filter_times: list[float] = []

    for k in range(n):
        env = _env_at(traj, k)
        ctrl = _ctrl_at(traj, k)
        hist = _hist_window(traj, k, window)

        # [RUNTIME FIX] NN inference is now inside the timed region.
        t_cycle = time.perf_counter()
        t_nn = time.perf_counter()
        feat = _build_feature(x, traj, k, hist)
        residual, _res_var = _predict_residual(method, feat, hist, pinn, trans)
        x = _inject_residual(x, residual, gamma)
        pinn_times.append(time.perf_counter() - t_nn)

        x = rk4_step(x, traj.time[k], project.config["synthetic"]["dt"], ctrl, env, params)
        cycle_times.append(time.perf_counter() - t_cycle)

        x[:4] = normalize_quaternion(x[:4])
        est[k] = x

    metrics = _finalize_metrics(traj.states, est, np.array([]), np.array([]), cycle_times, pinn_times, filter_times)
    return est, metrics, np.array([]), np.array([]), np.array([])


def _run_hybrid_ukf_mekf(traj: Trajectory, project, pinn=None):
    """Proposed method: PINN prior + parallel UKF (translational) and MEKF (rotational).

    [TRUTH-LEAK FIX] v1 used a single UKF plus a correction toward the TRUE
    quaternion (truth leakage). This version runs a genuine MEKF for the
    rotational partition, updated only from real (noisy) measurements, in
    parallel with a UKF for the translational partition, and fuses the
    disjoint partitions - exactly the architecture described in the paper.
    """
    params = SpacecraftParams(**project.config["simulation"])
    include_rd = bool(project.config["synthetic"]["include_range_doppler"])
    R = measurement_noise_cov(project.config["measurement_noise"], include_rd) * float(
        project.config["filter"]["r_scale"]
    )
    Q = _process_noise(project.config) * float(project.config["filter"]["q_scale"])

    x0 = _init_state(traj)
    ukf = UKF(
        x0,
        np.diag([1e-2] * 25),
        params,
        Q,
        R,
        include_range_doppler=include_rd,
        alpha=project.config["filter"]["ukf_alpha"],
        beta=project.config["filter"]["ukf_beta"],
        kappa=project.config["filter"]["ukf_kappa"],
    )
    mekf = MEKF(x0, np.diag([1e-2] * 25), params, Q, R, include_range_doppler=include_rd)

    r_scale_pinn = float(project.config.get("fusion", {}).get("pinn_r_scale", 1.0))
    window = int(project.config["training"]["window"])

    est = np.zeros_like(traj.states)
    n = len(traj.time)
    nees = np.full(n, np.nan)
    nis = np.full(n, np.nan)
    innovations: list[np.ndarray] = []
    cycle_times: list[float] = []
    pinn_times: list[float] = []
    filter_times: list[float] = []

    for k in range(n):
        env = _env_at(traj, k)
        ctrl = _ctrl_at(traj, k)
        hist = _hist_window(traj, k, window)

        t_cycle = time.perf_counter()

        # PINN disturbance prior (timed)
        t_nn = time.perf_counter()
        feat = _build_feature(est[k - 1] if k > 0 else ukf.x, traj, k, hist)
        residual, res_var = _predict_residual("PINN+UKF+MEKF", feat, hist, pinn, None)
        _fuse_pinn_prior(ukf, residual, res_var, r_scale_pinn)
        _fuse_pinn_prior(mekf, residual, res_var, r_scale_pinn)
        pinn_times.append(time.perf_counter() - t_nn)

        # Cross-feed: each filter receives the other's partition estimate
        # so both propagate a consistent full state.
        ukf.x[:4] = normalize_quaternion(mekf.x[:4])
        ukf.x[ROT_IDX] = mekf.x[ROT_IDX]
        mekf.x[TRANS_IDX] = ukf.x[TRANS_IDX]

        t_f = time.perf_counter()
        step_u = ukf.step(
            traj.time[k], project.config["synthetic"]["dt"], env, ctrl,
            traj.measurements[k], traj.measurement_mask[k],
        )
        step_m = mekf.step(
            traj.time[k], project.config["synthetic"]["dt"], env, ctrl,
            traj.measurements[k], traj.measurement_mask[k],
        )
        filter_times.append(time.perf_counter() - t_f)

        # Fuse disjoint partitions: rotational from MEKF, translational from UKF.
        fused = ukf.x.copy()
        fused[:4] = normalize_quaternion(mekf.x[:4])
        fused[ROT_IDX] = mekf.x[ROT_IDX]
        cycle_times.append(time.perf_counter() - t_cycle)

        est[k] = fused

        if getattr(step_u, "S", None) is not None and np.size(step_u.S) > 0:
            innov = np.asarray(step_u.innovation).reshape(-1, 1)
            innovations.append(innov.squeeze().copy())
            try:
                nis[k] = float(innov.T @ np.linalg.solve(step_u.S, innov))
            except np.linalg.LinAlgError:
                nis[k] = np.nan

        # Composite error covariance: rotational rows/cols from MEKF,
        # translational from UKF (cross-partition terms approximated by UKF).
        e = _state_error_vec(traj.states[k], est[k])
        P_u = _error_covariance(ukf.P)
        P_m = _error_covariance(mekf.P)
        P24 = P_u.copy()
        P24[np.ix_(ROT_ERR_IDX, ROT_ERR_IDX)] = P_m[np.ix_(ROT_ERR_IDX, ROT_ERR_IDX)]
        if P24.shape[0] == e.size:
            try:
                nees[k] = float(e.T @ np.linalg.solve(P24, e))
            except np.linalg.LinAlgError:
                nees[k] = np.nan

    metrics = _finalize_metrics(traj.states, est, nees, nis, cycle_times, pinn_times, filter_times)
    return est, metrics, nees, nis, np.asarray(innovations, dtype=object)


def run_all_experiments(project):
    test = _load_split(project.data_dir / "test.npz")

    pinn = load_pinn(project.model_dir / "pinn.pt", project.config["device"])
    trans = load_transformer(project.model_dir / "transformer.pt", project.config["device"])

    pred_dir = ensure_dir(project.output_dir / "predictions")
    all_metrics: dict[str, pd.DataFrame] = {}

    for method in METHODS:
        metrics_all: list[dict[str, Any]] = []

        for ti, traj in enumerate(test):
            if method in {"EKF", "Adaptive-EKF", "UKF", "MEKF"}:
                est, m, nees, nis, innovations = _run_filter(method, traj, project)
            elif method == "PINN-only":
                est, m, nees, nis, innovations = _run_learned(method, traj, project, pinn=pinn)
            elif method == "Transformer-only":
                est, m, nees, nis, innovations = _run_learned(method, traj, project, trans=trans)
            elif method in {"PINN+EKF", "PINN+UKF", "PINN+MEKF"}:
                est, m, nees, nis, innovations = _run_filter(method, traj, project, pinn=pinn)
            elif method == "Transformer+MEKF":
                est, m, nees, nis, innovations = _run_filter(method, traj, project, trans=trans)
            elif method == "PINN+UKF+MEKF":
                est, m, nees, nis, innovations = _run_hybrid_ukf_mekf(traj, project, pinn=pinn)
            else:
                raise ValueError(method)

            if ti == 0:
                np.savez_compressed(
                    pred_dir / f"{method.replace('+', '_').replace('-', '_')}.npz",
                    time=traj.time,
                    truth=traj.states,
                    est=est,
                    nees=nees,
                    nis=nis,
                    innovations=innovations,
                )

            m["trial"] = ti
            metrics_all.append(m)

        all_metrics[method] = pd.DataFrame(metrics_all)

    # ---- [MC-STATS FIX] per-trial table for boxplots / variance / CIs ----
    per_trial = pd.concat(
        [df.assign(method=method) for method, df in all_metrics.items()],
        ignore_index=True,
    )
    per_trial.to_csv(project.table_dir / "metrics_per_trial.csv", index=False)

    key_metrics = [
        "attitude_geodesic_rmse",
        "position_rmse",
        "velocity_rmse",
        "angular_rate_rmse",
        "runtime_per_step_s",
    ]

    base = all_metrics["EKF"].mean(numeric_only=True)
    rows: list[dict[str, Any]] = []

    for method, df in all_metrics.items():
        summary = df.mean(numeric_only=True).to_dict()
        summary["method"] = method
        summary["n_traj"] = len(test)

        for k, v in df.std(numeric_only=True).to_dict().items():
            summary[f"{k}_std"] = v
        for k, v in df.var(numeric_only=True).to_dict().items():
            summary[f"{k}_var"] = v

        # [MC-STATS FIX] 95% confidence intervals over Monte Carlo trials
        for metric in key_metrics:
            if metric in df.columns:
                lo, hi = confidence_interval(df[metric].values)
                summary[f"{metric}_ci95_lo"] = lo
                summary[f"{metric}_ci95_hi"] = hi

        for metric in key_metrics[:4]:
            if metric in summary and metric in base:
                summary[f"{metric}_rel_impr_vs_ekf_pct"] = improvement(base[metric], summary[metric])

        # [MC-STATS FIX] paired significance tests vs EKF (parametric + non-parametric)
        if method != "EKF":
            for metric in ["position_rmse", "attitude_geodesic_rmse"]:
                if metric in df.columns:
                    a = all_metrics["EKF"][metric].values
                    b = df[metric].values
                    summary[f"p_ttest_vs_ekf_{metric}"] = paired_ttest(a, b)
                    summary[f"p_wilcoxon_vs_ekf_{metric}"] = wilcoxon_test(a, b)

        rows.append(summary)

    out = pd.DataFrame(rows)
    out.to_csv(project.table_dir / "metrics_summary.csv", index=False)
    out.to_latex(project.table_dir / "metrics_summary.tex", index=False, float_format="%.6g")

    # ---- [RUNTIME FIX] runtime breakdown table justifying reported runtimes ----
    breakdown = pd.DataFrame(
        [
            {
                "method": method,
                "pinn_ms": 1e3 * df["pinn_time_per_step_s"].mean(),
                "filter_ms": 1e3 * df["filter_time_per_step_s"].mean(),
                "total_ms": 1e3 * df["runtime_per_step_s"].mean(),
                "total_ms_std": 1e3 * df["runtime_per_step_s"].std(),
                "total_ms_median": 1e3 * df["runtime_per_step_s_median"].mean(),
            }
            for method, df in all_metrics.items()
        ]
    )
    breakdown.to_csv(project.table_dir / "runtime_breakdown.csv", index=False)
    breakdown.to_latex(project.table_dir / "runtime_breakdown.tex", index=False, float_format="%.3f")

    return rows
