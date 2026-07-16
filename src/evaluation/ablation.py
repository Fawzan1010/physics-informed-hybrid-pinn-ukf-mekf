from __future__ import annotations

"""Hyperparameter sensitivity / ablation study.

Addresses the reviewer comment: "There is no discussion of PINN depth,
window length, loss weights, learning rate, coupling coefficient."

One-at-a-time sweep around the baseline configuration. Axes that affect
training (depth, hidden, window, lr, lambda_norm) retrain the PINN with a
reduced epoch budget over multiple seeds; the coupling coefficient reuses
the trained baseline model. Each setting is evaluated with the proposed
PINN+UKF+MEKF pipeline on a subset of test trajectories.

Outputs:
  outputs/tables/ablation_results.csv
  outputs/figures/ablation_<axis>.pdf

Run with: python main.py --mode ablate
"""

import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.experiments import _load_split, _run_hybrid_ukf_mekf
from src.pinn.train import train_pinn, load_pinn
from src.utils.reproducibility import set_seed

DEFAULT_AXES = {
    "depth": [2, 4, 6],
    "hidden": [64, 128, 256],
    "window": [2, 4, 8],
    "lr": [3e-4, 1e-3, 3e-3],
    "lambda_norm": [0.0, 0.1, 1.0],
    "pinn_r_scale": [0.25, 1.0, 4.0, 16.0],
}


def _evaluate(project, pinn, test) -> dict[str, float]:
    att, pos, run = [], [], []
    for traj in test:
        _est, m, *_ = _run_hybrid_ukf_mekf(traj, project, pinn=pinn)
        att.append(m["attitude_geodesic_rmse"])
        pos.append(m["position_rmse"])
        run.append(m["runtime_per_step_s"])
    return {
        "attitude_rmse_mean": float(np.mean(att)),
        "attitude_rmse_std": float(np.std(att, ddof=1)) if len(att) > 1 else 0.0,
        "position_rmse_mean": float(np.mean(pos)),
        "position_rmse_std": float(np.std(pos, ddof=1)) if len(pos) > 1 else 0.0,
        "runtime_per_step_ms": float(1e3 * np.mean(run)),
    }


def run_ablation(project) -> pd.DataFrame:
    abl_cfg = project.config.get("ablation", {})
    axes = {k: abl_cfg.get(k, v) for k, v in DEFAULT_AXES.items()}
    seeds = abl_cfg.get("seeds", [0, 1, 2])
    n_test = int(abl_cfg.get("subset_test_trajectories", 20))
    n_train = int(abl_cfg.get("subset_train_trajectories", 40))
    epochs = int(abl_cfg.get("epochs", 15))

    train = _load_split(project.data_dir / "train.npz")[:n_train]
    val = _load_split(project.data_dir / "val.npz")
    test = _load_split(project.data_dir / "test.npz")[:n_test]

    rows = []
    scratch = project.output_dir / "ablation_models"
    scratch.mkdir(parents=True, exist_ok=True)

    for axis, values in axes.items():
        for value in values:
            for seed in seeds:
                cfg = copy.deepcopy(project.config)
                cfg["training"]["pinn_epochs"] = epochs
                set_seed(int(seed))

                if axis == "pinn_r_scale":
                    # No retraining needed: reuse the baseline model.
                    pinn = load_pinn(project.model_dir / "pinn.pt", cfg["device"])
                    proj = copy.copy(project)
                    proj.config = copy.deepcopy(project.config)
                    proj.config.setdefault("fusion", {})["pinn_r_scale"] = float(value)
                else:
                    if axis == "lambda_norm":
                        cfg["training"]["lambda_norm"] = float(value)
                    elif axis == "lr":
                        cfg["training"]["lr"] = float(value)
                    else:
                        cfg["training"][axis] = int(value)
                    pinn = train_pinn(train, val, cfg, scratch / f"{axis}_{value}_{seed}", cfg["device"])
                    proj = copy.copy(project)
                    proj.config = cfg

                res = _evaluate(proj, pinn, test)
                res.update({"axis": axis, "value": value, "seed": seed})
                rows.append(res)
                print(f"[ablation] {axis}={value} seed={seed}: "
                      f"att={res['attitude_rmse_mean']:.4f} pos={res['position_rmse_mean']:.4f}")

    df = pd.DataFrame(rows)
    project.table_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(project.table_dir / "ablation_results.csv", index=False)

    # Aggregate over seeds and plot mean +/- std per axis.
    agg = df.groupby(["axis", "value"]).agg(
        att_mean=("attitude_rmse_mean", "mean"),
        att_std=("attitude_rmse_mean", "std"),
        pos_mean=("position_rmse_mean", "mean"),
        pos_std=("position_rmse_mean", "std"),
    ).reset_index()
    agg.to_csv(project.table_dir / "ablation_summary.csv", index=False)

    for axis in axes:
        sub = agg[agg["axis"] == axis]
        if sub.empty:
            continue
        fig, ax1 = plt.subplots(figsize=(6, 4))
        x = np.arange(len(sub))
        ax1.errorbar(x, sub["att_mean"], yerr=sub["att_std"], marker="o", capsize=4, label="Attitude RMSE")
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(v) for v in sub["value"]])
        ax1.set_xlabel(axis)
        ax1.set_ylabel("Attitude geodesic RMSE")
        ax2 = ax1.twinx()
        ax2.errorbar(x, sub["pos_mean"], yerr=sub["pos_std"], marker="s", color="tab:red", capsize=4, label="Position RMSE")
        ax2.set_ylabel("Position RMSE")
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(project.figure_dir / f"ablation_{axis}.pdf")
        plt.close(fig)

    print(f"Ablation results written to {project.table_dir / 'ablation_results.csv'}")
    return df
