from __future__ import annotations

"""Monte Carlo statistics helpers.

[MC-STATS FIX] Added bootstrap confidence intervals, Wilcoxon signed-rank
test, and a per-method summary helper (mean / std / var / 95% CI).
"""

import numpy as np
import pandas as pd
from scipy import stats


def confidence_interval(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Student-t confidence interval for the mean."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        m = float(np.mean(x)) if n else float("nan")
        return m, m
    m = np.mean(x)
    h = stats.sem(x) * stats.t.ppf(1 - alpha / 2, n - 1)
    return float(m - h), float(m + h)


def bootstrap_ci(x: np.ndarray, alpha: float = 0.05, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    """Non-parametric bootstrap CI for the mean (does not assume normality)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 2:
        m = float(np.mean(x)) if x.size else float("nan")
        return m, m
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_ttest(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    return float(stats.ttest_rel(a, b, nan_policy="omit").pvalue)


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> float:
    """Non-parametric paired test; robust to non-Gaussian RMSE distributions."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2 or np.allclose(a, b):
        return float("nan")
    try:
        return float(stats.wilcoxon(a, b, nan_policy="omit").pvalue)
    except ValueError:
        return float("nan")


def improvement(a: float, b: float) -> float:
    return float(100.0 * (a - b) / max(abs(a), 1e-12))


def mc_summary(df: pd.DataFrame, metric: str) -> dict[str, float]:
    """Full Monte Carlo summary for one metric column."""
    x = df[metric].values
    lo, hi = confidence_interval(x)
    blo, bhi = bootstrap_ci(x)
    return {
        f"{metric}_mean": float(np.nanmean(x)),
        f"{metric}_median": float(np.nanmedian(x)),
        f"{metric}_std": float(np.nanstd(x, ddof=1)),
        f"{metric}_var": float(np.nanvar(x, ddof=1)),
        f"{metric}_ci95_lo": lo,
        f"{metric}_ci95_hi": hi,
        f"{metric}_boot_ci95_lo": blo,
        f"{metric}_boot_ci95_hi": bhi,
        f"{metric}_iqr": float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25)),
    }
