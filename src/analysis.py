"""
analysis.py — Regression discontinuity helpers for the sunset analysis.

Extracted from notebooks/11_sunset_rd.ipynb so the condensed analysis
notebook can import them cleanly.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _sig(p: float) -> str:
    """Return significance stars for a p-value."""
    if np.isnan(p):
        return "n/a"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def local_linear_rdd(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float,
    kernel: str = "triangular",
) -> dict | None:
    """Local linear RD estimator at cutoff x = 0.

    Model: y = α + β·x + τ·D + γ·(x·D)  where D = 1[x ≥ 0]
    τ is the jump at x = 0 (the RD treatment effect).

    Parameters
    ----------
    x : array of running-variable values (bin midpoints or crash-level)
    y : array of outcomes
    bandwidth : float — kernel truncation point
    kernel : 'triangular' (recommended) or 'uniform'

    Returns dict with keys: tau, se, t, p, n_obs, bw, beta — or None if
    there are too few observations.
    """
    mask = (x >= -bandwidth) & (x <= bandwidth)
    xi, yi = x[mask], y[mask]
    n = len(xi)
    if n < 8:
        return None

    if kernel == "triangular":
        wi = np.maximum(0, 1 - np.abs(xi) / bandwidth)
    else:
        wi = np.ones(n)
    wi = np.maximum(wi, 1e-12)

    D = (xi >= 0).astype(float)
    X_mat = np.column_stack([np.ones(n), xi, D, xi * D])  # [1, x, D, x·D]

    # WLS: (X'WX) β = X'Wy
    XtW = X_mat.T * wi            # (k, n)
    XtWX = XtW @ X_mat            # (k, k)
    XtWy = XtW @ yi               # (k,)
    try:
        beta = np.linalg.solve(XtWX, XtWy)
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        return None

    # HC0 sandwich variance
    resid = yi - X_mat @ beta
    score = (wi * resid)[:, None] * X_mat
    V = XtWX_inv @ (score.T @ score) @ XtWX_inv
    se_vec = np.sqrt(np.clip(np.diag(V), 0, None))

    tau, tau_se = float(beta[2]), float(se_vec[2])
    t_stat = tau / tau_se if tau_se > 1e-12 else np.nan
    p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=max(n - 4, 1))))

    return {
        "tau":   tau,
        "se":    tau_se,
        "t":     float(t_stat),
        "p":     p_val,
        "n_obs": n,
        "bw":    bandwidth,
        "beta":  beta,
    }


def binned_rd(
    series,
    bandwidth: float,
    bin_min: float = 2,
    min_crashes: int = 60,
) -> dict | None:
    """Bin a running-variable series into fixed-width bins and run LLR.

    Parameters
    ----------
    series : array-like or pd.Series of running-variable values
    bandwidth : float — half-width of the analysis window (minutes)
    bin_min : float — width of each bin in minutes (default 2)
    min_crashes : int — minimum crash count required to run (default 60)

    Returns result dict from local_linear_rdd, or None.
    """
    import pandas as pd
    x = pd.Series(series).dropna().values
    x = x[(x >= -bandwidth) & (x <= bandwidth)]
    if len(x) < min_crashes:
        return None
    bins = np.arange(-bandwidth, bandwidth + bin_min, bin_min)
    midpoints = (bins[:-1] + bins[1:]) / 2
    counts, _ = np.histogram(x, bins=bins)
    return local_linear_rdd(midpoints, counts.astype(float), bandwidth)
