"""Evaluation metrics required by the course: Information Coefficient + Sharpe."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

TRADING_DAYS = 252


def information_coefficient(
    predicted: np.ndarray, realized: np.ndarray, method: str = "pearson"
) -> float:
    """Correlation between predicted and realized returns (predictive power).

    `pearson` is the standard IC; `spearman` (rank IC) is more robust to
    outliers. Returns NaN if inputs are degenerate (no variance).
    """
    p = np.asarray(predicted, dtype=float)
    r = np.asarray(realized, dtype=float)
    mask = np.isfinite(p) & np.isfinite(r)
    p, r = p[mask], r[mask]
    if len(p) < 3 or np.std(p) == 0 or np.std(r) == 0:
        return float("nan")
    corr = spearmanr(p, r)[0] if method == "spearman" else pearsonr(p, r)[0]
    return float(corr)


def sharpe_ratio(returns: np.ndarray, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe ratio of a daily return stream (risk-free = 0)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or np.std(r) == 0:
        return float("nan")
    return float(np.mean(r) / np.std(r) * np.sqrt(periods))


def annualised_return(returns: np.ndarray, periods: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    return float(np.mean(r) * periods)


def summary(predicted: np.ndarray, realized: np.ndarray, strat_returns: np.ndarray) -> pd.Series:
    """One-line scorecard for a model on a given split."""
    return pd.Series(
        {
            "IC": information_coefficient(predicted, realized, "pearson"),
            "RankIC": information_coefficient(predicted, realized, "spearman"),
            "Sharpe": sharpe_ratio(strat_returns),
            "AnnReturn": annualised_return(strat_returns),
        }
    )
