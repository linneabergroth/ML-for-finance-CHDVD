"""Turn return prediction into a traded strategy and compare to the benchmark.

Strategy rule (simple, transparent for the paper):
  position_t = clip(sign or scaled forecast, -1, 1), applied to the *next*
  period's realised return. The forecast at t predicts r_{t+1}, so the P&L of
  acting on it is position_t * realised_{t+1}.

Benchmarks:
  * buy-and-hold CHDVD.SW (the index itself)
  * random-walk / historical-mean (predict 0 excess -> always long): this is the
    canonical hard-to-beat benchmark (Goyal-Welch 2008).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.metrics import annualised_return, sharpe_ratio


def forecast_to_position(
    forecast: np.ndarray, mode: str = "sign", cost_bps: float = 0.0
) -> np.ndarray:
    """Map predicted returns to positions in [-1, 1].

    `sign`       : long/short 1 on the sign of the forecast (classic rule).
    `scaled`     : proportional to forecast, capped at +/-1 (conviction sizing).
    `long`       : long-only, 1 if forecast > 0 else 0 (no shorting).
    `cost_aware` : sign rule with a no-trade band — trade only when
                   |forecast| > cost_bps/1e4, else hold the previous position
                   (identical to `sign` when cost_bps=0).
    """
    f = np.asarray(forecast, dtype=float)
    if mode == "sign":
        return np.sign(f)
    if mode == "cost_aware":
        return cost_aware_positions(f, cost_bps)
    if mode == "long":
        return (f > 0).astype(float)
    if mode == "scaled":
        s = f / (np.std(f) + 1e-12)
        return np.clip(s, -1.0, 1.0)
    raise ValueError(f"unknown mode {mode!r}")


def cost_aware_positions(forecast: np.ndarray, cost_bps: float = 0.0) -> np.ndarray:
    """Sign positions with a no-trade band: hold instead of trading small edges.

    Switching from the current position to sign(forecast) costs
    |Δposition| * cost_bps/1e4, while the expected gain of the switch is
    |Δposition| * |forecast|. The switch is therefore only worth it when
    |forecast| > cost_bps/1e4; otherwise the previous position is held.
    With cost_bps=0 this reduces to the plain 'sign' rule.
    """
    f = np.asarray(forecast, dtype=float)
    thr = cost_bps / 1e4
    pos = np.zeros(len(f))
    prev = 0.0
    for i, fi in enumerate(f):
        if np.isfinite(fi) and np.sign(fi) != prev and abs(fi) > thr:
            prev = np.sign(fi)
        pos[i] = prev
    return pos


def strategy_returns(
    forecast: np.ndarray,
    realized: np.ndarray,
    mode: str = "sign",
    cost_bps: float = 0.0,
) -> np.ndarray:
    """Daily strategy returns, optionally net of turnover transaction costs."""
    pos = forecast_to_position(forecast, mode, cost_bps)
    pnl = pos * np.asarray(realized, dtype=float)
    if cost_bps > 0:
        turnover = np.abs(np.diff(pos, prepend=0.0))
        pnl = pnl - turnover * (cost_bps / 1e4)
    return pnl


def backtest_table(
    forecast: np.ndarray,
    realized: np.ndarray,
    dates: pd.DatetimeIndex,
    mode: str = "sign",
    cost_bps: float = 0.0,
    horizon: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """Equity curves + summary stats for the strategy vs buy-and-hold.

    Returns (curves_df, stats_series). `curves_df` holds cumulative log returns
    indexed by date for plotting in the notebook.

    `horizon` sets the observation frequency (trading days per period). At h>1
    each return covers h days, so there are 252/h observations per year — both
    AnnRet and Sharpe are scaled by 252/h, not 252.
    """
    from utils.metrics import TRADING_DAYS
    periods = round(TRADING_DAYS / max(1, horizon))

    strat = strategy_returns(forecast, realized, mode, cost_bps)
    bench = np.asarray(realized, dtype=float)  # buy-and-hold the index

    curves = pd.DataFrame(
        {"strategy": np.cumsum(strat), "buy_and_hold": np.cumsum(bench)},
        index=dates,
    )
    stats = pd.Series(
        {
            "strat_Sharpe": sharpe_ratio(strat, periods=periods),
            "strat_AnnReturn": annualised_return(strat, periods=periods),
            "bench_Sharpe": sharpe_ratio(bench, periods=periods),
            "bench_AnnReturn": annualised_return(bench, periods=periods),
        }
    )
    return curves, stats
