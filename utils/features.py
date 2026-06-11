"""Feature engineering for the benchmark and its holdings.

For each ticker (benchmark + holdings), for each window X in roll_windows:
  {tkr}_pct_rank_{X}   : percentile rank of today's price in [0,1]
  {tkr}_vs_max_{X}     : price / rolling_max  - 1
  {tkr}_vs_min_{X}     : price / rolling_min  - 1
  {tkr}_vs_mean_{X}    : price / rolling_mean - 1  (valuation proxy)
  {tkr}_vol_{X}        : annualised realised volatility
  {tkr}_sharpe_{X}     : rolling_mean_ret / rolling_std (quality-adjusted momentum)
  {tkr}_fft_1_{X} .. {tkr}_fft_3_{X} : top-3 FFT amplitudes, normalised by sqrt(X)

Same set over the full available history (expanding window, suffix _global).

Cross-sectional (one per day, only when holding_prices is provided):
  xs_dispersion   : cross-sectional std of holding log-returns
  xs_mean_corr_50 : mean pairwise Pearson correlation of holdings over last 50 days

Benchmark only:
  ret_lag_0 .. ret_lag_{past_window-1} : trailing daily log-returns
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from utils.config import PAST_WINDOW, ROLL_WINDOWS


def _pfx(ticker: str) -> str:
    return ticker.replace(".", "_")


def _rolling_fft(returns: np.ndarray, window: int) -> np.ndarray:
    """Vectorised rolling top-5 FFT amplitudes. Returns (n, 5) array, NaN where invalid."""
    n = len(returns)
    out = np.full((n, 5), np.nan)
    if n < window:
        return out
    valid = ~np.isnan(returns)
    filled = np.where(valid, returns, 0.0)
    wins = sliding_window_view(filled, window)               # (n-W+1, W)
    has_nan = sliding_window_view(~valid, window).any(axis=1)
    fft_mags = np.abs(np.fft.rfft(wins, axis=1))[:, 1:]     # skip DC
    fft_mags /= np.sqrt(window)
    top5 = np.sort(fft_mags, axis=1)[:, ::-1][:, :5]
    top5[has_nan] = np.nan
    out[window - 1:] = top5
    return out


def _ticker_rolling(
    prices: pd.Series, returns: pd.Series, window: int, prefix: str
) -> pd.DataFrame:
    w = str(window)
    roll_p = prices.rolling(window, min_periods=window)
    roll_r = returns.rolling(window, min_periods=window)
    pct_rank = prices.rolling(window, min_periods=window).apply(
        lambda x: (x < x[-1]).mean(), raw=True
    )
    fft = _rolling_fft(returns.values.astype(float), window)
    return pd.DataFrame(
        {
            f"{prefix}_pct_rank_{w}": pct_rank,
            f"{prefix}_vs_max_{w}":   prices / roll_p.max() - 1,
            f"{prefix}_vs_min_{w}":   prices / roll_p.min() - 1,
            f"{prefix}_vs_mean_{w}":  prices / roll_p.mean() - 1,
            f"{prefix}_vol_{w}":      roll_r.std() * np.sqrt(252),
            f"{prefix}_sharpe_{w}":   roll_r.mean() / roll_r.std(),
            f"{prefix}_fft_1_{w}":    fft[:, 0],
            f"{prefix}_fft_2_{w}":    fft[:, 1],
            f"{prefix}_fft_3_{w}":    fft[:, 2],
            f"{prefix}_fft_4_{w}":    fft[:, 3],
            f"{prefix}_fft_5_{w}":    fft[:, 4],
        },
        index=prices.index,
    )


def _ticker_global(
    prices: pd.Series, returns: pd.Series, prefix: str
) -> pd.DataFrame:
    exp_p = prices.expanding(min_periods=2)
    exp_r = returns.expanding(min_periods=2)
    pct_rank = prices.expanding(min_periods=2).apply(
        lambda x: (x < x[-1]).mean(), raw=True
    )
    arr = returns.values.astype(float)
    n = len(arr)
    fft = np.full((n, 5), np.nan)
    valid_mask = ~np.isnan(arr)
    if valid_mask.any():
        first_valid = int(np.argmax(valid_mask))
        for i in range(first_valid + 5, n):
            chunk = arr[first_valid : i + 1]
            if np.any(np.isnan(chunk)):
                continue
            mags = np.abs(np.fft.rfft(chunk))[1:]
            if len(mags) < 5:
                continue
            mags /= np.sqrt(len(chunk))
            fft[i] = np.sort(mags)[::-1][:5]
    return pd.DataFrame(
        {
            f"{prefix}_pct_rank_global": pct_rank,
            f"{prefix}_vs_max_global":   prices / exp_p.max() - 1,
            f"{prefix}_vs_min_global":   prices / exp_p.min() - 1,
            f"{prefix}_vs_mean_global":  prices / exp_p.mean() - 1,
            f"{prefix}_vol_global":      exp_r.std() * np.sqrt(252),
            f"{prefix}_sharpe_global":   exp_r.mean() / exp_r.std(),
            f"{prefix}_fft_1_global":    fft[:, 0],
            f"{prefix}_fft_2_global":    fft[:, 1],
            f"{prefix}_fft_3_global":    fft[:, 2],
            f"{prefix}_fft_4_global":    fft[:, 3],
            f"{prefix}_fft_5_global":    fft[:, 4],
        },
        index=prices.index,
    )


def _xs_features(holding_returns: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional dispersion + mean pairwise rolling correlation."""
    xs_disp = holding_returns.std(axis=1, skipna=True).rename("xs_dispersion")

    arr = holding_returns.values.astype(float)
    n, k = arr.shape
    window = 50  # noqa: F841 (used below)
    corr_arr = np.full(n, np.nan)
    if k >= 2:
        for i in range(window - 1, n):
            chunk = arr[i - window + 1 : i + 1]
            valid_cols = ~np.isnan(chunk).any(axis=0)
            kv = valid_cols.sum()
            if kv < 2:
                continue
            c = np.corrcoef(chunk[:, valid_cols].T)
            corr_arr[i] = c[np.triu_indices(kv, k=1)].mean()

    xs_corr = pd.Series(corr_arr, index=holding_returns.index, name="xs_mean_corr_50")
    return pd.concat([xs_disp, xs_corr], axis=1)


def make_features(
    benchmark_prices: pd.Series,
    benchmark_returns: pd.Series,
    holding_prices: pd.DataFrame | None = None,
    roll_windows: tuple[int, ...] = ROLL_WINDOWS,
    past_window: int = PAST_WINDOW,
) -> pd.DataFrame:
    """Assemble the full feature matrix.

    benchmark_prices/returns : CHDVD.SW aligned price and log-return series.
    holding_prices           : DataFrame with one column per holding (optional).
                               When None, only benchmark features are computed.
    """
    if holding_prices is None:
        holding_prices = pd.DataFrame(index=benchmark_prices.index)

    parts: list[pd.DataFrame] = []

    # ── Per-ticker hofding features (benchmark + each holding) ──────────────────────
    bench_name = str(benchmark_prices.name or "CHDVD_SW")
    all_prices = pd.concat(
        [benchmark_prices.rename(bench_name), holding_prices], axis=1
    )
    all_returns = np.log(all_prices).diff()

    for tkr in all_prices.columns:
        prefix = _pfx(str(tkr))
        p, r = all_prices[tkr], all_returns[tkr]
        for w in roll_windows:
            parts.append(_ticker_rolling(p, r, w, prefix))
        parts.append(_ticker_global(p, r, prefix))

    # ── Cross-sectional features (only when holdings are provided) ───────────
    if not holding_prices.empty:
        holding_returns = np.log(holding_prices).diff()
        parts.append(_xs_features(holding_returns))

    # ── Benchmark lagged returns ─────────────────────────────────────────────
    lags = pd.DataFrame(
        {f"ret_lag_{k}": benchmark_returns.shift(k) for k in range(past_window)},
        index=benchmark_returns.index,
    )
    parts.append(lags)

    return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- #
# Feature catalogue
# --------------------------------------------------------------------------- #
def feature_summary(feats_df: pd.DataFrame, bench_name: str = "CHDVD.SW") -> pd.DataFrame:
    """Return a table describing the raw feature matrix by group.

    To add a new feature group
    --------------------------
    1. Write a helper function _my_feature(prices, returns) -> pd.DataFrame
       with clearly named columns.
    2. Append its result to `parts` inside make_features() above.
    3. (Optional) add a row to the `rows` list here for documentation.
    """
    pfx = bench_name.replace(".", "_") + "_"
    bench_rolling = [c for c in feats_df.columns
                     if c.startswith(pfx) and not c.startswith("ret_lag_")]
    lag_rets   = [c for c in feats_df.columns if c.startswith("ret_lag_")]
    xs_cols    = [c for c in feats_df.columns if c in ("xs_dispersion", "xs_mean_corr_50")]
    hold_cols  = [c for c in feats_df.columns
                  if c not in bench_rolling and c not in lag_rets and c not in xs_cols]

    n_lags = len(lag_rets)
    rows = [
        ("Benchmark rolling",  len(bench_rolling), f"{pfx}{{type}}_{{window}}",
         "pct_rank, vs_max/min/mean, vol, Sharpe, FFT  ×  roll_windows + global"),
        ("Holdings rolling",   len(hold_cols),      "[ticker]_{type}_{window}",
         "same 9 features × each holding × roll_windows  (→ PCA in make_splits)"),
        ("Cross-sectional",    len(xs_cols),         "xs_*",
         "xs_dispersion (holding ret std), xs_mean_corr_50 (mean pairwise corr)"),
        ("Lagged returns",     n_lags,               "ret_lag_{k}",
         f"trailing {n_lags} daily log-returns of {bench_name}"),
    ]
    return pd.DataFrame(rows, columns=["Group", "Count", "Col pattern", "Description"])
