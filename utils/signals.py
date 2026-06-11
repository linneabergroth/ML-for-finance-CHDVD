"""Economically-motivated signal sets (paper requires >= 2 sets).

Each signal set function returns a DataFrame of *raw indicator values* aligned
to the return index.  signals_to_positions() converts those values into discrete
position rules {-1, 0, +1} that are used as ML features and standalone baselines.

Adding a new signal set
-----------------------
1. Write signal_set_e(prices, returns) -> pd.DataFrame  (raw values, named columns)
2. Add a block inside signals_to_positions(sig_e=None, ...) converting each column
   to a position rule following the existing pattern.
3. Wire it into build_feature_frame in utils/dataset.py (use_set_e flag pattern).

Set A — Trend / Momentum
    Jegadeesh-Titman (1993) momentum; MA-crossover rule.
Set B — Mean-reversion / Volatility
    RSI, Bollinger bands, volatility-clustering regime.
Set C — Statistical / Regime
    Autocorrelation, skewness, volatility-of-volatility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Signal catalogue  (hyperparameters + position-rule description per signal)
# --------------------------------------------------------------------------- #
# Each entry documents the numerical parameters baked into the signal functions
# below.  When you change a window or threshold in the code, update it here too.
# To add Set E: add entries with 'set': 'E' and wire them into signals_to_positions.
SIGNAL_HYPERPARAMS: dict[str, dict] = {
    # ── Set A: Trend / Momentum ──────────────────────────────────────────────
    "ma_cross_50_200":    {"set": "A", "fast_window": 50,  "slow_window": 200,
                           "position_rule": "sign(fast_SMA - slow_SMA)"},
    "mom_12_1":           {"set": "A", "lookback_days": 252, "skip_days": 21,
                           "position_rule": "sign(price[-21] / price[-252] - 1)"},
    "trend_strength_20":  {"set": "A", "window": 20,
                           "position_rule": "mean(sign(daily_ret)) ∈ [-1, 1] (continuous)"},
    # ── Set B: Mean-reversion / Volatility ──────────────────────────────────
    "rsi_14":             {"set": "B", "window": 14, "oversold_thr": 30, "overbought_thr": 70,
                           "position_rule": "+1 if RSI < 30 (oversold), -1 if RSI > 70 (overbought), 0 otherwise"},
    "bollinger_z_20":     {"set": "B", "window": 20, "lower_thr": -1, "upper_thr": 1,
                           "position_rule": "+1 if z < -1 (below band), -1 if z > 1 (above band), 0 otherwise"},
    "vol_ratio_20_100":   {"set": "B", "fast_window": 20, "slow_window": 100, "exit_thr": 1.5,
                           "position_rule": "-1 (risk-off) if ratio > 1.5, else 0"},
    # ── Set C: Statistical / Regime ──────────────────────────────────────────
    "ret_autocorr_20":    {"set": "C", "window": 20,
                           "position_rule": "sign(autocorr): +1=momentum regime, -1=mean-rev regime"},
    "ret_skew_60":        {"set": "C", "window": 60,
                           "position_rule": "sign(skew): -1=crash-risk tail"},
    "vov_ratio_20_60":    {"set": "C", "fast_window": 20, "slow_window": 60, "flat_thr": 1.2,
                           "position_rule": "0 (flat) if ratio > 1.2 (regime instability), else +1"},
}


def signal_metadata_table() -> pd.DataFrame:
    """Return a DataFrame summarising signal hyperparameters.  Useful for display."""
    rows = []
    for signal, params in SIGNAL_HYPERPARAMS.items():
        hypers = {k: v for k, v in params.items()
                  if k not in ("set", "position_rule")}
        rows.append({
            "signal":        signal,
            "set":           params.get("set", "?"),
            "hyperparameters": ", ".join(f"{k}={v}" for k, v in hypers.items()),
            "position_rule": params.get("position_rule", ""),
        })
    return pd.DataFrame(rows).set_index("signal")


# --------------------------------------------------------------------------- #
# Set A — Trend / Momentum
# --------------------------------------------------------------------------- #
def signal_set_a(prices: pd.Series, returns: pd.Series) -> pd.DataFrame:
    """Trend / momentum signals (raw values)."""
    out: dict[str, pd.Series] = {}

    # Moving-average crossover: fast SMA above slow SMA -> uptrend.
    sma_fast = prices.rolling(50).mean()
    sma_slow = prices.rolling(200).mean()
    out["ma_cross_50_200"] = (sma_fast - sma_slow) / sma_slow

    # 12-1 momentum (skip most recent month to avoid short-term reversal).
    out["mom_12_1"] = prices.shift(21) / prices.shift(252) - 1.0

    # Sign-consistency of recent returns (trend strength).
    out["trend_strength_20"] = np.sign(returns).rolling(20).mean()

    df = pd.DataFrame(out).reindex(returns.index)
    return df.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- #
# Set B — Mean-reversion / Volatility
# --------------------------------------------------------------------------- #
def signal_set_b(prices: pd.Series, returns: pd.Series) -> pd.DataFrame:
    """Mean-reversion / volatility signals (raw values)."""
    out: dict[str, pd.Series] = {}

    # RSI(14): >70 overbought, <30 oversold (contrarian).
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - 100 / (1 + rs)

    # Bollinger z-score: distance from 20d mean in std units (mean reversion).
    ma20 = prices.rolling(20).mean()
    sd20 = prices.rolling(20).std()
    out["bollinger_z_20"] = (prices - ma20) / sd20

    # Volatility regime: short vol relative to long vol (risk state).
    out["vol_ratio_20_100"] = returns.rolling(20).std() / returns.rolling(100).std()

    df = pd.DataFrame(out).reindex(returns.index)
    return df.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- #
# Set C — Statistical / Regime
# --------------------------------------------------------------------------- #
def signal_set_c(prices: pd.Series, returns: pd.Series) -> pd.DataFrame:
    """Statistical / regime signals (raw values, derived from returns only)."""
    out: dict[str, pd.Series] = {}

    # Rolling 1-lag autocorrelation (efficient computation via rolling cov/std).
    r_lag = returns.shift(1)
    cov_20 = returns.rolling(20).cov(r_lag)
    std_20 = returns.rolling(20).std()
    std_lag_20 = r_lag.rolling(20).std()
    out["ret_autocorr_20"] = cov_20 / (std_20 * std_lag_20)

    # Rolling skewness of returns (60-day window for stability).
    out["ret_skew_60"] = returns.rolling(60).skew()

    # Volatility-of-volatility: std of 5-day realized vol over [20d / 60d].
    vol_5 = returns.rolling(5).std()
    vov_20 = vol_5.rolling(20).std()
    vov_60 = vol_5.rolling(60).std().replace(0, np.nan)
    out["vov_ratio_20_60"] = vov_20 / vov_60

    df = pd.DataFrame(out).reindex(returns.index)
    return df.replace([np.inf, -np.inf], np.nan)

# --------------------------------------------------------------------------- #
# Position rules — convert raw signals to {-1, 0, +1} trading rules
# --------------------------------------------------------------------------- #
def signals_to_positions(
    sig_a: pd.DataFrame | None = None,
    sig_b: pd.DataFrame | None = None,
    sig_c: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert raw signal values to discrete position rules in {-1, 0, +1}.

    +1 = long, -1 = short/hedge, 0 = flat.  Each column is a directly
    interpretable trading rule.  ML models learn to weight and combine these
    rather than re-discovering thresholds from raw indicator values.

    To add Set E:
        1. Compute raw values with signal_set_e().
        2. Add a block below (copy the Set A pattern) translating each column.
        3. Pass sig_e=... when calling this function.
    """
    out: dict[str, pd.Series] = {}
    idx: pd.Index | None = None

    # ── Set A: Trend / Momentum ──────────────────────────────────────────────
    if sig_a is not None:
        idx = sig_a.index
        out["pos_ma_cross"]  = np.sign(sig_a["ma_cross_50_200"])   # uptrend → long
        out["pos_mom_12_1"]  = np.sign(sig_a["mom_12_1"])          # TSMOM: +1 if yr-ret > 0
        out["pos_trend_20"]  = np.sign(sig_a["trend_strength_20"])           # sign majority ∈ [-1, 1]

    # ── Set B: Mean-Reversion / Volatility ──────────────────────────────────
    if sig_b is not None:
        if idx is None:
            idx = sig_b.index
        rsi = sig_b["rsi_14"]
        # RSI < 30 = oversold → long;  RSI > 70 = overbought → short
        out["pos_rsi"] = pd.Series(
            np.where(rsi < 30, 1.0, np.where(rsi > 70, -1.0, 0.0)),
            index=sig_b.index,
        )
        bz = sig_b["bollinger_z_20"]
        # Below lower band → mean-reversion long;  above upper band → short
        out["pos_bollinger"] = pd.Series(
            np.where(bz < -1, 1.0, np.where(bz > 1, -1.0, 0.0)),
            index=sig_b.index,
        )
        # High short-/long-vol ratio = risk-off (exit);  no forced long signal
        out["pos_vol_regime"] = pd.Series(
            np.where(sig_b["vol_ratio_20_100"] > 1.5, -1.0, 0.0),
            index=sig_b.index,
        )

    # ── Set C: Statistical / Regime ──────────────────────────────────────────
    if sig_c is not None:
        if idx is None:
            idx = sig_c.index
        # Positive autocorr = momentum regime (follow trend); negative = mean-rev
        out["pos_autocorr"] = np.sign(sig_c["ret_autocorr_20"])
        # Negative skew = crash-risk premium; positive skew = upside asymmetry
        out["pos_skew"] = np.sign(sig_c["ret_skew_60"])
        # High vol-of-vol = regime instability → flat;  stable vol → long
        out["pos_vov_stable"] = pd.Series(
            np.where(sig_c["vov_ratio_20_60"] > 1.2, 0.0, 1.0),
            index=sig_c.index,
        )


    if idx is None:
        raise ValueError("At least one signal set (sig_a/b/c/d) must be provided.")

    return pd.DataFrame(
        {
            k: v.reindex(idx) if isinstance(v, pd.Series) else pd.Series(v, index=idx)
            for k, v in out.items()
        }
    ).replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- #
# Builder (kept for direct raw-signal access / ablations)
# --------------------------------------------------------------------------- #
def build_signals(
    prices: pd.Series,
    returns: pd.Series,
    use_set_a: bool = True,
    use_set_b: bool = True,
    use_set_c: bool = False,
) -> pd.DataFrame:
    """Concatenate raw signal values for the requested sets (A/B/C).

    Prefer signals_to_positions() when building ML feature matrices.
    Use this when you need the raw indicator values (e.g. ablation plots).
    """
    parts = []
    if use_set_a:
        parts.append(signal_set_a(prices, returns))
    if use_set_b:
        parts.append(signal_set_b(prices, returns))
    if use_set_c:
        parts.append(signal_set_c(prices, returns))
    if not parts:
        raise ValueError("Select at least one signal set (A, B, or C).")
    return pd.concat(parts, axis=1)
