"""Load cached raw prices and turn them into clean return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.config import BENCHMARK_TICKER, RAW_DIR


def _csv_name(ticker: str) -> str:
    return f"{ticker.replace('.', '_')}.csv"


def validate_prices(s: pd.Series, ffill_limit: int = 3) -> pd.Series:
    """Audit and clean a price series. Prints a quality report; returns cleaned series.

    Steps (in order):
      1. Drop NaN (pre-history missing rows in the panel).
      2. Remove non-positive prices (data corruption).
      3. Deduplicate dates (keep first occurrence).
      4. Forward-fill gaps up to `ffill_limit` consecutive days (short holidays/halts).
      5. Report zero-return streaks longer than ffill_limit (possible stale data).
    """
    name = s.name or "series"
    issues: list[str] = []
    original_len = len(s)

    # 1. NaN
    n_nan = s.isna().sum()
    if n_nan:
        s = s.dropna()
        issues.append(f"dropped {n_nan} NaN rows")

    # 2. Non-positive prices
    n_bad = (s <= 0).sum()
    if n_bad:
        s = s[s > 0]
        issues.append(f"dropped {n_bad} non-positive prices")

    # 3. Duplicate dates
    n_dup = s.index.duplicated().sum()
    if n_dup:
        s = s[~s.index.duplicated(keep="first")]
        issues.append(f"dropped {n_dup} duplicate dates")

    # 4. Forward-fill short gaps (trading halts / holidays with stale price)
    n_missing = s.isna().sum()  # after ffill this should be 0
    s = s.ffill(limit=ffill_limit)
    n_filled = s.isna().sum()
    if n_missing:
        issues.append(f"forward-filled up to {ffill_limit}-day gaps")
    if n_filled:
        issues.append(f"{n_filled} gaps longer than {ffill_limit} days remain as NaN — dropped")
        s = s.dropna()

    # 5. Zero-return streak check (stale/repeated prices beyond ffill window)
    zero_ret = (s.diff().abs() < 1e-8)
    rle = zero_ret.ne(zero_ret.shift()).cumsum()
    max_streak = zero_ret.groupby(rle).sum().max()
    if max_streak > ffill_limit:
        issues.append(f"longest zero-return streak: {int(max_streak)} days "
                      f"(possible stale data beyond ffill window)")

    status = "OK" if not issues else f"{len(issues)} issue(s)"
    print(f"  [{name}] {len(s):,} obs  "
          f"{s.index.min().date()} → {s.index.max().date()}  [{status}]")
    for msg in issues:
        print(f"    ⚠  {msg}")

    return s


def load_close_panel() -> pd.DataFrame:
    """Load the combined adjusted-close panel saved by extract_data.py."""
    path = RAW_DIR / "close_panel.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python extract_data.py` first."
        )
    panel = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return panel


def load_ohlcv(ticker: str = BENCHMARK_TICKER) -> pd.DataFrame:
    """Load full OHLCV (Open, High, Low, Close, Volume) for one ticker.

    Required for Set D (volume/microstructure) signals.
    Returns a validated DataFrame aligned to trading days.
    """
    path = RAW_DIR / _csv_name(ticker)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python extract_data.py` first."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()

    # Basic sanity: keep only rows where Close is positive and finite.
    df = df[df["Close"].notna() & (df["Close"] > 0)]

    # Forward-fill short gaps (holidays/halts) for all OHLCV columns.
    df = df.ffill(limit=3)

    # Report
    n_zero_vol = (df["Volume"] == 0).sum()
    print(f"  [OHLCV {ticker}] {len(df):,} obs  "
          f"{df.index.min().date()} → {df.index.max().date()}  "
          f"[zero-volume days: {n_zero_vol}]")
    return df


def load_benchmark_close(ticker: str = BENCHMARK_TICKER) -> pd.Series:
    """Adjusted close of the traded index, validated and cleaned."""
    panel = load_close_panel()
    if ticker not in panel.columns:
        raise KeyError(f"{ticker} missing from panel: {list(panel.columns)}")
    s = panel[ticker].copy()
    s.name = ticker
    return validate_prices(s)


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Daily log returns. First row (NaN) is dropped."""
    return np.log(prices).diff().dropna()


def benchmark_returns(ticker: str = BENCHMARK_TICKER) -> pd.Series:
    """Daily log returns of the traded index."""
    r = log_returns(load_benchmark_close(ticker))
    r.name = "ret"
    return r
