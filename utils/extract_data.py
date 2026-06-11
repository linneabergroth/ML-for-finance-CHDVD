"""Download raw price data (OHLCV) for CHDVD.SW and its equity holdings.

Run once, or whenever you want to refresh prices:

    python extract_data.py [--start YYYY-MM-DD] [--end YYYY-MM-DD] # default from utils.config (2010-today)

What this script does
─────────────────────
1. Fetches the CHDVD.SW currents holdings from yfinance (best-effort).
   It uses the list in utils.config.HOLDINGS for the individual holdings.
2. For tickers with a ticker rename (Roche, Helvetia Baloise) in the timeframe,
   it downloads both the old and new symbols and combine them together them at the switch date.
3. Saves one OHLCV CSV per equity to data/raw/<TICKER>.csv.
4. Saves the combined adjusted-close panel to data/raw/close_panel.csv.
5. Prints a per-ticker coverage report (date range, row count, longest gap).
6. Warns about non-equity positions found in the ETF (cash, collateral, futures).
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd
import yfinance as yf

from config import (
    BENCHMARK_TICKER,
    END_DATE,
    HOLDING_WEIGHTS,
    HOLDINGS,
    RAW_DIR,
    START_DATE,
    STITCHES,
)

# Non-equity position types that appear in ETF composition reports.
_NON_EQUITY_KEYWORDS = ("cash", "collateral", "future", "forward", "index")


# ──────────────────────────────────────────────────────────────────────────────
# ETF composition inference
# ──────────────────────────────────────────────────────────────────────────────

def try_infer_holdings(etf_ticker: str) -> pd.DataFrame | None:
    """Try to read the ETF's top-holdings table from yfinance.

    Returns a DataFrame (columns: holdingName, holdingPercent, symbol) or None
    if yfinance doesn't have the data for this ETF.
    """
    try:
        fd = yf.Ticker(etf_ticker).funds_data
        top = fd.top_holdings
        if top is not None and not top.empty:
            return top
    except Exception:
        pass
    return None


def print_etf_composition(etf_ticker: str, config_holdings: list[str]) -> None:
    """Print what yfinance knows about the ETF, noting non-equity positions."""
    print(f"\n── ETF composition check ({etf_ticker}) ──")
    top = try_infer_holdings(etf_ticker)
    if top is None:
        print("  yfinance has no holdings data for this ETF; using utils.config.HOLDINGS.")
    else:
        non_equity = []
        equity_symbols = []
        for idx, row in top.iterrows():
            name = str(row.get("holdingName", idx) or idx).lower()
            sym = row.get("symbol", "")
            pct = row.get("holdingPercent", float("nan"))
            is_non_eq = any(kw in name for kw in _NON_EQUITY_KEYWORDS)
            if is_non_eq:
                non_equity.append((row.get("holdingName", idx), pct))
            else:
                equity_symbols.append(sym)

        print(f"  yfinance returned {len(top)} positions.")
        if non_equity:
            print("  ⚠  Non-equity positions (excluded from download):")
            for label, pct in non_equity:
                pct_str = f"{pct*100:.2f}%" if pd.notna(pct) else "n/a"
                print(f"       {label}  ({pct_str})")

        # Show which config holdings yfinance doesn't recognise
        yf_symbols = set(s for s in equity_symbols if s)
        config_set = set(config_holdings)
        unrecognised = config_set - yf_symbols
        if unrecognised:
            print(f"  ℹ  {len(unrecognised)} config holding(s) not found in yfinance data "
                  f"(normal for SIX-listed ETFs): {sorted(unrecognised)}")

    print(f"  Using {len(config_holdings)} tickers from utils.config.HOLDINGS.")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Download helpers
# ──────────────────────────────────────────────────────────────────────────────

def _download(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """Raw yfinance download. Returns empty DataFrame on failure."""
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end or date.today().isoformat(),
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        print(f"  [error] {ticker}: {exc}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def download_one(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """Download OHLCV for a single ticker. Returns empty frame on failure."""
    return _download(ticker, start, end)


def download_stitched(
    new_ticker: str,
    old_ticker: str,
    switch_date: str,
    start: str,
    end: str | None,
) -> pd.DataFrame:
    """Download new_ticker and stitch with old_ticker before switch_date.

    If new_ticker already carries history back to `start` (yfinance migrated
    the history), no stitching is done.
    """
    df_new = _download(new_ticker, start, end)
    switch_ts = pd.Timestamp(switch_date)
    start_ts = pd.Timestamp(start)

    # If the new ticker reaches close to the start date, history was migrated.
    if not df_new.empty and df_new.index.min() <= start_ts + pd.Timedelta(days=400):
        print(f"    [{new_ticker}] history already continuous back to "
              f"{df_new.index.min().date()} — no stitch needed.")
        return df_new

    df_old = _download(old_ticker, start, switch_date)

    if df_old.empty and df_new.empty:
        return pd.DataFrame()
    if df_old.empty:
        print(f"    [{new_ticker}] ⚠  no data for predecessor {old_ticker}; "
              f"returning {new_ticker} only (history from {df_new.index.min().date() if not df_new.empty else 'N/A'}).")
        return df_new
    if df_new.empty:
        print(f"    [{new_ticker}] ⚠  no data for {new_ticker}; "
              f"returning {old_ticker} only.")
        return df_old

    pre = df_old[df_old.index < switch_ts]
    post = df_new[df_new.index >= switch_ts]
    combined = pd.concat([pre, post])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    print(f"    [{new_ticker}] stitched {old_ticker} (<{switch_date}) + "
          f"{new_ticker} (≥{switch_date}): {len(pre)} + {len(post)} = {len(combined)} rows.")
    return combined


# ──────────────────────────────────────────────────────────────────────────────
# Coverage report
# ──────────────────────────────────────────────────────────────────────────────

def coverage_row(ticker: str, df: pd.DataFrame, start: str) -> dict:
    start_ts = pd.Timestamp(start)
    if df.empty:
        return {
            "ticker": ticker, "first": None, "last": None, "rows": 0,
            "reaches_start": False, "max_gap_days": None,
            "missing_pct_total": 100.0, "missing_pct_active": 100.0,
        }
    idx = df.index
    first_ts, last_ts = idx.min(), idx.max()

    # Total missing %: over the full requested window (start → last).
    total_span = max(1, (last_ts - start_ts).days * 252 / 365)
    missing_pct_total = max(0.0, (1 - len(df) / total_span) * 100)

    # Active missing %: within the ticker's own history (first → last).
    active_span = max(1, (last_ts - first_ts).days * 252 / 365)
    missing_pct_active = max(0.0, (1 - len(df) / active_span) * 100)

    gaps = idx.to_series().diff().dt.days.dropna()
    real_gaps = gaps[gaps > 5]
    return {
        "ticker": ticker,
        "first": first_ts.date().isoformat(),
        "last": last_ts.date().isoformat(),
        "rows": len(df),
        "reaches_start": first_ts <= start_ts + pd.Timedelta(days=400),
        "max_gap_days": int(real_gaps.max()) if not real_gaps.empty else 0,
        "missing_pct_total": round(missing_pct_total, 1),
        "missing_pct_active": round(missing_pct_active, 1),
    }


def print_coverage_report(
    rows: list[dict], requested_start: str, weight_map: dict[str, float]
) -> None:
    print(f"\n── Coverage report  (requested start: {requested_start}) ─────────────────────")
    print(f"  {'Ticker':<12} {'ETF wt%':>7} {'First':>12} {'Last':>12} {'Rows':>6} "
          f"{'Max gap':>8}  {'Miss%(total)':>13}  {'Miss%(active)':>13}")
    print("  " + "─" * 93)
    for r in rows:
        wt = weight_map.get(r["ticker"])
        wt_str = f"{wt:>5.2f}%" if wt is not None else "     -"
        if r["rows"] == 0:
            print(f"  ⚠  {r['ticker']:<12} {wt_str}  NO DATA")
            continue
        late_note = ""
        if not r["reaches_start"]:
            late_note = f"  <- starts {r['first']} (later than requested)"
        print(f"  {r['ticker']:<12} {wt_str} {r['first']:>12} {r['last']:>12} {r['rows']:>6} "
              f"  {r['max_gap_days']:>6}d  {r['missing_pct_total']:>11.1f}%  "
              f"{r['missing_pct_active']:>11.1f}%{late_note}")
    no_data = [r["ticker"] for r in rows if r["rows"] == 0]
    short = [r["ticker"] for r in rows if r["rows"] > 0 and not r["reaches_start"]]
    if no_data:
        print(f"\n  ⚠  No data at all: {no_data}")
    if short:
        print(f"  ⚠  Late-start tickers (Miss%total includes pre-IPO/pre-listing period): "
              f"{short}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _build_weight_map() -> dict[str, float]:
    """Return ticker → ETF weight (%). Tries yfinance live data, falls back to config."""
    top = try_infer_holdings(BENCHMARK_TICKER)
    if top is not None:
        wm: dict[str, float] = {}
        for idx, row in top.iterrows():
            sym = row.get("symbol", str(idx))
            pct = row.get("holdingPercent", None)
            if sym and pct is not None:
                wm[sym] = float(pct) * 100
        if wm:
            return wm
    return dict(HOLDING_WEIGHTS)


def main(start: str, end: str | None) -> None:
    # 1. ETF composition check + weight map (used in coverage report)
    weight_map = _build_weight_map()
    print_etf_composition(BENCHMARK_TICKER, HOLDINGS)

    # 2. Download benchmark + holdings
    tickers = [BENCHMARK_TICKER] + HOLDINGS
    closes: dict[str, pd.Series] = {}
    coverage: list[dict] = []

    for t in tickers:
        stitch_cfg = STITCHES.get(t)
        if stitch_cfg:
            print(f"  {t}  [stitched from {stitch_cfg['old']}]")
            print(f"    note: {stitch_cfg['note']}")
            df = download_stitched(t, stitch_cfg["old"], stitch_cfg["switch"], start, end)
        else:
            df = download_one(t, start, end)

        if df.empty:
            print(f"  ⚠  {t}: no data — skipping")
            coverage.append(coverage_row(t, df, start))
            continue

        out = RAW_DIR / f"{t.replace('.', '_')}.csv"
        df.to_csv(out)
        closes[t] = df["Close"]
        coverage.append(coverage_row(t, df, start))
        if stitch_cfg:
            print(f"    saved {len(df)} rows → {out.name}")
        else:
            print(f"  {t}: {len(df)} rows  "
                  f"{df.index.min().date()} → {df.index.max().date()}  → {out.name}")

    # 3. Combined close panel
    panel = pd.DataFrame(closes).sort_index()
    panel.to_csv(RAW_DIR / "close_panel.csv")
    print(f"\n  Saved close panel: {panel.shape} → close_panel.csv")
    print(f"  Run `python clean_data.py` next to impute any gaps.")

    # 4. Coverage report
    print_coverage_report(coverage, start, weight_map)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download CHDVD.SW + holdings OHLCV data.")
    p.add_argument("--start", default=START_DATE)
    p.add_argument("--end", default=END_DATE)
    args = p.parse_args()
    main(args.start, args.end)
