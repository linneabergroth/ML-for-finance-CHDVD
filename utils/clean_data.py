"""Impute missing values in the raw close panel and save to processed/.

Usage:
    python clean_data.py [--method {ffill,bfill,xgboost}]
                         [--input  PATH]   (default: data/raw/close_panel.csv)
                         [--output PATH]   (default: data/processed/close_panel_clean.csv)
                         [--xgb-iters N]  (IterativeImputer max_iter, default: 10)

Methods:
    ffill    : carry last known price forward (good for short trading halts)
    bfill    : carry next known price backward
    xgboost  : sklearn IterativeImputer with XGBRegressor (MICE-style, cross-sectional)
               Works best when other tickers are correlated proxies for the missing one.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR


# ──────────────────────────────────────────────────────────────────────────────
# Missingness report
# ──────────────────────────────────────────────────────────────────────────────

def print_missingness(panel: pd.DataFrame) -> None:
    total = panel.size
    missing_total = panel.isna().sum().sum()
    per_col_count = panel.isna().sum()
    per_col_pct = per_col_count / len(panel) * 100

    print(f"\n── Missingness report  ({panel.shape[0]} rows × {panel.shape[1]} columns) ──")
    print(f"  Overall: {missing_total:,} / {total:,} cells missing "
          f"({missing_total / total * 100:.2f}%)\n")
    print(f"  {'Ticker':<20} {'Missing':>8}  {'%':>6}  {'First valid':>12}  {'Last valid':>12}")
    print("  " + "─" * 66)

    for col in panel.columns:
        n = per_col_count[col]
        pct = per_col_pct[col]
        first = panel[col].first_valid_index()
        last = panel[col].last_valid_index()
        first_s = first.date().isoformat() if first is not None else "N/A"
        last_s = last.date().isoformat() if last is not None else "N/A"
        flag = "  ⚠" if pct > 5 else ""
        print(f"  {col:<20} {n:>8d}  {pct:>5.1f}%  {first_s:>12}  {last_s:>12}{flag}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Imputation strategies
# ──────────────────────────────────────────────────────────────────────────────

def impute_ffill(panel: pd.DataFrame) -> pd.DataFrame:
    filled = panel.ffill()
    remaining = filled.isna().sum().sum()
    print(f"  ffill applied.  Remaining NaNs after ffill: {remaining} "
          f"(rows before any valid observation — use bfill to fill leading NaNs).")
    return filled


def impute_bfill(panel: pd.DataFrame) -> pd.DataFrame:
    filled = panel.bfill()
    remaining = filled.isna().sum().sum()
    print(f"  bfill applied.  Remaining NaNs: {remaining} "
          f"(rows after last valid observation).")
    return filled


def impute_xgboost(panel: pd.DataFrame, max_iter: int = 10) -> pd.DataFrame:
    """MICE-style imputation using XGBRegressor as the base estimator.

    Strategy:
      - Convert prices to log-returns (stationary, better for cross-sectional regression).
      - Impute missing log-returns with IterativeImputer(XGBRegressor).
      - Reconstruct prices from cumulative imputed returns, anchoring each gap to
        the last known price before it.
    """
    try:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        from xgboost import XGBRegressor
    except ImportError as e:
        raise ImportError(
            f"XGBoost imputation requires scikit-learn and xgboost: {e}"
        ) from e

    print(f"  XGBoost imputation (IterativeImputer, max_iter={max_iter}) ...")

    # Log-returns: skip the first row (always NaN after diff).
    ret_body = np.log(panel).diff().iloc[1:]

    all_nan_cols = ret_body.columns[ret_body.isna().all()].tolist()
    if all_nan_cols:
        print(f"  ⚠  Columns entirely missing (cannot impute from neighbors): {all_nan_cols}")

    imputer = IterativeImputer(
        estimator=XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=0, verbosity=0,
        ),
        max_iter=max_iter,
        random_state=0,
        verbose=0,
    )
    imputed_arr = imputer.fit_transform(ret_body.values)
    imputed_ret = pd.DataFrame(imputed_arr, index=ret_body.index, columns=ret_body.columns)

    # cumret[t] = total log-return from panel.index[0] to t (via imputed returns).
    # We anchor at index[0] with value 0 so the subtraction below works uniformly.
    cumret = pd.concat([
        pd.DataFrame([[0.0] * len(panel.columns)], index=[panel.index[0]], columns=panel.columns),
        imputed_ret.cumsum(),
    ])

    filled = panel.copy()
    for col in panel.columns:
        if col in all_nan_cols or not panel[col].isna().any():
            continue

        s = panel[col].copy()
        missing = s.isna()

        # Group consecutive NaN blocks; process each block with its own anchor.
        gap_id = missing.ne(missing.shift()).cumsum()
        for _, block in s[missing].groupby(gap_id[missing]):
            gap_start_pos = panel.index.get_loc(block.index[0])
            if gap_start_pos == 0:
                continue  # No preceding price to anchor to.
            anchor_idx = panel.index[gap_start_pos - 1]
            anchor_price = panel[col][anchor_idx]
            if pd.isna(anchor_price):
                continue
            anchor_cumret = cumret.loc[anchor_idx, col]
            for idx in block.index:
                s[idx] = anchor_price * np.exp(cumret.loc[idx, col] - anchor_cumret)

        filled[col] = s

    remaining = filled.isna().sum().sum()
    print(f"  XGBoost imputation done.  Remaining NaNs: {remaining} "
          f"(gaps with no preceding known price, or entirely missing columns).")
    return filled


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(method: str, input_path: str, output_path: str, xgb_iters: int) -> None:
    panel = pd.read_csv(input_path, index_col=0, parse_dates=True).sort_index()
    print(f"\nLoaded:  {input_path}  →  {panel.shape[0]} rows × {panel.shape[1]} columns")
    print(f"Date range: {panel.index.min().date()} → {panel.index.max().date()}")

    before = panel.isna().sum().sum()
    print_missingness(panel)

    if before == 0:
        print("  No missing values — nothing to impute. Saving as-is.")
        filled = panel
    else:
        print(f"  Applying method: {method}")
        if method == "ffill":
            filled = impute_ffill(panel)
        elif method == "bfill":
            filled = impute_bfill(panel)
        elif method == "xgboost":
            filled = impute_xgboost(panel, max_iter=xgb_iters)
        else:
            raise ValueError(f"Unknown method: {method}")

    after = filled.isna().sum().sum()
    print(f"\n  Before: {before:,} missing  →  After: {after:,} missing  "
          f"(filled {before - after:,} cells)\n")

    filled.to_csv(output_path)
    print(f"  Saved → {output_path}")


if __name__ == "__main__":
    default_in = str(RAW_DIR / "close_panel.csv")
    default_out = str(PROCESSED_DIR / "close_panel_clean.csv")

    p = argparse.ArgumentParser(description="Impute missing values in close_panel.csv.")
    p.add_argument("--method", choices=["ffill", "bfill", "xgboost"], default="ffill")
    p.add_argument("--input", default=default_in, metavar="PATH")
    p.add_argument("--output", default=default_out, metavar="PATH")
    p.add_argument("--xgb-iters", type=int, default=10,
                   help="IterativeImputer max_iter (xgboost method only)")
    args = p.parse_args()
    main(args.method, args.input, args.output, args.xgb_iters)
