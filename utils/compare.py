#!/usr/bin/env python
"""Compare saved model runs from results/.

Usage:
    python compare.py              # all runs
    python compare.py --latest     # only the most recent run per model
    python compare.py --results-dir /path/to/results

Can also be imported from the notebook:
    from compare import compare_runs
    compare_runs()
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from results import load_all_runs

_METRICS = [
    ("IC_in",      "IC  in-sample",     "{:+.4f}"),
    ("IC_out",     "IC  out-of-sample", "{:+.4f}"),
    ("RankIC_out", "Rank-IC  OOS",      "{:+.4f}"),
    ("Sharpe_out", "Sharpe  OOS",       "{:+.3f}"),
    ("AnnRet_out", "Ann.Ret OOS",       "{:+.2%}"),
]
_W = 72  # output width


def _fmt(v, fmt):
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return "—"


def _line(char="─"):
    return "  " + char * _W


def compare_runs(results_dir: str | None = None, latest_only: bool = False) -> pd.DataFrame:
    """Load results, print a formatted comparison, return the DataFrame."""
    df = load_all_runs(results_dir)
    if df.empty:
        print("  No runs found. Run the notebook with TRAIN=True first.")
        return df

    if latest_only:
        df = (df.sort_values("timestamp")
                .groupby("model").tail(1)
                .reset_index(drop=True))

    n = len(df)
    _ts = df["timestamp"].dropna()
    ts_range = f"{_ts.min().date()} → {_ts.max().date()}" if not _ts.empty else "—"

    print(f"\n{'═' * _W}")
    print(f"  Model Comparison Report — {n} run{'s' if n != 1 else ''} "
          f"({ts_range})")
    print(f"{'═' * _W}\n")

    # ── All-runs table ──────────────────────────────────────────────────────
    show = df.sort_values("IC_out", ascending=False).reset_index(drop=True)

    print(f"  {'#':<3} {'Model':<22} {'IC_in[-1,1]':>13} {'IC_out[-1,1]':>13} "
          f"{'Rank-IC[-1,1]':>14} {'Sharpe[ann]':>12} {'AnnRet[%pa]':>12}")
    print(_line())
    for i, row in show.iterrows():
        print(f"  {i+1:<3} {row['model']:<22}"
              f" {_fmt(row.get('IC_in'),      '{:+.4f}'):>13}"
              f" {_fmt(row.get('IC_out'),     '{:+.4f}'):>13}"
              f" {_fmt(row.get('RankIC_out'), '{:+.4f}'):>14}"
              f" {_fmt(row.get('Sharpe_out'), '{:+.3f}'):>12}"
              f" {_fmt(row.get('AnnRet_out'), '{:+.2%}'):>12}")
    print()

    # ── Best per metric ─────────────────────────────────────────────────────
    print("  Best per metric:")
    for col, label, fmt in _METRICS:
        if col not in df.columns:
            continue
        valid = df.dropna(subset=[col])
        if valid.empty:
            continue
        idx = valid[col].idxmax()
        best = valid.loc[idx]
        print(f"  ★  {label:<24}  {best['model']}  ({_fmt(best[col], fmt)})")
    print()

    # ── Signal-set pivot (if ablation runs present) ─────────────────────────
    if {"use_set_a", "use_set_b"}.issubset(df.columns):
        _signal_pivot(df)

    # ── Qualitative interpretation ──────────────────────────────────────────
    _qualitative(df)

    return df


def _signal_pivot(df: pd.DataFrame) -> None:
    df = df[df["source"] == "training"].copy() if "source" in df.columns else df.copy()
    if df.empty:
        return
    df["signals"] = df.apply(
        lambda r: ("A+B" if (r.get("use_set_a") and r.get("use_set_b"))
                   else "A only" if r.get("use_set_a")
                   else "B only" if r.get("use_set_b")
                   else "none"),
        axis=1,
    )
    pivot = (df.groupby(["model", "signals"])[["IC_out", "Sharpe_out"]]
               .mean()
               .round(4)
               .unstack("signals"))
    if not pivot.empty:
        print("  IC_out by model × signal set:")
        print("  " + pivot["IC_out"].to_string().replace("\n", "\n  "))
        print()


def _qualitative(df: pd.DataFrame) -> None:
    print(_line())
    print("  Qualitative Summary\n")

    # Overfitting gap
    if {"IC_in", "IC_out"}.issubset(df.columns):
        v = df.dropna(subset=["IC_in", "IC_out"]).copy()
        v["gap"] = v["IC_in"] - v["IC_out"]
        worst = v.loc[v["gap"].idxmax()]
        best_gen = v.loc[v["gap"].idxmin()]
        print("  In/out-of-sample IC gap (smaller = less overfitting):")
        print(f"    Largest  → {worst['model']:<22}  ΔIC = {worst['gap']:+.4f}"
              f"  (in={worst['IC_in']:+.4f}, out={worst['IC_out']:+.4f})")
        print(f"    Smallest → {best_gen['model']:<22}  ΔIC = {best_gen['gap']:+.4f}"
              f"  (in={best_gen['IC_in']:+.4f}, out={best_gen['IC_out']:+.4f})")
        print()

    # Signal-set hypothesis (training runs only — Optuna rows lack use_set_* cols)
    _tr = df[df["source"] == "training"] if "source" in df.columns else df
    if {"use_set_a", "use_set_b"}.issubset(_tr.columns):
        _a = _tr["use_set_a"].fillna(False).astype(bool)
        _b = _tr["use_set_b"].fillna(False).astype(bool)
        ab     = _tr[ _a &  _b]["IC_out"].mean()
        a_only = _tr[ _a & ~_b]["IC_out"].mean()
        b_only = _tr[~_a &  _b]["IC_out"].mean()
        if not any(np.isnan(x) for x in [ab, a_only, b_only] if x is not None):
            best = max([("A+B", ab), ("A only", a_only), ("B only", b_only)],
                       key=lambda t: t[1])
            supported = "SUPPORTED" if best[0] == "A+B" else "NOT SUPPORTED"
            print("  Signal sets (avg IC_out across models):")
            print(f"    A only = {a_only:+.4f}   B only = {b_only:+.4f}   A+B = {ab:+.4f}")
            print(f"    Hypothesis (combining sets helps): {supported}")
            print()

    # Model ranking
    if "IC_out" in df.columns:
        ranked = df.groupby("model")["IC_out"].mean().sort_values(ascending=False)
        print("  Average IC_out ranking (predictive power):")
        for i, (model, ic) in enumerate(ranked.items(), 1):
            star = "  ★" if i == 1 else ""
            print(f"    {i}. {model:<22}  IC={ic:+.4f} [-1,1]{star}")
        print()

    if "Sharpe_out" in df.columns:
        ranked_s = df.groupby("model")["Sharpe_out"].mean().sort_values(ascending=False)
        print("  Average Sharpe ranking (risk-adjusted return):")
        for i, (model, s) in enumerate(ranked_s.items(), 1):
            star = "  ★" if i == 1 else ""
            print(f"    {i}. {model:<22}  Sharpe={s:+.3f} [ann.]{star}")
        print()

    print(_line("═"))
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare saved model runs.")
    p.add_argument("--latest", action="store_true",
                   help="Only the most recent run per model")
    p.add_argument("--results-dir", default=None,
                   help="Path to results/ directory (default: repo root/results/)")
    args = p.parse_args()
    compare_runs(args.results_dir, args.latest)
