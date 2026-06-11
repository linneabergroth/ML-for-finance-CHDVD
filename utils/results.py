"""Persist and reload per-run metrics for cross-run comparison."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def save_run(
    name: str,
    res: dict,
    split,
    seed: int,
    signals_meta: dict | None = None,
    weights_path: str | None = None,
    position_mode: str = "sign",
    cost_bps: float = 1.0,
    horizon: int = 1,
) -> str:
    """Save run metadata + metrics to results/<run_id>.json. Returns the path."""
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now()
    safe = (name.replace(" ", "_").replace("(", "").replace(")", "")
                .replace("/", "_").replace("+", "plus"))
    run_id = f"{ts.strftime('%Y-%m-%d_%H-%M-%S')}_{safe}_seed{seed}"

    def _float(v):
        try:
            f = float(v)
            return None if f != f else f  # NaN -> None
        except (TypeError, ValueError):
            return None

    record = {
        "run_id": run_id,
        "timestamp": ts.isoformat(),
        "model": name,
        "seed": seed,
        "horizon": horizon,
        "signals_meta": signals_meta or {},
        "position_mode": position_mode,
        "cost_bps": cost_bps,
        "n_features": int(split.n_features),
        "n_train": int(len(split.y_train)),
        "n_val": int(len(split.y_val)),
        "n_test": int(len(split.y_test)),
        "date_test_start": str(split.dates_test.min().date()),
        "date_test_end": str(split.dates_test.max().date()),
        "weights_path": str(weights_path) if weights_path else None,
        "metrics": {k: _float(v) for k, v in res.items()},
    }
    path = RESULTS_DIR / f"{run_id}.json"
    path.write_text(json.dumps(record, indent=2))
    return str(path)


def load_all_runs(results_dir: str | None = None) -> pd.DataFrame:
    """Load all result JSONs into a flat DataFrame (one row per run).

    Handles two file formats:
      - Training run JSONs  (have ``run_id``)
      - Optuna study JSONs  (have ``best_value``; filename = optuna_<category>.json)
    """
    d = Path(results_dir or RESULTS_DIR)
    rows = []
    for p in sorted(d.glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue

        if "run_id" in r:
            # ── Training run ───────────────────────────────────────────────
            row = {
                "run_id":    r["run_id"],
                "timestamp": r["timestamp"],
                "model":     r["model"],
                "seed":      r["seed"],
                "horizon":   r.get("horizon", 1),
                "source":    "training",
                **r.get("signals_meta", {}),
                **{k: v for k, v in r.get("metrics", {}).items()},
            }
        elif "best_value" in r:
            # ── Optuna study (best trial only) ─────────────────────────────
            # filename pattern: optuna_<category>.json
            category = p.stem.replace("optuna_", "").replace("_", "-")
            bp = r.get("best_params", {})
            row = {
                "run_id":        p.stem,
                "timestamp":     None,
                "model":         f"Optuna-{category}",
                "seed":          None,
                "source":        "optuna",
                "signal_config": bp.get("signal_config"),
                "IC_out":        r["best_value"],   # val-set IC is the objective
                "n_trials":      len(r.get("trials", [])),
                **{k: v for k, v in bp.items() if k != "signal_config"},
            }
        else:
            continue  # unknown format — skip

        rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df
