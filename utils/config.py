"""Configurations and reproducibility helpers.

Everything that another module might want to tune (paths, tickers, window
sizes, model hyper-parameters, the global seed) lives here so the notebook and
the .py modules share a single files where they can rely on.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

for _d in (RAW_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fix *all* RNGs we rely on so a run is reproducible.

    Covers Python `random`, NumPy, and (if installed) PyTorch on CPU/GPU plus
    cuDNN determinism. XGBoost is made deterministic via its own `seed` arg in
    `model.py`.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
# Benchmark / traded instrument: iShares Swiss Dividend ETF (the "index").
BENCHMARK_TICKER = "^SPX"

# Full equity holdings of CHDVD.SW (iShares Swiss Dividend ETF).
# Source: https://www.ishares.com/ch/individual/en/products/264108/
# Cash, collateral and futures positions are excluded.
# Re-run extract_data.py after changing this list.
HOLDINGS = []
# HOLDINGS = [
#     # Top 10 by ETF weight (~96% of NAV) — full history available from START_DATE.
#     "ZURN.SW",  # Zurich Insurance Group    ~15.1%
#     "ROP.SW",   # Roche PS (participation cert, ISIN CH1499059983) ~15.0% — was ROG.SW before 2026-03-17
#     "NOVN.SW",  # Novartis                  ~14.1%
#     "NESN.SW",  # Nestlé                    ~14.1%
#     "HOLN.SW",  # Holcim                    ~11.1%
#     "SREN.SW",  # Swiss Re                  ~ 9.6%
#     "SLHN.SW",  # Swiss Life                ~ 6.0%
#     "SIKA.SW",  # Sika                      ~ 3.4%
#     "HBAN.SW",  # Helvetia Baloise Holding  ~ 3.2% — was HELN.SW before 2025-12-05
#     "GIVN.SW",  # Givaudan                  ~ 3.1%
#     # Remaining small-cap holdings excluded (each <1% of NAV):
#     # "GALE.SW",  # 0.6% — IPO 2017, shortens usable history by ~4 years
#     # "BION.SW",  # 0.6%
#     # "DKSH.SW",  # 0.5%
#     # "BKW.SW",   # 0.4%
#     # "SUN.SW",   # 0.4%
#     # "EFGN.SW",  # 0.4%
#     # "BUCN.SW",  # 0.4%
#     # "BCHN.SW",  # 0.3%
#     # "ALSN.SW",  # 0.2%
#     # "KARN.SW",  # 0.2%
# ]

# Handle when a SIX ticker changed symbol, we download both
# halves and concatenate at the switch date. Key = current (new) ticker in HOLDINGS.
# Approximate ETF weights from the iShares fact sheet (as of 2026-06).
# Used as fallback when yfinance can't return live weights for CHDVD.SW.
HOLDING_WEIGHTS: dict[str, float] = {}
# HOLDING_WEIGHTS: dict[str, float] = {
#     "ZURN.SW": 15.06,
#     "ROP.SW":  14.97,
#     "NOVN.SW": 14.07,
#     "NESN.SW": 14.07,
#     "HOLN.SW": 11.07,
#     "SREN.SW":  9.61,
#     "SLHN.SW":  5.98,
#     "SIKA.SW":  3.41,
#     "HBAN.SW":  3.19,
#     "GIVN.SW":  3.10,
#     # "GALE.SW": 0.63,  # dropped: IPO 2017, shortens usable history by ~4 years
#     "BION.SW":  0.56,
#     "DKSH.SW":  0.53,
#     "BKW.SW":   0.45,
#     "SUN.SW":   0.43,
#     "EFGN.SW":  0.40,
#     "BUCN.SW":  0.37,
#     "BCHN.SW":  0.28,
#     "ALSN.SW":  0.23,
#     "KARN.SW":  0.23,
# }

# STITCHES: dict[str, dict[str, str]] = {
#     "ROP.SW": {
#         "old": "ROG.SW",
#         "switch": "2026-03-17",
#         "note": "Roche Genussschein (ROG) relisted as participation cert (ROP) on 2026-03-17",
#     },
#     "HBAN.SW": {
#         "old": "HELN.SW",
#         "switch": "2025-12-05",
#         "note": "Helvetia Holding merged with Baloise; new entity HBAN listed on 2025-12-05",
#     },
# }
STITCHES: dict[str, dict[str, str]] = {}

START_DATE = "2010-01-04"
END_DATE = None  # None -> up to today

# --------------------------------------------------------------------------- #
# Feature / dataset construction
# --------------------------------------------------------------------------- #
PAST_WINDOW = 20          # trailing daily returns fed to the model
HORIZON = 1               # predict the 1-day (daily) compounded return
HORIZONS = (1, 5, 10, 30) # horizon ablation values (trading days) (1, 5, 10, 30)  
ROLL_WINDOWS = (50, 200, 365)  # windows for rolling indicators

# Chronological split fractions (no shuffling -> no look-ahead leakage).
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test = 1 - TRAIN_FRAC - VAL_FRAC

# Signal ablation configuration
SIGNAL_CONFIGS: dict[str, dict] = {
    'A only':     dict(use_set_a=True,  use_set_b=False, use_set_c=False, use_set_m=False),
    'B only':     dict(use_set_a=False, use_set_b=True,  use_set_c=False, use_set_m=False),
    'C only':     dict(use_set_a=False, use_set_b=False, use_set_c=True,  use_set_m=False),
    'M only':     dict(use_set_a=False, use_set_b=False, use_set_c=False, use_set_m=True),
    'A + B':      dict(use_set_a=True,  use_set_b=True,  use_set_c=False, use_set_m=False),
    'A + C':      dict(use_set_a=True,  use_set_b=False, use_set_c=True,  use_set_m=False),
    'B + C':      dict(use_set_a=False, use_set_b=True,  use_set_c=True,  use_set_m=False),
    'A + M':      dict(use_set_a=True,  use_set_b=False, use_set_c=False, use_set_m=True),
    'B + M':      dict(use_set_a=False, use_set_b=True,  use_set_c=False, use_set_m=True),
    'C + M':      dict(use_set_a=False, use_set_b=False, use_set_c=True,  use_set_m=True),
    'A + B + C':  dict(use_set_a=True,  use_set_b=True,  use_set_c=True, use_set_m=False),
    'A + B + C + M': dict(use_set_a=True,  use_set_b=True,  use_set_c=True, use_set_m=True),
    'No signals': dict(use_set_a=False, use_set_b=False, use_set_c=False, use_set_m=False),
}
ABLATION_MODELS = ('Linear', 'XGBoost', 'VAE-medium')

# Backtest settings (see utils/backtest.py forecast_to_position for modes)
POSITION_MODE = 'cost_aware'  # default: sign rule with no-trade band (= plain 'sign' when COST_BPS=0);
                              # alternatives: 'sign', 'long', 'scaled' — see utils/backtest.py
COST_BPS = 1.0          # transaction cost in basis points per unit turnover
COST_BPS_GRID = (0.0, 1.0, 5.0, 10.0, 20.0)  # cost-sensitivity sweep; 5-20 bps ≈ realistic retail range

# PCA compression of holdings features
PCA_N_COMPONENTS_FULL = 10      # full-signal (A + B + C) runs
PCA_N_COMPONENTS_ABLATION = 20  # signal-ablation and Optuna runs
PCA_VARIANCE_THRESHOLD = 0.95

# Notebook analysis settings
OPTUNA_N_TRIALS = 50
FORECAST_PLOT_DAYS = 60  # trading days per forecast-vs-realized panel

# Plot colour per model family (extend when adding a family to MODEL_FAMILIES)
FAMILY_COLORS: dict[str, str] = {
    'Linear':   'tab:blue',
    'XGBoost':  'tab:orange',
    'VAE-MLP':  'tab:green',
    'VAE-Attn': 'tab:red',
}


@dataclass
class VAEConfig:
    """Hyper-parameters for the Masked-VAE forecaster."""

    hidden_dims: tuple[int, ...] = (128, 64)
    latent_dim: int = 16
    beta: float = 1.0          # KL weight (beta-VAE)
    mask_prob: float = 0.30    # fraction of input features masked during training
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    batch_size: int = 64
    patience: int = 15         # early-stopping patience on val loss
    use_attention: bool = False  # swap MLP encoder for a TransformerEncoder


# VaAE different onfigurations e.g. MaskedVAE(n_features, VAE_CONFIGS["large"]).
VAE_CONFIGS: dict[str, "VAEConfig"] = {}  # filled after class definition below


@dataclass
class XGBConfig:
    """Hyper-parameters for the XGBoost baseline."""

    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 30   # stop if val RMSE doesn't improve


@dataclass
class Config:
    """Top-level bundle passed around the pipeline."""

    seed: int = SEED
    benchmark: str = BENCHMARK_TICKER
    holdings: list[str] = field(default_factory=lambda: list(HOLDINGS))
    start: str = START_DATE
    end: str | None = END_DATE
    past_window: int = PAST_WINDOW
    horizon: int = HORIZON
    roll_windows: tuple[int, ...] = ROLL_WINDOWS
    train_frac: float = TRAIN_FRAC
    val_frac: float = VAL_FRAC
    vae: VAEConfig = field(default_factory=VAEConfig)
    xgb: XGBConfig = field(default_factory=XGBConfig)


# Populate after Config so forward-ref VAEConfig is resolved.
VAE_CONFIGS.update({
    # ── MLP encoder (use_attention=False) ──────────────────────────────────
    "tiny":        VAEConfig(hidden_dims=(32, 16),          latent_dim=8),
    "small":       VAEConfig(hidden_dims=(64, 32),          latent_dim=16),
    "medium":      VAEConfig(hidden_dims=(128, 64),         latent_dim=16),   # default
    "large":       VAEConfig(hidden_dims=(256, 128, 64),    latent_dim=32),
    "xlarge":      VAEConfig(hidden_dims=(512, 256, 128),   latent_dim=64),
    # ── Transformer encoder (use_attention=True) ────────────────────────────
    "attn_tiny":   VAEConfig(hidden_dims=(32, 16),          latent_dim=8,  use_attention=True),
    "attn_small":  VAEConfig(hidden_dims=(64, 32),          latent_dim=16, use_attention=True),
    "attn_medium": VAEConfig(hidden_dims=(128, 64),         latent_dim=16, use_attention=True),
    "attn_large":  VAEConfig(hidden_dims=(256, 128),        latent_dim=32, use_attention=True),
})

# Display name → VAE_CONFIGS key (single source of truth for VAE model naming).
VAE_KEY_MAP: dict[str, str] = {
    "VAE-tiny":        "tiny",
    "VAE-small":       "small",
    "VAE-medium":      "medium",
    "VAE-large":       "large",
    "VAE-xlarge":      "xlarge",
    "VAE-attn-small":  "attn_small",
    "VAE-attn-medium": "attn_medium",
    "VAE-attn-large":  "attn_large",
}

# Model family groupings (used for plots and best-per-family selection).
MODEL_FAMILIES: dict[str, list[str]] = {
    "Linear":   ["Linear"],
    "XGBoost":  ["XGBoost"],
    "VAE-MLP":  ["VAE-tiny", "VAE-small", "VAE-medium", "VAE-large", "VAE-xlarge"],
    "VAE-Attn": ["VAE-attn-small", "VAE-attn-medium", "VAE-attn-large"],
}

# Inverse of MODEL_FAMILIES: model name → family string.
MODEL_CATEGORY: dict[str, str] = {
    m: fam for fam, models in MODEL_FAMILIES.items() for m in models
}
