"""Turn prices/returns + features + signals into a model-ready supervised set.

Design (masked-VAE forecasting):
  * At decision time t we know returns up to t. The feature row at t contains
    PAST_WINDOW lagged returns + rolling indicators + signal positions (all trailing).
  * The label y_t is the *next* return r_{t+HORIZON} -- this is the token the
    Masked-VAE reconstructs while it is masked, i.e. the forecast.
  * Splits are strictly chronological (train -> val -> test) so the test set is
    genuinely out-of-sample. Scalers are fit on TRAIN ONLY (no leakage).

Feature columns in the final matrix (default: pca_all_features=False)
------------------------------------------------------------------------
  [CHDVD_SW_* | ret_lag_*]   — benchmark rolling features + lagged returns (scaled)
  [pos_*]                    — signal position rules {-1, 0, +1} (scaled)
  [pca_0 … pca_k]            — PCA of holding + cross-sectional rolling features

When pca_all_features=True: all columns go through a single scaler → PCA,
producing only pca_0 … pca_k.

Extending the pipeline
-----------------------
  * New feature group: add it in utils/features.py; no changes needed here.
  * New signal set: add signal_set_e();
    wire it in build_feature_frame with a use_set_e flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils.config import Config
from utils.features import make_features
from utils.signals import signal_set_a, signal_set_b, signal_set_c, signal_set_macro, signals_to_positions   


@dataclass
class SplitData:
    """Scaled arrays + bookkeeping needed for evaluation / backtest."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    dates_train: pd.DatetimeIndex
    dates_val: pd.DatetimeIndex
    dates_test: pd.DatetimeIndex
    feature_names: list[str]
    x_scaler: StandardScaler   # fitted on benchmark + position columns (or all if pca_all_features)
    y_scaler: StandardScaler
    # PCA pipeline — stored for out-of-sample transforms and diagnostics
    hold_scaler: StandardScaler | None = field(default=None)
    hold_pca: PCA | None = field(default=None)
    pca_explained_var: np.ndarray | None = field(default=None)  # per-component variance ratio

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """Map a (scaled) target/prediction back to raw return units."""
        y_scaled = np.asarray(y_scaled).reshape(-1, 1)
        return self.y_scaler.inverse_transform(y_scaled).ravel()

def build_feature_frame(
    prices: pd.Series,
    returns: pd.Series,
    cfg: Config,
    use_set_a: bool = True,
    use_set_b: bool = True,
    use_set_c: bool = True,
    use_set_m: bool = True,
    macro_df: pd.DataFrame | None = None,
    extra_signals: pd.DataFrame | None = None,
    holding_prices: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Assemble the aligned (X, y) frame before scaling / splitting."""
    feats = make_features(
        prices, returns,
        holding_prices=holding_prices,
        roll_windows=cfg.roll_windows,
        past_window=cfg.past_window,
    )

    # Build position rules from the requested signal sets
    sig_kwargs: dict[str, pd.DataFrame] = {}
    if use_set_a:
        sig_kwargs = {**sig_kwargs, **signal_set_a(prices, returns)}
    if use_set_b:
        sig_kwargs = {**sig_kwargs, **signal_set_b(prices, returns)}
    if use_set_c:
        sig_kwargs = {**sig_kwargs, **signal_set_c(prices, returns)}
    if use_set_m and macro_df is not None:
        sig_kwargs = {**sig_kwargs, **signal_set_macro(macro_df, prices.index)}


    merged_signals = pd.DataFrame(sig_kwargs, index=prices.index).add_prefix("pos_raw_")

    X = pd.concat([feats, merged_signals], axis=1)
    # k-day compounded forward return: (P_{t+k} - P_t) / P_t
    y = prices.pct_change(cfg.horizon).shift(-cfg.horizon).rename("y")

    frame = pd.concat([X, y], axis=1).dropna()
    return frame[X.columns], frame["y"]


def make_splits(
    prices: pd.Series,
    returns: pd.Series,
    cfg: 'Config',
    use_set_a: bool = True,
    use_set_b: bool = True,
    use_set_c: bool = True,
    use_set_m: bool = True,
    extra_signals: pd.DataFrame | None = None,
    holding_prices: pd.DataFrame | None = None,
    macro_df: pd.DataFrame | None = None,  # NEW ARGUMENT
    pca_n_components: int = 20,
    pca_variance_threshold: float = 0.95,
    pca_all_features: bool = False,
) -> 'SplitData':
    
    # 1. Build base technical and holding features
    # FIX: Explicitly name all arguments to prevent positional misalignments
    X, y = build_feature_frame(
        prices=prices, 
        returns=returns, 
        cfg=cfg, 
        use_set_a=use_set_a, 
        use_set_b=use_set_b, 
        use_set_c=use_set_c, 
        extra_signals=extra_signals, 
        holding_prices=holding_prices,
        macro_df=macro_df if use_set_m else None,
    )

    # 2. NEW: Safely integrate macro features preventing look-ahead bias
    if macro_df is not None:
        # Strip timezones and ensure strict datetime indices
        X.index = pd.to_datetime(X.index).tz_localize(None).normalize()
        macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None).normalize()
        
        X = X.sort_index()
        macro_df = macro_df.sort_index()
        
        # Map weekly macro data onto daily pipeline dates
        X_merged = pd.merge_asof(
            X, 
            macro_df, 
            left_index=True, 
            right_index=True, 
            direction='backward'
        )
        
        # Drop rows where macro data isn't available yet, and re-align y
        X = X_merged.dropna()
        y = y.loc[X.index]
        print(f"  Merged macro signals. Final dataset size: {len(X)} rows.")

    # 3. Non-overlapping subsampling for horizon > 1
    if cfg.horizon > 1:
        X = X.iloc[::cfg.horizon]
        y = y.iloc[::cfg.horizon]
        print(f"  Subsampled every {cfg.horizon} days → {len(X)} non-overlapping observations")

    n = len(X)
    i_tr = int(n * cfg.train_frac)
    i_va = int(n * (cfg.train_frac + cfg.val_frac))

    Xtr, Xva, Xte = X.iloc[:i_tr], X.iloc[i_tr:i_va], X.iloc[i_va:]
    ytr, yva, yte = y.iloc[:i_tr], y.iloc[i_tr:i_va], y.iloc[i_va:]

    # ── Column groups ────────────────────────────────────────────────────────
    bench_pfx = str(prices.name or "CHDVD.SW").replace(".", "_") + "_"
    bench_cols = [c for c in X.columns if c.startswith(bench_pfx) or c.startswith("ret_lag_")]
    pos_cols   = [c for c in X.columns if c.startswith("pos_")]
    macro_cols = [c for c in X.columns if c.startswith("xs_macro_")] # Isolate macros
    
    hold_cols  = [c for c in X.columns if c not in bench_cols and c not in pos_cols and c not in macro_cols]

    if pca_all_features:
        pca_cols = list(X.columns)
        bp_cols  = []
    else:
        pca_cols = hold_cols
        # Bypass PCA for bench, pos, AND macro columns
        bp_cols  = bench_cols + pos_cols + macro_cols

    # ── Scalers (fit on train only) ──────────────────────────────────────────
    y_scaler = StandardScaler().fit(ytr.values.reshape(-1, 1))

    # ── PCA pipeline ─────────────────────────────────────────────────────────
    hold_sc = hold_pca = None
    pca_expvar: np.ndarray | None = None
    n_pca = 0

    if pca_cols and pca_n_components > 0:
        max_comp = min(len(pca_cols), i_tr - 1)
        hold_sc = StandardScaler().fit(Xtr[pca_cols].fillna(0).values)
        _pca_tmp = PCA(n_components=max_comp).fit(hold_sc.transform(Xtr[pca_cols].fillna(0).values))
        cumvar = np.cumsum(_pca_tmp.explained_variance_ratio_)
        n_thresh = int(np.searchsorted(cumvar, min(pca_variance_threshold, 1.0)) + 1)
        n_pca = min(pca_n_components, n_thresh, max_comp)
        hold_pca = PCA(n_components=n_pca).fit(hold_sc.transform(Xtr[pca_cols].fillna(0).values))
        pca_expvar = hold_pca.explained_variance_ratio_

    if bp_cols:
        x_scaler = StandardScaler().fit(Xtr[bp_cols].fillna(0).values)
    else:
        x_scaler = hold_sc or StandardScaler()

    # ── Transform helpers ────────────────────────────────────────────────────
    def sx(df: pd.DataFrame) -> np.ndarray:
        parts: list[np.ndarray] = []
        if bp_cols:
            parts.append(x_scaler.transform(df[bp_cols].fillna(0).values))
        if hold_pca is not None:
            parts.append(hold_pca.transform(hold_sc.transform(df[pca_cols].fillna(0).values)))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def sy(d: pd.Series) -> np.ndarray:
        return y_scaler.transform(d.values.reshape(-1, 1)).ravel().astype(np.float32)

    pca_names = [f"pca_{i}" for i in range(n_pca)]
    feat_names = bp_cols + pca_names

    return SplitData(
        X_train=sx(Xtr), y_train=sy(ytr),
        X_val=sx(Xva), y_val=sy(yva),
        X_test=sx(Xte), y_test=sy(yte),
        dates_train=Xtr.index, dates_val=Xva.index, dates_test=Xte.index,
        feature_names=feat_names,
        x_scaler=x_scaler, y_scaler=y_scaler,
        hold_scaler=hold_sc, hold_pca=hold_pca,
        pca_explained_var=pca_expvar,
    )
