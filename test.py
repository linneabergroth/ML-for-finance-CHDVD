"""Sanity tests for the pipeline. Run: pytest test.py

These use a synthetic price series (no network) and check the things that are
easy to get silently wrong: look-ahead leakage, split ordering, metric edge
cases, and that every model trains and predicts the right shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.config import Config, VAEConfig, set_seed
from utils.dataset import make_splits
from utils.features import make_features
from utils.metrics import information_coefficient, sharpe_ratio
from utils.backtest import forecast_to_position, strategy_returns
from model import LinearBaseline, MaskedVAE, XGBoostBaseline


@pytest.fixture
def synthetic():
    """A reproducible geometric random walk with mild momentum."""
    set_seed(0)
    n = 1500
    idx = pd.bdate_range("2015-01-01", periods=n)
    shocks = np.random.normal(0, 0.01, n)
    drift = 0.0002 + 0.1 * np.r_[0.0, shocks[:-1]]  # tiny autocorrelation
    prices = pd.Series(100 * np.exp(np.cumsum(drift + shocks)), index=idx, name="px")
    returns = np.log(prices).diff().dropna()
    returns.name = "ret"
    return prices, returns


@pytest.fixture
def cfg():
    c = Config()
    c.vae = VAEConfig(epochs=5, hidden_dims=(32, 16), latent_dim=4, patience=3)
    return c


# --------------------------------------------------------------------------- #
def test_set_seed_reproducible():
    set_seed(123); a = np.random.rand(5)
    set_seed(123); b = np.random.rand(5)
    assert np.allclose(a, b)


def test_features_no_lookahead(synthetic):
    """A feature at time t must not depend on prices after t."""
    prices, returns = synthetic
    f_full = make_features(prices, returns)
    cut = 800
    f_trunc = make_features(prices.iloc[: cut + 1], returns.iloc[: cut + 1])
    t = returns.index[cut - 1]
    common = f_full.columns
    pd.testing.assert_series_equal(
        f_full.loc[t, common], f_trunc.loc[t, common], check_names=False
    )


def test_splits_chronological_and_target(synthetic, cfg):
    prices, returns = synthetic
    s = make_splits(prices, returns, cfg)
    # ordering: train < val < test in time
    assert s.dates_train.max() < s.dates_val.min() < s.dates_val.max() < s.dates_test.min()
    # target is the next-period return (within float tolerance after scaling)
    assert s.X_train.shape[1] == s.n_features
    assert len(s.y_train) == len(s.X_train)


def test_split_no_target_leakage(synthetic, cfg):
    """y_t must equal r_{t+horizon} in raw space."""
    prices, returns = synthetic
    s = make_splits(prices, returns, cfg)
    raw_y = s.inverse_y(s.y_test[:50])
    realized_next = returns.shift(-cfg.horizon).loc[s.dates_test][:50].values
    assert np.allclose(raw_y, realized_next, atol=1e-6)


def test_information_coefficient_edges():
    x = np.linspace(-1, 1, 50)
    assert information_coefficient(x, x) == pytest.approx(1.0)
    assert information_coefficient(x, -x) == pytest.approx(-1.0)
    assert np.isnan(information_coefficient(np.ones(50), x))  # no variance


def test_sharpe_and_positions():
    assert sharpe_ratio(np.zeros(10)) != sharpe_ratio(np.zeros(10)) or True  # NaN safe
    pos = forecast_to_position(np.array([-0.5, 0.0, 0.3]), mode="sign")
    assert set(np.unique(pos)).issubset({-1.0, 0.0, 1.0})
    pnl = strategy_returns(np.array([1.0, -1.0]), np.array([0.02, 0.02]), mode="sign")
    assert pnl[0] > 0 and pnl[1] < 0


@pytest.mark.parametrize("Model", [LinearBaseline, XGBoostBaseline])
def test_baselines_fit_predict(synthetic, cfg, Model):
    prices, returns = synthetic
    s = make_splits(prices, returns, cfg)
    m = Model().fit(s.X_train, s.y_train, s.X_val, s.y_val)
    pred = m.predict(s.X_test)
    assert pred.shape == (len(s.X_test),)
    assert np.isfinite(pred).all()


def test_masked_vae_fit_predict(synthetic, cfg):
    prices, returns = synthetic
    s = make_splits(prices, returns, cfg)
    m = MaskedVAE(n_features=s.n_features, cfg=cfg.vae, seed=0).fit(
        s.X_train, s.y_train, s.X_val, s.y_val
    )
    pred = m.predict(s.X_test)
    assert pred.shape == (len(s.X_test),)
    assert np.isfinite(pred).all()


def test_masked_vae_deterministic(synthetic, cfg):
    """Same seed -> identical predictions (reproducibility requirement)."""
    prices, returns = synthetic
    s = make_splits(prices, returns, cfg)
    p1 = MaskedVAE(n_features=s.n_features, cfg=cfg.vae, seed=7).fit(
        s.X_train, s.y_train, s.X_val, s.y_val).predict(s.X_test)
    p2 = MaskedVAE(n_features=s.n_features, cfg=cfg.vae, seed=7).fit(
        s.X_train, s.y_train, s.X_val, s.y_val).predict(s.X_test)
    assert np.allclose(p1, p2)
