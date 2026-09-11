"""
Contract tests for HistoricalLOBEnv.

The central property under test is L1 from the audit: the traded price must be
independent of the label. These tests fail loudly if anyone reintroduces a
label-derived price path.
"""

import numpy as np
import pytest

from src.envs.historical_lob_env import (
    HistoricalLOBEnv,
    PriceSourceError,
    N_AGENT_STATE_FEATURES,
)


class FakeLoader:
    """Minimal stand-in for FI2010DataLoader."""

    def __init__(self, n=500, d=144, seed=0):
        rng = np.random.default_rng(seed)
        self.train_data = rng.normal(0, 1, size=(n, d)).astype(np.float32)
        self.train_labels = rng.integers(0, 3, size=n).astype(np.int64)
        self.test_data = self.train_data
        self.test_labels = self.train_labels

    def load(self, split):
        return self.train_data, self.train_labels


def make_prices(n=500, seed=1):
    rng = np.random.default_rng(seed)
    return 100.0 * np.cumprod(1.0 + rng.normal(0, 1e-3, size=n))


def make_env(**kw):
    loader = FakeLoader()
    kw.setdefault("prices", make_prices())
    kw.setdefault("episode_length", 100)
    return HistoricalLOBEnv(loader, split="train", **kw)


# ---------------------------------------------------------------- price source

def test_refuses_to_run_without_a_price_source():
    """L1: the env must not synthesise a price path."""
    with pytest.raises(PriceSourceError, match="will not synthesise one"):
        HistoricalLOBEnv(FakeLoader(), split="train", episode_length=100)


def test_error_message_names_the_leak_and_the_zscore_problem():
    with pytest.raises(PriceSourceError) as e:
        HistoricalLOBEnv(FakeLoader(), split="train", episode_length=100)
    msg = str(e.value)
    assert "FUTURE mid-price direction" in msg
    assert "z-scored per column" in msg


def test_rejects_zscored_column_as_price():
    """A standardised feature column crosses zero; that must be caught."""
    with pytest.raises(PriceSourceError, match="non-positive"):
        HistoricalLOBEnv(FakeLoader(), split="train", episode_length=100, price_column=0)


def test_rejects_misaligned_price_series():
    with pytest.raises(PriceSourceError, match="align 1:1"):
        make_env(prices=make_prices(n=499))


def test_rejects_constant_price_series():
    with pytest.raises(PriceSourceError, match="constant"):
        make_env(prices=np.full(500, 100.0))


def test_rejects_both_price_sources():
    with pytest.raises(PriceSourceError, match="not both"):
        make_env(prices=make_prices(), price_column=0)


# ---------------------------------------------------------------- the leak test

def test_price_path_is_independent_of_labels():
    """
    L1 regression guard. Run the same seed twice with completely different
    labels. If the price or reward stream changes, a label is driving price.
    """
    prices = make_prices()

    def rollout(label_seed):
        loader = FakeLoader()
        rng = np.random.default_rng(label_seed)
        loader.train_labels = rng.integers(0, 3, size=len(loader.train_data)).astype(np.int64)
        env = HistoricalLOBEnv(loader, split="train", episode_length=50, prices=prices)
        env.reset(seed=42)
        seen_prices, rewards = [], []
        for _ in range(30):
            obs, r, term, trunc, info = env.step(np.array([0.5], dtype=np.float32))
            seen_prices.append(info["current_price"])
            rewards.append(r)
            if term:
                break
        return np.array(seen_prices), np.array(rewards)

    p_a, r_a = rollout(label_seed=11)
    p_b, r_b = rollout(label_seed=99)

    assert np.array_equal(p_a, p_b), "price path changed with labels -- leak reintroduced"
    assert np.allclose(r_a, r_b), "reward changed with labels -- leak reintroduced"


def test_observed_price_matches_supplied_series():
    prices = make_prices()
    env = make_env(prices=prices)
    _, info = env.reset(seed=7)
    start = info["idx"]
    for k in range(1, 11):
        _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
        assert info["current_price"] == pytest.approx(prices[start + k])


# ---------------------------------------------------------------- markov state

def test_observation_includes_agent_state():
    """D6: reward depends on inventory, so inventory must be observable."""
    env = make_env()
    assert env.observation_space.shape == (144 + N_AGENT_STATE_FEATURES,)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (147,)


def test_position_weight_is_reflected_in_observation():
    env = make_env()
    env.reset(seed=0)
    obs, _, _, _, _ = env.step(np.array([1.0], dtype=np.float32))
    assert obs[144] == pytest.approx(1.0, abs=1e-2)  # fully long
    obs, _, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
    assert obs[144] == pytest.approx(0.0, abs=1e-6)  # flat


# ---------------------------------------------------------------- reproducibility

def test_reset_seed_is_honoured():
    """The old code called np.random.randint, so seeds did nothing."""
    a = make_env().reset(seed=123)[1]["idx"]
    b = make_env().reset(seed=123)[1]["idx"]
    c = make_env().reset(seed=456)[1]["idx"]
    assert a == b
    assert a != c, "different seeds produced the same episode start"


def test_episode_length_is_not_mutated_by_reset():
    env = make_env(episode_length=100)
    for s in range(5):
        env.reset(seed=s)
    assert env.episode_length == 100


# ---------------------------------------------------------------- mechanics

def test_flat_position_earns_nothing():
    env = make_env()
    env.reset(seed=3)
    for _ in range(20):
        _, r, term, _, info = env.step(np.array([0.0], dtype=np.float32))
        assert r == pytest.approx(0.0, abs=1e-12)
        assert info["portfolio_value"] == pytest.approx(100000.0)
        if term:
            break


def test_fees_are_charged_on_turnover():
    env = make_env(transaction_fee=0.01)
    env.reset(seed=3)
    _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))
    assert info["fee_paid"] > 0
    assert info["portfolio_value"] < 100000.0


def test_long_position_gains_when_price_rises():
    prices = np.linspace(100.0, 120.0, 500)
    env = make_env(prices=prices, transaction_fee=0.0)
    env.reset(seed=3)
    total = 0.0
    for _ in range(30):
        _, r, term, _, _ = env.step(np.array([1.0], dtype=np.float32))
        total += r
        if term:
            break
    assert total > 0


def test_regime_length_mismatch_raises(tmp_path):
    bad = tmp_path / "regimes.npy"
    np.save(bad, np.zeros((10, 2)))
    with pytest.raises(ValueError, match="Refusing to run with misaligned regimes"):
        make_env(regime_path=str(bad))
