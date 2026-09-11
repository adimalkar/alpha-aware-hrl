"""Guards against the silent-degeneracy failures found in the audit."""

import warnings

import numpy as np
import pytest

from src.envs.historical_lob_env import HistoricalLOBEnv


class FakeLoader:
    def __init__(self, n=300, d=144, seed=0):
        rng = np.random.default_rng(seed)
        self.train_data = rng.normal(0, 1, size=(n, d)).astype(np.float32)
        self.train_labels = rng.integers(0, 3, size=n).astype(np.int64)
        self.test_data, self.test_labels = self.train_data, self.train_labels

    def load(self, split):
        return self.train_data, self.train_labels


def prices(n=300, seed=1):
    rng = np.random.default_rng(seed)
    return 100.0 * np.cumprod(1.0 + rng.normal(0, 1e-3, size=n))


def test_constant_regime_warns(tmp_path):
    """D2: every committed regime artefact was a single repeated value."""
    path = tmp_path / "regimes.npy"
    np.save(path, np.stack([np.full(300, 2.0), np.full(300, 0.43)], axis=1))
    with pytest.warns(RuntimeWarning, match="is constant"):
        HistoricalLOBEnv(
            FakeLoader(), split="train", episode_length=100,
            prices=prices(), regime_path=str(path),
        )


def test_varying_regime_does_not_warn(tmp_path):
    path = tmp_path / "regimes.npy"
    rng = np.random.default_rng(0)
    np.save(path, np.stack([rng.integers(0, 3, 300).astype(float), rng.random(300)], axis=1))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        HistoricalLOBEnv(
            FakeLoader(), split="train", episode_length=100,
            prices=prices(), regime_path=str(path),
        )


def test_fi2010_loader_warns_on_raw_price_levels(tmp_path):
    """R2: crypto snapshots loaded as FI-2010 without complaint."""
    import pandas as pd
    from src.utils.data_loader import FI2010DataLoader

    d = tmp_path / "FI2010"
    d.mkdir()
    rng = np.random.default_rng(0)
    feats = rng.normal(70000, 100, size=(50, 144))  # raw price levels
    df = pd.DataFrame(feats)
    for k in range(5):
        df[144 + k] = 2
    df.to_csv(d / "FI2010_train.csv")

    with pytest.warns(RuntimeWarning, match="looks like raw price levels"):
        FI2010DataLoader(data_dir=str(d)).load("train")


def test_hierarchical_wrapper_is_deprecated():
    """D1: the no-gradient wrapper must announce itself."""
    import gymnasium as gym
    from src.envs.hierarchical_wrapper import HierarchicalEnvWrapper

    class Dummy:
        def get_output_dim(self):
            return 8

    env = HistoricalLOBEnv(FakeLoader(), split="train", episode_length=100, prices=prices())
    with pytest.warns(DeprecationWarning, match="never receive a gradient"):
        HierarchicalEnvWrapper(env, mamba_extractor=Dummy(), llm_analyst=None, device="cpu")
