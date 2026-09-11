"""
D1 regression guard: the event encoder must receive gradients from the RL loss.

The original architecture ran the encoder inside the environment under
torch.no_grad(), so it was a frozen random projection and was never saved with
the policy. These tests fail if that arrangement comes back.
"""

import numpy as np
import pytest
import torch

from src.agents.lem_extractor import (
    GRUWindowExtractor,
    LEMFeaturesExtractor,
    MLPWindowExtractor,
)
from src.envs.historical_lob_env import HistoricalLOBEnv
from src.envs.sequence_wrapper import SequenceWindowWrapper


class FakeLoader:
    def __init__(self, n=400, d=144, seed=0):
        rng = np.random.default_rng(seed)
        self.train_data = rng.normal(0, 1, size=(n, d)).astype(np.float32)
        self.train_labels = rng.integers(0, 3, size=n).astype(np.int64)
        self.test_data, self.test_labels = self.train_data, self.train_labels

    def load(self, split):
        return self.train_data, self.train_labels


def make_wrapped(seq_len=16):
    rng = np.random.default_rng(1)
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0, 1e-3, size=400))
    env = HistoricalLOBEnv(FakeLoader(), split="train", episode_length=60, prices=prices)
    return SequenceWindowWrapper(env, seq_len=seq_len)


def stepped_obs(env, n=20):
    """Advance past reset so the window holds distinct rows, not seq_len copies."""
    obs, _ = env.reset(seed=0)
    for _ in range(n):
        obs, _, term, _, _ = env.step(np.array([0.1], dtype=np.float32))
        if term:
            obs, _ = env.reset(seed=0)
    return obs


def test_wrapper_emits_dict_window():
    env = make_wrapped(seq_len=16)
    obs, _ = env.reset(seed=0)
    assert set(obs.keys()) == {"window", "agent", "regime"}
    assert obs["window"].shape == (16, 144)
    assert obs["agent"].shape == (3,)
    assert obs["regime"].shape == (4,)


def test_window_slides_with_time():
    env = make_wrapped(seq_len=8)
    obs0, _ = env.reset(seed=0)
    first = obs0["window"].copy()
    obs1, _, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
    # oldest row dropped, newest appended
    assert not np.array_equal(first[-1], obs1["window"][-1])
    assert np.array_equal(first[1:], obs1["window"][:-1])


@pytest.mark.parametrize(
    "cls", [LEMFeaturesExtractor, MLPWindowExtractor, GRUWindowExtractor]
)
def test_extractor_receives_gradient(cls):
    """Every trainable parameter must get a non-None, non-zero gradient."""
    env = make_wrapped(seq_len=16)
    extractor = cls(env.observation_space)
    obs = stepped_obs(env)

    batch = {
        k: torch.as_tensor(np.stack([v, v]), dtype=torch.float32) for k, v in obs.items()
    }
    out = extractor(batch)
    assert out.shape == (2, extractor.features_dim)

    # NOT out.sum(): the encoder terminates in a LayerNorm, and the sum of a
    # LayerNorm output across the normalised dimension is identically constant,
    # so .sum() has an exactly-zero gradient everywhere upstream. A real policy
    # head applies learned weights, so the test must too.
    torch.manual_seed(0)
    head = torch.nn.Linear(extractor.features_dim, 1)
    head(out).sum().backward()

    # intensity_head / base_intensity / type_classifier belong to the Hawkes
    # PRETRAINING objective, not the pooled representation the policy consumes,
    # so they are legitimately inert here. Everything on the representation
    # path must be reached.
    PRETRAIN_ONLY = ("encoder.intensity_head", "encoder.base_intensity", "encoder.type_classifier")
    named = [(n, p) for n, p in extractor.named_parameters() if p.requires_grad]
    assert named, "extractor has no trainable parameters"

    path_params = [(n, p) for n, p in named if not n.startswith(PRETRAIN_ONLY)]
    missing = [n for n, p in path_params if p.grad is None or p.grad.abs().sum() == 0]
    assert not missing, f"parameters on the representation path got no gradient: {missing}"


def test_lem_encoder_specifically_gets_gradient():
    """Narrow the claim to the Hawkes encoder itself, not just the head."""
    env = make_wrapped(seq_len=16)
    extractor = LEMFeaturesExtractor(env.observation_space)
    obs = stepped_obs(env)
    batch = {k: torch.as_tensor(v[None], dtype=torch.float32) for k, v in obs.items()}

    torch.manual_seed(0)
    head = torch.nn.Linear(extractor.features_dim, 1)
    head(extractor(batch)).sum().backward()

    encoder_grads = {
        name: p.grad for name, p in extractor.encoder.named_parameters() if p.requires_grad
    }
    assert encoder_grads, "encoder exposes no trainable parameters"
    ungrad = [n for n, g in encoder_grads.items() if g is None or g.abs().sum() == 0]
    # The intensity head is not on the pooled-representation path, so it is
    # legitimately unused here; everything else must be reached.
    unexpected = [n for n in ungrad if not n.startswith(("intensity_head", "base_intensity", "type_classifier"))]
    assert not unexpected, f"encoder params received no gradient: {unexpected}"


def test_freeze_flag_is_explicit_and_reproduces_old_behaviour():
    env = make_wrapped(seq_len=16)
    extractor = LEMFeaturesExtractor(env.observation_space, freeze_encoder=True)
    assert extractor.frozen
    assert all(not p.requires_grad for p in extractor.encoder.parameters())


def test_pretrained_shape_mismatch_is_rejected(tmp_path):
    ckpt = tmp_path / "bad.pt"
    torch.save(
        {"state_dict": {}, "config": {"feature_dim": 999, "d_model": 128}}, ckpt
    )
    env = make_wrapped(seq_len=16)
    with pytest.raises(ValueError, match="was trained with feature_dim"):
        LEMFeaturesExtractor(env.observation_space, pretrained_path=str(ckpt))
