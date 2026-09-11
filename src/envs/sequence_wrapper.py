"""
Observation wrapper that emits a RAW sequence window for the policy to encode.

Why this replaces the encode-inside-the-wrapper design
=====================================================
`HierarchicalEnvWrapper` ran the feature extractor itself:

    with torch.no_grad():
        self.mamba_extractor.eval()
        extractor_out = self.mamba_extractor(lob_tensor)
    mamba_feats = final_state.cpu().numpy().squeeze()

Because the encoder lived in the *environment* and its output was detached to
numpy, no gradient from the RL loss could ever reach it. TQC optimised only the
MLP downstream of a frozen, randomly initialised encoder. The saved
`best_model.zip` contains just `policy.pth`, so the encoder was not even
persisted -- it was re-randomised on every load.

This wrapper instead hands the policy a raw window and lets the policy's
features extractor do the encoding, which is where SB3 expects it and where
autograd can reach it.
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class SequenceWindowWrapper(gym.Wrapper):
    """
    Buffers the last `seq_len` observations and emits them as a Dict space.

    Observation:
        window : (seq_len, market_dim) recent market feature vectors
        agent  : (n_agent,)            current inventory / cash / progress
        regime : (4,)                  macro regime one-hot + confidence

    The market and agent parts are split so the sequence encoder sees only
    market history, while inventory state is fed directly to the policy head.
    """

    N_REGIME = 4

    def __init__(
        self,
        env: gym.Env,
        seq_len: int = 50,
        n_agent_features: int = 3,
        regime_provider: Optional[Any] = None,
    ):
        super().__init__(env)
        self.seq_len = int(seq_len)
        self.n_agent = int(n_agent_features)
        self.regime_provider = regime_provider

        base_dim = int(env.observation_space.shape[0])
        if base_dim <= self.n_agent:
            raise ValueError(
                f"Base observation has {base_dim} dims but {self.n_agent} were "
                "declared as agent-state features."
            )
        self.market_dim = base_dim - self.n_agent

        self.observation_space = gym.spaces.Dict(
            {
                "window": gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.seq_len, self.market_dim), dtype=np.float32,
                ),
                "agent": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.n_agent,), dtype=np.float32
                ),
                "regime": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.N_REGIME,), dtype=np.float32
                ),
            }
        )
        self._buffer: list = []
        self._regime = np.zeros(self.N_REGIME, dtype=np.float32)

    def _split(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        obs = np.asarray(obs, dtype=np.float32)
        return obs[: self.market_dim], obs[self.market_dim :]

    def _regime_vector(self, info: Dict[str, Any]) -> np.ndarray:
        pr = info.get("precomputed_regime")
        vec = np.zeros(self.N_REGIME, dtype=np.float32)
        if pr is None:
            vec[0] = 1.0  # default: safe, full confidence
            vec[3] = 1.0
            return vec
        regime = int(pr.get("regime", 0))
        vec[min(max(regime, 0), 2)] = 1.0
        vec[3] = float(pr.get("confidence", 1.0))
        return vec

    def _observation(self) -> Dict[str, np.ndarray]:
        return {
            "window": np.asarray(self._buffer, dtype=np.float32),
            "agent": self._agent.astype(np.float32),
            "regime": self._regime,
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        market, self._agent = self._split(obs)
        self._buffer = [market.copy() for _ in range(self.seq_len)]
        self._regime = self._regime_vector(info)
        return self._observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        market, self._agent = self._split(obs)
        self._buffer.append(market)
        if len(self._buffer) > self.seq_len:
            self._buffer.pop(0)
        self._regime = self._regime_vector(info)
        return self._observation(), reward, terminated, truncated, info
