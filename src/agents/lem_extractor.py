"""
SB3 features extractors that actually receive gradients.

`LEMFeaturesExtractor` puts the Transformer Hawkes encoder inside the policy,
which is the fix for audit finding D1: previously the encoder ran in the
environment under `torch.no_grad()`, so it was a frozen random projection for
the entire run and was never saved with the model.

Optionally loads a checkpoint produced by scripts/pretrain_event_encoder.py,
giving the pretrain-then-finetune path: the Hawkes likelihood teaches arrival
dynamics, then the RL objective adapts the representation to the trading task.
"""

from typing import Dict, Optional

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.models.event_encoder import TransformerHawkesEncoder
from src.utils.event_pipeline import NUM_EVENT_TYPES


class _WindowToEvents(nn.Module):
    """
    Converts a raw market window (B, L, D) into the event tensors the THP takes.

    Event types are derived from observable mid-return and spread changes only.
    No label, and nothing from the future, is used anywhere in this mapping.
    """

    MID_RETURN_COL = 41  # LiveMarketDataLoader writes one-step realised return here
    SPREAD_COL = 40

    def __init__(self, move_threshold: float = 0.5, spread_threshold: float = 0.5):
        super().__init__()
        self.move_threshold = move_threshold
        self.spread_threshold = spread_threshold

    def forward(self, window: torch.Tensor):
        B, L, D = window.shape
        device = window.device

        mid_ret = window[:, :, self.MID_RETURN_COL]
        spread = window[:, :, self.SPREAD_COL]
        spread_diff = torch.zeros_like(spread)
        spread_diff[:, 1:] = spread[:, 1:] - spread[:, :-1]

        # 10 = mid_price_up, 11 = mid_price_down, 14 = volatility_spike (flat),
        # 8 = spread_widen, 9 = spread_narrow  (see EVENT_TYPES)
        types = torch.where(
            mid_ret > self.move_threshold,
            torch.full_like(mid_ret, 10, dtype=torch.long),
            torch.where(
                mid_ret < -self.move_threshold,
                torch.full_like(mid_ret, 11, dtype=torch.long),
                torch.full_like(mid_ret, 14, dtype=torch.long),
            ),
        )
        types = torch.where(
            spread_diff > self.spread_threshold, torch.full_like(types, 8),
            torch.where(spread_diff < -self.spread_threshold, torch.full_like(types, 9), types),
        )
        types = types.clamp(0, NUM_EVENT_TYPES - 1)

        # Uniform sampling grid: the wrapper delivers snapshots at a fixed poll
        # interval, so inter-arrival time is constant by construction. This is a
        # real limitation of REST-polled data and is stated rather than hidden
        # behind a synthetic dt.
        dt = torch.full((B, L), 1.0, dtype=torch.float32, device=device)
        timestamps = torch.cumsum(dt, dim=1)
        return types, timestamps, dt


class LEMFeaturesExtractor(BaseFeaturesExtractor):
    """
    Transformer Hawkes encoder over the market window, concatenated with the
    agent-state and regime vectors.

    Args:
        observation_space: Dict space from SequenceWindowWrapper.
        d_model: encoder width.
        pretrained_path: optional checkpoint from pretrain_event_encoder.py.
        freeze_encoder: if True the encoder is frozen (ablation control only --
            this reproduces the old broken behaviour on purpose).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = False,
    ):
        seq_len, market_dim = observation_space["window"].shape
        n_agent = observation_space["agent"].shape[0]
        n_regime = observation_space["regime"].shape[0]
        super().__init__(observation_space, features_dim=d_model + n_agent + n_regime)

        self.encoder = TransformerHawkesEncoder(
            num_event_types=NUM_EVENT_TYPES,
            feature_dim=market_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
        )
        self.to_events = _WindowToEvents()
        self.pretrained_loaded = False

        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            cfg = ckpt.get("config", {})
            if cfg.get("feature_dim") != market_dim or cfg.get("d_model") != d_model:
                raise ValueError(
                    f"Checkpoint {pretrained_path} was trained with feature_dim="
                    f"{cfg.get('feature_dim')}, d_model={cfg.get('d_model')}, but this "
                    f"policy needs feature_dim={market_dim}, d_model={d_model}."
                )
            self.encoder.load_state_dict(ckpt["state_dict"])
            self.pretrained_loaded = True

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.frozen = freeze_encoder

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        window = observations["window"]
        types, timestamps, dt = self.to_events(window)
        _, _, pooled = self.encoder(
            event_types=types,
            timestamps=timestamps,
            inter_arrival_times=dt,
            features=window,
            mask=None,
        )
        return torch.cat([pooled, observations["agent"], observations["regime"]], dim=-1)


class MLPWindowExtractor(BaseFeaturesExtractor):
    """Flatten-and-MLP control arm, for ablations against the THP."""

    def __init__(self, observation_space: gym.spaces.Dict, d_model: int = 128):
        seq_len, market_dim = observation_space["window"].shape
        n_agent = observation_space["agent"].shape[0]
        n_regime = observation_space["regime"].shape[0]
        super().__init__(observation_space, features_dim=d_model + n_agent + n_regime)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * market_dim, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [self.net(observations["window"]), observations["agent"], observations["regime"]],
            dim=-1,
        )


class GRUWindowExtractor(BaseFeaturesExtractor):
    """Recurrent control arm (the 'lstm' backbone, named honestly)."""

    def __init__(self, observation_space: gym.spaces.Dict, d_model: int = 128, n_layers: int = 2):
        seq_len, market_dim = observation_space["window"].shape
        n_agent = observation_space["agent"].shape[0]
        n_regime = observation_space["regime"].shape[0]
        super().__init__(observation_space, features_dim=d_model + n_agent + n_regime)
        self.rnn = nn.GRU(market_dim, d_model, num_layers=n_layers, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        out, _ = self.rnn(observations["window"])
        return torch.cat(
            [self.norm(out[:, -1]), observations["agent"], observations["regime"]], dim=-1
        )
