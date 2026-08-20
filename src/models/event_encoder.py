"""
Large Event Model (LEM) - Transformer Hawkes Process (THP) Event Encoder.

Processes continuous-time event sequences from Limit Order Book order flow
and macroeconomic streams to extract 128-dimensional event-aware state representations.
"""

import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.event_pipeline import NUM_EVENT_TYPES, EventStreamPipeline


class TemporalEncoding(nn.Module):
    """
    Continuous temporal encoding using sinusoidal harmonics and learned projection.
    Maps continuous time intervals / timestamps to d_model-dimensional vectors.
    """
    def __init__(self, d_model: int, max_period: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period
        
        # Linear projection for learnable time-scale transformation
        self.time_proj = nn.Linear(1, d_model)
        self.dt_proj = nn.Linear(1, d_model)
        self.combine = nn.Linear(d_model * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def _sinusoidal_embed(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B, L, 1) or (B, L)
        Returns:
            (B, L, d_model)
        """
        if t.dim() == 2:
            t = t.unsqueeze(-1)
            
        half_dim = self.d_model // 2
        div_term = torch.exp(
            torch.arange(0, half_dim, dtype=torch.float32, device=t.device)
            * -(math.log(self.max_period) / max(1, half_dim - 1))
        )
        # (B, L, half_dim)
        sin_part = torch.sin(t * div_term)
        cos_part = torch.cos(t * div_term)
        emb = torch.cat([sin_part, cos_part], dim=-1)
        if emb.shape[-1] < self.d_model:
            # Handle odd dimensions if any
            emb = F.pad(emb, (0, self.d_model - emb.shape[-1]))
        return emb

    def forward(self, timestamps: torch.Tensor, inter_arrival_times: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: (B, L) continuous timestamps
            inter_arrival_times: (B, L) delta times dt
        Returns:
            (B, L, d_model) combined temporal embedding
        """
        if timestamps.dim() == 2:
            timestamps = timestamps.unsqueeze(-1)
        if inter_arrival_times.dim() == 2:
            inter_arrival_times = inter_arrival_times.unsqueeze(-1)

        t_sin = self._sinusoidal_embed(timestamps)
        t_learned = self.time_proj(timestamps)
        t_emb = t_sin + t_learned

        dt_sin = self._sinusoidal_embed(inter_arrival_times)
        dt_learned = self.dt_proj(inter_arrival_times)
        dt_emb = dt_sin + dt_learned

        combined = self.combine(torch.cat([t_emb, dt_emb], dim=-1))
        return self.layer_norm(F.silu(combined))


class EventEmbedding(nn.Module):
    """
    Fuses discrete event type embeddings, continuous microstructural features,
    and continuous temporal encodings.
    """
    def __init__(
        self,
        num_event_types: int = NUM_EVENT_TYPES,
        feature_dim: int = 144,
        d_model: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.type_embed = nn.Embedding(num_event_types, d_model)
        self.feature_proj = nn.Linear(feature_dim, d_model)
        self.temporal_encoder = TemporalEncoding(d_model)
        
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        event_types: torch.Tensor,
        timestamps: torch.Tensor,
        inter_arrival_times: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            event_types: (B, L) int64
            timestamps: (B, L) float32
            inter_arrival_times: (B, L) float32
            features: (B, L, feature_dim) float32
        Returns:
            (B, L, d_model) fused event tokens
        """
        type_emb = self.type_embed(event_types)
        feat_emb = self.feature_proj(features)
        time_emb = self.temporal_encoder(timestamps, inter_arrival_times)

        fused = self.fusion(torch.cat([type_emb, feat_emb, time_emb], dim=-1))
        return fused


class TransformerHawkesEncoder(nn.Module):
    """
    Transformer Hawkes Process (THP) Core Sequence Encoder.
    Applies multi-head self-attention over event sequence tokens and predicts
    event intensity λ_k(t).
    """
    def __init__(
        self,
        num_event_types: int = NUM_EVENT_TYPES,
        feature_dim: int = 144,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_event_types = num_event_types
        
        self.event_embed = EventEmbedding(
            num_event_types=num_event_types,
            feature_dim=feature_dim,
            d_model=d_model,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Hawkes Intensity Head: predicts conditional intensity for each event type
        self.intensity_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, num_event_types),
        )
        # Base intensity parameters
        self.base_intensity = nn.Parameter(torch.zeros(num_event_types))

        # Event type prediction head for auxiliary self-supervised learning
        self.type_classifier = nn.Linear(d_model, num_event_types)

    def forward(
        self,
        event_types: torch.Tensor,
        timestamps: torch.Tensor,
        inter_arrival_times: torch.Tensor,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            event_types: (B, L) int64
            timestamps: (B, L) float32
            inter_arrival_times: (B, L) float32
            features: (B, L, 144) float32
            mask: (B, L) boolean mask (True for padding)
        Returns:
            hidden_states: (B, L, d_model)
            intensities: (B, L, num_event_types)
            pooled_rep: (B, d_model) representation of sequence end
        """
        # (B, L, d_model)
        tokens = self.event_embed(event_types, timestamps, inter_arrival_times, features)
        
        # Causal mask so events can only attend to past events
        seq_len = tokens.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=tokens.device), diagonal=1).bool()

        hidden = self.transformer(tokens, mask=causal_mask, src_key_padding_mask=mask)
        hidden = self.norm(hidden)

        # Compute Hawkes intensities: λ_k(t) = softplus(W h(t) + μ_k)
        raw_intensity = self.intensity_head(hidden) + self.base_intensity
        intensities = F.softplus(raw_intensity)

        # Pooled representation: last valid event token
        if mask is not None:
            # Select last unmasked token
            lengths = (~mask).sum(dim=-1) - 1
            lengths = torch.clamp(lengths, min=0)
            batch_indices = torch.arange(hidden.size(0), device=hidden.device)
            pooled = hidden[batch_indices, lengths]
        else:
            pooled = hidden[:, -1, :]

        return hidden, intensities, pooled

    def compute_log_likelihood(
        self,
        event_types: torch.Tensor,
        timestamps: torch.Tensor,
        inter_arrival_times: torch.Tensor,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the log-likelihood of the event sequence under the Hawkes model:
        LL = sum_{i} log λ_{k_i}(t_i) - int_0^T sum_k λ_k(s) ds
        """
        hidden, intensities, _ = self.forward(
            event_types, timestamps, inter_arrival_times, features, mask
        )
        
        # 1. Event log-intensity term: log λ_{k_i}(t_i)
        # intensities: (B, L, K), event_types: (B, L)
        event_intensities = intensities.gather(dim=-1, index=event_types.unsqueeze(-1)).squeeze(-1)
        log_term = torch.log(event_intensities + 1e-8)

        # 2. Integral survival term approximation via Riemann sum: sum_k λ_k(t_i) * dt_i
        total_intensity = intensities.sum(dim=-1)  # (B, L)
        survival_term = total_intensity * inter_arrival_times

        ll = log_term - survival_term

        if mask is not None:
            ll = ll.masked_fill(mask, 0.0)
            return ll.sum() / (~mask).sum().clamp(min=1)
        return ll.mean()


class EventFeatureExtractor(nn.Module):
    """
    Drop-in replacement / alternative for MambaFeatureExtractor.
    Complies with the exact interface required by HierarchicalEnvWrapper:
    - `get_output_dim()` -> returns d_model (128)
    - `forward(x)` -> returns (batch, 128)
    
    Supports both:
    1. Direct LOB window sequences `(batch, seq_len, 144)` (converts dynamically)
    2. Structured Event sequences via `forward_events(...)`
    """
    def __init__(
        self,
        input_dim: int = 144,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        sensitivity_threshold: float = 0.03,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        self.encoder = TransformerHawkesEncoder(
            num_event_types=NUM_EVENT_TYPES,
            feature_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.pipeline = EventStreamPipeline(
            sensitivity_threshold=sensitivity_threshold,
            feature_dim=input_dim,
        )
        
        # Classification head for fine-tuning
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),  # down, stable, up
        )

    def get_output_dim(self) -> int:
        """Returns feature vector dimension (128)."""
        return self.d_model

    def _convert_lob_batch_to_events(
        self,
        lob_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dynamically converts raw LOB sliding window tensors (B, L, 144) to event tensors.
        """
        device = lob_batch.device
        B, L, D = lob_batch.shape

        # Approximate event detection in batch vector operations
        # Feature 0: Ask Price 1, Feature 2: Bid Price 1, Feature 1: Ask Vol, Feature 3: Bid Vol
        diffs = lob_batch[:, 1:, :] - lob_batch[:, :-1, :]
        mid_diffs = ((lob_batch[:, 1:, 0] + lob_batch[:, 1:, 2]) - (lob_batch[:, :-1, 0] + lob_batch[:, :-1, 2])) / 2.0
        spread_diffs = (lob_batch[:, 1:, 0] - lob_batch[:, 1:, 2]) - (lob_batch[:, :-1, 0] - lob_batch[:, :-1, 2])

        # Vectorized event assignment:
        # Default mid price moves
        types = torch.where(mid_diffs > 0.01, 10, torch.where(mid_diffs < -0.01, 11, 14))
        # Spread moves
        types = torch.where(spread_diffs > 0.02, 8, torch.where(spread_diffs < -0.02, 9, types))

        # Pad first step to maintain sequence length L
        first_step_types = torch.full((B, 1), 10, dtype=torch.long, device=device)
        event_types = torch.cat([first_step_types, types], dim=1)

        # Continuous delta times
        dts = torch.clamp(0.01 * torch.exp(-torch.abs(mid_diffs) * 3.0), min=1e-4, max=0.1)
        first_step_dt = torch.full((B, 1), 0.01, dtype=torch.float32, device=device)
        inter_arrival_times = torch.cat([first_step_dt, dts], dim=1)

        timestamps = torch.cumsum(inter_arrival_times, dim=1)

        return event_types, timestamps, inter_arrival_times, lob_batch

    def forward(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Standard forward pass for reinforcement learning and feature extraction.
        
        Args:
            x: (batch, seq_len, 144) raw LOB sequence OR dictionary of event tensors
        Returns:
            (batch, 128) state representation
        """
        if isinstance(x, dict):
            event_types = x["event_types"]
            timestamps = x["timestamps"]
            inter_arrival_times = x["inter_arrival_times"]
            features = x["features"]
            mask = x.get("mask")
        else:
            # Raw LOB tensor: convert dynamically
            event_types, timestamps, inter_arrival_times, features = self._convert_lob_batch_to_events(x)
            mask = None

        hidden, _, pooled_rep = self.encoder(
            event_types=event_types,
            timestamps=timestamps,
            inter_arrival_times=inter_arrival_times,
            features=features,
            mask=mask,
        )
        return hidden, pooled_rep

    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts 3-class price movement (down, stable, up)."""
        _, feats = self.forward(x)
        return self.classifier(feats)
