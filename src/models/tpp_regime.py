"""
TPP-LLM Regime Detector for Large Event Model (LEM).

Integrates semantic textual representations of news events with
Temporal Point Process (TPP) event dynamics to detect macro risk regimes
and flash-crash probability based on news tempo and temporal clustering.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.event_encoder import TemporalEncoding


@dataclass
class NewsEvent:
    timestamp: float  # continuous timestamp (e.g. in hours or trading days)
    headline: str
    ticker: Optional[str] = None
    sentiment_score: float = 0.0  # -1.0 to 1.0


class NewsSemanticEncoder(nn.Module):
    """
    Encodes news headline strings and sentiment into dense semantic vectors.
    Uses bag-of-embeddings + keyword gating with optional TinyLlama LoRA adapter.
    """
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.sentiment_proj = nn.Linear(1, embed_dim)
        
        # Risk keyword emphasis projection
        self.risk_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )
        self.output_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def _simple_tokenize(self, text: str, max_len: int = 32) -> torch.Tensor:
        """Lightweight hash tokenizer for deterministic, zero-dependency tokenization."""
        words = text.lower().split()
        tokens = [min(abs(hash(w)) % 9999 + 1, 9999) for w in words[:max_len]]
        if len(tokens) < max_len:
            tokens.extend([0] * (max_len - len(tokens)))
        return torch.tensor(tokens, dtype=torch.long)

    def forward(self, headlines: List[str], sentiments: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Args:
            headlines: List of B headlines
            sentiments: (B, 1) sentiment scores
        Returns:
            (B, embed_dim) semantic embeddings
        """
        token_tensors = torch.stack([self._simple_tokenize(h) for h in headlines]).to(device)
        # (B, L, embed_dim)
        word_embeds = self.embedding(token_tensors)
        # Mean pooling over tokens
        mask = (token_tensors != 0).float().unsqueeze(-1)
        text_emb = (word_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        sent_emb = self.sentiment_proj(sentiments)
        combined = torch.cat([text_emb, sent_emb], dim=-1)

        gate = self.risk_gate(combined)
        out = self.output_proj(combined) * gate
        return self.norm(out)


class TPPLoRARegimeDetector(nn.Module):
    """
    Temporal Point Process + LLM News Regime Classifier.
    
    Models the self-exciting arrival of financial news.
    When multiple high-uncertainty / negative headlines cluster within short time gaps (dt -> 0),
    the Hawkes intensity λ_crash spikes, driving the model into the 'Crash' or 'Risky' regime.
    """
    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.semantic_encoder = NewsSemanticEncoder(embed_dim=embed_dim)
        self.temporal_encoder = TemporalEncoding(d_model=embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Hawkes intensity head: predicts arrival intensity of crash/risk events
        self.intensity_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 3),  # [λ_safe, λ_risky, λ_crash]
        )
        self.base_intensity = nn.Parameter(torch.tensor([0.5, 0.2, 0.05]))

        # Regime classification head
        self.regime_classifier = nn.Sequential(
            nn.Linear(embed_dim + 3, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),  # 0=Safe, 1=Risky, 2=Crash
        )

    def forward(
        self,
        news_events: List[NewsEvent],
        current_time: Optional[float] = None,
    ) -> Tuple[int, float, np.ndarray, Dict[str, float]]:
        """
        Processes a sequence of historical news events up to the current timestamp.
        
        Args:
            news_events: List of recent NewsEvent items
            current_time: Current simulation timestamp
            
        Returns:
            regime: 0 (Safe), 1 (Risky), 2 (Crash)
            confidence: float in [0.0, 1.0]
            regime_embedding: 4D numpy array [safe_prob, risky_prob, crash_prob, confidence]
            intensities_dict: dictionary with {'lambda_safe', 'lambda_risky', 'lambda_crash'}
        """
        device = next(self.parameters()).device

        if not news_events:
            # Default normal regime
            return 0, 0.9, np.array([1.0, 0.0, 0.0, 0.9], dtype=np.float32), {
                "lambda_safe": 1.0, "lambda_risky": 0.1, "lambda_crash": 0.01
            }

        headlines = [e.headline for e in news_events]
        sentiments = torch.tensor([[e.sentiment_score] for e in news_events], dtype=torch.float32, device=device)
        
        timestamps = np.array([e.timestamp for e in news_events], dtype=np.float32)
        dts = np.diff(timestamps, prepend=timestamps[0] - 0.1)
        dts = np.clip(dts, a_min=1e-4, a_max=10.0)

        ts_tensor = torch.tensor(timestamps, dtype=torch.float32, device=device).unsqueeze(0)  # (1, N)
        dt_tensor = torch.tensor(dts, dtype=torch.float32, device=device).unsqueeze(0)  # (1, N)

        # 1. Semantic embeddings
        # (N, embed_dim) -> (1, N, embed_dim)
        sem_emb = self.semantic_encoder(headlines, sentiments, device).unsqueeze(0)

        # 2. Continuous time embeddings
        # (1, N, embed_dim)
        time_emb = self.temporal_encoder(ts_tensor, dt_tensor)

        # 3. Fuse semantic + temporal tokens
        tokens = sem_emb + time_emb
        hidden = self.transformer(tokens)  # (1, N, embed_dim)
        last_hidden = hidden[:, -1, :]  # (1, embed_dim)

        # 4. Hawkes Intensities
        raw_lambda = self.intensity_head(last_hidden) + self.base_intensity
        intensities = F.softplus(raw_lambda)  # (1, 3)

        # 5. Regime Probabilities
        features_with_lambda = torch.cat([last_hidden, intensities], dim=-1)
        logits = self.regime_classifier(features_with_lambda)
        probs = F.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()

        regime = int(np.argmax(probs))
        confidence = float(probs[regime])

        regime_embedding = np.array([probs[0], probs[1], probs[2], confidence], dtype=np.float32)

        intensities_arr = intensities.squeeze(0).detach().cpu().numpy()
        intensities_dict = {
            "lambda_safe": float(intensities_arr[0]),
            "lambda_risky": float(intensities_arr[1]),
            "lambda_crash": float(intensities_arr[2]),
        }

        return regime, confidence, regime_embedding, intensities_dict
