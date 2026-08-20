"""
Unit tests for TPP-LLM Regime Detector and EventAwareRegimeAnalyst.
"""

import torch
import numpy as np
import pytest

from src.models.tpp_regime import NewsEvent, NewsSemanticEncoder, TPPLoRARegimeDetector
from src.agents.llm_analyst import EventAwareRegimeAnalyst, RegimeSignal


def test_news_semantic_encoder():
    encoder = NewsSemanticEncoder(embed_dim=128)
    headlines = [
        "Fed raises interest rates by 50 basis points amidst inflation surge",
        "Tech earnings beat expectations as AI demand accelerates",
    ]
    sentiments = torch.tensor([[-0.8], [0.6]], dtype=torch.float32)
    device = torch.device("cpu")

    out = encoder(headlines, sentiments, device)
    assert out.shape == (2, 128)
    assert not torch.isnan(out).any()


def test_tpp_regime_detector():
    detector = TPPLoRARegimeDetector(embed_dim=128)
    events = [
        NewsEvent(timestamp=0.0, headline="Market opens flat with low volume", sentiment_score=0.1),
        NewsEvent(timestamp=0.5, headline="Rumors of major hedge fund liquidation circulate", sentiment_score=-0.7),
        NewsEvent(timestamp=0.55, headline="Emergency halt called on S&P futures after flash dump", sentiment_score=-0.95),
    ]

    regime, conf, emb, intensities = detector(events, current_time=0.6)
    assert regime in [0, 1, 2]
    assert 0.0 <= conf <= 1.0
    assert emb.shape == (4,)
    assert "lambda_crash" in intensities
    assert "lambda_safe" in intensities
    assert "lambda_risky" in intensities


def test_event_aware_regime_analyst():
    analyst = EventAwareRegimeAnalyst(device="cpu")
    events = [
        NewsEvent(timestamp=1.0, headline="Global markets rally as inflation cools", sentiment_score=0.8),
    ]
    signal, intensities = analyst.analyze_events(events)
    assert isinstance(signal, RegimeSignal)
    assert signal.regime in [0, 1, 2]
    assert len(signal.reasoning) > 0

    emb = analyst.get_regime_embedding(signal)
    assert emb.shape == (4,)


if __name__ == "__main__":
    test_news_semantic_encoder()
    test_tpp_regime_detector()
    test_event_aware_regime_analyst()
    print("All TPP Regime tests passed!")
