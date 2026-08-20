"""
Unit tests for Transformer Hawkes Process Event Encoder and Feature Extractor.
"""

import torch
import numpy as np
import pytest

from src.models.event_encoder import (
    TemporalEncoding,
    EventEmbedding,
    TransformerHawkesEncoder,
    EventFeatureExtractor,
)


def test_temporal_encoding():
    te = TemporalEncoding(d_model=64)
    ts = torch.linspace(0, 10, steps=20).unsqueeze(0)  # (1, 20)
    dts = torch.full((1, 20), 0.5)

    out = te(ts, dts)
    assert out.shape == (1, 20, 64)
    assert not torch.isnan(out).any()


def test_event_embedding():
    ee = EventEmbedding(num_event_types=20, feature_dim=144, d_model=128)
    types = torch.randint(0, 20, (2, 32))
    ts = torch.cumsum(torch.rand(2, 32), dim=1)
    dts = torch.rand(2, 32) * 0.1
    feats = torch.randn(2, 32, 144)

    emb = ee(types, ts, dts, feats)
    assert emb.shape == (2, 32, 128)
    assert not torch.isnan(emb).any()


def test_transformer_hawkes_encoder():
    thp = TransformerHawkesEncoder(num_event_types=20, feature_dim=144, d_model=128, n_layers=2)
    types = torch.randint(0, 20, (2, 32))
    ts = torch.cumsum(torch.rand(2, 32), dim=1)
    dts = torch.rand(2, 32) * 0.1
    feats = torch.randn(2, 32, 144)

    hidden, intensities, pooled = thp(types, ts, dts, feats)
    assert hidden.shape == (2, 32, 128)
    assert intensities.shape == (2, 32, 20)
    assert pooled.shape == (2, 128)
    assert (intensities >= 0).all()  # Intensities must be non-negative

    ll = thp.compute_log_likelihood(types, ts, dts, feats)
    assert torch.isfinite(ll)


def test_event_feature_extractor_interface():
    extractor = EventFeatureExtractor(input_dim=144, d_model=128)
    assert extractor.get_output_dim() == 128

    # Test with raw LOB tensor (Batch=4, SeqLen=50, Feat=144)
    raw_lob = torch.randn(4, 50, 144)
    hidden, out = extractor(raw_lob)
    assert hidden.shape == (4, 50, 128)
    assert out.shape == (4, 128)


if __name__ == "__main__":
    test_temporal_encoding()
    test_event_embedding()
    test_transformer_hawkes_encoder()
    test_event_feature_extractor_interface()
    print("All Event Encoder tests passed!")
