"""
Unit tests for Event Pipeline and Dataset.
"""

import numpy as np
import torch
import pytest

from src.utils.event_pipeline import (
    EventStreamPipeline,
    EventSequence,
    EventDataset,
    EVENT_TYPES,
    NUM_EVENT_TYPES,
)


def test_event_vocabulary():
    assert NUM_EVENT_TYPES == 20
    assert len(EVENT_TYPES) == 20
    assert 0 in EVENT_TYPES
    assert 19 in EVENT_TYPES


def test_event_detection_synthetic():
    pipeline = EventStreamPipeline(sensitivity_threshold=0.02)
    lob_data = np.zeros((100, 144), dtype=np.float32)
    # Simulate price jumps
    lob_data[10:, 0] = 1.0  # Ask up
    lob_data[20:, 2] = -1.0  # Bid down
    lob_data[30:, 1] = 5.0  # Ask vol surge
    
    events = pipeline.detect_lob_events(lob_data)
    assert len(events) > 0
    assert isinstance(events, EventSequence)
    assert events.event_types.ndim == 1
    assert events.timestamps.ndim == 1
    assert events.inter_arrival_times.ndim == 1
    assert events.features.shape == (len(events), 144)


def test_event_dataset_windowing():
    n_events = 200
    types = np.random.randint(0, NUM_EVENT_TYPES, size=n_events)
    dts = np.random.exponential(0.01, size=n_events).astype(np.float32)
    ts = np.cumsum(dts)
    feats = np.random.randn(n_events, 144).astype(np.float32)
    labels = np.random.randint(0, 3, size=n_events)

    seq = EventSequence(types, ts, dts, feats, labels)
    dataset = EventDataset(seq, seq_len=64, stride=16)

    assert len(dataset) > 0
    sample = dataset[0]
    assert sample["event_types"].shape == (64,)
    assert sample["inter_arrival_times"].shape == (64,)
    assert sample["timestamps"].shape == (64,)
    assert sample["features"].shape == (64, 144)
    assert "target_label" in sample


if __name__ == "__main__":
    test_event_vocabulary()
    test_event_detection_synthetic()
    test_event_dataset_windowing()
    print("All Event Pipeline tests passed!")
