"""
Event Data Pipeline for Large Event Model (LEM) integration.

Transforms fixed-interval Limit Order Book (LOB) data (such as FI-2010)
and macro news streams into continuous-time event sequences for
Temporal Point Process (TPP) and Transformer Hawkes Process (THP) models.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# 20-class Market Microstructure & Macro Event Vocabulary
EVENT_TYPES = {
    0: "bid_price_up",
    1: "bid_price_down",
    2: "ask_price_up",
    3: "ask_price_down",
    4: "bid_volume_surge",
    5: "bid_volume_drain",
    6: "ask_volume_surge",
    7: "ask_volume_drain",
    8: "spread_widen",
    9: "spread_narrow",
    10: "mid_price_up",
    11: "mid_price_down",
    12: "imbalance_buy",
    13: "imbalance_sell",
    14: "volatility_spike",
    15: "volatility_drop",
    16: "news_positive",
    17: "news_negative",
    18: "news_neutral",
    19: "regime_shift",
}

EVENT_TYPE_TO_ID = {v: k for k, v in EVENT_TYPES.items()}
NUM_EVENT_TYPES = len(EVENT_TYPES)


class EventSequence:
    """
    Data structure representing a sequence of continuous-time events.
    """
    def __init__(
        self,
        event_types: np.ndarray,
        timestamps: np.ndarray,
        inter_arrival_times: np.ndarray,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ):
        self.event_types = np.asarray(event_types, dtype=np.int64)
        self.timestamps = np.asarray(timestamps, dtype=np.float32)
        self.inter_arrival_times = np.asarray(inter_arrival_times, dtype=np.float32)
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64) if labels is not None else None

    def __len__(self) -> int:
        return len(self.event_types)

    def slice(self, start: int, end: int) -> "EventSequence":
        return EventSequence(
            event_types=self.event_types[start:end],
            timestamps=self.timestamps[start:end] - self.timestamps[start],
            inter_arrival_times=self.inter_arrival_times[start:end],
            features=self.features[start:end],
            labels=self.labels[start:end] if self.labels is not None else None,
        )

    def to_dict(self) -> Dict[str, np.ndarray]:
        d = {
            "event_types": self.event_types,
            "timestamps": self.timestamps,
            "inter_arrival_times": self.inter_arrival_times,
            "features": self.features,
        }
        if self.labels is not None:
            d["labels"] = self.labels
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, np.ndarray]) -> "EventSequence":
        return cls(
            event_types=data["event_types"],
            timestamps=data["timestamps"],
            inter_arrival_times=data["inter_arrival_times"],
            features=data["features"],
            labels=data.get("labels"),
        )


class EventStreamPipeline:
    """
    Transforms LOB matrices and market news into discrete, continuous-time event streams.
    """
    def __init__(
        self,
        sensitivity_threshold: float = 0.05,
        base_time_step: float = 0.01,
        feature_dim: int = 144,
    ):
        self.sensitivity_threshold = sensitivity_threshold
        self.base_time_step = base_time_step
        self.feature_dim = feature_dim

    def detect_lob_events(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> EventSequence:
        """
        Converts a sequence of normalized LOB feature vectors into an EventSequence.
        
        Args:
            features: (N, 144) LOB feature matrix
            labels: (N,) optional price movement labels
            timestamps: (N,) REAL wall-clock seconds for each snapshot. Strongly
                recommended: a temporal point process is a model of when events
                actually arrive, so fitting one to synthetic inter-arrival times
                fits nothing. When omitted a uniform clock is used and a warning
                is issued -- the previous implementation silently manufactured
                dt as `base_time_step * exp(-|mid_diff|*5) + Exponential(...)`,
                a deterministic function of the price move plus noise.

        Returns:
            EventSequence with detected events, continuous timestamps, and feature vectors.
        """
        n_samples = len(features)
        if n_samples == 0:
            return EventSequence(
                event_types=np.zeros(0, dtype=np.int64),
                timestamps=np.zeros(0, dtype=np.float32),
                inter_arrival_times=np.zeros(0, dtype=np.float32),
                features=np.zeros((0, self.feature_dim), dtype=np.float32),
            )

        event_types = []
        event_times = []
        inter_arrival_times = []
        selected_features = []
        selected_labels = []

        current_time = 0.0

        clock = None
        if timestamps is not None:
            clock = np.asarray(timestamps, dtype=np.float64).ravel()
            if len(clock) != n_samples:
                raise ValueError(
                    f"timestamps has length {len(clock)} but features has {n_samples}."
                )
            clock = clock - clock[0]
            if np.any(np.diff(clock) < 0):
                raise ValueError("timestamps must be non-decreasing.")
        else:
            warnings.warn(
                "detect_lob_events called without real timestamps; using a uniform "
                "clock. Any Hawkes/TPP intensity fitted on this is modelling an "
                "artefact of the sampling grid, not market event timing.",
                RuntimeWarning,
                stacklevel=2,
            )
        last_event_time = 0.0

        # Feature mapping from FI-2010 normalization:
        # 0: Ask price 1, 1: Ask volume 1, 2: Bid price 1, 3: Bid volume 1
        for i in range(1, n_samples):
            curr_feat = features[i]
            prev_feat = features[i - 1]

            ask_p_diff = curr_feat[0] - prev_feat[0]
            ask_v_diff = curr_feat[1] - prev_feat[1]
            bid_p_diff = curr_feat[2] - prev_feat[2]
            bid_v_diff = curr_feat[3] - prev_feat[3]

            curr_spread = curr_feat[0] - curr_feat[2]
            prev_spread = prev_feat[0] - prev_feat[2]
            spread_diff = curr_spread - prev_spread

            curr_mid = (curr_feat[0] + curr_feat[2]) / 2.0
            prev_mid = (prev_feat[0] + prev_feat[2]) / 2.0
            mid_diff = curr_mid - prev_mid

            curr_imbalance = (curr_feat[3] - curr_feat[1]) / (curr_feat[3] + curr_feat[1] + 1e-6)
            prev_imbalance = (prev_feat[3] - prev_feat[1]) / (prev_feat[3] + prev_feat[1] + 1e-6)
            imbalance_diff = curr_imbalance - prev_imbalance

            # Classify event based on largest microstructural shift
            detected_type = None

            if abs(mid_diff) > self.sensitivity_threshold:
                detected_type = 10 if mid_diff > 0 else 11  # mid_price_up / down
            elif abs(spread_diff) > self.sensitivity_threshold:
                detected_type = 8 if spread_diff > 0 else 9  # spread_widen / narrow
            elif abs(bid_p_diff) > self.sensitivity_threshold:
                detected_type = 0 if bid_p_diff > 0 else 1  # bid_price_up / down
            elif abs(ask_p_diff) > self.sensitivity_threshold:
                detected_type = 2 if ask_p_diff > 0 else 3  # ask_price_up / down
            elif abs(bid_v_diff) > 2.0 * self.sensitivity_threshold:
                detected_type = 4 if bid_v_diff > 0 else 5  # bid_volume_surge / drain
            elif abs(ask_v_diff) > 2.0 * self.sensitivity_threshold:
                detected_type = 6 if ask_v_diff > 0 else 7  # ask_volume_surge / drain
            elif abs(imbalance_diff) > 3.0 * self.sensitivity_threshold:
                detected_type = 12 if imbalance_diff > 0 else 13  # imbalance_buy / sell
            else:
                # Background micro-tick event based on high-order feature drift
                feature_drift = np.linalg.norm(curr_feat[:40] - prev_feat[:40])
                if feature_drift > self.sensitivity_threshold * 2:
                    detected_type = 14 if feature_drift > 1.0 else 15  # volatility_spike / drop

            if detected_type is not None:
                if clock is not None:
                    event_time = float(clock[i])
                    dt = max(1e-6, event_time - last_event_time)
                    last_event_time = event_time
                else:
                    dt = self.base_time_step
                    current_time += dt
                    event_time = current_time

                event_types.append(detected_type)
                event_times.append(event_time)
                inter_arrival_times.append(dt)
                selected_features.append(curr_feat)
                if labels is not None:
                    selected_labels.append(labels[i])

        if len(event_types) == 0:
            # Fallback if threshold was too strict: record periodic micro-ticks
            for i in range(0, n_samples, 5):
                if clock is not None:
                    event_time = float(clock[i])
                    dt = max(1e-6, event_time - last_event_time)
                    last_event_time = event_time
                else:
                    dt = self.base_time_step
                    current_time += dt
                    event_time = current_time
                event_types.append(10 if features[i, 0] >= features[max(0, i-1), 0] else 11)
                event_times.append(event_time)
                inter_arrival_times.append(dt)
                selected_features.append(features[i])
                if labels is not None:
                    selected_labels.append(labels[i])

        return EventSequence(
            event_types=np.array(event_types, dtype=np.int64),
            timestamps=np.array(event_times, dtype=np.float32),
            inter_arrival_times=np.array(inter_arrival_times, dtype=np.float32),
            features=np.array(selected_features, dtype=np.float32),
            labels=np.array(selected_labels, dtype=np.int64) if labels is not None else None,
        )

    def process_and_save(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray],
        output_path: Union[str, Path],
        timestamps: Optional[np.ndarray] = None,
    ) -> EventSequence:
        """
        Processes LOB features into events and saves to an .npz cache file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        event_seq = self.detect_lob_events(features, labels, timestamps=timestamps)
        np.savez_compressed(output_path, **event_seq.to_dict())
        print(f"[EventPipeline] Generated {len(event_seq)} events saved to {output_path}")
        return event_seq


class EventDataset(Dataset):
    """
    PyTorch Dataset for batched training of Temporal Point Process models.
    Slices event sequences into fixed context windows (e.g. 128 or 256 events).
    """
    def __init__(
        self,
        event_sequence: EventSequence,
        seq_len: int = 128,
        stride: int = 16,
    ):
        self.event_sequence = event_sequence
        self.seq_len = seq_len
        self.stride = stride

        self.num_windows = max(1, (len(event_sequence) - seq_len) // stride + 1)

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.stride
        end = start + self.seq_len

        if end > len(self.event_sequence):
            start = max(0, len(self.event_sequence) - self.seq_len)
            end = len(self.event_sequence)

        types = self.event_sequence.event_types[start:end]
        dts = self.event_sequence.inter_arrival_times[start:end]
        ts = self.event_sequence.timestamps[start:end] - self.event_sequence.timestamps[start]
        feats = self.event_sequence.features[start:end]

        # Padding if sequence shorter than seq_len. The mask marks padded
        # positions (True = padding) so they can be excluded from the Hawkes
        # log-likelihood; without it, padded slots were counted as real events
        # and contributed a spurious survival term.
        n_real = len(types)
        mask = np.zeros(self.seq_len, dtype=bool)
        if n_real < self.seq_len:
            pad_len = self.seq_len - n_real
            types = np.pad(types, (0, pad_len), mode="constant", constant_values=0)
            dts = np.pad(dts, (0, pad_len), mode="constant", constant_values=1.0)
            ts = np.pad(ts, (0, pad_len), mode="edge")
            feats = np.pad(feats, ((0, pad_len), (0, 0)), mode="edge")
            mask[n_real:] = True

        item = {
            "event_types": torch.tensor(types, dtype=torch.long),
            "inter_arrival_times": torch.tensor(dts, dtype=torch.float32),
            "timestamps": torch.tensor(ts, dtype=torch.float32),
            "features": torch.tensor(feats, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
        }

        if self.event_sequence.labels is not None:
            lbl = self.event_sequence.labels[end - 1] if end - 1 < len(self.event_sequence.labels) else 1
            item["target_label"] = torch.tensor(lbl, dtype=torch.long)

        return item
