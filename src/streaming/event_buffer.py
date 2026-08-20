"""
Thread-safe Ring Buffer for Continuous Event Streaming in Alpha-Aware HRL.
"""

import threading
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch


class EventRingBuffer:
    """
    Circular ring buffer designed for low-latency, lock-free or minimal-lock
    ingestion of continuous-time market microstructure and news events.
    """
    def __init__(self, capacity: int = 1024, feature_dim: int = 144):
        self.capacity = capacity
        self.feature_dim = feature_dim
        
        self.event_types = np.zeros(capacity, dtype=np.int64)
        self.timestamps = np.zeros(capacity, dtype=np.float32)
        self.inter_arrival_times = np.zeros(capacity, dtype=np.float32)
        self.features = np.zeros((capacity, feature_dim), dtype=np.float32)
        
        self.head = 0
        self.count = 0
        self.last_timestamp = 0.0
        self._lock = threading.Lock()

    def append(
        self,
        event_type: int,
        timestamp: Optional[float] = None,
        feature_vector: Optional[np.ndarray] = None,
    ) -> None:
        """
        Appends a single event to the circular buffer.
        """
        with self._lock:
            if timestamp is None:
                timestamp = time.time()

            if self.last_timestamp > 0:
                dt = max(1e-5, float(timestamp - self.last_timestamp))
            else:
                dt = 0.01

            self.last_timestamp = timestamp
            idx = self.head

            self.event_types[idx] = int(event_type)
            self.timestamps[idx] = float(timestamp)
            self.inter_arrival_times[idx] = float(dt)

            if feature_vector is not None:
                self.features[idx] = feature_vector
            else:
                self.features[idx] = 0.0

            self.head = (self.head + 1) % self.capacity
            self.count = min(self.capacity, self.count + 1)

    def get_latest_window(self, window_size: int = 128) -> Dict[str, np.ndarray]:
        """
        Retrieves the most recent window of events in chronological order.
        """
        with self._lock:
            if self.count == 0:
                return {
                    "event_types": np.zeros(window_size, dtype=np.int64),
                    "timestamps": np.zeros(window_size, dtype=np.float32),
                    "inter_arrival_times": np.full(window_size, 0.01, dtype=np.float32),
                    "features": np.zeros((window_size, self.feature_dim), dtype=np.float32),
                }

            actual_len = min(window_size, self.count)
            # Reconstruct chronological slice from circular buffer
            indices = [(self.head - actual_len + i) % self.capacity for i in range(actual_len)]

            types = self.event_types[indices]
            ts = self.timestamps[indices]
            dts = self.inter_arrival_times[indices]
            feats = self.features[indices]

            # Zero pad to exact window_size if count < window_size
            if actual_len < window_size:
                pad_len = window_size - actual_len
                types = np.pad(types, (pad_len, 0), mode="edge")
                ts = np.pad(ts, (pad_len, 0), mode="edge")
                dts = np.pad(dts, (pad_len, 0), mode="constant", constant_values=0.01)
                feats = np.pad(feats, ((pad_len, 0), (0, 0)), mode="edge")

            return {
                "event_types": types,
                "timestamps": ts,
                "inter_arrival_times": dts,
                "features": feats,
            }

    def get_event_rate(self, time_window_seconds: float = 5.0) -> float:
        """
        Calculates instantaneous event arrival rate (events / sec).
        """
        with self._lock:
            if self.count < 2:
                return 0.0
            
            now = self.last_timestamp
            recent_count = 0
            for i in range(self.count):
                idx = (self.head - 1 - i) % self.capacity
                if now - self.timestamps[idx] <= time_window_seconds:
                    recent_count += 1
                else:
                    break

            return float(recent_count) / max(0.1, time_window_seconds)
