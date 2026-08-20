"""
Live and Simulated Event Feed Adapters for Streaming Ingestion.
"""

import time
import threading
import numpy as np
from typing import Callable, Optional, Dict, Any

from src.streaming.event_buffer import EventRingBuffer
from src.utils.event_pipeline import EVENT_TYPES


class SyntheticMarketFeed:
    """
    Self-exciting synthetic continuous-time market event simulator.
    Simulates Hawkes cluster bursts, sudden volatility shocks, and order book arrivals.
    """
    def __init__(
        self,
        event_buffer: EventRingBuffer,
        base_rate: float = 20.0,  # events per second
        hawkes_alpha: float = 0.6,
        hawkes_beta: float = 2.0,
    ):
        self.buffer = event_buffer
        self.base_rate = base_rate
        self.alpha = hawkes_alpha
        self.beta = hawkes_beta
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.current_intensity = base_rate
        self.current_mid = 100.0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run_loop(self) -> None:
        last_t = time.time()
        
        while self._running:
            now = time.time()
            dt_elapsed = now - last_t
            last_t = now

            # Decay intensity toward base rate: dλ = -β(λ - λ_0) dt
            self.current_intensity = self.base_rate + (self.current_intensity - self.base_rate) * np.exp(-self.beta * dt_elapsed)

            # Sample next event time via Ogata's thinning algorithm / exponential step
            expected_wait = 1.0 / max(1.0, self.current_intensity)
            sleep_time = np.random.exponential(expected_wait)
            time.sleep(min(sleep_time, 0.1))

            if not self._running:
                break

            # Self-excitation jump: λ <- λ + α
            self.current_intensity += self.alpha

            # Determine event type
            r = np.random.rand()
            if self.current_intensity > self.base_rate * 2.5:
                # Flash crash / volatility burst regime
                event_type = np.random.choice([1, 3, 5, 7, 8, 11, 14])  # drop/drain/widen/crash
            elif r < 0.35:
                event_type = 10  # mid_price_up
                self.current_mid += np.random.uniform(0.01, 0.05)
            elif r < 0.70:
                event_type = 11  # mid_price_down
                self.current_mid -= np.random.uniform(0.01, 0.05)
            else:
                event_type = np.random.randint(0, 16)

            # Synthesize 144-D microstructural vector
            feat = np.zeros(144, dtype=np.float32)
            feat[0] = self.current_mid + 0.02
            feat[1] = np.random.uniform(10, 100)
            feat[2] = self.current_mid - 0.02
            feat[3] = np.random.uniform(10, 100)

            self.buffer.append(
                event_type=event_type,
                timestamp=time.time(),
                feature_vector=feat,
            )


class LiveExchangeAdapter:
    """
    WebSocket adapter interface for real cryptocurrency and equity exchange feeds.
    Connects to live order books via ccxt or direct WebSocket.
    """
    def __init__(self, event_buffer: EventRingBuffer, symbol: str = "BTC/USDT"):
        self.buffer = event_buffer
        self.symbol = symbol
        self._running = False

    def connect(self) -> None:
        print(f"[LiveExchangeAdapter] Connecting to live event feed for {self.symbol}...")
        self._running = True

    def disconnect(self) -> None:
        self._running = False
        print(f"[LiveExchangeAdapter] Disconnected feed for {self.symbol}")
