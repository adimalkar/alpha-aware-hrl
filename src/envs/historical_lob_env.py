"""
Historical Limit Order Book replay environment.

IMPORTANT -- on price paths and FI-2010
=======================================
This environment requires a REAL, CAUSAL price series. It will not invent one.

The previous implementation derived the traded price from the FI-2010 label:

    label = self.labels[data_idx]          # the k-step-ahead direction
    if label == 2:   self.current_price *= 1.001
    elif label == 0: self.current_price *= 0.999

FI-2010 labels encode the *future* mid-price direction, so reward at step t
was a deterministic function of information from t+k while the observation was
precisely the feature vector that label describes. A perfect observation ->
reward map existed by construction, and every metric downstream of it measured
that leak.

The obvious repair -- reconstruct the mid from the book -- does not work on the
distributed FI-2010 CSVs either. Those files are z-scored PER COLUMN, which
destroys order-book geometry. Measured on the first 50k training rows:

    ask_1 > bid_1      in  51.3% of rows   (should be 100%)
    ask_1 < ask_2      in  27.6% of rows   (should be ~100%)
    bid_1 > bid_2      in  24.2% of rows   (should be ~100%)
    reconstructed mid  crosses zero (min -1.065, max +0.473)

Every price column was independently standardised to the same distribution
(mean ~= -0.215, sd ~= 0.676), so no mid, spread, or return survives. FI-2010
as distributed supports exactly one task: supervised classification of the
supplied label from the supplied features. That is what DeepLOB uses it for.

Therefore: supply `prices` (or `price_column`) from a source that really has
them. If you cannot, this dataset cannot support a trading simulation, and the
honest move is to evaluate the model as a classifier instead.
"""

import warnings
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Observation = LOB features + agent state. The agent state terms make the
# decision problem Markov: reward depends on inventory, so a policy that
# cannot see its inventory is being scored on state it has no access to.
N_AGENT_STATE_FEATURES = 3  # [position_weight, cash_fraction, step_fraction]


class PriceSourceError(ValueError):
    """Raised when no causal price series is available for the requested data."""


class HistoricalLOBEnv(gym.Env):
    """
    Replays historical LOB snapshots and tracks a single-asset portfolio.

    Args:
        data_loader: An FI2010DataLoader that has loaded (or can load) data.
        split: 'train' or 'test'.
        episode_length: Steps per episode.
        starting_cash: Initial portfolio cash.
        transaction_fee: Fee per trade as a fraction of traded notional.
        prices: Explicit causal mid-price series aligned 1:1 with the split's
            rows. This is the supported way to supply prices.
        price_column: Alternatively, the index of a feature column that holds a
            real (un-normalised, strictly positive) price.
        regime_path: Optional .npy of shape (N,) or (N, 2) with regime signals.
        reward: 'log_return' (default) or 'simple_return'. Absolute PnL deltas
            are not offered: on a 1e5 portfolio they are O(1e-2), which is
            numerically negligible as a learning signal and was a contributing
            cause of the all-zero ablation results.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_loader: Any,
        split: str = "train",
        episode_length: int = 1000,
        starting_cash: float = 100000.0,
        transaction_fee: float = 0.0001,
        prices: Optional[np.ndarray] = None,
        price_column: Optional[int] = None,
        regime_path: Optional[str] = None,
        reward: str = "log_return",
    ):
        super().__init__()

        self.data_loader = data_loader
        self.split = split
        self.episode_length = int(episode_length)
        self.starting_cash = float(starting_cash)
        self.transaction_fee = float(transaction_fee)
        if reward not in ("log_return", "simple_return"):
            raise ValueError(f"reward must be 'log_return' or 'simple_return', got {reward!r}")
        self.reward_kind = reward

        if split == "train":
            if self.data_loader.train_data is None:
                self.data_loader.load("train")
            self.data = self.data_loader.train_data
            self.labels = self.data_loader.train_labels
        else:
            if self.data_loader.test_data is None:
                self.data_loader.load("test")
            self.data = self.data_loader.test_data
            self.labels = self.data_loader.test_labels

        self.n_samples = len(self.data)
        self.n_features = int(self.data.shape[1])

        self.prices = self._resolve_prices(prices, price_column)

        # Regimes are optional, but a length mismatch is a hard error. The
        # previous code swallowed the assertion in a try/except and printed a
        # warning, so a misaligned regime array silently became no regime.
        self.regimes = None
        if regime_path:
            regimes = np.load(regime_path)
            if len(regimes) != self.n_samples:
                raise ValueError(
                    f"Regime array from {regime_path} has length {len(regimes)} "
                    f"but the '{split}' split has {self.n_samples} rows. Refusing "
                    "to run with misaligned regimes."
                )
            # D2: every committed regime artefact is a single repeated value
            # (train_regimes.npy = 2 x 362,400; test = 2 x 31,937; the live
            # splits = 0 with confidence 0.95). The one-hot of a constant is an
            # intercept, so a constant regime contributes nothing a bias term
            # does not. Say so rather than letting it look like a live signal.
            reg_col = regimes[:, 0] if regimes.ndim > 1 else regimes
            if len(np.unique(reg_col)) == 1:
                warnings.warn(
                    f"Regime array from {regime_path} is constant "
                    f"(value {reg_col[0]!r} across all {len(reg_col)} rows). It "
                    "carries no information beyond an intercept; any 'regime-aware' "
                    "claim based on it is unsupported.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.regimes = regimes

        if self.n_samples <= self.episode_length:
            raise ValueError(
                f"Dataset size ({self.n_samples}) must exceed episode length "
                f"({self.episode_length})."
            )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features + N_AGENT_STATE_FEATURES,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.start_idx = 0
        self.cash = self.starting_cash
        self.position = 0.0

    # ------------------------------------------------------------------ #
    # price resolution
    # ------------------------------------------------------------------ #
    def _resolve_prices(
        self, prices: Optional[np.ndarray], price_column: Optional[int]
    ) -> np.ndarray:
        if prices is not None and price_column is not None:
            raise PriceSourceError("Pass either `prices` or `price_column`, not both.")

        if prices is not None:
            p = np.asarray(prices, dtype=np.float64).ravel()
            if len(p) != self.n_samples:
                raise PriceSourceError(
                    f"`prices` has length {len(p)} but the '{self.split}' split has "
                    f"{self.n_samples} rows; they must align 1:1."
                )
        elif price_column is not None:
            if not (0 <= price_column < self.n_features):
                raise PriceSourceError(
                    f"price_column {price_column} is outside [0, {self.n_features})."
                )
            p = np.asarray(self.data[:, price_column], dtype=np.float64)
        else:
            raise PriceSourceError(
                "No price series supplied. This environment requires a real, causal "
                "price path and will not synthesise one.\n\n"
                "The previous implementation derived the price from the FI-2010 "
                "label, which encodes the FUTURE mid-price direction -- that is a "
                "lookahead leak, not a price.\n\n"
                "The distributed FI-2010 CSVs are z-scored per column, which destroys "
                "order-book geometry (ask_1 > bid_1 in only ~51% of rows), so a mid "
                "price cannot be reconstructed from them either.\n\n"
                "Supply `prices=` from a source that has genuine prices, or use "
                "FI-2010 for the supervised classification task it actually supports."
            )

        self._validate_prices(p)
        return p

    def _validate_prices(self, p: np.ndarray) -> None:
        if not np.all(np.isfinite(p)):
            raise PriceSourceError("Price series contains non-finite values.")
        if np.any(p <= 0):
            n = int((p <= 0).sum())
            raise PriceSourceError(
                f"Price series has {n} non-positive values, so simple returns are "
                "undefined. This is the signature of a z-scored feature column being "
                "passed as a price."
            )
        if np.allclose(p, p[0]):
            raise PriceSourceError(
                "Price series is constant; no trading result can be measured from it."
            )

    # ------------------------------------------------------------------ #
    # gym API
    # ------------------------------------------------------------------ #
    def _observation(self, data_idx: int) -> np.ndarray:
        """LOB features concatenated with normalised agent state."""
        market = self.data[data_idx].astype(np.float32)
        equity = self._get_portfolio_value(self.prices[data_idx])
        position_value = self.position * self.prices[data_idx]

        agent_state = np.array(
            [
                position_value / max(equity, 1e-9),   # current portfolio weight
                self.cash / max(equity, 1e-9),        # cash fraction
                self.current_step / self.episode_length,  # time remaining
            ],
            dtype=np.float32,
        )
        return np.concatenate([market, agent_state]).astype(np.float32)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Use the seeded generator gymnasium installs. The previous code called
        # np.random.randint here, so `seed=` had no effect on episode placement
        # and no run was reproducible.
        max_start = self.n_samples - self.episode_length - 1
        self.start_idx = int(self.np_random.integers(0, max_start)) if max_start > 0 else 0

        self.current_step = 0
        self.cash = self.starting_cash
        self.position = 0.0

        obs = self._observation(self.start_idx)
        info = {
            "current_price": float(self.prices[self.start_idx]),
            "portfolio_value": self._get_portfolio_value(self.prices[self.start_idx]),
            "position": 0.0,
            "cash": self.cash,
            "idx": self.start_idx,
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        idx = self.start_idx + self.current_step
        price_now = self.prices[idx]

        prev_value = self._get_portfolio_value(price_now)

        # Rebalance to the target weight at the CURRENT (observed) price.
        target_weight = float(np.clip(action[0], -1.0, 1.0))
        target_position = (target_weight * prev_value) / price_now
        trade_shares = target_position - self.position
        trade_notional = abs(trade_shares * price_now)
        fee = trade_notional * self.transaction_fee

        self.position = target_position
        self.cash -= trade_shares * price_now + fee

        # Advance; the price move is revealed only after the trade is committed.
        self.current_step += 1
        next_idx = self.start_idx + self.current_step
        price_next = self.prices[next_idx]

        new_value = self._get_portfolio_value(price_next)

        terminated = False
        if new_value <= 0.0:
            # Leveraged short positions can wipe the book out. Terminate rather
            # than propagating a negative equity into the metrics.
            new_value = 0.0
            terminated = True
            reward = -10.0
        elif self.reward_kind == "log_return":
            reward = float(np.log(max(new_value, 1e-12) / max(prev_value, 1e-12)))
        else:
            reward = float((new_value - prev_value) / max(prev_value, 1e-12))

        if self.current_step >= self.episode_length - 1:
            terminated = True

        obs = self._observation(next_idx)
        info = {
            "current_price": float(price_next),
            "portfolio_value": float(new_value),
            "position": float(self.position),
            "position_weight": float(self.position * price_next / max(new_value, 1e-9)),
            "cash": float(self.cash),
            "fee_paid": float(fee),
            "idx": next_idx,
            # The label is carried for supervised/diagnostic use only. It must
            # never influence price, reward, or observation.
            "label": int(self.labels[next_idx]),
        }

        if self.regimes is not None:
            reg = np.atleast_1d(self.regimes[next_idx])
            info["precomputed_regime"] = {
                "regime": int(reg[0]),
                "confidence": float(reg[1]) if reg.size > 1 else 1.0,
            }

        return obs, float(reward), terminated, False, info

    def _get_portfolio_value(self, price: float) -> float:
        return float(self.cash + self.position * price)
