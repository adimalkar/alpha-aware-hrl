import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional

class HistoricalLOBEnv(gym.Env):
    """
    Simulated trading environment that replays historical Limit Order Book (LOB) data.
    
    This replaces the synthetic ABIDESEnv for conference-level RL training.
    It reads normalized FI-2010 LOB snapshots and tracks an agent's portfolio.
    
    Args:
        data_loader: An instance of FI2010DataLoader that has already loaded data.
        split: 'train' or 'test' (which dataset to iterate over).
        episode_length: Number of timesteps per episode.
        starting_cash: Initial portfolio cash.
        transaction_fee: Fee per trade as a fraction of trade value (e.g., 0.0001 for 1 bps).
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        data_loader: Any,
        split: str = "train",
        episode_length: int = 1000,
        starting_cash: float = 100000.0,
        transaction_fee: float = 0.0001,
        regime_path: Optional[str] = None,
    ):
        super().__init__()
        
        self.data_loader = data_loader
        self.split = split
        self.episode_length = episode_length
        self.starting_cash = starting_cash
        self.transaction_fee = transaction_fee
        
        # Load the data if not already loaded
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
        
        # Load precomputed LLM regimes if provided
        self.regimes = None
        if regime_path:
            try:
                self.regimes = np.load(regime_path)
                assert len(self.regimes) == self.n_samples, "Regime array length must match data length"
            except Exception as e:
                print(f"Warning: Failed to load regimes from {regime_path}: {e}")
        
        if self.n_samples < self.episode_length:
            raise ValueError(f"Dataset size ({self.n_samples}) is smaller than episode length ({self.episode_length})")
        
        # Action space: Continuous [-1, 1] mapped to [-max_trade, max_trade]
        # For simplicity, action is desired portfolio weight (-1.0 to 1.0)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space: 144 features from FI-2010
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(144,), dtype=np.float32
        )
        
        # Internal state
        self.current_step = 0
        self.start_idx = 0
        self.cash = self.starting_cash
        self.position = 0.0  # Number of shares held
        
        # Since FI-2010 features are Z-score normalized, we don't have true absolute prices.
        # We will simulate a synthetic price path based on the labels or just use a dummy price
        # for PnL calculation. 
        # Label 0 = Down, 1 = Stable, 2 = Up.
        # Let's start the synthetic price at 100.0 and move it slightly based on the label/future horizon
        # For a truly realistic setup, we'd need the un-normalized mid-prices which FI-2010 drops.
        # As a proxy for this implementation, we will create a synthetic price that tracks the label.
        self.current_price = 100.0
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to a random starting point in the dataset."""
        super().reset(seed=seed)
        
        # Pick a random starting index that allows for a full episode
        self.start_idx = np.random.randint(0, self.n_samples - self.episode_length)
        self.current_step = 0
        
        self.cash = self.starting_cash
        self.position = 0.0
        self.current_price = 100.0
        
        obs = self.data[self.start_idx].astype(np.float32)
        
        info = {
            "current_price": self.current_price,
            "portfolio_value": self._get_portfolio_value(),
            "idx": self.start_idx,
        }
        
        return obs, info
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute trading action and advance one tick in historical data."""
        # Current portfolio value before action
        prev_portfolio_value = self._get_portfolio_value()
        
        # Map action [-1, 1] to target position weight
        target_weight = np.clip(action[0], -1.0, 1.0)
        
        # Calculate target position size in shares
        target_value = target_weight * prev_portfolio_value
        target_position = target_value / self.current_price
        
        # Calculate required trade
        trade_shares = target_position - self.position
        trade_value = trade_shares * self.current_price
        
        # Apply transaction fees
        fee = abs(trade_value) * self.transaction_fee
        
        # Execute trade
        self.position = target_position
        self.cash -= (trade_value + fee)
        
        # Advance time
        self.current_step += 1
        data_idx = self.start_idx + self.current_step
        
        # Update synthetic price based on actual label (0=down, 1=stable, 2=up)
        # In a real scenario with unnormalized data, this would be the actual mid-price.
        label = self.labels[data_idx]
        if label == 2:
            self.current_price *= (1 + 0.001)  # Up 10 bps
        elif label == 0:
            self.current_price *= (1 - 0.001)  # Down 10 bps
            
        # Get new observation
        obs = self.data[data_idx].astype(np.float32)
        
        # Calculate reward (change in portfolio value)
        current_portfolio_value = self._get_portfolio_value()
        reward = current_portfolio_value - prev_portfolio_value
        
        # Check termination
        terminated = self.current_step >= self.episode_length - 1
        truncated = False
        
        info = {
            "current_price": self.current_price,
            "portfolio_value": current_portfolio_value,
            "position": self.position,
            "cash": self.cash,
            "idx": data_idx,
            "label": label,
        }
        
        # Inject LLM regime if available
        if self.regimes is not None:
            # We assume the array has shape (N, 2) [regime_id, confidence]
            reg = self.regimes[data_idx]
            info["precomputed_regime"] = {
                "regime": int(reg[0]),
                "confidence": float(reg[1]),
                "reasoning": "Precomputed from FNSPID"
            }
        
        return obs, float(reward), terminated, truncated, info
        
    def _get_portfolio_value(self) -> float:
        """Calculate total value of cash + held positions."""
        return self.cash + (self.position * self.current_price)
