"""
ABIDES Environment Wrapper for Gymnasium

Wraps the ABIDES market simulator for use with standard RL libraries.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
import gymnasium as gym
from gymnasium import spaces


class ABIDESEnv(gym.Env):
    """
    Gymnasium wrapper for ABIDES market simulator.
    
    ABIDES (Agent-Based Interactive Discrete Event Simulation) is a
    high-fidelity market simulator that models realistic market dynamics
    including:
    - Order book mechanics
    - Market impact
    - Latency
    - Multiple agent types (market makers, noise traders, etc.)
    
    Args:
        ticker: Stock ticker to trade
        starting_cash: Initial cash balance
        max_position: Maximum allowed position size
        episode_length: Number of steps per episode
        market_impact: Whether to model market impact
        latency_ms: Simulated latency in milliseconds
        volatility_regime: Market volatility level ('low', 'medium', 'high', 'crash')
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        ticker: str = "AAPL",
        starting_cash: float = 100000.0,
        max_position: int = 1000,
        episode_length: int = 1000,
        market_impact: bool = True,
        latency_ms: int = 1,
        volatility_regime: str = "medium",
    ):
        super().__init__()
        
        self.ticker = ticker
        self.starting_cash = starting_cash
        self.max_position = max_position
        self.episode_length = episode_length
        self.market_impact = market_impact
        self.latency_ms = latency_ms
        self.volatility_regime = volatility_regime
        
        # Volatility parameters by regime
        self._vol_params = {
            "low":    {"sigma": 0.003, "spread_mult": 0.0005, "impact_mult": 0.0005},
            "medium": {"sigma": 0.010, "spread_mult": 0.001,  "impact_mult": 0.001},
            "high":   {"sigma": 0.025, "spread_mult": 0.003,  "impact_mult": 0.002},
            "crash":  {"sigma": 0.050, "spread_mult": 0.008,  "impact_mult": 0.005},
        }
        self._vp = self._vol_params.get(volatility_regime, self._vol_params["medium"])
        
        # State: [bid_prices(10), ask_prices(10), bid_volumes(10), ask_volumes(10)]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(40,),
            dtype=np.float32,
        )
        
        # Action: [order_type, quantity, price_offset]
        # order_type: -1 (sell) to 1 (buy), continuous
        # quantity: 0 to 1 (fraction of max position)
        # price_offset: -1 to 1 (offset from mid price)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )
        
        # Internal state
        self.cash = starting_cash
        self.position = 0
        self.step_count = 0
        self.current_price = 100.0  # Placeholder
        
        # ABIDES kernel (to be initialized)
        self.kernel = None
        
    def _init_abides(self):
        """Initialize ABIDES simulation kernel."""
        # TODO: Implement actual ABIDES initialization
        # from abides_markets.configs import rmsc04
        # self.kernel = rmsc04.build_config(...)
        pass
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        self.cash = self.starting_cash
        self.position = 0
        self.step_count = 0
        self.current_price = 100.0 + np.random.randn() * 5
        
        # TODO: Reset ABIDES kernel
        # self._init_abides()
        
        obs = self._get_observation()
        info = {
            "cash": self.cash,
            "position": self.position,
            "portfolio_value": self._get_portfolio_value(),
        }
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: [order_direction, quantity_fraction, price_offset]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1
        
        # Parse action
        order_direction = action[0]  # -1 to 1
        quantity_fraction = (action[1] + 1) / 2  # 0 to 1
        price_offset = action[2]  # -1 to 1
        
        # Calculate order
        order_quantity = int(quantity_fraction * self.max_position)
        
        # Execute order (simplified - actual ABIDES would handle this)
        previous_value = self._get_portfolio_value()
        
        if order_direction > 0.3:  # Buy
            max_buyable = int(self.cash / self.current_price)
            buy_quantity = min(order_quantity, max_buyable)
            
            # Apply market impact
            if self.market_impact:
                impact = self._vp["impact_mult"] * buy_quantity / self.max_position
                self.current_price *= (1 + impact)
            
            self.position += buy_quantity
            self.cash -= buy_quantity * self.current_price
            
        elif order_direction < -0.3:  # Sell
            sell_quantity = min(order_quantity, self.position)
            
            if self.market_impact:
                impact = self._vp["impact_mult"] * sell_quantity / self.max_position
                self.current_price *= (1 - impact)
            
            self.position -= sell_quantity
            self.cash += sell_quantity * self.current_price
        
        # Price random walk with regime-dependent volatility
        self.current_price *= (1 + np.random.randn() * self._vp["sigma"])
        
        # Calculate reward (change in portfolio value)
        current_value = self._get_portfolio_value()
        reward = (current_value - previous_value) / self.starting_cash
        
        # Check termination
        terminated = self.step_count >= self.episode_length
        truncated = self.cash < 0  # Bankrupt
        
        obs = self._get_observation()
        info = {
            "cash": self.cash,
            "position": self.position,
            "portfolio_value": current_value,
            "current_price": self.current_price,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Generate observation (LOB state)."""
        # Simplified: Generate synthetic LOB data
        # Real implementation would get this from ABIDES
        
        mid_price = self.current_price
        spread = mid_price * self._vp["spread_mult"]
        
        # 10 levels of bid/ask prices and volumes
        bid_prices = mid_price - spread * np.arange(1, 11)
        ask_prices = mid_price + spread * np.arange(1, 11)
        bid_volumes = np.random.exponential(100, 10)
        ask_volumes = np.random.exponential(100, 10)
        
        # Normalize
        obs = np.concatenate([
            (bid_prices - mid_price) / mid_price,
            (ask_prices - mid_price) / mid_price,
            bid_volumes / 1000,
            ask_volumes / 1000,
        ]).astype(np.float32)
        
        return obs
    
    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        return self.cash + self.position * self.current_price
    
    def render(self):
        """Render the environment state."""
        print(f"Step {self.step_count}: "
              f"Cash=${self.cash:.2f}, "
              f"Position={self.position}, "
              f"Price=${self.current_price:.2f}, "
              f"Portfolio=${self._get_portfolio_value():.2f}")
