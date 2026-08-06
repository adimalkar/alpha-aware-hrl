"""
DSAC/TQC Trader Agent for Risk-Aware Execution

Implements Distributional Soft Actor-Critic with Truncated Quantile Critics
for learning the full risk distribution of trading actions.
"""

import numpy as np
import torch
from typing import Optional, Dict, Any, Tuple
import gymnasium as gym


class DSACTrader:
    """
    Distributional SAC (DSAC) / TQC agent for trade execution.
    
    Uses Truncated Quantile Critics to learn the risk distribution
    (not just expected value) of trading actions. This allows the
    agent to account for tail risks (1% crash probability).
    
    Args:
        observation_dim: Dimension of state observations
        action_dim: Dimension of action space
        regime_dim: Dimension of regime signal embedding
        learning_rate: Learning rate for all networks
        buffer_size: Size of replay buffer
        batch_size: Batch size for training
        gamma: Discount factor
        tau: Soft update coefficient
        n_critics: Number of critic networks
        top_quantiles_to_drop: TQC-specific parameter
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        regime_dim: int = 4,  # 3 one-hot + 1 confidence
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_critics: int = 2,
        top_quantiles_to_drop: int = 2,
        device: str = "cuda",
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.regime_dim = regime_dim
        self.device = device
        
        # Combined input dimension (features + regime)
        self.total_obs_dim = observation_dim + regime_dim
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.n_critics = n_critics
        self.top_quantiles_to_drop = top_quantiles_to_drop
        
        self.model = None
        self._initialized = False
        
    def initialize(self, env: gym.Env):
        """Initialize the TQC model with the environment."""
        if self._initialized:
            return
            
        from sb3_contrib import TQC
        
        # Determine device string for SB3
        device_str = "cuda" if "cuda" in self.device else "cpu"
        
        # SB3 TQC initialization
        self.model = TQC(
            "MlpPolicy",
            env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            gamma=self.gamma,
            tau=self.tau,
            top_quantiles_to_drop_per_net=self.top_quantiles_to_drop,
            verbose=1,
            device=device_str,
            policy_kwargs=dict(
                net_arch=[256, 256],
                n_critics=self.n_critics
            )
        )
        
        self._initialized = True
        print("[DSACTrader] Initialized Full TQC (DSAC) model from sb3_contrib")
        
    def select_action(
        self,
        observation: np.ndarray,
        regime_embedding: torch.Tensor,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select trading action given state and regime.
        
        Args:
            observation: Market state features from Mamba
            regime_embedding: Regime signal from LLM analyst
            deterministic: Whether to use deterministic policy
            
        Returns:
            action: Trading action (e.g., buy/sell/hold amounts)
        """
        # Concatenate observation with regime embedding
        regime_np = regime_embedding.cpu().numpy().squeeze()
        
        # Ensure flat observation
        obs_flat = observation.flatten()
        
        full_obs = np.concatenate([obs_flat, regime_np], axis=0)
        
        if self.model is not None:
             action, _ = self.model.predict(full_obs, deterministic=deterministic)
             return action
        else:
             # Random fallback if not initialized
             return np.random.uniform(-1, 1, size=self.action_dim)
    
    def train_step(
        self,
        replay_buffer: Any,
    ) -> Dict[str, float]:
        """
        [DEPRECATED] Training is done through model.learn() via HierarchicalAgent.
        This method is kept for API compatibility but is not used.
        """
        return {
            "critic_loss": 0.0,
            "actor_loss": 0.0,
            "alpha_loss": 0.0,
        }
    
    def save(self, path: str):
        """Save model checkpoint."""
        if self.model is not None:
            self.model.save(path)
            
    def load(self, path: str, env: Optional[gym.Env] = None):
        """Load model checkpoint."""
        if not path.endswith('.zip'):
             path = f"{path}.zip"
        
        from sb3_contrib import TQC
        self.model = TQC.load(path, env=env)
        self._initialized = True
        print(f"[DSACTrader] Loaded TQC model from {path}")
    
    def get_risk_distribution(
        self,
        observation: np.ndarray,
        action: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the learned risk distribution for a state-action pair.
        
        This is the key advantage of DSAC/TQC over standard SAC:
        we can see the full distribution of returns, not just the mean.
        
        Args:
            observation: State observation (flattened)
            action: Optional action. If None, uses current policy.
            
        Returns:
            quantiles: Quantile levels (e.g., 0.1, 0.25, 0.5, 0.75, 0.9)
            values: Corresponding Q-values at each quantile
        """
        if self.model is None:
            quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
            return quantiles, np.zeros_like(quantiles)
        
        import torch as th
        
        # Get action from policy if not provided
        obs_tensor = th.tensor(observation, dtype=th.float32).unsqueeze(0)
        obs_tensor = obs_tensor.to(self.model.device)
        
        if action is None:
            with th.no_grad():
                action_tensor = self.model.actor(obs_tensor)
        else:
            action_tensor = th.tensor(action, dtype=th.float32).unsqueeze(0)
            action_tensor = action_tensor.to(self.model.device)
        
        # Extract quantile values from each critic network
        all_quantile_values = []
        with th.no_grad():
            for critic in self.model.critic.quantile_critics:
                qv = critic(obs_tensor, action_tensor)
                all_quantile_values.append(qv.cpu().numpy().flatten())
        
        # Combine all critic quantile values and sort
        combined = np.concatenate(all_quantile_values)
        combined.sort()
        
        # Select representative quantile levels
        n = len(combined)
        if n >= 5:
            indices = [int(n * p) for p in [0.1, 0.25, 0.5, 0.75, 0.9]]
            indices = [min(i, n - 1) for i in indices]
            quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
            values = combined[indices]
        else:
            quantiles = np.linspace(0, 1, max(n, 1))
            values = combined
        
        return quantiles, values
