"""
Hierarchical Agent: Combines Mamba, LLM Analyst, Alpha Model, and DSAC Trader

The main agent class that orchestrates the three-tier hierarchy:
1. Analyst (LLM) - High-level regime detection
2. Manager (Mamba + Alpha) - Feature extraction and state representation
3. Trader (DSAC) - Low-level execution
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, List
import gymnasium as gym

from src.envs.hierarchical_wrapper import HierarchicalEnvWrapper
from .mamba_extractor import MambaFeatureExtractor
from .llm_analyst import LLMAnalyst, RegimeSignal
from .dsac_trader import DSACTrader
from src.models.timesfm_wrapper import SimpleAlphaModel


class HierarchicalAgent:
    """
    Hierarchical agent for alpha-aware trading.
    
    Architecture:
    ```
    News/Context → [LLM Analyst] → Regime Signal (4D)
                         ↓
    Price History → [Alpha Model] → Alpha Signal (4D)
                         ↓
    LOB Data → [Mamba Extractor] → Features (128D)
                         ↓
    (Features + Regime + Alpha) → [DSAC Trader] → Actions
    ```
    
    Args:
        lob_input_dim: Dimension of LOB features
        action_dim: Dimension of action space
        mamba_config: Configuration for Mamba extractor
        llm_config: Configuration for LLM analyst
        dsac_config: Configuration for DSAC trader
        alpha_config: Configuration for alpha model
        use_alpha: Whether to include alpha signals in observation
        device: Torch device
    """
    
    def __init__(
        self,
        lob_input_dim: int = 40,
        action_dim: int = 3,  # buy, sell, hold amounts
        mamba_config: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        dsac_config: Optional[Dict[str, Any]] = None,
        alpha_config: Optional[Dict[str, Any]] = None,
        use_alpha: bool = True,
        device: str = "cuda",
    ):
        self.device = device
        self.action_dim = action_dim
        self.use_alpha = use_alpha
        
        # Default configs
        mamba_config = mamba_config or {}
        llm_config = llm_config or {}
        dsac_config = dsac_config or {}
        alpha_config = alpha_config or {}
        
        # Initialize components
        self.mamba_extractor = MambaFeatureExtractor(
            input_dim=lob_input_dim,
            **mamba_config,
        ).to(device)
        
        self.llm_analyst = LLMAnalyst(**llm_config)
        
        # Alpha model (SimpleAlphaModel as lightweight alternative to TimesFM)
        if use_alpha:
            self.alpha_model = SimpleAlphaModel(
                input_dim=alpha_config.get("input_dim", 1),
                hidden_dim=alpha_config.get("hidden_dim", 64),
                n_layers=alpha_config.get("n_layers", 2),
                prediction_horizons=alpha_config.get("prediction_horizons", [10, 20, 50, 100]),
            ).to(device)
            self.alpha_model.eval()
        else:
            self.alpha_model = None
        
        # DSAC observation = Mamba output + regime embedding + alpha
        mamba_dim = self.mamba_extractor.get_output_dim()
        regime_dim = 4  # 3 one-hot + 1 confidence
        alpha_dim = 4 if use_alpha else 0
        total_obs_dim = mamba_dim + regime_dim + alpha_dim
        
        self.dsac_trader = DSACTrader(
            observation_dim=total_obs_dim,  # This is the raw obs dim the TQC sees
            action_dim=action_dim,
            regime_dim=0,  # Regime is already baked into observation via wrapper
            **dsac_config,
        )
        
        # Track the wrapper's env for evaluation
        self._wrapped_env = None
        
        # Current regime
        self.current_regime: Optional[RegimeSignal] = None
        
    def reset(self):
        """Reset agent state for new episode."""
        self.current_regime = None
        
    def train(
        self,
        env: gym.Env,
        total_timesteps: int,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        news_articles: Optional[List[str]] = None,
        callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Train the hierarchical agent.
        
        Args:
            env: Training environment
            total_timesteps: Total training steps
            eval_freq: Evaluation frequency
            n_eval_episodes: Episodes per evaluation
            news_articles: Optional news for regime signal (pre-computed once)
            callback: Optional training callback
            
        Returns:
            Training metrics and history
        """
        # Pre-compute regime signal from news if provided (much faster than per-step LLM)
        if news_articles is not None:
            self.llm_analyst.initialize()
            self.current_regime = self.llm_analyst.analyze(news_articles)
            print(f"[HierarchicalAgent] Pre-computed regime: {self.current_regime.regime} "
                  f"(confidence={self.current_regime.confidence:.2f})")
            # Inject regime into environment so the wrapper can access it
            env.current_regime = self.current_regime
        
        # Initialize DSAC with wrapped environment
        device_for_wrapper = "cpu"  # Wrapper ops are light, keep on CPU to avoid SB3 issues
        
        hierarchical_env = HierarchicalEnvWrapper(
            env=env,
            mamba_extractor=self.mamba_extractor.cpu(),
            llm_analyst=self.llm_analyst,
            alpha_model=self.alpha_model.cpu() if self.alpha_model is not None else None,
            device=device_for_wrapper,
        )
        self._wrapped_env = hierarchical_env
        
        # Initialize DSAC trader with the wrapped environment
        self.dsac_trader.initialize(hierarchical_env)
        
        # Training loop
        print(f"[HierarchicalAgent] Training DSAC trader for {total_timesteps} steps")
        print(f"[HierarchicalAgent] Observation space: {hierarchical_env.observation_space.shape}")
        
        # Uses SB3's built-in learn method
        self.dsac_trader.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
        )
        
        # Move models back to original device
        self.mamba_extractor.to(self.device)
        if self.alpha_model is not None:
            self.alpha_model.to(self.device)
        
        history = {
            "status": "completed",
            "total_timesteps": total_timesteps,
            "obs_dim": hierarchical_env.total_dim,
        }
        
        return history
    
    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int = 10,
        deterministic: bool = True,
        news_articles: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the agent using the same wrapper pipeline as training.
        
        Args:
            env: Evaluation environment (unwrapped)
            n_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic policy
            news_articles: Optional news for regime detection
            
        Returns:
            Dictionary with evaluation metrics (reward, sharpe, etc.)
        """
        if self.dsac_trader.model is None:
            raise RuntimeError("Agent must be trained before evaluation")
        
        # Pre-compute regime if news provided
        if news_articles is not None:
            self.llm_analyst.initialize()
            self.current_regime = self.llm_analyst.analyze(news_articles)
            env.current_regime = self.current_regime
        
        # Build the same wrapper used during training
        device_for_wrapper = "cpu"
        eval_env = HierarchicalEnvWrapper(
            env=env,
            mamba_extractor=self.mamba_extractor.cpu(),
            llm_analyst=self.llm_analyst,
            alpha_model=self.alpha_model.cpu() if self.alpha_model is not None else None,
            device=device_for_wrapper,
        )
        
        rewards = []
        portfolio_values = []
        
        for ep in range(n_episodes):
            obs, info = eval_env.reset()
            self.reset()
            
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.dsac_trader.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                
                episode_reward += reward
                done = terminated or truncated
            
            rewards.append(episode_reward)
            portfolio_values.append(info.get("portfolio_value", 0))
        
        # Move models back
        self.mamba_extractor.to(self.device)
        if self.alpha_model is not None:
            self.alpha_model.to(self.device)
        
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "mean_portfolio": float(np.mean(portfolio_values)),
        }
    
    def save(self, path: str):
        """Save all components."""
        state = {
            "mamba_state_dict": self.mamba_extractor.state_dict(),
        }
        if self.alpha_model is not None:
            state["alpha_state_dict"] = self.alpha_model.state_dict()
        
        torch.save(state, f"{path}_mamba.pt")
        self.dsac_trader.save(f"{path}_dsac")
        
    def load(self, path: str):
        """Load all components."""
        checkpoint = torch.load(f"{path}_mamba.pt", map_location=self.device)
        self.mamba_extractor.load_state_dict(checkpoint["mamba_state_dict"])
        
        if self.alpha_model is not None and "alpha_state_dict" in checkpoint:
            self.alpha_model.load_state_dict(checkpoint["alpha_state_dict"])
        
        self.dsac_trader.load(f"{path}_dsac")
