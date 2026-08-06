"""
Wrapper to connect ABIDES market simulation to Hierarchical Agent components.

Transforms the raw Limit Order Book (LOB) state into a combined feature
representation utilizing the MambaFeatureExtractor, the LLM Analyst regime,
and optionally alpha signals from TimesFM or SimpleAlphaModel.
"""

import numpy as np
import torch
import gymnasium as gym
from typing import Dict, Any, Tuple, Optional


class HierarchicalEnvWrapper(gym.ObservationWrapper):
    """
    Gymnasium Observation Wrapper for Hierarchical trading.
    
    This wrapper intercepts the raw LOB state from the underlying ABIDESEnv,
    passes it through the Mamba feature extractor to get a compressed state
    representation, concatenates it with the current market regime embedding
    and optional alpha signal features.
    
    This produces a flat vector observation space compatible with standard
    SB3/SB3-Contrib reinforcement learning algorithms (e.g., TQC/DSAC).
    
    Args:
        env: The underlying Gymnasium environment (e.g., ABIDESEnv)
        mamba_extractor: The MambaFeatureExtractor instance
        llm_analyst: The LLMAnalyst instance for regime embeddings
        alpha_model: Optional alpha signal model (TimesFMWrapper or SimpleAlphaModel)
        seq_len: Length of the LOB sequence window to maintain
        device: Torch device for inference
    """
    
    def __init__(
        self,
        env: gym.Env,
        mamba_extractor: Any,
        llm_analyst: Any,
        alpha_model: Optional[Any] = None,
        seq_len: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(env)
        self.mamba_extractor = mamba_extractor
        self.llm_analyst = llm_analyst
        self.alpha_model = alpha_model
        self.seq_len = seq_len
        self.device = device
        
        # Ensure PyTorch modules are on the correct device
        if hasattr(self.mamba_extractor, 'to'):
            self.mamba_extractor = self.mamba_extractor.to(self.device)
        if hasattr(self.alpha_model, 'to') and self.alpha_model is not None:
            self.alpha_model = self.alpha_model.to(self.device)
            
        # Determine the dimensions
        self.mamba_dim = mamba_extractor.get_output_dim()
        
        # Default regime is [0, 1, 0] + [confidence], shape=4
        self.regime_dim = 4
        
        # Alpha dimension: 4 horizons if alpha model provided, else 0
        self.alpha_dim = 4 if alpha_model is not None else 0
        
        self.total_dim = self.mamba_dim + self.regime_dim + self.alpha_dim
        
        # The new observation space is a flat vector
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.total_dim,),
            dtype=np.float32,
        )
        
        # Internal buffer for the sliding window of LOB states
        self.state_buffer = []
        
        # Price history buffer for alpha signal generation
        self.price_history = []
        self._price_context_len = 128  # Number of price ticks to keep for alpha
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and the sequence buffer."""
        obs, info = self.env.reset(**kwargs)
        
        # Handle precomputed regime if injected by the environment
        if "precomputed_regime" in info:
            pr = info["precomputed_regime"]
            from src.agents.llm_analyst import RegimeSignal
            self.env.unwrapped.current_regime = RegimeSignal(
                regime=pr["regime"],
                confidence=pr["confidence"],
                reasoning=pr.get("reasoning", "Precomputed")
            )
        
        # Fill the buffer with the initial observation to reach required seq_len
        self.state_buffer = [obs for _ in range(self.seq_len)]
        
        # Initialize price history with current price
        current_price = info.get("current_price", 100.0)
        self.price_history = [current_price] * self._price_context_len
        
        # Reset mamba hidden state (if applicable, e.g. LSTM fallback)
        if hasattr(self.mamba_extractor, 'hidden_state'):
            self.mamba_extractor.hidden_state = None
            
        return self.observation(obs, info), info
    
    def step(self, action):
        """Override step to pass info to observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track price history for alpha model
        current_price = info.get("current_price", 100.0)
        self.price_history.append(current_price)
        if len(self.price_history) > self._price_context_len:
            self.price_history.pop(0)
        
        # Handle precomputed regime if injected by the environment
        if "precomputed_regime" in info:
            pr = info["precomputed_regime"]
            from src.agents.llm_analyst import RegimeSignal
            self.env.unwrapped.current_regime = RegimeSignal(
                regime=pr["regime"],
                confidence=pr["confidence"],
                reasoning=pr.get("reasoning", "Precomputed")
            )
            
        return self.observation(obs, info), reward, terminated, truncated, info
        
    def observation(self, observation: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        """
        Process a single raw observation from the environment into the 
        combined (Mamba + Regime + Alpha) feature vector.
        """
        # 1. Update sliding window
        self.state_buffer.append(observation)
        if len(self.state_buffer) > self.seq_len:
            self.state_buffer.pop(0)
            
        # 2. Convert to tensor: shape (batch=1, seq_len, features)
        lob_sequence = np.array(self.state_buffer, dtype=np.float32)
        lob_tensor = torch.tensor(lob_sequence).unsqueeze(0).to(self.device)
        
        # 3. Extract features using Mamba
        with torch.no_grad():
            self.mamba_extractor.eval()
            features, final_state = self.mamba_extractor(lob_tensor)
            mamba_feats = final_state.cpu().numpy().squeeze()
            
        # 4. Get current macro regime embedding
        if hasattr(self.env, "current_regime") and self.env.current_regime is not None:
            regime_embedding = self.llm_analyst.get_regime_embedding(self.env.current_regime)
        else:
            from src.agents.llm_analyst import RegimeSignal
            default_signal = RegimeSignal(regime=1, confidence=0.8, reasoning="Default wrapper regime")
            regime_embedding = self.llm_analyst.get_regime_embedding(default_signal)
            
        regime_feats = regime_embedding.cpu().numpy().squeeze()
        
        # 5. Get alpha signal if model is available
        parts = [mamba_feats, regime_feats]
        
        if self.alpha_model is not None and len(self.price_history) >= 10:
            try:
                with torch.no_grad():
                    price_tensor = torch.tensor(
                        [self.price_history], dtype=torch.float32, device=self.device
                    )
                    current_price = self.price_history[-1]
                    
                    # SimpleAlphaModel: forward() returns (features, predictions)
                    if hasattr(self.alpha_model, 'get_alpha_signal'):
                        alpha = self.alpha_model.get_alpha_signal(price_tensor, current_price)
                    else:
                        _, alpha = self.alpha_model(price_tensor)
                    
                    alpha_feats = alpha.cpu().numpy().squeeze()
                    
                    # Ensure correct shape (4,) for the 4 horizons
                    if alpha_feats.ndim == 0:
                        alpha_feats = np.zeros(self.alpha_dim, dtype=np.float32)
                    elif len(alpha_feats) != self.alpha_dim:
                        alpha_feats = alpha_feats[:self.alpha_dim] if len(alpha_feats) > self.alpha_dim else \
                            np.pad(alpha_feats, (0, self.alpha_dim - len(alpha_feats)))
                    
                    parts.append(alpha_feats)
            except Exception:
                # Fallback to zeros if alpha computation fails
                parts.append(np.zeros(self.alpha_dim, dtype=np.float32))
        elif self.alpha_dim > 0:
            parts.append(np.zeros(self.alpha_dim, dtype=np.float32))
        
        # 6. Concatenate together
        full_obs = np.concatenate(parts, axis=0).astype(np.float32)
        
        return full_obs
