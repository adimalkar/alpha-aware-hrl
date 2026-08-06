"""
TimesFM Wrapper for Alpha Signal Generation

Wraps Google's Time Series Foundation Model (TimesFM) as a frozen
feature extractor for price prediction and alpha signal generation.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List


class TimesFMWrapper(nn.Module):
    """
    Wrapper for TimesFM foundation model.
    
    TimesFM is a pre-trained time series model from Google Research
    that can predict future values and extract useful features.
    
    We use it as a FROZEN feature extractor - no fine-tuning.
    This is the "Foundation Model" approach mentioned in the paper.
    
    Args:
        model_name: TimesFM checkpoint name
        prediction_horizons: List of prediction horizons
        device: Torch device
    """
    
    def __init__(
        self,
        model_name: str = "google/timesfm-1.0-200m-pytorch",
        prediction_horizons: List[int] = [10, 20, 50, 100],
        device: str = "cuda",
    ):
        super().__init__()
        
        self.model_name = model_name
        self.prediction_horizons = prediction_horizons
        self.device = device
        
        self.model = None
        self._initialized = False
        
        # Feature projection (TimesFM output to our dimension)
        # Will be initialized after model loading
        self.feature_proj = None
        
    def initialize(self):
        """Lazy initialization of TimesFM model."""
        import timesfm
        
        # Determine device for TimesFM (0 for cuda:0, -1 for cpu/metal)
        backend = "gpu" if "cuda" in self.device else "cpu"
        
        # Initialize the TimesFm model instance.
        self.model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                context_len=512,       # Max context the model can swallow
                horizon_len=max(self.prediction_horizons), # We need predictions out to the max horizon
                input_patch_len=32,
                output_patch_len=128,
                num_layers=20,
                model_dims=1280,
                backend=backend,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self.model_name
            )
        )
        
        # Placeholder feature projection
        # TimesFM typically outputs 768-dim features
        self.feature_proj = nn.Linear(768, 128).to(self.device)
        
        self._initialized = True
        print(f"[TimesFM] Initialized with {self.model_name}")
        
    def forward(
        self,
        price_history: torch.Tensor,
        return_predictions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract features and optionally predictions from price history.
        
        Args:
            price_history: Price series (batch, seq_len)
            return_predictions: Whether to return price predictions
            
        Returns:
            features: Extracted features (batch, feature_dim)
            predictions: Optional price predictions (batch, n_horizons)
        """
        if not self._initialized:
            self.initialize()
        
        batch_size = price_history.shape[0]
        
        import timesfm
        
        # TimesFM expects input in shape (batch_size, context_len) as numpy arrays or lists
        # We assume price_history is (batch, seq_len)
        price_np = price_history.cpu().numpy()
        
        # Create zero frequencies (since LOB mid-price is high-frequency, frequency handling is agnostic)
        frequencies = [0] * batch_size
        
        # 1. Provide prediction arrays
        predictions = None
        if return_predictions:
            # forecast() outputs a tuple: (forecast_values, forecast_quantiles)
            # We want point forecasts. Result shape is (batch_size, horizon_len)
            point_forecasts, _ = self.model.forecast(
                inputs=price_np.tolist(),
                freq=frequencies
            )
            
            # Select specific horizons required by our RL state space
            # Note: prediction horizons are usually 1-indexed (e.g., 10 ticks ahead)
            # So index 9 is exactly horizon 10.
            selected_preds = []
            for h in self.prediction_horizons:
                idx = min(h - 1, point_forecasts.shape[1] - 1)
                selected_preds.append(point_forecasts[:, idx:idx+1])
                
            predictions = np.concatenate(selected_preds, axis=1) # (batch, len(horizons))
            predictions = torch.tensor(predictions, device=self.device, dtype=torch.float32)

        # 2. Extract intermediate features
        # The timesfm python API doesn't cleanly expose the internal hidden states 
        # required to freeze a representation vector out-of-the-box like transformers do.
        # Instead, we will simulate a robust feature representation from the dense predictions
        # to feed the RL module, which gives it long-horizon macro context.
        
        # We'll take the first 128 elements of the point forecasts (or pad/trim)
        # and project it to our expected feature dimension.
        dense_forecasts, _ = self.model.forecast(
             inputs=price_np.tolist(),
             freq=frequencies
        )
        dense_tensor = torch.tensor(dense_forecasts, device=self.device, dtype=torch.float32)
        
        # Trim or pad to 768 to match the feature projection head
        target_len = 768
        if dense_tensor.shape[1] >= target_len:
             m_features = dense_tensor[:, :target_len]
        else:
             pad_size = target_len - dense_tensor.shape[1]
             m_features = torch.nn.functional.pad(dense_tensor, (0, pad_size))
             
        # Project raw macro context into our normalized feature size (128)
        features = self.feature_proj(m_features)
        
        return features, predictions
    
    def get_alpha_signal(
        self,
        price_history: torch.Tensor,
        current_price: float,
    ) -> torch.Tensor:
        """
        Generate alpha signal (expected return direction).
        
        The alpha signal indicates whether the model predicts
        price increase (positive) or decrease (negative).
        
        Args:
            price_history: Historical prices
            current_price: Current market price
            
        Returns:
            alpha: Alpha signal for each horizon (batch, n_horizons)
        """
        _, predictions = self.forward(price_history, return_predictions=True)
        
        if predictions is None:
            return torch.zeros(
                price_history.shape[0], len(self.prediction_horizons),
                device=self.device
            )
        
        # Alpha = (predicted - current) / current
        alpha = (predictions - current_price) / current_price
        
        return alpha
    
    def get_feature_dim(self) -> int:
        """Return output feature dimension."""
        return 128


class SimpleAlphaModel(nn.Module):
    """
    Simple baseline alpha model for comparison.
    
    Uses LSTM to predict future price direction.
    Used when TimesFM is not available or for ablation studies.
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        n_layers: int = 2,
        prediction_horizons: List[int] = [10, 20, 50, 100],
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        
        self.prediction_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1)
            for _ in prediction_horizons
        ])
        
        self.feature_proj = nn.Linear(hidden_dim, 128)
        
    def forward(
        self,
        price_history: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            price_history: (batch, seq_len) or (batch, seq_len, 1)
            
        Returns:
            features: (batch, 128)
            predictions: (batch, n_horizons)
        """
        if price_history.dim() == 2:
            price_history = price_history.unsqueeze(-1)
        
        lstm_out, (h_n, _) = self.lstm(price_history)
        
        # Last hidden state
        hidden = h_n[-1]
        
        features = self.feature_proj(hidden)
        
        predictions = torch.cat([
            head(hidden) for head in self.prediction_heads
        ], dim=-1)
        
        return features, predictions
