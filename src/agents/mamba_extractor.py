"""
Mamba Feature Extractor for Limit Order Book Processing

Replaces traditional TCN/LSTM with Mamba SSM for "infinite context" 
processing of high-frequency trading data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal

# Try to import Mamba, fall back to LSTM if not available
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("[MambaExtractor] mamba-ssm not installed, using LSTM fallback")


class MambaFeatureExtractor(nn.Module):
    """
    Mamba-based feature extractor for LOB (Limit Order Book) data.
    
    Uses State Space Models (SSM) to process sequential market data
    with O(n) complexity instead of O(n²) for Transformers.
    
    Args:
        input_dim: Dimension of LOB features (144 for FI-2010 full features)
        d_model: Hidden dimension for Mamba layers
        n_layers: Number of Mamba blocks
        d_state: SSM state dimension
        expand: Expansion factor for inner MLP
        dropout: Dropout rate
        backend: 'mamba' or 'lstm' (auto-selects based on availability)
    """
    
    def __init__(
        self,
        input_dim: int = 144,  # FI-2010 has 144 features
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        expand: int = 2,
        dropout: float = 0.1,
        backend: Literal["mamba", "lstm", "auto"] = "auto",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Determine backend
        if backend == "auto":
            self.backend = "mamba" if MAMBA_AVAILABLE else "lstm"
        else:
            self.backend = backend
            
        if self.backend == "mamba" and not MAMBA_AVAILABLE:
            print("[MambaExtractor] Mamba requested but not available, using LSTM")
            self.backend = "lstm"
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Build backbone based on backend
        if self.backend == "mamba":
            self._build_mamba_backbone(d_model, n_layers, d_state, expand)
        else:
            self._build_lstm_backbone(d_model, n_layers, dropout)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        print(f"[MambaExtractor] Initialized with backend='{self.backend}', "
              f"input_dim={input_dim}, d_model={d_model}, n_layers={n_layers}")
    
    def _build_mamba_backbone(self, d_model: int, n_layers: int, d_state: int, expand: int):
        """Build Mamba SSM backbone."""
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, expand=expand)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
    
    def _build_lstm_backbone(self, d_model: int, n_layers: int, dropout: float):
        """Build LSTM backbone as fallback."""
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False,
        )
        
    def forward(
        self, 
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process LOB sequence through Mamba/LSTM layers.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            hidden: Optional hidden state for recurrent processing (LSTM only)
            
        Returns:
            features: Extracted features (batch, seq_len, d_model)
            final_state: Final hidden state (batch, d_model)
        """
        # Project input to model dimension
        x = self.input_proj(x)
        x = self.dropout(x)
        
        if self.backend == "mamba":
            features = self._forward_mamba(x)
        else:
            features = self._forward_lstm(x, hidden)
        
        # Normalize output
        features = self.norm(features)
        
        # Get final state (last timestep)
        final_state = features[:, -1, :]
        
        return features, final_state
    
    def _forward_mamba(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Mamba layers with residual connections."""
        for layer, norm in zip(self.layers, self.layer_norms):
            residual = x
            x = norm(x)
            x = layer(x)
            x = x + residual
        return x
    
    def _forward_lstm(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Forward pass through LSTM backbone."""
        if hidden is None:
            features, _ = self.lstm(x)
        else:
            features, _ = self.lstm(x, hidden)
        return features
    
    def get_output_dim(self) -> int:
        """Return the output dimension for downstream modules."""
        return self.d_model


class LOBClassifier(nn.Module):
    """
    Complete LOB prediction model with feature extractor and classification head.
    
    Used for mid-price movement prediction (up/stable/down) from LOB data.
    This is useful for testing the feature extractor on FI-2010 benchmark.
    
    Args:
        input_dim: Dimension of LOB features
        d_model: Hidden dimension
        n_layers: Number of encoder layers
        n_classes: Number of output classes (3 for up/stable/down)
        dropout: Dropout rate
        backend: 'mamba', 'lstm', or 'auto'
    """
    
    def __init__(
        self,
        input_dim: int = 144,
        d_model: int = 128,
        n_layers: int = 4,
        n_classes: int = 3,
        dropout: float = 0.1,
        backend: Literal["mamba", "lstm", "auto"] = "auto",
    ):
        super().__init__()
        
        self.feature_extractor = MambaFeatureExtractor(
            input_dim=input_dim,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            backend=backend,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )
        
        self.n_classes = n_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            
        Returns:
            logits: Classification logits (batch, n_classes)
        """
        _, final_state = self.feature_extractor(x)
        logits = self.classifier(final_state)
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class labels."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
