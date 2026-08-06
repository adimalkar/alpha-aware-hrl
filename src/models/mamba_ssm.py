"""
Mamba State Space Model Components

Core neural network modules for the Mamba-based feature extractor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MambaBlock(nn.Module):
    """
    Single Mamba block with selective state space mechanism.
    
    This is a simplified implementation. For production, use the
    official mamba-ssm library which has optimized CUDA kernels.
    
    The Mamba architecture uses:
    - Linear projections for input/output
    - 1D convolution for local context
    - Selective SSM for long-range dependencies
    
    Args:
        d_model: Model dimension
        d_state: SSM state dimension
        d_conv: Convolution width
        expand: Expansion factor for inner dimension
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        self.d_inner = d_model * expand
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution layer
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        
        # SSM parameters
        # These would be learned/selective in real Mamba
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)
        
        # SSM state matrices (simplified)
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Mamba block.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        residual = x
        
        x = self.norm(x)
        
        # Project input
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]  # Trim to original length
        x = x.transpose(1, 2)
        
        x = F.silu(x)
        
        # Simplified SSM (real Mamba uses selective scan)
        # This is a placeholder - use official mamba-ssm for proper implementation
        y = self._simple_ssm(x)
        
        # Gate with z
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output + residual
    
    def _simple_ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simplified SSM computation (placeholder for selective scan).
        
        Real Mamba uses input-dependent A, B, C matrices.
        """
        # Just use a simple linear transformation as placeholder
        # The real SSM would do: y = C @ (A @ h + B @ x)
        return x * torch.sigmoid(self.D)


class MambaEncoder(nn.Module):
    """
    Stack of Mamba blocks for sequence encoding.
    
    Args:
        input_dim: Input feature dimension
        d_model: Model dimension
        n_layers: Number of Mamba blocks
        d_state: SSM state dimension
        d_conv: Convolution width
        expand: Expansion factor
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        return_all_hiddens: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input sequence with Mamba layers.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            return_all_hiddens: Whether to return all layer outputs
            
        Returns:
            features: Encoded features (batch, seq_len, d_model)
            final_state: Final hidden state (batch, d_model)
        """
        x = self.input_proj(x)
        x = self.dropout(x)
        
        hiddens = []
        for layer in self.layers:
            x = layer(x)
            if return_all_hiddens:
                hiddens.append(x)
        
        x = self.norm(x)
        final_state = x[:, -1, :]  # Last timestep
        
        return x, final_state


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network block for baseline comparison.
    
    Used in ablation studies to compare against Mamba.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size, padding=padding, dilation=dilation
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, seq_len)"""
        residual = self.residual(x)
        
        x = self.conv1(x)
        x = x[:, :, :-self.conv1.padding[0]]  # Causal trim
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = x[:, :, :-self.conv2.padding[0]]  # Causal trim
        x = self.relu(x)
        x = self.dropout(x)
        
        return self.relu(x + residual)


class TCNEncoder(nn.Module):
    """
    TCN Encoder for baseline comparison in ablation studies.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** i
            self.layers.append(
                TCNBlock(hidden_dim, hidden_dim, kernel_size, dilation, dropout)
            )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, seq_len, input_dim)"""
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = x.transpose(1, 2)  # (batch, seq_len, hidden_dim)
        x = self.norm(x)
        
        final_state = x[:, -1, :]
        
        return x, final_state
