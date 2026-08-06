#!/usr/bin/env python3
"""
Test Script for TimesFM Alpha Signal Generator (Step 6)

This script validates that the `TimesFMWrapper` successfully loads
the Google TimesFM checkpoint, processes a synthetic price history,
and outputs properly formatted Alpha signals for the specified horizons.
"""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from src.models.timesfm_wrapper import TimesFMWrapper

def main():
    print("="*70)
    print("Testing TimesFM Feature Wrapper (Step 6)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. Initialize Wrapper
    print("\nInitializing TimesFM Wrapper...")
    wrapper = TimesFMWrapper(
        model_name="google/timesfm-1.0-200m-pytorch",
        prediction_horizons=[10, 20, 50, 100],
        device=device,
    )
    
    # 2. Create Synthetic Data
    # LOB mid-prices usually fluctuate around a baseline.
    batch_size = 4
    context_len = 512
    
    print(f"\nGenerating {batch_size} synthetic price histories of length {context_len}...")
    baseline = 150.0
    noise = np.random.randn(batch_size, context_len) * 0.5
    trend = np.linspace(0, 2, context_len)[None, :]
    
    # Synthetic prices
    price_history = baseline + noise + trend
    current_prices = price_history[:, -1]
    
    price_tensor = torch.tensor(price_history, dtype=torch.float32, device=device)
    
    # 3. Test Feature Extraction
    print("\nTesting: feature extraction (forward)...")
    try:
        features, predictions = wrapper(price_tensor, return_predictions=True)
        print(f"  Feature shape: {features.shape} (Expected: ({batch_size}, 128))")
        print(f"  Predictions shape: {predictions.shape} (Expected: ({batch_size}, 4))")
        
        assert features.shape == (batch_size, 128), "Feature shape mismatch!"
        assert predictions.shape == (batch_size, 4), "Predictions shape mismatch!"
        assert not torch.isnan(features).any(), "Features contain NaNs!"
        assert not torch.isnan(predictions).any(), "Predictions contain NaNs!"
        print("  ✅ Feature extraction passed.")
    except Exception as e:
        print(f"  ❌ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # 4. Test Alpha Signal Generation
    print("\nTesting: Alpha signal generation...")
    try:
        alpha = wrapper.get_alpha_signal(price_tensor, current_prices[0])
        print(f"  Alpha shape: {alpha.shape} (Expected: ({batch_size}, 4))")
        
        # Display sample alpha for batch 0
        sample_alpha = alpha[0].cpu().numpy()
        print(f"  Sample Alpha vector (Horizons 10, 20, 50, 100):")
        print(f"  {sample_alpha}")
        
        assert alpha.shape == (batch_size, 4), "Alpha shape mismatch!"
        assert not torch.isnan(alpha).any(), "Alpha contains NaNs!"
        print("  ✅ Alpha signal logic passed.")
    except Exception as e:
        print(f"  ❌ Alpha generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("\n" + "="*70)
    print("All TimesFM tests completed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()
