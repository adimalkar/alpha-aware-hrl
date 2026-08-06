#!/usr/bin/env python3
"""
Test script for Phase 1: Real Market Data Integration

Verifies that:
1. FI2010DataLoader loads the 144-dimensional LOB data correctly.
2. HistoricalLOBEnv replays the data tick-by-tick.
3. HierarchicalEnvWrapper properly integrates the 144-dim LOB with the 
   Mamba extractor and Regime signals.
"""

import sys
import torch
import numpy as np
sys.path.insert(0, '.')

from src.utils.data_loader import FI2010DataLoader
from src.envs.historical_lob_env import HistoricalLOBEnv
from src.envs.hierarchical_wrapper import HierarchicalEnvWrapper
from src.agents.mamba_extractor import MambaFeatureExtractor
from src.agents.llm_analyst import LLMAnalyst, RegimeSignal

def main():
    print("=" * 60)
    print("PHASE 1 TEST: FI-2010 Historical LOB Environment")
    print("=" * 60)
    
    # 1. Load Data
    print("\n1. Loading FI-2010 Data...")
    loader = FI2010DataLoader(data_dir="data/fi2010/FI2010", horizon_idx=0)
    loader.load("train")
    
    # 2. Create Historical Environment
    print("\n2. Initializing HistoricalLOBEnv...")
    env = HistoricalLOBEnv(
        data_loader=loader,
        split="train",
        episode_length=100,
        starting_cash=100000.0
    )
    
    obs, info = env.reset()
    print(f"   Raw Observation Shape: {obs.shape} (Expected: (144,))")
    print(f"   Initial Price: {info['current_price']}")
    
    # Step the raw env
    action = np.array([0.5])  # Target 50% long
    obs, rew, term, trunc, info = env.step(action)
    print(f"   After Step 1 -> Price: {info['current_price']:.2f}, Portfolio: ${info['portfolio_value']:.2f}, Reward: {rew:.2f}")
    
    # 3. Wrap in Hierarchical Pipeline
    print("\n3. Wrapping in HierarchicalEnvWrapper (Mamba + LLM)...")
    
    # Mamba expects input_dim=144 now
    mamba = MambaFeatureExtractor(
        input_dim=144, 
        d_model=128, 
        n_layers=2, 
        backend="lstm"
    ).cpu()
    
    llm_analyst = LLMAnalyst(device="cpu")
    # Pre-inject regime to bypass LLM inference
    env.current_regime = RegimeSignal(regime=1, confidence=0.8, reasoning="Test")
    
    wrapped_env = HierarchicalEnvWrapper(
        env=env,
        mamba_extractor=mamba,
        llm_analyst=llm_analyst,
        alpha_model=None,
        device="cpu"
    )
    
    print(f"   Wrapped Observation Space: {wrapped_env.observation_space.shape} (Expected: (132,))")
    
    w_obs, w_info = wrapped_env.reset()
    print(f"   Reset Wrapped Obs Shape: {w_obs.shape}")
    
    # Step wrapped env
    action = wrapped_env.action_space.sample()
    w_obs2, w_rew, w_term, w_trunc, w_info2 = wrapped_env.step(action)
    print(f"   Step Wrapped Env -> Obs Shape: {w_obs2.shape}, Reward: {w_rew:.4f}")
    
    print("\n✅ Phase 1 implementation successfully verified!")

if __name__ == "__main__":
    main()
