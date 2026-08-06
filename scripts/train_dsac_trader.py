#!/usr/bin/env python3
"""
Training Script for DSAC/TQC Trader (Step 5)

This script tests the integration of Mamba feature extraction with
the SB3-Contrib TQC reinforcement learning algorithm.
"""

import sys
sys.path.insert(0, '.')

import argparse
import torch
import os
from pathlib import Path

# Fix library path if libstdc++ is missing (common with Conda + SB3)
# We don't necessarily need it, but it prevents some warnings.

from src.envs.abides_wrapper import ABIDESEnv
from src.agents.hierarchical_agent import HierarchicalAgent

def main():
    parser = argparse.ArgumentParser(description="Train DSAC/TQC Trader")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--save_path", type=str, default="checkpoints/dsac_trader")
    args = parser.parse_args()
    
    print("="*70)
    print("Training DSAC/TQC Trader (Step 5 Integration Test)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. Initialize base environment (ABIDES Simulator mockup)
    print("\nInitializing base market environment...")
    base_env = ABIDESEnv(
        ticker="AAPL",
        starting_cash=100000.0,
        episode_length=200,  # Short episodes for testing
        market_impact=True,
    )
    
    # 2. Initialize Hierarchical Agent
    # This automatically instantiates MambaFeatureExtractor, LLMAnalyst, and DSACTrader.
    print("\nInitializing Hierarchical Agent architecture...")
    agent = HierarchicalAgent(
        lob_input_dim=40,  # Match ABIDESEnv observation_space
        action_dim=3,      # Default buy/sell/price
        mamba_config={"d_model": 128, "n_layers": 2, "backend": "mamba"},
        device=device,
    )
    
    # Ensure save directory exists
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    
    # 3. Train
    # The agent's train method wraps the base_env in HierarchicalEnvWrapper internally
    print(f"\nStarting training for {args.timesteps} total timesteps...")
    try:
        agent.train(env=base_env, total_timesteps=args.timesteps)
        
        # 4. Save
        print(f"\nSaving model to {args.save_path}...")
        agent.save(args.save_path)
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
