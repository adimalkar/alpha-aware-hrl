#!/usr/bin/env python3
"""
Cluster Execution & Scale Training (Phase 4)

This script is designed to run on a high-performance GPU cluster (e.g., Slurm or AWS).
It trains the Alpha-Aware Hierarchical RL agent on the real FI-2010 historical 
LOB dataset, fully integrated with the precomputed LLM Regime signals.

Key Features:
- Vectorized Environments (DummyVecEnv/SubprocVecEnv) for faster data collection.
- Callbacks for periodic evaluation on the FI-2010 Test set.
- Model checkpointing and Tensorboard logging for cluster monitoring.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib import TQC

sys.path.insert(0, '.')
from src.utils.data_loader import FI2010DataLoader
from src.envs.historical_lob_env import HistoricalLOBEnv
from src.envs.hierarchical_wrapper import HierarchicalEnvWrapper
from src.agents.mamba_extractor import MambaFeatureExtractor
from src.agents.llm_analyst import LLMAnalyst

def make_env(env_kwargs, mamba_kwargs, rank, seed=0):
    """
    Utility function for multiplexed multiprocessing.
    Creates a callable that instantiates the wrapped environment.
    """
    def _init():
        # 1. Base Historical Env
        env = HistoricalLOBEnv(**env_kwargs)
        env.reset(seed=seed + rank)
        
        # 2. Extractors
        # We instantiate Mamba and LLM inside the process to avoid IPC tensor issues
        mamba = MambaFeatureExtractor(**mamba_kwargs)
        llm = LLMAnalyst(device="cpu") # Not actually used for inference due to precomputation
        
        # 3. Wrapper
        wrapped_env = HierarchicalEnvWrapper(
            env=env,
            mamba_extractor=mamba,
            llm_analyst=llm,
            alpha_model=None, # SimpleAlphaModel could go here
            device="cuda" # Ensure Mamba extracts on GPU
        )
        return wrapped_env
    return _init

def parse_args():
    parser = argparse.ArgumentParser(description="Cluster Training Script")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of vectorized environments")
    parser.add_argument("--save-dir", type=str, default="experiments/cluster_run", help="Output directory")
    parser.add_argument("--vec-env", type=str, default="dummy", choices=["dummy", "subproc"])
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 60)
    print("PHASE 4: CLUSTER TRAINING EXECUTION")
    print("=" * 60)
    print(f"Target Timesteps: {args.timesteps}")
    print(f"Number of Envs:   {args.n_envs} ({args.vec_env})")
    print(f"Random Seed:      {args.seed}")
    print(f"Output Directory: {args.save_dir}")
    
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize Data Loader
    # Note: Loading data before fork() saves RAM on Linux via copy-on-write
    print("\nLoading FI-2010 Train/Test Splits...")
    loader = FI2010DataLoader(data_dir="data/fi2010/FI2010", horizon_idx=0)
    loader.load("train")
    loader.load("test")
    
    # Environment Configurations
    train_env_kwargs = {
        "data_loader": loader,
        "split": "train",
        "episode_length": 5000, # Truncate episodes to force regular resets
        "starting_cash": 100000.0,
        "transaction_fee": 0.0001,
        "regime_path": "data/precomputed_regimes/train_combined.npy"
    }
    
    test_env_kwargs = {
        "data_loader": loader,
        "split": "test",
        "episode_length": 10000,
        "starting_cash": 100000.0,
        "transaction_fee": 0.0001,
        "regime_path": "data/precomputed_regimes/test_combined.npy"
    }
    
    mamba_kwargs = {
        "input_dim": 144, # Real FI-2010 dimensions
        "d_model": 128,
        "n_layers": 2,
        "backend": "lstm" # Fallback since mamba_ssm isn't compiled in this env
    }
    
    # 2. Create Vectorized Environments
    print("\nCreating Vectorized Environments...")
    env_fns = [make_env(train_env_kwargs, mamba_kwargs, i, args.seed) for i in range(args.n_envs)]
    
    if args.vec_env == "subproc":
        train_env = SubprocVecEnv(env_fns)
    else:
        train_env = DummyVecEnv(env_fns)
        
    eval_env = DummyVecEnv([make_env(test_env_kwargs, mamba_kwargs, 0, args.seed + 999)])
    
    # 3. Setup Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / "models"),
        log_path=str(save_path / "logs"),
        eval_freq=max(10000 // args.n_envs, 1),
        deterministic=True,
        render=False
    )
    
    # 4. Initialize TQC Agent
    print("\nInitializing TQC (Distributional RL) Agent...")
    model = TQC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=5000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        top_quantiles_to_drop_per_net=2,
        policy_kwargs=dict(n_quantiles=25, net_arch=[256, 256]),
        tensorboard_log=str(save_path / "tensorboard"),
        verbose=1,
        seed=args.seed,
        device="cuda"
    )
    
    # 5. Execute Training
    print(f"\nStarting training for {args.timesteps} timesteps...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=eval_callback,
            tb_log_name="TQC_FI2010_Hierarchical",
            reset_num_timesteps=True
        )
        print("\n✅ Training Complete!")
        
        # Save Final Model
        final_model_path = save_path / "models" / "final_model"
        model.save(str(final_model_path))
        print(f"Final model saved to {final_model_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
        interrupted_path = save_path / "models" / "interrupted_model"
        model.save(str(interrupted_path))
        print(f"Saved to {interrupted_path}")

if __name__ == "__main__":
    main()
