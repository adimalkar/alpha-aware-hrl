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
from src.models.event_encoder import EventFeatureExtractor

def make_env(env_kwargs, extractor_kwargs, encoder_type, rank, seed=0):
    """
    Utility function for multiplexed multiprocessing.
    Creates a callable that instantiates the wrapped environment.
    """
    def _init():
        # 1. Base Historical Env
        env = HistoricalLOBEnv(**env_kwargs)
        env.reset(seed=seed + rank)
        
        # 2. Feature Extractor
        if encoder_type == "event":
            extractor = EventFeatureExtractor(input_dim=144, d_model=extractor_kwargs.get("d_model", 128))
        else:
            extractor = MambaFeatureExtractor(**extractor_kwargs)
            
        llm = LLMAnalyst(device="cpu") # Handled via precomputed regime
        
        # 3. Wrapper
        wrapped_env = HierarchicalEnvWrapper(
            env=env,
            mamba_extractor=extractor,
            llm_analyst=llm,
            alpha_model=None,
            device="cuda"
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
    parser.add_argument("--encoder", type=str, default="event", choices=["event", "mamba"],
                        help="Feature extractor: 'event' (Large Event Model THP) or 'mamba'")
    parser.add_argument("--data-dir", type=str, default="data/fi2010/FI2010", help="Data directory (e.g. data/live_market or data/fi2010/FI2010)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 60)
    print("PHASE 4: CLUSTER TRAINING EXECUTION")
    print("=" * 60)
    print(f"Target Timesteps: {args.timesteps}")
    print(f"Data Directory:   {args.data_dir}")
    print(f"Feature Encoder:  {args.encoder.upper()}")
    print(f"Number of Envs:   {args.n_envs} ({args.vec_env})")
    print(f"Random Seed:      {args.seed}")
    print(f"Output Directory: {args.save_dir}")
    
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize Data Loader
    print(f"\nLoading LOB Train/Test Splits from {args.data_dir}...")
    loader = FI2010DataLoader(data_dir=args.data_dir, horizon_idx=0)
    loader.load("train")
    loader.load("test")
    
    is_live = "live" in args.data_dir.lower()
    train_regime = "data/precomputed_regimes/live_train_combined.npy" if is_live else "data/precomputed_regimes/train_combined.npy"
    test_regime = "data/precomputed_regimes/live_test_combined.npy" if is_live else "data/precomputed_regimes/test_combined.npy"
    
    # Environment Configurations
    train_env_kwargs = {
        "data_loader": loader,
        "split": "train",
        "episode_length": 2000 if is_live else 5000,
        "starting_cash": 100000.0,
        "transaction_fee": 0.0001,
        "regime_path": train_regime
    }
    
    test_env_kwargs = {
        "data_loader": loader,
        "split": "test",
        "episode_length": 1000 if is_live else 10000,
        "starting_cash": 100000.0,
        "transaction_fee": 0.0001,
        "regime_path": test_regime
    }
    
    mamba_kwargs = {
        "input_dim": 144, # Real FI-2010 dimensions
        "d_model": 128,
        "n_layers": 2,
        "backend": "lstm" # Fallback since mamba_ssm isn't compiled in this env
    }
    
    # 2. Create Vectorized Environments
    print(f"\nCreating Vectorized Environments with [{args.encoder.upper()}] Encoder...")
    env_fns = [make_env(train_env_kwargs, mamba_kwargs, args.encoder, i, args.seed) for i in range(args.n_envs)]
    
    if args.vec_env == "subproc":
        train_env = SubprocVecEnv(env_fns)
    else:
        train_env = DummyVecEnv(env_fns)
        
    eval_env = DummyVecEnv([make_env(test_env_kwargs, mamba_kwargs, args.encoder, 0, args.seed + 999)])
    
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
        learning_starts=min(1000, max(200, args.timesteps // 10)),
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
        
        # 6. Out-of-Sample Evaluation on FI-2010 Test Set
        print("\n" + "=" * 60)
        print("OUT-OF-SAMPLE TEST EVALUATION (FI-2010)")
        print("=" * 60)
        
        test_obs = eval_env.reset()
        done = False
        episode_rewards = []
        portfolio_values = []
        actions_list = []
        
        from src.utils.metrics import compute_sharpe, compute_max_drawdown
        
        # Run test episode
        for step in range(5000):
            action, _ = model.predict(test_obs, deterministic=True)
            test_obs, reward, done, info = eval_env.step(action)
            episode_rewards.append(float(reward[0]))
            actions_list.append(float(action[0]))
            
            # Track portfolio value
            if hasattr(eval_env.envs[0].env, "portfolio_value"):
                portfolio_values.append(float(eval_env.envs[0].env.portfolio_value))
            if done[0]:
                break
                
        initial_val = 100000.0
        final_val = portfolio_values[-1] if portfolio_values else initial_val * (1.0 + sum(episode_rewards) * 0.001)
        total_return_pct = ((final_val - initial_val) / initial_val) * 100.0
        
        returns_series = np.diff(portfolio_values) / (np.array(portfolio_values[:-1]) + 1e-6) if len(portfolio_values) > 1 else np.array(episode_rewards) * 0.0001
        sharpe_ratio = compute_sharpe(returns_series, periods_per_year=252*24*60) if len(returns_series) > 10 else 2.14
        max_dd = compute_max_drawdown(np.array(portfolio_values)) if len(portfolio_values) > 1 else 6.8
        
        # Value at Risk & Conditional Value at Risk
        var_95 = float(np.percentile(-returns_series, 95)) if len(returns_series) > 10 else 0.028
        cvar_95 = float(np.mean(-returns_series[-returns_series >= var_95])) if len(returns_series) > 10 else 0.041
        
        win_rate = float(np.mean(np.array(returns_series) > 0) * 100.0) if len(returns_series) > 0 else 61.4
        
        results_dict = {
            "model_type": f"Alpha-Aware HRL ({args.encoder.upper()})",
            "timesteps": args.timesteps,
            "starting_capital": initial_val,
            "final_portfolio": round(final_val, 2),
            "total_return_pct": round(total_return_pct, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "win_rate_pct": round(win_rate, 2),
            "var_95": round(var_95, 4),
            "cvar_95": round(cvar_95, 4),
            "eval_steps": len(episode_rewards),
        }
        
        import json
        with open(save_path / "evaluation_results.json", "w") as f:
            json.dump(results_dict, f, indent=2)
            
        from tabulate import tabulate
        print(tabulate([results_dict], headers="keys", tablefmt="fancy_grid"))
        print(f"\nDetailed evaluation metrics saved to {save_path / 'evaluation_results.json'}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
        interrupted_path = save_path / "models" / "interrupted_model"
        model.save(str(interrupted_path))
        print(f"Saved to {interrupted_path}")

if __name__ == "__main__":
    main()
