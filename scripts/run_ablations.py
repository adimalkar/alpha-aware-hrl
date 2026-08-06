#!/usr/bin/env python3
"""
Ablation Study Script (Step 7)

Trains and evaluates 4 model configurations to compare their performance:
  1. TCN + PPO   — TCN backbone + PPO (baseline)
  2. LSTM + PPO  — LSTM backbone + PPO (baseline)
  3. LSTM + DSAC — LSTM backbone + TQC (no LLM, no alpha)
  4. Full Model  — LSTM + TQC + LLM regime + Alpha signals

Usage:
    python scripts/run_ablations.py --timesteps 5000 --seed 42
    python scripts/run_ablations.py --timesteps 50000 --save-dir experiments/ablations
"""

import sys
sys.path.insert(0, '.')

import argparse
import time
import json
import numpy as np
import torch
from pathlib import Path

from src.envs.abides_wrapper import ABIDESEnv
from src.envs.hierarchical_wrapper import HierarchicalEnvWrapper
from src.agents.mamba_extractor import MambaFeatureExtractor
from src.agents.llm_analyst import LLMAnalyst, RegimeSignal
from src.agents.dsac_trader import DSACTrader
from src.models.timesfm_wrapper import SimpleAlphaModel
from src.models.mamba_ssm import TCNEncoder


class TCNFeatureExtractor:
    """Adapter to make TCNEncoder compatible with HierarchicalEnvWrapper."""
    
    def __init__(self, input_dim=40, hidden_dim=128, n_layers=4, dropout=0.1):
        self.encoder = TCNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )
        self._d_model = hidden_dim
    
    def __call__(self, x):
        return self.encoder(x)
    
    def eval(self):
        self.encoder.eval()
        return self
    
    def cpu(self):
        self.encoder.cpu()
        return self
    
    def to(self, device):
        self.encoder.to(device)
        return self
    
    def get_output_dim(self):
        return self._d_model
    
    def state_dict(self):
        return self.encoder.state_dict()
    
    def parameters(self):
        return self.encoder.parameters()


def build_config(config_name, device="cpu"):
    """
    Build components for a specific ablation configuration.
    
    Returns:
        dict with 'extractor', 'alpha_model', 'llm_analyst', 'rl_algo', 'obs_dim'
    """
    mamba_dim = 128
    regime_dim = 4
    alpha_dim = 0
    
    if config_name == "tcn_ppo":
        extractor = TCNFeatureExtractor(input_dim=40, hidden_dim=mamba_dim, n_layers=4)
        alpha_model = None
        use_regime = False
        rl_algo = "PPO"
        
    elif config_name == "lstm_ppo":
        extractor = MambaFeatureExtractor(
            input_dim=40, d_model=mamba_dim, n_layers=2, backend="lstm"
        )
        alpha_model = None
        use_regime = False
        rl_algo = "PPO"
        
    elif config_name == "lstm_dsac":
        extractor = MambaFeatureExtractor(
            input_dim=40, d_model=mamba_dim, n_layers=2, backend="lstm"
        )
        alpha_model = None
        use_regime = False
        rl_algo = "TQC"
        
    elif config_name == "full":
        extractor = MambaFeatureExtractor(
            input_dim=40, d_model=mamba_dim, n_layers=2, backend="lstm"
        )
        alpha_model = SimpleAlphaModel(input_dim=1, hidden_dim=64, n_layers=2)
        alpha_model.eval()
        use_regime = True
        rl_algo = "TQC"
        alpha_dim = 4
        
    else:
        raise ValueError(f"Unknown config: {config_name}")
    
    total_obs_dim = mamba_dim + regime_dim + alpha_dim
    
    return {
        "extractor": extractor,
        "alpha_model": alpha_model,
        "use_regime": use_regime,
        "rl_algo": rl_algo,
        "obs_dim": total_obs_dim,
        "mamba_dim": mamba_dim,
        "regime_dim": regime_dim,
        "alpha_dim": alpha_dim,
    }


def train_and_evaluate(config_name, timesteps, seed, save_dir, episode_length=200):
    """Train a single ablation configuration and return metrics."""
    
    print(f"\n{'='*70}")
    print(f"  Configuration: {config_name.upper()}")
    print(f"  Timesteps: {timesteps}, Seed: {seed}")
    print(f"{'='*70}")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = "cpu"  # SB3 training on CPU for stability
    
    # Build components
    cfg = build_config(config_name, device)
    
    # Build environment
    base_env = ABIDESEnv(
        ticker="AAPL",
        starting_cash=100000.0,
        episode_length=episode_length,
        market_impact=True,
        volatility_regime="medium",
    )
    
    # Build LLM analyst (lightweight — only used for regime embedding)
    llm_analyst = LLMAnalyst(device="cpu")
    
    # Inject a static regime if this config uses it
    if cfg["use_regime"]:
        base_env.current_regime = RegimeSignal(
            regime=0, confidence=0.9, reasoning="Pre-set for ablation"
        )
    
    # Wrap environment
    wrapped_env = HierarchicalEnvWrapper(
        env=base_env,
        mamba_extractor=cfg["extractor"],
        llm_analyst=llm_analyst,
        alpha_model=cfg["alpha_model"],
        device=device,
    )
    
    print(f"  Observation space: {wrapped_env.observation_space.shape}")
    print(f"  Action space: {wrapped_env.action_space.shape}")
    print(f"  RL Algorithm: {cfg['rl_algo']}")
    
    # Build RL model
    start_time = time.time()
    
    if cfg["rl_algo"] == "TQC":
        from sb3_contrib import TQC
        model = TQC(
            "MlpPolicy", wrapped_env,
            learning_rate=3e-4,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            verbose=0,
            device=device,
            seed=seed,
            policy_kwargs=dict(net_arch=[256, 256], n_critics=2),
        )
    else:  # PPO
        from stable_baselines3 import PPO
        model = PPO(
            "MlpPolicy", wrapped_env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=0,
            device=device,
            seed=seed,
            policy_kwargs=dict(net_arch=[256, 256]),
        )
    
    # Train
    print(f"  Training...")
    model.learn(total_timesteps=timesteps, progress_bar=False)
    train_time = time.time() - start_time
    print(f"  Training completed in {train_time:.1f}s")
    
    # Evaluate
    print(f"  Evaluating...")
    n_eval_episodes = 10
    rewards = []
    portfolio_values = []
    
    for ep in range(n_eval_episodes):
        obs, info = wrapped_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        rewards.append(episode_reward)
        portfolio_values.append(info.get("portfolio_value", 0))
    
    results = {
        "config": config_name,
        "seed": seed,
        "timesteps": timesteps,
        "train_time_s": round(train_time, 1),
        "mean_reward": round(float(np.mean(rewards)), 6),
        "std_reward": round(float(np.std(rewards)), 6),
        "min_reward": round(float(np.min(rewards)), 6),
        "max_reward": round(float(np.max(rewards)), 6),
        "mean_portfolio": round(float(np.mean(portfolio_values)), 2),
        "obs_dim": cfg["obs_dim"],
        "rl_algo": cfg["rl_algo"],
    }
    
    # Save model
    config_save_path = Path(save_dir) / config_name
    config_save_path.mkdir(parents=True, exist_ok=True)
    model.save(str(config_save_path / f"model_seed{seed}"))
    
    print(f"  Results: mean_reward={results['mean_reward']:.6f}, "
          f"mean_portfolio=${results['mean_portfolio']:.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Ablation Studies (Step 7)")
    parser.add_argument("--timesteps", type=int, default=5000, help="Training timesteps per config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="experiments/ablations", help="Save directory")
    parser.add_argument("--episode-length", type=int, default=200, help="Episode length")
    parser.add_argument("--configs", type=str, default="all",
                        help="Comma-separated configs to run (tcn_ppo,lstm_ppo,lstm_dsac,full) or 'all'")
    args = parser.parse_args()
    
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    if args.configs == "all":
        configs = ["tcn_ppo", "lstm_ppo", "lstm_dsac", "full"]
    else:
        configs = [c.strip() for c in args.configs.split(",")]
    
    print("=" * 70)
    print("ABLATION STUDY — Alpha-Aware Hierarchical RL")
    print("=" * 70)
    print(f"Configs:   {configs}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Seed:      {args.seed}")
    print(f"Save dir:  {args.save_dir}")
    
    all_results = []
    
    for config_name in configs:
        try:
            result = train_and_evaluate(
                config_name=config_name,
                timesteps=args.timesteps,
                seed=args.seed,
                save_dir=args.save_dir,
                episode_length=args.episode_length,
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n  ❌ Config '{config_name}' FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"config": config_name, "error": str(e)})
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Config':<15} {'RL Algo':<8} {'Obs Dim':<8} {'Mean Reward':<14} {'Std Reward':<12} {'Portfolio':<12} {'Time(s)':<8}")
    print("-" * 70)
    
    for r in all_results:
        if "error" in r:
            print(f"{r['config']:<15} FAILED: {r['error']}")
        else:
            print(f"{r['config']:<15} {r['rl_algo']:<8} {r['obs_dim']:<8} "
                  f"{r['mean_reward']:<14.6f} {r['std_reward']:<12.6f} "
                  f"${r['mean_portfolio']:<11.2f} {r['train_time_s']:<8.1f}")
    
    # Save results to JSON
    results_path = Path(args.save_dir) / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
