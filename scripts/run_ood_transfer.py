#!/usr/bin/env python3
"""
Out-of-Distribution Transfer Study (Step 8)

Trains the Full Model on a low-volatility environment and evaluates
on progressively harder volatility regimes (medium, high, crash).

Usage:
    python scripts/run_ood_transfer.py --timesteps 5000
    python scripts/run_ood_transfer.py --timesteps 50000 --seed 42
"""

import sys
sys.path.insert(0, '.')

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path

from src.envs.abides_wrapper import ABIDESEnv
from src.envs.hierarchical_wrapper import HierarchicalEnvWrapper
from src.agents.mamba_extractor import MambaFeatureExtractor
from src.agents.llm_analyst import LLMAnalyst, RegimeSignal
from src.models.timesfm_wrapper import SimpleAlphaModel


def evaluate_on_regime(model, extractor, llm_analyst, alpha_model,
                       volatility_regime, n_episodes=10, episode_length=200,
                       device="cpu"):
    """Evaluate a trained model on a specific volatility regime."""
    
    env = ABIDESEnv(
        ticker="AAPL",
        starting_cash=100000.0,
        episode_length=episode_length,
        market_impact=True,
        volatility_regime=volatility_regime,
    )
    
    wrapped_env = HierarchicalEnvWrapper(
        env=env,
        mamba_extractor=extractor,
        llm_analyst=llm_analyst,
        alpha_model=alpha_model,
        device=device,
    )
    
    rewards = []
    portfolio_values = []
    
    for ep in range(n_episodes):
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
    
    return {
        "volatility_regime": volatility_regime,
        "mean_reward": round(float(np.mean(rewards)), 6),
        "std_reward": round(float(np.std(rewards)), 6),
        "min_reward": round(float(np.min(rewards)), 6),
        "max_reward": round(float(np.max(rewards)), 6),
        "mean_portfolio": round(float(np.mean(portfolio_values)), 2),
        "std_portfolio": round(float(np.std(portfolio_values)), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="OOD Transfer Study (Step 8)")
    parser.add_argument("--timesteps", type=int, default=5000, help="Training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="experiments/ood_transfer", help="Save directory")
    parser.add_argument("--episode-length", type=int, default=200, help="Episode length")
    parser.add_argument("--n-eval-episodes", type=int, default=10, help="Evaluation episodes per regime")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cpu"
    train_regime = "low"
    test_regimes = ["low", "medium", "high", "crash"]
    
    print("=" * 70)
    print("OUT-OF-DISTRIBUTION TRANSFER STUDY")
    print("=" * 70)
    print(f"Train regime:  {train_regime}")
    print(f"Test regimes:  {test_regimes}")
    print(f"Timesteps:     {args.timesteps}")
    print(f"Seed:          {args.seed}")
    
    # Build components (Full Model configuration)
    mamba_dim = 128
    
    extractor = MambaFeatureExtractor(
        input_dim=40, d_model=mamba_dim, n_layers=2, backend="lstm"
    )
    
    alpha_model = SimpleAlphaModel(input_dim=1, hidden_dim=64, n_layers=2)
    alpha_model.eval()
    
    llm_analyst = LLMAnalyst(device="cpu")
    
    # Train on LOW volatility
    print(f"\n--- Training on '{train_regime}' volatility ---")
    
    train_env = ABIDESEnv(
        ticker="AAPL",
        starting_cash=100000.0,
        episode_length=args.episode_length,
        market_impact=True,
        volatility_regime=train_regime,
    )
    
    wrapped_train = HierarchicalEnvWrapper(
        env=train_env,
        mamba_extractor=extractor,
        llm_analyst=llm_analyst,
        alpha_model=alpha_model,
        device=device,
    )
    
    from sb3_contrib import TQC
    
    model = TQC(
        "MlpPolicy", wrapped_train,
        learning_rate=3e-4,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        verbose=0,
        device=device,
        seed=args.seed,
        policy_kwargs=dict(net_arch=[256, 256], n_critics=2),
    )
    
    start_time = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.1f}s")
    
    # Save trained model
    model.save(str(save_dir / "ood_full_model"))
    
    # Evaluate on each volatility regime
    print(f"\n--- Evaluating across volatility regimes ---")
    
    ood_results = []
    
    for regime in test_regimes:
        print(f"\n  Testing on '{regime}' volatility...")
        result = evaluate_on_regime(
            model=model,
            extractor=extractor,
            llm_analyst=llm_analyst,
            alpha_model=alpha_model,
            volatility_regime=regime,
            n_episodes=args.n_eval_episodes,
            episode_length=args.episode_length,
            device=device,
        )
        ood_results.append(result)
        print(f"    Mean reward: {result['mean_reward']:+.6f} ± {result['std_reward']:.6f}")
        print(f"    Mean portfolio: ${result['mean_portfolio']:.2f} ± ${result['std_portfolio']:.2f}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("OOD TRANSFER RESULTS (Trained on 'low' volatility)")
    print("=" * 70)
    print(f"{'Regime':<10} {'Mean Reward':<18} {'Std Reward':<14} {'Mean Portfolio':<16} {'Degradation':<12}")
    print("-" * 70)
    
    baseline_reward = ood_results[0]["mean_reward"]  # In-distribution (low)
    
    for r in ood_results:
        if baseline_reward != 0:
            degradation = ((r["mean_reward"] - baseline_reward) / abs(baseline_reward)) * 100
        else:
            degradation = 0
        
        marker = "  (in-dist)" if r["volatility_regime"] == train_regime else ""
        print(f"{r['volatility_regime']:<10} {r['mean_reward']:+.6f}       "
              f"{r['std_reward']:<14.6f} ${r['mean_portfolio']:<15.2f} "
              f"{degradation:+.1f}%{marker}")
    
    # Save results
    output = {
        "train_regime": train_regime,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "train_time_s": round(train_time, 1),
        "results": ood_results,
    }
    
    results_path = save_dir / "ood_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Generate plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        regimes = [r["volatility_regime"] for r in ood_results]
        mean_rewards = [r["mean_reward"] for r in ood_results]
        std_rewards = [r["std_reward"] for r in ood_results]
        mean_portfolios = [r["mean_portfolio"] for r in ood_results]
        std_portfolios = [r["std_portfolio"] for r in ood_results]
        
        colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Reward transfer
        ax1 = axes[0]
        bars1 = ax1.bar(regimes, mean_rewards, yerr=std_rewards,
                        capsize=5, color=colors, alpha=0.8,
                        edgecolor='black', linewidth=0.5)
        ax1.set_ylabel('Mean Episode Reward')
        ax1.set_title(f'OOD Transfer: Reward (Trained on "{train_regime}")')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Highlight in-distribution bar
        bars1[0].set_edgecolor('#000000')
        bars1[0].set_linewidth(2)
        
        # Portfolio transfer
        ax2 = axes[1]
        bars2 = ax2.bar(regimes, mean_portfolios, yerr=std_portfolios,
                        capsize=5, color=colors, alpha=0.8,
                        edgecolor='black', linewidth=0.5)
        ax2.set_ylabel('Mean Final Portfolio ($)')
        ax2.set_title(f'OOD Transfer: Portfolio (Trained on "{train_regime}")')
        ax2.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial capital')
        ax2.legend()
        
        bars2[0].set_edgecolor('#000000')
        bars2[0].set_linewidth(2)
        
        plt.tight_layout()
        plot_path = plots_dir / "ood_transfer.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {plot_path}")
        
    except ImportError:
        print("matplotlib not available — skipping plot generation")


if __name__ == "__main__":
    main()
