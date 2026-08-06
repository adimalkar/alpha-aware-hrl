#!/usr/bin/env python3
"""
Multi-Seed Robustness Study (Step 8)

Runs each ablation configuration across multiple seeds and reports
mean ± std with confidence intervals. Generates comparison plots.

Usage:
    python scripts/run_robustness.py --timesteps 5000 --seeds 42,1,100,7,2024
    python scripts/run_robustness.py --timesteps 50000 --seeds 42,1,100
"""

import sys
sys.path.insert(0, '.')

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Reuse the ablation training function
from scripts.run_ablations import train_and_evaluate


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Robustness Study (Step 8)")
    parser.add_argument("--timesteps", type=int, default=5000, help="Training timesteps per config")
    parser.add_argument("--seeds", type=str, default="42,1,100", help="Comma-separated seeds")
    parser.add_argument("--save-dir", type=str, default="experiments/robustness", help="Save directory")
    parser.add_argument("--episode-length", type=int, default=200, help="Episode length")
    parser.add_argument("--configs", type=str, default="all",
                        help="Comma-separated configs or 'all'")
    args = parser.parse_args()
    
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    if args.configs == "all":
        configs = ["tcn_ppo", "lstm_ppo", "lstm_dsac", "full"]
    else:
        configs = [c.strip() for c in args.configs.split(",")]
    
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    plots_dir = Path(args.save_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("MULTI-SEED ROBUSTNESS STUDY — Alpha-Aware Hierarchical RL")
    print("=" * 70)
    print(f"Configs:   {configs}")
    print(f"Seeds:     {seeds}")
    print(f"Timesteps: {args.timesteps}")
    
    # Collect results across seeds
    all_results = []  # List of dicts
    config_results = defaultdict(list)  # config_name -> list of results
    
    for config_name in configs:
        for seed in seeds:
            try:
                result = train_and_evaluate(
                    config_name=config_name,
                    timesteps=args.timesteps,
                    seed=seed,
                    save_dir=args.save_dir,
                    episode_length=args.episode_length,
                )
                all_results.append(result)
                config_results[config_name].append(result)
            except Exception as e:
                print(f"\n  ❌ Config '{config_name}' seed={seed} FAILED: {e}")
                import traceback
                traceback.print_exc()
    
    # Compute aggregate statistics
    print("\n" + "=" * 70)
    print("ROBUSTNESS RESULTS — Mean ± Std across seeds")
    print("=" * 70)
    print(f"{'Config':<15} {'Seeds':<6} {'Mean Reward':<18} {'Mean Portfolio':<18} {'Mean Time(s)':<12}")
    print("-" * 70)
    
    summary = []
    
    for config_name in configs:
        results = config_results[config_name]
        if not results:
            print(f"{config_name:<15} NO RESULTS")
            continue
        
        rewards = [r["mean_reward"] for r in results]
        portfolios = [r["mean_portfolio"] for r in results]
        times = [r["train_time_s"] for r in results]
        
        n = len(rewards)
        mean_rew = np.mean(rewards)
        std_rew = np.std(rewards)
        ci_rew = 1.96 * std_rew / np.sqrt(n) if n > 1 else 0
        
        mean_port = np.mean(portfolios)
        std_port = np.std(portfolios)
        
        mean_time = np.mean(times)
        
        print(f"{config_name:<15} {n:<6} {mean_rew:+.6f} ± {std_rew:.6f}  "
              f"${mean_port:.2f} ± ${std_port:.2f}  {mean_time:.1f}")
        
        summary.append({
            "config": config_name,
            "n_seeds": n,
            "mean_reward": round(mean_rew, 6),
            "std_reward": round(std_rew, 6),
            "ci_95_reward": round(ci_rew, 6),
            "mean_portfolio": round(mean_port, 2),
            "std_portfolio": round(std_port, 2),
            "mean_train_time_s": round(mean_time, 1),
            "per_seed_results": results,
        })
    
    # Save results
    results_path = Path(args.save_dir) / "robustness_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed results saved to {results_path}")
    
    # Generate plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Plot 1: Mean Reward with Error Bars
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        config_names = [s["config"] for s in summary]
        mean_rewards = [s["mean_reward"] for s in summary]
        std_rewards = [s["std_reward"] for s in summary]
        mean_portfolios = [s["mean_portfolio"] for s in summary]
        std_portfolios = [s["std_portfolio"] for s in summary]
        
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
        
        # Reward bar chart
        ax1 = axes[0]
        bars1 = ax1.bar(config_names, mean_rewards, yerr=std_rewards,
                        capsize=5, color=colors[:len(config_names)], alpha=0.8,
                        edgecolor='black', linewidth=0.5)
        ax1.set_ylabel('Mean Episode Reward')
        ax1.set_title('Ablation: Mean Reward (± 1 std)')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.tick_params(axis='x', rotation=15)
        
        # Portfolio bar chart
        ax2 = axes[1]
        bars2 = ax2.bar(config_names, mean_portfolios, yerr=std_portfolios,
                        capsize=5, color=colors[:len(config_names)], alpha=0.8,
                        edgecolor='black', linewidth=0.5)
        ax2.set_ylabel('Mean Final Portfolio ($)')
        ax2.set_title('Ablation: Final Portfolio Value (± 1 std)')
        ax2.tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        plot_path = plots_dir / "robustness_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {plot_path}")
        
        # Plot 2: Per-seed scatter
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, config_name in enumerate(configs):
            results = config_results[config_name]
            if results:
                seed_rewards = [r["mean_reward"] for r in results]
                seed_labels = [r["seed"] for r in results]
                x_positions = [i + np.random.uniform(-0.15, 0.15) for _ in seed_rewards]
                ax.scatter(x_positions, seed_rewards, color=colors[i], s=60,
                          alpha=0.8, edgecolors='black', linewidth=0.5, zorder=3)
        
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs)
        ax.set_ylabel('Mean Episode Reward')
        ax.set_title('Per-Seed Reward Distribution')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        scatter_path = plots_dir / "per_seed_scatter.png"
        plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {scatter_path}")
        
    except ImportError:
        print("matplotlib not available — skipping plot generation")


if __name__ == "__main__":
    main()
