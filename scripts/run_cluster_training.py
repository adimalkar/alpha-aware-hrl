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

def provenance(args) -> dict:
    """
    Record what produced a result so it can be tied back to code and data.
    Without this an experiments/ JSON is an assertion, not a measurement.
    """
    import subprocess, hashlib, platform, datetime

    def _git(*cmd):
        try:
            return subprocess.check_output(
                ["git", *cmd], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return None

    def _digest(path):
        f = Path(path)
        if not f.is_file():
            return None
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return f"sha256:{h.hexdigest()[:16]}"

    data_dir = Path(args.data_dir)
    return {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "seed": args.seed,
        "encoder": args.encoder,
        "data_dir": str(data_dir),
        "train_csv": _digest(data_dir / "FI2010_train.csv"),
        "test_csv": _digest(data_dir / "FI2010_test.csv"),
    }


def run_eval_episode(model, eval_env, max_steps: int):
    """
    Run one deterministic evaluation episode and return the REAL equity curve.

    Portfolio value is read from the env's info dict, which is where
    HistoricalLOBEnv actually publishes it. The previous implementation probed
    `eval_env.envs[0].env.portfolio_value` -- an attribute that does not exist
    -- so the list stayed empty and every metric fell through to a constant.
    """
    obs = eval_env.reset()
    equity, positions, rewards = [], [], []

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = eval_env.step(action)
        rewards.append(float(reward[0]))

        info = infos[0]
        if "portfolio_value" not in info:
            raise RuntimeError(
                "Environment info dict has no 'portfolio_value' key; the "
                "evaluation cannot measure anything. Keys present: "
                f"{sorted(info.keys())}"
            )
        equity.append(float(info["portfolio_value"]))
        positions.append(float(info.get("position", 0.0)))

        if done[0]:
            break

    return np.asarray(equity), np.asarray(positions), rewards


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
    parser.add_argument("--eval-steps", type=int, default=5000,
                        help="Max steps in the out-of-sample evaluation episode")
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
        
        from src.utils.metrics import compute_all_metrics, MetricUnitError

        equity_curve, positions, episode_rewards = run_eval_episode(
            model, eval_env, max_steps=args.eval_steps
        )

        # No fallback constants. An evaluation that collected nothing is a
        # failed evaluation, not a result. The previous version substituted
        # sharpe=2.14 / max_dd=6.8 / var=0.028 / cvar=0.041 / win_rate=61.4
        # whenever portfolio tracking came up empty -- which was always,
        # because HistoricalLOBEnv has no `portfolio_value` attribute, only
        # an info dict key. Those five constants are the numbers that were
        # published as results.
        if len(equity_curve) < 2:
            raise RuntimeError(
                f"Evaluation collected {len(equity_curve)} portfolio observations "
                f"over {len(episode_rewards)} steps. Nothing can be measured from "
                "this. Check that the environment emits 'portfolio_value' in its "
                "info dict and that the episode ran."
            )

        initial_val = float(equity_curve[0])
        metrics = compute_all_metrics(
            prices=equity_curve,
            positions=np.ones_like(equity_curve),  # equity curve is already the book
            initial_capital=initial_val,
            periods_per_year=None,  # tick-sampled: not annualised. See metrics.py.
        )

        results_dict = {
            "model_type": f"Alpha-Aware HRL ({args.encoder.upper()})",
            "provenance": provenance(args),
            "timesteps": args.timesteps,
            "starting_capital": initial_val,
            "final_portfolio": round(float(equity_curve[-1]), 2),
            "total_return_pct": round(metrics["total_return_pct"], 4),
            "sharpe_ratio_per_step": round(metrics["sharpe_ratio"], 4),
            "annualised": metrics["annualised"],
            "max_drawdown_pct": round(metrics["max_drawdown_pct"], 4),
            "win_rate_pct": round(metrics["win_rate_pct"], 4),
            "var_95": round(metrics["var_95"], 6),
            "cvar_95": round(metrics["cvar_95"], 6),
            "eval_steps": len(episode_rewards),
            "n_return_periods": metrics["n_periods"],
            "mean_abs_position": round(float(np.mean(np.abs(positions))), 6),
            "n_position_changes": int(np.sum(np.abs(np.diff(positions)) > 1e-9)),
        }

        # An agent that never moved its position has not been evaluated on
        # anything. Say so in the artefact rather than reporting 0.0 return.
        if results_dict["n_position_changes"] == 0:
            results_dict["WARNING"] = (
                "Agent held a constant position for the entire evaluation. "
                "Returns reflect buy-and-hold (or cash), not a learned policy."
            )

        import json
        with open(save_path / "evaluation_results.json", "w") as f:
            json.dump(results_dict, f, indent=2)

        for k, v in results_dict.items():
            print(f"  {k:24s}: {v}")
        print(f"\nDetailed evaluation metrics saved to {save_path / 'evaluation_results.json'}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
        interrupted_path = save_path / "models" / "interrupted_model"
        model.save(str(interrupted_path))
        print(f"Saved to {interrupted_path}")

if __name__ == "__main__":
    main()
