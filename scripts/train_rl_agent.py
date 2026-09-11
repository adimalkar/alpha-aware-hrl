#!/usr/bin/env python3
"""
Train the trading agent on real collected market data.

Replaces scripts/run_cluster_training.py, which could not produce a valid
result: it ran against FI-2010, whose label-derived price path leaked the
future into the reward, and its evaluation block substituted hardcoded
constants (sharpe=2.14, max_dd=6.8, var=0.028, cvar=0.041, win_rate=61.4)
whenever portfolio tracking came up empty -- which was always, because it
probed an attribute the environment does not define.

Everything here measures or fails.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from sb3_contrib import TQC
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.lem_extractor import (
    GRUWindowExtractor,
    LEMFeaturesExtractor,
    MLPWindowExtractor,
)
from src.envs.historical_lob_env import HistoricalLOBEnv
from src.envs.sequence_wrapper import SequenceWindowWrapper
from src.utils.data_loader import LiveMarketDataLoader
from src.utils.metrics import compute_all_metrics
from src.utils.provenance import provenance

EXTRACTORS = {
    "lem": LEMFeaturesExtractor,
    "gru": GRUWindowExtractor,
    "mlp": MLPWindowExtractor,
}


def get_features_extractor(model):
    """
    Return the policy's features extractor.

    SB3's off-policy algorithms (TQC/SAC) build separate actor and critic
    extractors and leave `policy.features_extractor` as None, so reading that
    attribute directly raises. On-policy algorithms (PPO) do populate it.
    """
    policy = model.policy
    for attr in ("features_extractor",):
        fe = getattr(policy, attr, None)
        if fe is not None:
            return fe
    for owner in ("actor", "critic"):
        sub = getattr(policy, owner, None)
        fe = getattr(sub, "features_extractor", None) if sub is not None else None
        if fe is not None:
            return fe
    raise AttributeError(
        f"Could not locate a features extractor on {type(policy).__name__}."
    )


def make_env(loader, split, seq_len, episode_length, fee, seed):
    def _init():
        env = HistoricalLOBEnv(
            data_loader=loader,
            split=split,
            episode_length=episode_length,
            starting_cash=100000.0,
            transaction_fee=fee,
            prices=loader.prices(split),
            reward="log_return",
        )
        wrapped = SequenceWindowWrapper(env, seq_len=seq_len)
        wrapped.reset(seed=seed)
        return Monitor(wrapped)

    return _init


def run_eval_episode(model, env, max_steps):
    """Deterministic rollout returning the REAL equity curve and positions."""
    obs = env.reset()
    equity, positions, rewards = [], [], []
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = env.step(action)
        info = infos[0]
        if "portfolio_value" not in info:
            raise RuntimeError(
                f"env info has no 'portfolio_value'; keys: {sorted(info.keys())}"
            )
        equity.append(float(info["portfolio_value"]))
        positions.append(float(info.get("position_weight", 0.0)))
        rewards.append(float(reward[0]))
        if done[0]:
            break
    return np.asarray(equity), np.asarray(positions), rewards


def buy_and_hold(prices, fee):
    """
    Fully-invested buy-and-hold equity curve on the same price path.

    M5: the project had no market benchmark anywhere. A trading return quoted
    without one says nothing -- a strategy that returns +5% while the asset
    returned +20% has destroyed value.
    """
    prices = np.asarray(prices, dtype=np.float64)
    equity = 100000.0 * (prices / prices[0]) * (1.0 - fee)
    return equity


def main():
    ap = argparse.ArgumentParser(description="Train the RL trading agent")
    ap.add_argument("--data-dir", default="data/live_market")
    ap.add_argument("--symbol", default="BTC/USD")
    ap.add_argument("--encoder", default="lem", choices=list(EXTRACTORS))
    ap.add_argument("--algo", default="TQC", choices=["TQC", "PPO"])
    ap.add_argument("--pretrained", default=None,
                    help="THP checkpoint from pretrain_event_encoder.py (lem only)")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="Ablation control: reproduce the old no-gradient behaviour")
    ap.add_argument("--timesteps", type=int, default=50_000)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--episode-length", type=int, default=500)
    ap.add_argument("--eval-episode-length", type=int, default=400)
    ap.add_argument("--eval-steps", type=int, default=2000)
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-dir", default="experiments/rl_run")
    ap.add_argument("--eval-only", default=None,
                    help="Path to a saved model; skip training and evaluate it")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_path = Path(args.save_dir)
    (save_path / "models").mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print(f"RL TRAINING  |  {args.symbol}  |  encoder={args.encoder}  algo={args.algo}")
    print("=" * 68)

    loader = LiveMarketDataLoader(args.data_dir, args.symbol)
    for sp in ("train", "val", "test"):
        loader.load(sp)
    print(f"  train {len(loader.train_data):>6} rows  "
          f"mid {loader.train_mid.min():.2f}-{loader.train_mid.max():.2f}")
    print(f"  val   {len(loader.val_data):>6} rows  "
          f"mid {loader.val_mid.min():.2f}-{loader.val_mid.max():.2f}   (model selection)")
    print(f"  test  {len(loader.test_data):>6} rows  "
          f"mid {loader.test_mid.min():.2f}-{loader.test_mid.max():.2f}   (sealed, used once)")

    # Train and eval draw from disjoint, temporally ordered splits with a purge
    # gap applied at collection time. The previous ablation script built ONE env
    # object and used it for both training and evaluation.
    train_env = DummyVecEnv([
        make_env(loader, "train", args.seq_len, args.episode_length, args.fee, args.seed)
    ])
    # Model selection runs on 'val'. 'test' is the sealed holdout and is
    # touched exactly once, after training is finished (audit X3).
    val_env = DummyVecEnv([
        make_env(loader, "val", args.seq_len, args.eval_episode_length, args.fee,
                 args.seed + 10_000)
    ])
    test_env = DummyVecEnv([
        make_env(loader, "test", args.seq_len, args.eval_episode_length, args.fee,
                 args.seed + 20_000)
    ])

    extractor_kwargs = {}
    if args.encoder == "lem":
        extractor_kwargs = {
            "pretrained_path": args.pretrained,
            "freeze_encoder": args.freeze_encoder,
        }

    policy_kwargs = dict(
        features_extractor_class=EXTRACTORS[args.encoder],
        features_extractor_kwargs=extractor_kwargs,
        net_arch=[256, 256],
    )

    if args.algo == "TQC":
        model = TQC("MultiInputPolicy", train_env, learning_rate=3e-4, batch_size=256,
                    gamma=0.99, tau=0.005, learning_starts=1000, verbose=0,
                    device=device, seed=args.seed, policy_kwargs=policy_kwargs,
                    tensorboard_log=str(save_path / "tensorboard"))
    else:
        model = PPO("MultiInputPolicy", train_env, learning_rate=3e-4, n_steps=512,
                    batch_size=64, n_epochs=10, gamma=0.99, verbose=0,
                    device=device, seed=args.seed, policy_kwargs=policy_kwargs,
                    tensorboard_log=str(save_path / "tensorboard"))

    fe = get_features_extractor(model)
    n_trainable = sum(p.numel() for p in fe.parameters() if p.requires_grad)
    print(f"  feature extractor: {type(fe).__name__}, {n_trainable:,} trainable params")
    if args.encoder == "lem":
        print(f"  pretrained loaded : {getattr(fe, 'pretrained_loaded', False)}")
        print(f"  encoder frozen    : {getattr(fe, 'frozen', False)}")

    eval_cb = EvalCallback(
        val_env,
        best_model_save_path=str(save_path / "models"),
        log_path=str(save_path / "logs"),
        eval_freq=max(args.timesteps // 10, 500),
        n_eval_episodes=3,
        deterministic=True,
        verbose=1,
    )

    if args.eval_only:
        print(f"\n  eval-only: loading {args.eval_only}")
        algo_cls = TQC if args.algo == "TQC" else PPO
        model = algo_cls.load(args.eval_only, env=train_env, device=device)
        fe = get_features_extractor(model)
    else:
        print(f"\n  training for {args.timesteps:,} timesteps on {device}...")
        model.learn(total_timesteps=args.timesteps, callback=eval_cb, progress_bar=False)
        model.save(str(save_path / "models" / "final_model"))

    # Confirm the encoder actually moved during training.
    encoder_changed = None
    if args.encoder == "lem" and not args.freeze_encoder and not args.eval_only:
        encoder_changed = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in fe.encoder.parameters()
            if p.requires_grad
        )

    print("\n  final evaluation on the SEALED test split (used once)...")
    equity, positions, rewards = run_eval_episode(model, test_env, args.eval_steps)
    if len(equity) < 2:
        raise RuntimeError(
            f"Evaluation collected {len(equity)} portfolio points over "
            f"{len(rewards)} steps. Nothing can be measured."
        )

    metrics = compute_all_metrics(
        prices=equity,
        positions=np.ones_like(equity),
        initial_capital=float(equity[0]),
        periods_per_year=None,
    )

    n_changes = int(np.sum(np.abs(np.diff(positions)) > 1e-9))

    # Buy-and-hold on the identical price window the agent just traded.
    test_prices = loader.prices("test")[: len(equity)]
    bh_equity = buy_and_hold(test_prices, args.fee)
    bh = compute_all_metrics(bh_equity, np.ones_like(bh_equity),
                             float(bh_equity[0]), periods_per_year=None)

    results = {
        "symbol": args.symbol,
        "encoder": args.encoder,
        "algo": args.algo,
        "pretrained": args.pretrained,
        "freeze_encoder": args.freeze_encoder,
        "encoder_received_gradients": encoder_changed,
        # Do not report a training budget for a run that did not train; the
        # default arg value would otherwise be recorded as if it were real.
        "timesteps": None if args.eval_only else args.timesteps,
        "evaluated_checkpoint": args.eval_only,
        "provenance": provenance(args, data_dir=args.data_dir),
        "starting_capital": float(equity[0]),
        "final_portfolio": round(float(equity[-1]), 2),
        "total_return_pct": round(metrics["total_return_pct"], 4),
        "sharpe_per_step": round(metrics["sharpe_ratio"], 4),
        "annualised": metrics["annualised"],
        "max_drawdown_pct": round(metrics["max_drawdown_pct"], 4),
        "win_rate_pct": round(metrics["win_rate_pct"], 4),
        "flat_rate_pct": round(metrics["flat_rate_pct"], 4),
        "loss_rate_pct": round(metrics["loss_rate_pct"], 4),
        "var_95": round(metrics["var_95"], 6),
        "cvar_95": round(metrics["cvar_95"], 6),
        "eval_steps": len(rewards),
        "n_return_periods": metrics["n_periods"],
        "mean_abs_position_weight": round(float(np.mean(np.abs(positions))), 6),
        "n_position_changes": n_changes,
        "benchmark_buy_and_hold": {
            "total_return_pct": round(bh["total_return_pct"], 4),
            "sharpe_per_step": round(bh["sharpe_ratio"], 4),
            "max_drawdown_pct": round(bh["max_drawdown_pct"], 4),
        },
        "excess_return_vs_buy_and_hold_pct": round(
            metrics["total_return_pct"] - bh["total_return_pct"], 4
        ),
        "beats_buy_and_hold": bool(metrics["total_return_pct"] > bh["total_return_pct"]),
    }
    if n_changes == 0:
        results["WARNING"] = (
            "Position never changed during evaluation; this measures buy-and-hold "
            "or cash, not a learned policy."
        )

    # Persist the equity curve so the result is inspectable and plottable,
    # rather than only summarised.
    np.savez_compressed(
        save_path / "equity_curve.npz",
        equity=equity, positions=positions,
        benchmark_equity=bh_equity, prices=test_prices,
    )

    with open(save_path / "evaluation_results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print("\n" + "=" * 68)
    for k, v in results.items():
        if k != "provenance":
            print(f"  {k:28s}: {v}")
    print("=" * 68)
    print(f"  saved -> {save_path / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
