#!/usr/bin/env python3
"""
Ablation study over feature extractors, with honest arms and a real eval split.

Two defects in the previous version:

S1 -- the study was presented as evidence that Mamba beats LSTM, but no arm
      ran Mamba. Every non-TCN arm constructed
      `MambaFeatureExtractor(..., backend="lstm")`; the class name was the only
      thing Mamba about it. Arms are now named for what they actually run.

X2 -- `wrapped_env` was built once at :167, trained at :185/:198 and evaluated
      at :223 on the SAME object. There was no eval env, no held-out split and
      no seed change between phases. Train and eval now use disjoint temporal
      splits.

Arms:
  lem         Transformer Hawkes encoder, trained end-to-end by the RL loss
  lem_frozen  same encoder, gradients disabled -- reproduces the original
              broken configuration on purpose, as the control that shows what
              the published numbers actually measured
  gru         recurrent baseline (this is what "mamba backend=lstm" really was)
  mlp         flatten-and-MLP baseline
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from sb3_contrib import TQC
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

ARMS = {
    "lem":        (LEMFeaturesExtractor, {}),
    "lem_frozen": (LEMFeaturesExtractor, {"freeze_encoder": True}),
    "gru":        (GRUWindowExtractor, {}),
    "mlp":        (MLPWindowExtractor, {}),
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


def build_env(loader, split, seq_len, episode_length, fee, seed):
    def _init():
        env = HistoricalLOBEnv(
            data_loader=loader, split=split, episode_length=episode_length,
            starting_cash=100000.0, transaction_fee=fee,
            prices=loader.prices(split), reward="log_return",
        )
        wrapped = SequenceWindowWrapper(env, seq_len=seq_len)
        wrapped.reset(seed=seed)
        return Monitor(wrapped)
    return _init


def evaluate(model, env, max_steps):
    obs = env.reset()
    equity, positions = [], []
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = env.step(action)
        equity.append(float(infos[0]["portfolio_value"]))
        positions.append(float(infos[0].get("position_weight", 0.0)))
        if done[0]:
            break
    return np.asarray(equity), np.asarray(positions)


def run_arm(arm, loader, args, seed):
    cls, kwargs = ARMS[arm]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_env = DummyVecEnv([
        build_env(loader, "train", args.seq_len, args.episode_length, args.fee, seed)
    ])
    eval_env = DummyVecEnv([
        build_env(loader, "test", args.seq_len, args.eval_episode_length, args.fee,
                  seed + 10_000)
    ])

    if arm.startswith("lem") and args.pretrained:
        kwargs = {**kwargs, "pretrained_path": args.pretrained}

    model = TQC(
        "MultiInputPolicy", train_env, learning_rate=3e-4, batch_size=256,
        learning_starts=min(1000, args.timesteps // 4), gamma=0.99, tau=0.005,
        verbose=0, device=device, seed=seed,
        policy_kwargs=dict(
            features_extractor_class=cls,
            features_extractor_kwargs=kwargs,
            net_arch=[256, 256],
        ),
    )

    fe = get_features_extractor(model)
    n_trainable = sum(p.numel() for p in fe.parameters() if p.requires_grad)

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    train_time = time.time() - t0

    equity, positions = evaluate(model, eval_env, args.eval_steps)
    if len(equity) < 2:
        raise RuntimeError(f"arm {arm} seed {seed}: evaluation collected {len(equity)} points")

    m = compute_all_metrics(equity, np.ones_like(equity), float(equity[0]), periods_per_year=None)
    n_changes = int(np.sum(np.abs(np.diff(positions)) > 1e-9))

    return {
        "arm": arm,
        "seed": seed,
        "timesteps": args.timesteps,
        "trainable_extractor_params": n_trainable,
        "encoder_frozen": bool(kwargs.get("freeze_encoder", False)),
        "train_time_s": round(train_time, 1),
        "total_return_pct": round(m["total_return_pct"], 4),
        "sharpe_per_step": round(m["sharpe_ratio"], 4),
        "max_drawdown_pct": round(m["max_drawdown_pct"], 4),
        "win_rate_pct": round(m["win_rate_pct"], 4),
        "final_portfolio": round(float(equity[-1]), 2),
        "eval_steps": len(equity),
        "mean_abs_position_weight": round(float(np.mean(np.abs(positions))), 6),
        "n_position_changes": n_changes,
        "traded": n_changes > 0,
    }


def main():
    ap = argparse.ArgumentParser(description="Feature-extractor ablation")
    ap.add_argument("--data-dir", default="data/live_market")
    ap.add_argument("--symbol", default="BTC/USD")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--timesteps", type=int, default=20_000)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--episode-length", type=int, default=500)
    ap.add_argument("--eval-episode-length", type=int, default=400)
    ap.add_argument("--eval-steps", type=int, default=2000)
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--pretrained", default=None)
    ap.add_argument("--save-dir", default="experiments/ablations")
    args = ap.parse_args()

    loader = LiveMarketDataLoader(args.data_dir, args.symbol)
    loader.load("train")
    loader.load("test")

    print("=" * 72)
    print(f"ABLATION  |  {args.symbol}  |  arms={args.arms}  seeds={args.seeds}")
    print(f"train rows {len(loader.train_data)}  test rows {len(loader.test_data)}")
    print("=" * 72)

    per_run = []
    for arm in args.arms:
        for seed in args.seeds:
            print(f"\n  {arm} (seed {seed}) ...", flush=True)
            row = run_arm(arm, loader, args, seed)
            per_run.append(row)
            print(f"    return {row['total_return_pct']:+.4f}%  "
                  f"sharpe/step {row['sharpe_per_step']:+.4f}  "
                  f"maxDD {row['max_drawdown_pct']:.4f}%  traded={row['traded']}")

    # Aggregate across seeds. A single-seed number is not a result.
    #
    # The half-width uses the t critical value, not 1.96. With a handful of
    # seeds the normal approximation is badly anti-conservative: at n=2 the
    # correct multiplier is t(0.975, df=1) = 12.706, so using 1.96 understates
    # the interval by 6.5x and makes indistinguishable arms look separated.
    from scipy import stats as _stats

    summary = []
    for arm in args.arms:
        rows = [r for r in per_run if r["arm"] == arm]
        rets = np.array([r["total_return_pct"] for r in rows])
        shps = np.array([r["sharpe_per_step"] for r in rows])
        n = len(rows)
        if n > 1:
            tcrit = float(_stats.t.ppf(0.975, n - 1))
            half = tcrit * float(rets.std(ddof=1)) / np.sqrt(n)
        else:
            tcrit, half = float("nan"), float("nan")
        summary.append({
            "arm": arm,
            "n_seeds": n,
            "return_pct_mean": round(float(rets.mean()), 5),
            "return_pct_std": round(float(rets.std(ddof=1)) if n > 1 else 0.0, 5),
            "return_pct_ci95_halfwidth": None if n < 2 else round(half, 5),
            "t_critical": None if n < 2 else round(tcrit, 3),
            "underpowered": n < 3,
            "sharpe_mean": round(float(shps.mean()), 4),
            "sharpe_std": round(float(shps.std(ddof=1)) if n > 1 else 0.0, 4),
            "arms_that_traded": int(sum(r["traded"] for r in rows)),
        })

    # Pairwise Welch tests against the primary arm, so "A beats B" is a claim
    # the data can be checked against rather than an eyeballed ordering.
    comparisons = []
    if len(args.arms) > 1 and len(args.seeds) > 1:
        ref = args.arms[0]
        ref_rets = np.array([r["total_return_pct"] for r in per_run if r["arm"] == ref])
        for arm in args.arms[1:]:
            other = np.array([r["total_return_pct"] for r in per_run if r["arm"] == arm])
            t_stat, p_val = _stats.ttest_ind(ref_rets, other, equal_var=False)
            comparisons.append({
                "reference": ref,
                "arm": arm,
                "mean_diff": round(float(ref_rets.mean() - other.mean()), 5),
                "welch_t": round(float(t_stat), 4),
                "p_value": round(float(p_val), 4),
                "significant_at_0.05": bool(p_val < 0.05),
                "note": "n is small; treat any p-value here as indicative only",
            })

    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": provenance(args, data_dir=args.data_dir),
        "summary": summary,
        "comparisons": comparisons,
        "per_run": per_run,
    }
    with open(out / "ablation_results.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    print("\n" + "=" * 72)
    hdr = (f"{'arm':12s} {'n':>2s} {'return% mean':>14s} {'+-ci95':>10s} "
           f"{'sharpe':>9s} {'traded':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for row in summary:
        hw = row["return_pct_ci95_halfwidth"]
        print(f"{row['arm']:12s} {row['n_seeds']:>2d} {row['return_pct_mean']:>14.5f} "
              f"{(hw if hw is not None else float('nan')):>10.5f} "
              f"{row['sharpe_mean']:>9.4f} "
              f"{row['arms_that_traded']:>4d}/{row['n_seeds']}")
    print("=" * 72)

    if comparisons:
        print("\nWelch tests vs " + comparisons[0]["reference"] + ":")
        for c in comparisons:
            verdict = "significant" if c["significant_at_0.05"] else "NOT significant"
            print(f"  vs {c['arm']:12s} diff {c['mean_diff']:+.5f}  "
                  f"p = {c['p_value']:.4f}  ({verdict} at 0.05)")

    if any(r["underpowered"] for r in summary):
        print("\n  WARNING: fewer than 3 seeds per arm. The t critical value is "
              "large\n  at this n and these intervals are wide; no arm ordering "
              "here is\n  established. Increase --seeds before quoting a winner.")
    print(f"saved -> {out / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
