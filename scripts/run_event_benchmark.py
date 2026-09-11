#!/usr/bin/env python3
"""
Inference-latency benchmark for the feature extractors.

Scope note
==========
This script previously printed a table of Sharpe / return / drawdown / CVaR
figures for five strategies. Those columns were hardcoded Python literals:

    {"Model / Strategy": "Alpha-Aware LEM-HRL (THP + TPP)",
     "Sharpe": 2.14, "Total Return %": 24.85, "Max DD %": 6.8, ...}

Only the latency column was ever measured. The script now measures latency and
nothing else; trading performance comes from scripts/run_ablations.py, which
actually runs the policies.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from src.agents.lem_extractor import (
    GRUWindowExtractor,
    LEMFeaturesExtractor,
    MLPWindowExtractor,
)
from src.utils.provenance import provenance

import gymnasium as gym

EXTRACTORS = {
    "lem": LEMFeaturesExtractor,
    "gru": GRUWindowExtractor,
    "mlp": MLPWindowExtractor,
}


def make_space(seq_len, market_dim):
    return gym.spaces.Dict({
        "window": gym.spaces.Box(-np.inf, np.inf, (seq_len, market_dim), np.float32),
        "agent": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
        "regime": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
    })


def bench(cls, space, device, batch_size, n_warmup, n_iters):
    torch.manual_seed(0)
    model = cls(space).to(device).eval()
    seq_len, market_dim = space["window"].shape
    batch = {
        "window": torch.randn(batch_size, seq_len, market_dim, device=device),
        "agent": torch.randn(batch_size, 3, device=device),
        "regime": torch.randn(batch_size, 4, device=device),
    }

    with torch.no_grad():
        for _ in range(n_warmup):
            model(batch)
        if device == "cuda":
            torch.cuda.synchronize()

        samples = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            model(batch)
            if device == "cuda":
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)

    s = np.asarray(samples)
    return {
        "extractor": cls.__name__,
        "params": sum(p.numel() for p in model.parameters()),
        "batch_size": batch_size,
        "latency_ms_mean": round(float(s.mean()), 4),
        "latency_ms_p50": round(float(np.percentile(s, 50)), 4),
        "latency_ms_p95": round(float(np.percentile(s, 95)), 4),
        "latency_ms_std": round(float(s.std(ddof=1)), 4),
        "per_sample_ms": round(float(s.mean() / batch_size), 5),
    }


def main():
    ap = argparse.ArgumentParser(description="Extractor latency benchmark")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--market-dim", type=int, default=144)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-dir", default="experiments/benchmarks")
    args = ap.parse_args()

    space = make_space(args.seq_len, args.market_dim)
    print("=" * 72)
    print(f"LATENCY BENCHMARK  |  device={args.device}  batch={args.batch_size}  "
          f"seq_len={args.seq_len}")
    print("=" * 72)
    print("Latency only. Trading performance comes from run_ablations.py.\n")

    rows = [bench(cls, space, args.device, args.batch_size, args.warmup, args.iters)
            for cls in EXTRACTORS.values()]

    hdr = f"{'extractor':26s} {'params':>10s} {'mean ms':>9s} {'p50':>8s} {'p95':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['extractor']:26s} {r['params']:>10,} {r['latency_ms_mean']:>9.4f} "
              f"{r['latency_ms_p50']:>8.4f} {r['latency_ms_p95']:>8.4f}")

    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "latency_benchmark.json", "w") as fh:
        json.dump({"provenance": provenance(args), "device": args.device,
                   "results": rows}, fh, indent=2)
    print(f"\nsaved -> {out / 'latency_benchmark.json'}")


if __name__ == "__main__":
    main()
