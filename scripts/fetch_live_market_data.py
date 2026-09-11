#!/usr/bin/env python3
"""
Live Level-2 order book collection, per symbol, with real prices preserved.

What changed and why
====================
The previous version had four defects that made its output unusable:

1. INTERLEAVED SYMBOLS. BTC/USD, ETH/USD and SOL/USD were appended to one
   list in round-robin, so consecutive "ticks" were different assets. Any
   price path built from it jumped between ~$100k, ~$3k and ~$200 every step.
   Symbols are now collected into separate series and never mixed.

2. PRICES DESTROYED AT SAVE TIME. Every price column was rescaled so the mid
   became exactly 100.0. That discards the only genuine price information in
   the file. Real prices are now preserved, and the mid is saved explicitly
   so the RL environment can consume a causal price path.

3. LABEL LEAKED FROM AN OBSERVED FEATURE. The label was thresholded level-1
   imbalance -- feature index 42, inside the observation the agent receives.
   Composing that with the old env's label->price rule made price an exact
   function of the input vector. Labels are now the realised k-step-ahead mid
   move, computed from the real mid, and are marked as diagnostic only.

4. TEST SET WAS THE TRAIN SET. Rows were tiled up to 5,000 and perturbed with
   sigma=0.003 noise BEFORE the 80/20 split, so every test row was a jittered
   copy of a training row. Measured on the committed artefacts: nearest
   neighbour distance from test to train had median 0.0447, indistinguishable
   from the train-half-to-train-half control of 0.0452. There is no tiling now,
   the split is strictly temporal, and a purge gap separates the two sides.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, ".")

N_LEVELS = 10
N_FEATURES = 144


def construct_features(order_book):
    """
    Build a 144-d feature vector from one L2 snapshot, preserving real prices.

    Layout:
        0..39   : 10 levels of (ask_price, ask_vol, bid_price, bid_vol)
        40      : spread
        41      : mid price          <- REAL price, not rescaled
        42..44  : L1 / L5 / L10 volume imbalance
        45..64  : per-level relative distance of ask/bid from mid
        65..143 : reserved (zero); kept so the vector width matches the
                  downstream 144-d model input.
    """
    bids = order_book.get("bids", [])[:N_LEVELS]
    asks = order_book.get("asks", [])[:N_LEVELS]
    if len(bids) < N_LEVELS or len(asks) < N_LEVELS:
        return None

    f = np.zeros(N_FEATURES, dtype=np.float64)
    for i in range(N_LEVELS):
        f[i * 4 + 0] = float(asks[i][0])
        f[i * 4 + 1] = float(asks[i][1])
        f[i * 4 + 2] = float(bids[i][0])
        f[i * 4 + 3] = float(bids[i][1])

    best_ask, best_bid = f[0], f[2]
    if not (best_ask > 0 and best_bid > 0 and best_ask >= best_bid):
        return None  # crossed or empty book

    mid = (best_ask + best_bid) / 2.0
    f[40] = best_ask - best_bid
    f[41] = mid

    ask_v1, bid_v1 = f[1], f[3]
    f[42] = (bid_v1 - ask_v1) / (bid_v1 + ask_v1 + 1e-9)
    bid_v5 = sum(f[i * 4 + 3] for i in range(5))
    ask_v5 = sum(f[i * 4 + 1] for i in range(5))
    f[43] = (bid_v5 - ask_v5) / (bid_v5 + ask_v5 + 1e-9)
    bid_v10 = sum(f[i * 4 + 3] for i in range(N_LEVELS))
    ask_v10 = sum(f[i * 4 + 1] for i in range(N_LEVELS))
    f[44] = (bid_v10 - ask_v10) / (bid_v10 + ask_v10 + 1e-9)

    for i in range(N_LEVELS):
        f[45 + i] = (f[i * 4 + 0] - mid) / mid
        f[55 + i] = (mid - f[i * 4 + 2]) / mid

    return f


def collect(exchange, symbol, n_snapshots, poll_sec, pbar):
    """Collect one symbol's series. Returns (features, mids, timestamps)."""
    feats, mids, stamps = [], [], []
    consecutive_errors = 0

    while len(feats) < n_snapshots:
        t0 = time.time()
        try:
            ob = exchange.fetch_order_book(symbol, limit=N_LEVELS * 2)
            f = construct_features(ob)
            consecutive_errors = 0
            if f is not None:
                feats.append(f)
                mids.append(f[41])
                stamps.append(time.time())
                pbar.update(1)
        except Exception as exc:  # network / rate limit
            consecutive_errors += 1
            if consecutive_errors >= 10:
                raise RuntimeError(
                    f"{symbol}: 10 consecutive fetch failures, aborting. Last: {exc}"
                ) from exc
            time.sleep(min(2.0 * consecutive_errors, 10.0))

        elapsed = time.time() - t0
        if elapsed < poll_sec:
            time.sleep(poll_sec - elapsed)

    return (
        np.asarray(feats, dtype=np.float64),
        np.asarray(mids, dtype=np.float64),
        np.asarray(stamps, dtype=np.float64),
    )


def make_labels(mids, horizon, threshold):
    """
    k-step-ahead realised mid move, as a 3-class label (0=down, 1=flat, 2=up).

    DIAGNOSTIC / SUPERVISED USE ONLY. This is forward-looking by construction
    and must never influence price, reward, or observation in the RL env.
    The last `horizon` rows have no label and are marked -1.
    """
    n = len(mids)
    labels = np.full(n, -1, dtype=np.int64)
    if n <= horizon:
        return labels
    future_ret = (mids[horizon:] - mids[:-horizon]) / mids[:-horizon]
    labels[:-horizon] = np.where(
        future_ret > threshold, 2, np.where(future_ret < -threshold, 0, 1)
    )
    return labels


def temporal_split(n, test_frac, purge):
    """
    Strictly temporal split with a purge gap.

    The gap drops `purge` rows at the boundary so a label whose horizon spans
    the split cannot put future test information into the training set.
    """
    split_idx = int(n * (1.0 - test_frac))
    train_end = split_idx - purge
    if train_end <= 0 or split_idx >= n:
        raise ValueError(
            f"Cannot split {n} rows with test_frac={test_frac} and purge={purge}: "
            "collect more data or reduce the purge."
        )
    return slice(0, train_end), slice(split_idx, n)


def main():
    ap = argparse.ArgumentParser(description="Collect live L2 order book data")
    ap.add_argument("--exchange", default="coinbase")
    ap.add_argument("--symbols", nargs="+", default=["BTC/USD"])
    ap.add_argument("--snapshots", type=int, default=6000,
                    help="Snapshots PER SYMBOL")
    ap.add_argument("--poll-sec", type=float, default=0.35,
                    help="Seconds between REST polls (respect the rate limit)")
    ap.add_argument("--horizon", type=int, default=10,
                    help="Label horizon in snapshots")
    ap.add_argument("--threshold", type=float, default=2e-4,
                    help="Return threshold separating flat from up/down")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--purge", type=int, default=None,
                    help="Rows dropped at the split boundary (default: 5x horizon)")
    ap.add_argument("--out-dir", default="data/live_market")
    args = ap.parse_args()

    purge = args.purge if args.purge is not None else args.horizon * 5

    import ccxt
    exchange = getattr(ccxt, args.exchange)({"timeout": 10000, "enableRateLimit": True})

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"LIVE L2 COLLECTION  |  {args.exchange}  |  {len(args.symbols)} symbol(s)")
    print(f"{args.snapshots} snapshots each @ {args.poll_sec}s  "
          f"(~{args.snapshots * args.poll_sec / 60:.1f} min per symbol)")
    print("=" * 70)

    manifest = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "exchange": args.exchange,
        "poll_sec": args.poll_sec,
        "horizon": args.horizon,
        "threshold": args.threshold,
        "test_frac": args.test_frac,
        "purge": purge,
        "symbols": {},
    }

    for symbol in args.symbols:
        safe = symbol.replace("/", "").lower()
        sym_dir = out_root / safe
        sym_dir.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(total=args.snapshots, desc=f"{symbol:10s}", unit="snap")
        feats, mids, stamps = collect(exchange, symbol, args.snapshots, args.poll_sec, pbar)
        pbar.close()

        labels = make_labels(mids, args.horizon, args.threshold)
        tr, te = temporal_split(len(feats), args.test_frac, purge)

        # No tiling, no augmentation, no shuffling. What was collected is what
        # is saved, split strictly in time.
        np.savez_compressed(
            sym_dir / "train.npz",
            features=feats[tr], mid=mids[tr], timestamp=stamps[tr], label=labels[tr],
        )
        np.savez_compressed(
            sym_dir / "test.npz",
            features=feats[te], mid=mids[te], timestamp=stamps[te], label=labels[te],
        )

        span_min = (stamps[-1] - stamps[0]) / 60.0
        ret = np.diff(mids) / mids[:-1]
        sym_meta = {
            "symbol": symbol,
            "n_total": int(len(feats)),
            "n_train": int(feats[tr].shape[0]),
            "n_test": int(feats[te].shape[0]),
            "purged_rows": purge,
            "span_minutes": round(span_min, 2),
            "mid_first": float(mids[0]),
            "mid_last": float(mids[-1]),
            "mid_min": float(mids.min()),
            "mid_max": float(mids.max()),
            "tick_return_std": float(ret.std()),
            "frac_ticks_moved": float(np.mean(ret != 0)),
            "label_counts_train": {
                str(k): int(v) for k, v in
                zip(*np.unique(labels[tr], return_counts=True))
            },
        }
        manifest["symbols"][symbol] = sym_meta

        print(f"  {symbol}: {sym_meta['n_train']} train / {sym_meta['n_test']} test "
              f"over {span_min:.1f} min, mid {mids.min():.2f}-{mids.max():.2f}, "
              f"tick sd {ret.std():.2e}, moved {100*sym_meta['frac_ticks_moved']:.1f}%")

    with open(out_root / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nManifest written to {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
