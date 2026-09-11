#!/usr/bin/env python3
"""
Pretrain the Transformer Hawkes Process (the "LEM") on its own likelihood.

Why this script exists
======================
`TransformerHawkesEncoder.compute_log_likelihood` -- the objective the whole
architecture is built around -- was never called anywhere outside a unit test.
The encoder was instantiated fresh inside the environment, run under
`torch.no_grad()`, and never checkpointed, so every result attributed to the
"Large Event Model" came from a randomly initialised, frozen feature map.

This trains it for real, on the Hawkes log-likelihood, against a temporally
held-out validation split, and reports a homogeneous Poisson baseline so the
learned model has to demonstrate it beats a constant intensity.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

from src.models.event_encoder import TransformerHawkesEncoder
from src.utils.data_loader import LiveMarketDataLoader
from src.utils.event_pipeline import NUM_EVENT_TYPES, EventDataset, EventStreamPipeline


def build_events(loader, split, sensitivity):
    """Turn a split into an EventSequence using REAL collected timestamps."""
    feats, labels = loader.load(split)
    # 'val' is carved from the tail of the train file by the loader, so the
    # timestamps must be sliced the same way rather than re-read wholesale.
    source = "train" if split in ("train", "val") else split
    path = loader.sym_dir / f"{source}.npz"
    with np.load(path) as z:
        stamps = z["timestamp"].astype(np.float64)
    if split in ("train", "val") and loader.val_frac > 0:
        n = len(stamps)
        cut = int(n * (1.0 - loader.val_frac))
        stamps = stamps[: cut - loader.purge] if split == "train" else stamps[cut:]
    if len(stamps) != len(feats):
        raise RuntimeError(
            f"timestamp/feature length mismatch for split {split!r}: "
            f"{len(stamps)} vs {len(feats)}"
        )

    pipeline = EventStreamPipeline(sensitivity_threshold=sensitivity, feature_dim=feats.shape[1])
    seq = pipeline.detect_lob_events(feats, labels, timestamps=stamps)
    return seq


def poisson_baseline_ll(seq, n_types=NUM_EVENT_TYPES):
    """
    Per-event log-likelihood of a homogeneous Poisson process fitted by MLE.

    For a constant intensity lambda_k per type, the MLE is n_k / T, and the
    per-event LL is  mean_i[ log lambda_{k_i} ] - (sum_k lambda_k) * T / N.
    Any THP that cannot beat this has learned nothing about event timing.
    """
    dts = np.asarray(seq.inter_arrival_times, dtype=np.float64)
    types = np.asarray(seq.event_types, dtype=np.int64)
    total_time = float(dts.sum())
    n = len(types)
    if n == 0 or total_time <= 0:
        return float("nan")

    counts = np.bincount(types, minlength=n_types).astype(np.float64)
    rates = counts / total_time
    log_rate_per_event = np.log(np.maximum(rates[types], 1e-12))
    compensator_per_event = rates.sum() * total_time / n
    return float(log_rate_per_event.mean() - compensator_per_event)


def evaluate(model, loader, device):
    model.eval()
    total, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            ll = model.compute_log_likelihood(
                event_types=batch["event_types"].to(device),
                timestamps=batch["timestamps"].to(device),
                inter_arrival_times=batch["inter_arrival_times"].to(device),
                features=batch["features"].to(device),
                mask=batch["mask"].to(device),
            )
            total += float(ll)
            n_batches += 1
    return total / max(n_batches, 1)


def main():
    ap = argparse.ArgumentParser(description="Pretrain the THP event encoder")
    ap.add_argument("--data-dir", default="data/live_market")
    ap.add_argument("--symbol", default="BTC/USD")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--sensitivity", type=float, default=0.03)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="checkpoints/event_encoder_thp.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 68)
    print("PRETRAIN: Transformer Hawkes Process on event log-likelihood")
    print("=" * 68)

    loader = LiveMarketDataLoader(args.data_dir, args.symbol)
    # Validation comes from 'val', NOT 'test'. Early stopping on the test split
    # is test-set reuse: the checkpoint is selected using the same data later
    # used to report the result (audit X3).
    train_seq = build_events(loader, "train", args.sensitivity)
    val_seq = build_events(loader, "val", args.sensitivity)

    print(f"  train events: {len(train_seq):>6}   val events: {len(val_seq):>6}")
    if len(train_seq) < args.seq_len * 2:
        raise SystemExit(
            f"Only {len(train_seq)} training events detected. Collect more data or "
            f"lower --sensitivity (currently {args.sensitivity})."
        )

    feature_dim = train_seq.features.shape[1]
    train_ds = EventDataset(train_seq, seq_len=args.seq_len, stride=args.stride)
    val_ds = EventDataset(val_seq, seq_len=args.seq_len, stride=args.seq_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"  train windows: {len(train_ds)}   val windows: {len(val_ds)}")

    base_train = poisson_baseline_ll(train_seq)
    base_val = poisson_baseline_ll(val_seq)
    print(f"  homogeneous Poisson baseline LL/event -- train {base_train:+.4f}  val {base_val:+.4f}")

    model = TransformerHawkesEncoder(
        num_event_types=NUM_EVENT_TYPES,
        feature_dim=feature_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val, best_epoch, bad_epochs = -float("inf"), -1, 0
    history = []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_ll, n_batches = 0.0, 0
        for batch in train_dl:
            opt.zero_grad(set_to_none=True)
            ll = model.compute_log_likelihood(
                event_types=batch["event_types"].to(device),
                timestamps=batch["timestamps"].to(device),
                inter_arrival_times=batch["inter_arrival_times"].to(device),
                features=batch["features"].to(device),
                mask=batch["mask"].to(device),
            )
            loss = -ll  # maximise likelihood
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_ll += float(ll)
            n_batches += 1
        sched.step()

        train_ll = epoch_ll / max(n_batches, 1)
        val_ll = evaluate(model, val_dl, device)
        history.append({"epoch": epoch, "train_ll": train_ll, "val_ll": val_ll})
        flag = ""

        if val_ll > best_val:
            best_val, best_epoch, bad_epochs = val_ll, epoch, 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {
                        "num_event_types": NUM_EVENT_TYPES,
                        "feature_dim": feature_dim,
                        "d_model": args.d_model,
                        "n_heads": args.n_heads,
                        "n_layers": args.n_layers,
                    },
                    "val_ll": val_ll,
                    "poisson_baseline_val_ll": base_val,
                    "epoch": epoch,
                    "symbol": args.symbol,
                },
                args.out,
            )
            flag = "  <- best, saved"
        else:
            bad_epochs += 1

        print(f"  epoch {epoch:3d}  train LL {train_ll:+.4f}  val LL {val_ll:+.4f}{flag}")
        if bad_epochs >= args.patience:
            print(f"  early stop: no val improvement for {args.patience} epochs")
            break

    elapsed = time.time() - t0
    beats = best_val > base_val
    print("\n" + "=" * 68)
    print(f"  best val LL/event   : {best_val:+.4f}  (epoch {best_epoch})")
    print(f"  Poisson baseline    : {base_val:+.4f}")
    print(f"  THP beats baseline  : {beats}  (delta {best_val - base_val:+.4f})")
    print(f"  trained in {elapsed:.1f}s, checkpoint -> {args.out}")
    print("=" * 68)
    if not beats:
        print("  WARNING: the learned model does not beat a constant intensity.")
        print("  Do not describe this encoder as having learned event dynamics.")

    with open(Path(args.out).with_suffix(".history.json"), "w") as fh:
        json.dump(
            {
                "history": history,
                "best_val_ll": best_val,
                "best_epoch": best_epoch,
                "poisson_baseline_val_ll": base_val,
                "poisson_baseline_train_ll": base_train,
                "beats_baseline": bool(beats),
                "n_params": n_params,
                "args": vars(args),
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
