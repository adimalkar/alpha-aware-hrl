# Alpha-Aware Hierarchical RL with a Transformer Hawkes Event Model

Continuous-time event modelling for limit-order-book trading: a Transformer
Hawkes Process (THP) encoder feeding a distributional RL agent (TQC), trained
and evaluated on collected Level-2 market data.

---

## Status

**The results previously published in this README were invalid and have been
withdrawn.** A validity audit ([AUDIT_FINDINGS.md](AUDIT_FINDINGS.md)) found 24
defects, 7 individually sufficient to void every number in `experiments/`. Those
artefacts are preserved under [`experiments/INVALIDATED/`](experiments/INVALIDATED/)
with a per-directory explanation.

The headline figure — `+6,260.91%` return, Sharpe `365.38` — was not an inflated
result. It came from an evaluation branch that was never reached:

```python
# scripts/run_cluster_training.py (removed)
if hasattr(eval_env.envs[0].env, "portfolio_value"):      # always False
    portfolio_values.append(...)
...
final_val    = portfolio_values[-1] if portfolio_values else initial_val * (1.0 + sum(episode_rewards) * 0.001)
sharpe_ratio = compute_sharpe(...) if len(returns_series) > 10 else 2.14
max_dd       = compute_max_drawdown(...) if len(portfolio_values) > 1 else 6.8
```

`HistoricalLOBEnv` never defined a `portfolio_value` attribute — only a method
and an info-dict key — so the list stayed empty, `final_portfolio` came from an
invented formula, and `max_drawdown_pct: 6.8` in both committed result files is
the literal `6.8` on the right-hand side of that `else`.

The codebase has been repaired. Current state:

| Component | State |
|---|---|
| Metrics | Fixed, 17 contract tests |
| Environment | Fixed, 16 contract tests incl. a leakage regression guard |
| Event encoder (LEM) | Trainable, receives gradients, 8 tests |
| Data pipeline | Rewritten; real prices, temporal split with purge |
| Ablation harness | Rewritten; honest arms, disjoint eval split |
| Dashboard | Rewritten; six pages rendered `Math.random()` as measurements |
| **Published results** | **First measurement below; RL evaluation in progress** |

No performance claim appears in this README until it is produced by the
repaired pipeline. That is the point.

### First measurement: the event encoder does not beat a Poisson baseline

Collected 3,950 train / 1,000 test BTC/USD Level-2 snapshots over 25 minutes
(mid $77,179.75–$77,332.08, tick return sd 9.87e-6, 4.7% of ticks moved the
mid). Pretraining the THP on its Hawkes log-likelihood for 60 epochs:

| | log-likelihood / event |
|---|---|
| THP (best val, epoch 59) | **−3.3171** |
| Homogeneous Poisson baseline | **−3.2030** |
| Delta | **−0.1140** — the THP loses |

This is explainable rather than surprising. REST polling on a fixed 0.3s grid
makes inter-arrival times nearly constant, so there is no arrival-time
structure for a Hawkes intensity to model — the encoder is being asked to
model a clock. A websocket feed giving true event-time arrivals is the change
that would make this test meaningful.

It is recorded here because it is the result. The previous version of this
project reported Sharpe 2.14 from this same code path.

---

## What the model is

| Tier | Component | Parameters | Trained here? |
|---|---|---|---|
| Micro | Transformer Hawkes encoder (`src/models/event_encoder.py`) | 514,172 | **Yes** — Hawkes NLL pretraining, then RL fine-tuning |
| Execution | TQC policy head, `net_arch=[256,256]` | ~2.4M | Yes |
| Macro | `TPPLoRARegimeDetector` (`src/models/tpp_regime.py`) | 1,662,154 | No — random init, not currently used in training |
| Macro | TinyLlama-1.1B (`src/agents/llm_analyst.py`) | 1.1B | No — pretrained, zero-shot prompting only |
| Macro | TimesFM-200M (`src/models/timesfm_wrapper.py`) | 200M | No — pretrained, frozen by design |

The previous README described this as a "1.3 Billion parameter hierarchical AI
stack". The trainable model is about **2.9M parameters**. The 1.3B figure summed
two off-the-shelf pretrained models that are not trained, not fine-tuned, and in
the case of TinyLlama bypassed entirely during training (`run_cluster_training.py`
constructed it and then used a precomputed regime). `TPPLoRARegimeDetector` has
no LoRA and no LLM — it is a hash-tokenised bag-of-embeddings.

### The event model

Order-book snapshots are mapped to a 20-type microstructure event vocabulary
(`src/utils/event_pipeline.py`), embedded with continuous inter-arrival times,
and encoded by a 3-layer causal transformer that predicts per-type conditional
intensities λ_k(t). The encoder is pretrained by maximising the Hawkes
log-likelihood

    LL = Σᵢ log λ_{kᵢ}(tᵢ) − ∫ Σ_k λ_k(s) ds

against a homogeneous-Poisson baseline, then fine-tuned by the RL objective.

---

## Data: why not FI-2010

FI-2010 as distributed **cannot support a trading simulation**, and the original
code hid this by deriving the traded price from the label:

```python
label = self.labels[data_idx]        # k-step-ahead direction
if   label == 2: self.current_price *= 1.001
elif label == 0: self.current_price *= 0.999
```

The label encodes the future, so reward at *t* was a deterministic function of
*t+k* while the observation was the feature vector that label describes.

Reconstructing a mid-price from the book does not work either. The CSVs are
z-scored **per column**, which destroys order-book geometry. Measured over the
first 50,000 training rows:

| Property | Should hold | Actually holds |
|---|---|---|
| `ask₁ > bid₁` | 100% | 51.3% |
| `ask₁ < ask₂` | ~100% | 27.6% |
| `bid₁ > bid₂` | ~100% | 24.2% |
| reconstructed mid > 0 | always | crosses zero (min −1.065) |

Every price column was standardised to the same distribution (mean ≈ −0.215,
sd ≈ 0.676). No mid, spread, or return survives. FI-2010 supports exactly one
task — supervised classification of the supplied label — which is what DeepLOB
uses it for.

`HistoricalLOBEnv` therefore requires an explicit price series and raises
`PriceSourceError` rather than fabricating one.

---

## Reproduce

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements.lock.txt   # pinned, Python 3.14.6

# 1. Collect real Level-2 data (per symbol; real prices preserved,
#    strictly temporal split with a purge gap, no tiling)
./.venv/bin/python scripts/fetch_live_market_data.py \
    --symbols BTC/USD ETH/USD --snapshots 5000 --poll-sec 0.3

# 2. Pretrain the event encoder on the Hawkes likelihood
./.venv/bin/python scripts/pretrain_event_encoder.py --symbol BTC/USD

# 3. Train the agent, fine-tuning the encoder end-to-end
./.venv/bin/python scripts/train_rl_agent.py --symbol BTC/USD --encoder lem \
    --pretrained checkpoints/event_encoder_thp.pt

# 4. Ablate against honest controls
./.venv/bin/python scripts/run_ablations.py --symbol BTC/USD \
    --arms lem lem_frozen gru mlp --seeds 0 1 2

# Tests
./.venv/bin/python -m pytest tests/ -q
```

`lem_frozen` deliberately reproduces the original broken configuration — encoder
frozen, no gradients — as the control that shows what the withdrawn numbers
actually measured.

Every results file carries a `provenance` block: commit SHA, dirty flag,
interpreter version, seed, and SHA-256 digests of the input data.

### Known limitation of the current data path

REST polling at ~3 snapshots/sec moves the BTC mid on roughly 8% of ticks, so
returns are sparse and mostly zero. This is a property of polled data, stated
rather than papered over. A websocket feed would give genuine event-time
arrivals and is the natural next step — it would also let the THP model real
inter-arrival times instead of a uniform sampling grid.

---

## Layout

```
├── src/
│   ├── envs/
│   │   ├── historical_lob_env.py   # replay env; refuses to fabricate a price
│   │   └── sequence_wrapper.py     # emits raw windows for the policy to encode
│   ├── models/
│   │   ├── event_encoder.py        # Transformer Hawkes Process (the LEM)
│   │   ├── tpp_regime.py           # news-event regime detector (untrained)
│   │   └── timesfm_wrapper.py      # pretrained alpha forecaster (frozen)
│   ├── agents/
│   │   └── lem_extractor.py        # SB3 feature extractors (gradients reach these)
│   └── utils/
│       ├── metrics.py              # unit-enforced financial metrics
│       ├── provenance.py           # commit + data digests for every artefact
│       ├── event_pipeline.py       # 20-type event vocabulary
│       └── data_loader.py          # FI-2010 + collected-market loaders
├── scripts/
│   ├── fetch_live_market_data.py   # L2 collection, real prices, purged split
│   ├── pretrain_event_encoder.py   # Hawkes NLL pretraining + Poisson baseline
│   ├── train_rl_agent.py           # RL training and out-of-sample evaluation
│   ├── run_ablations.py            # honest arms, disjoint eval split
│   └── run_event_benchmark.py      # latency only
├── tests/                          # 51 tests
├── AUDIT_FINDINGS.md               # the 24 defects
└── experiments/INVALIDATED/        # withdrawn results, retained for the record
```

---

## Audit summary

Full detail in [AUDIT_FINDINGS.md](AUDIT_FINDINGS.md). The defects that voided
the published results:

| ID | Defect | Status |
|---|---|---|
| L1 | Price path computed from the forward-looking label | Fixed — env requires a real price; regression test |
| L2 | Live label was thresholded `feature[42]`, inside the observation | Fixed — labels are realised future moves, diagnostic only |
| X1 | Test set tiled from train set + σ=0.003 noise **before** the split | Fixed — no tiling; temporal split with purge |
| X2 | Ablation trained and evaluated on the same env object | Fixed — disjoint splits, different seeds |
| D1 | Feature extractor never received a gradient | Fixed — moved into the policy; 8 tests |
| D2 | Regime signal constant across all 362,400 rows | Documented; component excluded from training |
| S1 | Mamba-vs-LSTM ablation contained no Mamba arm | Fixed — arms named for what they run |
| M1 | Daily risk-free rate subtracted from tick returns, inverting Sharpe | Fixed — raises on frequency mismatch |
| M2 | Max drawdown double-scaled (6619% reported) | Fixed — asserted into [0, 100] |
| M3 | Hardcoded metric constants substituted for measurement | Fixed — evaluation raises instead |
| R1 | Committed venv lacked sb3, pandas, flask; could not run the scripts | Fixed — `requirements.lock.txt` |
