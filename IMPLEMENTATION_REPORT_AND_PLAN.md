# Alpha-Aware Hierarchical Reinforcement Learning — Implementation Report & Plan

**Project:** Alpha-Aware Hierarchical RL (Mamba-DSAC)  
**Target:** ICML Workshop Submission  
**Report Date:** January 2025  
**Status:** Steps 1–4 Complete; Steps 5–8 Pending  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Implementation Status](#3-implementation-status)
4. [Detailed Implementation Report](#4-detailed-implementation-report)
5. [Datasets](#5-datasets)
6. [Training Results](#6-training-results)
7. [Configuration](#7-configuration)
8. [Remaining Plan (Steps 5–8)](#8-remaining-plan-steps-58)
9. [How to Run](#9-how-to-run)
10. [File Reference](#10-file-reference)

---

## 1. Executive Summary

This document summarizes everything implemented so far in the **Alpha-Aware Hierarchical Reinforcement Learning** project and provides a downloadable plan for the remaining work.

**Completed (Steps 1–4):**

- Project structure, configs, and dependencies
- **Feature extractor** for LOB data: Mamba SSM with **LSTM fallback** (Mamba not installable on current environment; LSTM used and fine-tuned)
- **FI-2010 data pipeline**: load, sequence creation, train/val/test splits
- **LOB classifier training**: LSTM, BiLSTM, TCN, Transformer with full training script
- **Achieved ~80% test accuracy** on FI-2010 mid-price prediction with LSTM (competitive with SOTA)
- Placeholder/stub implementations for: LLM Analyst, DSAC Trader, Hierarchical Agent, ABIDES wrapper, TimesFM, metrics

**Pending (Steps 5–8):**

- Integrate TQC/DSAC from SB3-Contrib with custom observation space
- Add TimesFM as frozen alpha signal extractor
- Implement LLM-based regime detector (TinyLlama/Phi-2)
- Wire full HierarchicalAgent and run end-to-end
- Ablation study (TCN+PPO, Mamba+PPO, TCN+DSAC, Full)
- OOD transfer (Tech → Energy)
- 5-seed robustness and confidence intervals

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     HIERARCHICAL AGENT                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  [Analyst Layer]   LLM + TimesFM  →  Regime signal from news + alpha      │
│  [Manager Layer]   Mamba/LSTM    →  State representation from LOB        │
│  [Trader Layer]    DSAC/TQC      →  Risk-aware trade execution           │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                     ABIDES SIMULATOR                                      │
│              (Market impact & latency modeling)                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Design choices:**

- **Mamba SSM:** Target backbone for LOB; not installable in current setup (Python 3.13, PyTorch 2.10, CUDA 12.8) → **LSTM fallback** used and tuned.
- **LLM:** Regime detection from financial text (TinyLlama/Phi-2).
- **TimesFM:** Frozen time-series foundation model for alpha signals.
- **DSAC/TQC:** Distributional RL for risk-aware execution in ABIDES.

---

## 3. Implementation Status

| Component              | Status   | Notes |
|------------------------|----------|--------|
| Project layout & config| Done     | `configs/default.yaml`, `requirements.txt` |
| MambaFeatureExtractor  | Done     | Backend: `mamba` \| `lstm` \| `auto`; LSTM used in practice |
| LOBClassifier         | Done     | Classification head on top of extractor |
| FI-2010 data loader    | Done     | 144 features, 5 horizons, label mapping 1,2,3 → 0,1,2 |
| Sequence creation      | Done     | Configurable `seq_len`, chronological splits |
| Full training script   | Done     | LSTM, BiLSTM, TCN, Transformer; OneCycleLR, early stop |
| Metrics (PnL, Sharpe, etc.) | Done | `src/utils/metrics.py` |
| LLMAnalyst             | Stub     | Placeholder for regime detection |
| DSACTrader             | Stub     | Placeholder for SB3 TQC/DSAC |
| HierarchicalAgent      | Stub     | Placeholder orchestration |
| ABIDES wrapper         | Stub     | Gymnasium wrapper; mock dynamics |
| TimesFM wrapper        | Stub     | Placeholder for alpha signals |
| TCNEncoder             | Done     | In `src/models/mamba_ssm.py` for ablations |
| Ablation pipeline      | Pending  | TCN+PPO, Mamba+PPO, TCN+DSAC, Full |
| OOD / 5-seed           | Pending  | Transfer + robustness |

---

## 4. Detailed Implementation Report

### 4.1 Feature Extractor (`src/agents/mamba_extractor.py`)

**Purpose:** Turn LOB sequences into a fixed-size state representation for the trader.

**Implemented:**

- **MambaFeatureExtractor**
  - `input_dim=144` (FI-2010), `d_model`, `n_layers`, optional `d_state`, `expand`, `dropout`.
  - `backend`: `"mamba"` | `"lstm"` | `"auto"`. If `mamba-ssm` is missing, falls back to LSTM.
  - Input projection → Mamba or LSTM backbone → LayerNorm → dropout → output `(batch, d_model)`.
- **LOBClassifier**
  - Wraps the extractor with a linear classifier for 3 classes (down / stable / up) for benchmarking.

**LSTM backbone (current production path):**

- 2-layer LSTM, hidden size `d_model`, dropout between layers, last hidden state used.

**Mamba backbone (when library available):**

- Stack of `Mamba` blocks from `mamba_ssm`, same `d_model` and layer count.

### 4.2 Data Pipeline (`src/utils/data_loader.py`)

**FI2010DataLoader:**

- Reads `FI2010_train.csv` and `FI2010_test.csv`.
- **Features:** columns 0–143 (144-D).
- **Labels:** columns 144–148 for horizons k=10,20,30,50,100; values 1,2,3 mapped to 0,1,2.
- `load(split)`, `load_all()`, `get_sequences(seq_len)` for sliding-window sequences.
- Chronological train/val/test handling (no shuffle across time).

**FNSPID:**

- Placeholder/notes for financial news; not wired into training yet.

### 4.3 LOB Classifier Training (`scripts/train_lob_classifier.py`)

**Purpose:** Train and evaluate LOB mid-price movement classifiers (and compare backbones).

**Implemented:**

- **Models:** `lstm`, `bilstm`, `tcn`, `transformer` (all consume sequences of shape `(batch, seq_len, 144)`).
- **Training:** CrossEntropyLoss, AdamW, OneCycleLR, gradient clipping, early stopping (patience 10).
- **Data:** FI-2010 from `data/fi2010/FI2010/`; optional `--max_samples` for memory control.
- **Splits:** 85% train, 15% val (from train), separate test set.
- **Checkpointing:** Best model by validation accuracy saved under `checkpoints/`.
- **Evaluation:** Test accuracy, per-class accuracy, prediction distribution.

**Usage examples:**

```bash
python scripts/train_lob_classifier.py --model lstm --epochs 30 --max_samples 100000
python scripts/train_lob_classifier.py --model transformer --epochs 30
./scripts/run_model_comparison.sh   # Runs lstm, bilstm, tcn, transformer
```

### 4.4 Metrics (`src/utils/metrics.py`)

**Implemented:**

- `compute_pnl(prices, positions, initial_capital)` → equity curve, total return.
- `compute_sharpe(returns, risk_free_rate, periods_per_year)` → annualized Sharpe.
- `compute_max_drawdown(equity_curve)` → max drawdown and related.
- Additional helpers for trading evaluation (as needed for ICML).

### 4.5 Models (`src/models/`)

- **mamba_ssm.py:** `MambaBlock`, `MambaEncoder` (simplified PyTorch-only SSM); **TCNEncoder** (causal convolutions) for TCN baseline and ablations.
- **timesfm_wrapper.py:** Stub for TimesFM alpha signals.

### 4.6 Agents (stubs)

- **llm_analyst.py:** LLM-based regime detection; interface only.
- **dsac_trader.py:** DSAC/TQC trader; to be wired to SB3-Contrib.
- **hierarchical_agent.py:** Top-level agent combining analyst, feature extractor, and trader; placeholder.

### 4.7 Environment (`src/envs/abides_wrapper.py`)

- Gymnasium wrapper for ABIDES; observation/action spaces and step logic sketched; mock market dynamics for now.

---

## 5. Datasets

| Dataset   | Purpose           | Location                    | Status |
|-----------|-------------------|-----------------------------|--------|
| FI-2010   | LOB features/labels | `data/fi2010/FI2010/`       | In use |
| FNSPID    | Financial news    | HuggingFace / local         | Not wired |
| ABIDES    | Simulation        | External repo + wrapper     | Wrapper stub |

**FI-2010 details:**

- 144 normalized LOB features; 5 label columns (mid-price movement over 5 horizons).
- Train/test CSVs; we use 85/15 train/val split from train and hold out test.

---

## 6. Training Results

**LOB mid-price prediction (3 classes: down / stable / up):**

| Setting        | Epochs | Samples | Model   | Val Acc | Test Acc |
|----------------|--------|---------|---------|---------|----------|
| Quick test     | 3      | 5,000   | LSTM 64d, 2L | ~50% | 48.7% |
| Full training  | 10     | 50,000  | LSTM 128d, 4L | **80.81%** | **80.32%** |

**Per-class test accuracy (LSTM, 10 epochs, 50k samples):**

- Down: 52.19%
- Stable: 91.45%
- Up: 64.08%

Conclusion: The **LSTM fallback**, when properly trained, reaches **~80% test accuracy**, which is competitive with FI-2010 benchmarks and sufficient to proceed with Step 5 (DSAC) and ablations.

---

## 7. Configuration

**File:** `configs/default.yaml`

- **model.mamba:** `d_model`, `n_layers`, `d_state`, `expand`, `dropout`
- **model.dsac:** lr, buffer_size, batch_size, tau, gamma, n_critics, top_quantiles_to_drop
- **model.llm:** model_name, max_tokens
- **training:** total_timesteps, eval_freq, n_eval_episodes, seeds
- **data:** lob_sequence_length, prediction_horizons, train/val/test splits
- **environment:** simulator, market_impact, latency_ms
- **experiment:** name, log_dir, save_dir

---

## 8. Remaining Plan (Steps 5–8)

### Step 5: DSAC/TQC Trader

- [ ] Implement custom Gymnasium observation space (LOB features + optional regime/alpha).
- [ ] Integrate SB3-Contrib TQC or DSAC with this observation space.
- [ ] Train in ABIDES wrapper (or simplified env) and log rewards, Sharpe, drawdown.

### Step 6: TimesFM & LLM

- [ ] Integrate TimesFM as frozen feature extractor for alpha signals.
- [ ] Implement LLMAnalyst (TinyLlama/Phi-2) for regime from news; output regime embedding or discrete label.
- [ ] Feed regime + alpha into observation or policy (e.g., concatenated to LOB state).

### Step 7: Full Hierarchical Agent & Ablation

- [ ] HierarchicalAgent: Analyst → state; Mamba/LSTM → state; concatenate with alpha/regime; Trader (DSAC) → action.
- [ ] Ablation: TCN+PPO, Mamba+PPO (or LSTM+PPO if Mamba unavailable), TCN+DSAC, Full (LSTM+DSAC+LLM+TimesFM).
- [ ] Same seeds and env settings across conditions.

### Step 8: OOD & Robustness

- [ ] OOD: Train on one sector (e.g., Tech), test on another (e.g., Energy) in simulator or data.
- [ ] 5-seed runs for each configuration; report mean and confidence intervals; generate plots.

---

## 9. How to Run

**Environment:**

```bash
cd alpha-aware-hrl
python3 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**LOB classifier (single model):**

```bash
python scripts/train_lob_classifier.py --model lstm --epochs 30 --max_samples 100000
python scripts/train_lob_classifier.py --model tcn --epochs 30
python scripts/train_lob_classifier.py --model transformer --epochs 30
```

**Compare all LOB models:**

```bash
./scripts/run_model_comparison.sh
```

**Quick tests (synthetic / small real data):**

```bash
python scripts/quick_test.py
python scripts/test_with_real_data.py
```

**Checkpoints:** Best model per run is saved under `checkpoints/<model>_best.pt`.

---

## 10. File Reference

| Path | Description |
|------|-------------|
| `configs/default.yaml` | Hyperparameters and experiment settings |
| `requirements.txt` | Python dependencies |
| `README.md` | Project overview and structure |
| `src/agents/mamba_extractor.py` | MambaFeatureExtractor, LOBClassifier (Mamba/LSTM) |
| `src/agents/llm_analyst.py` | LLM regime analyst (stub) |
| `src/agents/dsac_trader.py` | DSAC/TQC trader (stub) |
| `src/agents/hierarchical_agent.py` | Hierarchical agent (stub) |
| `src/envs/abides_wrapper.py` | ABIDES Gymnasium wrapper (stub) |
| `src/models/mamba_ssm.py` | MambaBlock, MambaEncoder, TCNEncoder |
| `src/models/timesfm_wrapper.py` | TimesFM wrapper (stub) |
| `src/utils/data_loader.py` | FI-2010 (and FNSPID) loaders |
| `src/utils/metrics.py` | PnL, Sharpe, drawdown, etc. |
| `scripts/train_lob_classifier.py` | Full LOB classifier training (lstm/bilstm/tcn/transformer) |
| `scripts/run_model_comparison.sh` | Run all LOB model comparisons |
| `scripts/test_feature_extractor.py` | Feature extractor test (FI-2010) |
| `scripts/quick_test.py` | Quick test with synthetic data |
| `scripts/test_with_real_data.py` | Test with chunked real FI-2010 |

---

**End of Report.**  
You can download this file as `IMPLEMENTATION_REPORT_AND_PLAN.md` from your project root:  
`alpha-aware-hrl/IMPLEMENTATION_REPORT_AND_PLAN.md`.
