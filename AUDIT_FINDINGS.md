# `alpha-aware-hrl` — Validity Audit

**Date:** 2026-09-01
**Scope:** `src/`, `scripts/`, `api/`, `experiments/`, `data/` (venv and `abides-jpmc-public/` excluded)
**Method:** static read of all 27 project Python files + direct measurement of committed artifacts
**Status:** audit only — no code changed
**Result:** **24 defects.** 7 are individually sufficient to invalidate every performance number in `experiments/`.

Each finding carries a taxonomy tag mapping it to a product check in `PRODUCT_PLAN.md` §3:

| Tag | Check | Status in plan |
|---|---|---|
| **C1** | Temporal leakage | Check 1 |
| **C2** | Evaluation contamination | Check 2 |
| **C3** | Degenerate components | Check 3 |
| **M** | Metric sanity | Free bundle |
| **S** | Claim–evidence mismatch | **Proposed Check 4 — not currently in the plan** |
| **R** | Reproducibility / provenance | **Proposed bundle — not currently in the plan** |

---

## Severity index

| ID | Sev | Tag | Finding | Location |
|---|---|---|---|---|
| L1 | Critical | C1 | Env price path is computed from a forward-looking label | `src/envs/historical_lob_env.py:148` |
| L2 | Critical | C1 | Live-data price is a deterministic function of an observed feature | `scripts/fetch_live_market_data.py:110` |
| X1 | Critical | C2 | Live test set is noise-perturbed duplicates of the train set | `scripts/fetch_live_market_data.py:135` |
| X2 | Critical | C2 | Ablation trains and evaluates on the same env instance | `scripts/run_ablations.py:167,223` |
| D1 | Critical | C3 | Feature extractor never receives a gradient | `src/envs/hierarchical_wrapper.py:148` |
| D2 | Critical | C3 | Regime signal is constant — measured, all splits | `data/precomputed_regimes/*.npy` |
| S1 | Critical | S | The ablation contains no Mamba arm | `scripts/run_ablations.py:82` |
| L3 | High | C1 | No purge/embargo at the train/val boundary | `src/utils/data_loader.py:154` |
| L4 | High | C1 | k-step label horizon not embargoed | `src/utils/data_loader.py:137` |
| X3 | High | C2 | No sealed holdout; test set reused across every run | repo-wide |
| D3 | High | C3 | Confidence is chunk-broadcast: 8 values over 362,400 rows | `data/precomputed_regimes/train_confidences.npy` |
| D4 | High | C3 | Alpha model is random-init, `.eval()`, never trained | `src/agents/hierarchical_agent.py:84` |
| D5 | High | C3 | Agent never trades — reward exactly 0.0 across configs | `experiments/*/[ablation,ood]_results.json` |
| M1 | High | M | Risk-free units mismatch inverts the sign of Sharpe | `src/utils/metrics.py:63` |
| M3 | High | M | Hardcoded metric fallbacks substitute for measurement | `scripts/run_cluster_training.py:215` |
| S2 | High | S | `full` vs `lstm_dsac` differ only by two dead inputs | `scripts/run_ablations.py:103` |
| S3 | High | S | 500-timestep runs presented as ablation/OOD/robustness results | `experiments/` |
| R1 | High | R | venv is broken; no result in `experiments/` can be regenerated | `venv/bin/python3` |
| D6 | Med | C3 | Position and cash absent from the observation — non-Markov | `src/envs/historical_lob_env.py:68` |
| M2 | Med | M | Max drawdown double-scaled → 986%, 6620% | `scripts/run_baselines.py:209` |
| M4 | Med | M | Annualisation inconsistent: 252 vs 252·24·60 | `metrics.py:46` vs `run_cluster_training.py:215` |
| M5 | Med | M | No buy-and-hold or market benchmark anywhere in the repo | repo-wide |
| S4 | Med | S | All five horizon columns identical on live data | `scripts/fetch_live_market_data.py:152` |
| R2 | Med | R | Dataset substitution is silent: crypto LOB named `FI2010_*.csv` | `data/live_market/` |

---

## C1 — Temporal leakage

### L1 (Critical) — The price the agent trades is manufactured from the label it is scored on

`src/envs/historical_lob_env.py:148-154`:
```python
label = self.labels[data_idx]
if label == 2:   self.current_price *= (1 + 0.001)
elif label == 0: self.current_price *= (1 - 0.001)
```

FI-2010 labels encode the **future** k-step mid-price direction. Reward at step *t* is therefore a deterministic function of information from *t+k*, while the observation `self.data[data_idx]` is precisely the feature vector that label describes. A perfect map from observation to future reward exists by construction.

The code documents its own defect at `historical_lob_env.py:79-83`: *"For a truly realistic setup, we'd need the un-normalized mid-prices which FI-2010 drops."*

**Reproduction:** instrument `step()` to log `(label, price_delta)`; correlation is exactly 1.0. Or invert: a policy that reads the label achieves a riskless 10 bp/tick.

**Blast radius:** every number in `experiments/baselines/`, and every metric downstream of `HistoricalLOBEnv`.

**Detection signature (artifact-only):** cross-correlate the realised return series against each feature column at lags −k…+k. A causal price path has no peak at negative lag; this one peaks at exactly the label horizon.

---

### L2 (Critical) — On live data, price is a deterministic function of an *observed* feature

`scripts/fetch_live_market_data.py:110-118`:
```python
imbalance = feat_144d[42]
if imbalance > 0.10:    lbl = 2   # up
elif imbalance < -0.10: lbl = 0   # down
else:                   lbl = 1   # stable
```

The label is thresholded order-book imbalance — feature index 42, **which is inside the observation the agent receives**. `HistoricalLOBEnv` then converts that label into the price path (L1). Composing the two: `price_delta = f(observation[42])`, exactly and noiselessly.

This is worse than ordinary lookahead. It is not that the agent sees the future; it is that the future is a two-line function of a number sitting in its input vector. Any learner that finds `feature[42]` earns a riskless return forever.

**Reproduction:** `np.corrcoef(features[:,42], price_deltas)` — deterministic step function.

---

### L3 (High) — No purge or embargo at the train/val boundary

`src/utils/data_loader.py:154-159` splits **already-overlapping** sliding windows chronologically:
```python
n_val   = int(len(X_train_full) * val_ratio)
n_train = len(X_train_full) - n_val
X_train = X_train_full[:n_train]
X_val   = X_train_full[n_train:]
```
With `sequence_length=100`, train window `n_train-1` spans timesteps `[n_train-1, n_train+98]` and val window `n_train` spans `[n_train, n_train+99]` — **99 shared timesteps**. Validation is scored partly on data the model fit.

**Fix class:** purge `sequence_length` samples at the boundary, then embargo a further `horizon`.

### L4 (High) — Label horizon extends past the boundary
The k-step label at the last train window is derived from prices up to `t+k`, which fall inside validation. Even a correct window purge leaves this. Requires `sequence_length + horizon` total separation.

---

## C2 — Evaluation contamination

### X1 (Critical) — The live test set is the live train set with noise added

`scripts/fetch_live_market_data.py:135-147`:
```python
if len(all_features) < 5000:
    repeats = int(np.ceil(5000 / len(all_features)))
    all_features = np.tile(all_features, (repeats, 1))[:5000]
    all_features += np.random.normal(0, 0.003, size=all_features.shape)
    all_labels   = np.tile(all_labels, repeats)[:5000]

split_idx = int(len(all_features) * 0.8)
train_feats, test_feats = all_features[:split_idx], all_features[split_idx:]
```

**The tiling happens before the split.** Every row in the 20% test set is a σ=0.003 perturbation of a row the model trained on. Held-out performance measures memorisation of ~1,000 genuine snapshots, nothing more.

Committed evidence: `data/live_market/FI2010_train.csv` = 4,000 rows, `FI2010_test.csv` = 1,000 rows, from a real sample smaller than 5,000.

**This is the single most demo-able finding in the repo** — a four-line block, a one-sentence explanation, and a catastrophic consequence.

**Detection signature (artifact-only):** nearest-neighbour distance from each test row to its closest train row. Under a clean split this distribution is broad; here it collapses to ≈ the injected noise scale.

---

### X2 (Critical) — Ablation evaluates on the training environment instance

`scripts/run_ablations.py`: `wrapped_env` is constructed at :167, trained at :185/:198, and evaluated at :223-229 — **the same object**. There is no eval env, no held-out split, no seed change between phases. Confirms the claim already recorded in `PRODUCT_PLAN.md` §3 Check 2.

### X3 (High) — No sealed holdout exists
Nothing in the repo reserves data before experimentation, records a hash, or enforces single-use. The FI-2010 test set has been scored by every baseline, every ablation arm, and every robustness seed. Multiple-comparison inflation is unbounded and unquantified.

---

## C3 — Degenerate components

### D1 (Critical) — The feature extractor never receives a gradient

`src/envs/hierarchical_wrapper.py:148-149` runs the extractor inside a `gym.ObservationWrapper` under `torch.no_grad()`, while `src/agents/dsac_trader.py:76-90` builds TQC with `"MlpPolicy"`, `net_arch=[256,256]`. The extractor sits **outside** SB3's computation graph and stays at initialisation for the entire run.

The project's headline component is a frozen random projection. `checkpoints/*_mamba.pt` therefore stores initialisation noise, not learned weights.

**Detection signature:** compare a component's weight distribution against a fresh init of the same class — identical distribution after N training steps means never-trained.

---

### D2 (Critical) — The regime signal is constant. Measured.

```
train_regimes.npy       (362400,)   n_unique = 1   value = 2  ×362,400
test_regimes.npy        (31937,)    n_unique = 1   value = 2  ×31,937
live_train_combined.npy (4000, 2)   n_unique = 1   value = 0  ×4,000
live_test_combined.npy  (1000, 2)   n_unique = 1   value = 0  ×1,000
```

Every split carries exactly one regime value. The one-hot of a constant is a constant vector; it contributes nothing an intercept does not. The "alpha-aware" tier is decorative in every experiment run to date.

Corroborated in code: `hierarchical_agent.py:130-135` calls `llm_analyst.analyze()` **once** before training and reuses the result for every timestep; `run_ablations.py:163` injects a hardcoded `RegimeSignal(regime=0, confidence=0.9)`.

Note also that the historical splits are regime 2 (*Crash*) while the live splits are regime 0 (*Safe*) — the constant is not even consistent across datasets.

> **Correction to `PRODUCT_PLAN.md` §3 Check 3.** The plan states *"`confidence` was `0.5` for all 362,400 rows — the parse-failure default."* That is not what the artifacts contain. Measured: `train_confidences.npy` holds **8 distinct values** (0.3848, 0.4018, 0.4032, 0.4040, …) in blocks of exactly 50,000, and `train_regimes.npy` is the constant — `n_unique = 1`, value `2`, all 362,400 rows. The row count is right and the conclusion is right, but the column and the value are wrong. Worth fixing before this is said aloud to a customer; a prospect who checks the number and finds it wrong loses the whole demo.

### D3 (High) — Confidence is chunk-broadcast, not per-row
The 50,000-row blocks show one LLM call per chunk, its scalar output broadcast across the chunk. Effective sample size of the "LLM signal" over the training set is **8**, not 362,400. On live data it is a single hardcoded 0.95.

### D4 (High) — Alpha model is random-init and never trained
`hierarchical_agent.py:84-91` constructs `SimpleAlphaModel`, calls `.eval()`, and never fits it. `TimesFMWrapper` documents itself as *"a FROZEN feature extractor - no fine-tuning"* — defensible for a pretrained foundation model, not for a randomly-initialised MLP. Its 4 output dims are random projections of price history.

### D5 (High) — The agent never trades
`experiments/test_ablations/ablation_results.json` and `experiments/test_ood/ood_results.json` report `mean_reward: 0.0`, `std_reward: 0.0`, `mean_portfolio: 100000.0` for nearly every configuration and every volatility regime — bit-identical to starting cash. Contributing causes: 500-step runs against TQC's `learning_starts=100` default (S3), and reward = raw PnL delta of order 1e-2 against a 1e5 portfolio, which is numerically negligible as a learning signal.

An OOD-transfer table whose every cell is the starting balance measures nothing about transfer.

### D6 (Med) — Observation omits position and cash
`historical_lob_env.py:68-70` sets `observation_space = Box(shape=(144,))` — the raw LOB snapshot only. The agent cannot observe its own inventory, cash, or unrealised PnL, so the decision problem is not Markov for any position-dependent policy. Reward depends on state the policy cannot see.

---

## M — Metric sanity

### M1 (High) — Risk-free units mismatch inverts Sharpe
`src/utils/metrics.py:63`: `rf_per_period = 0.02 / 252 ≈ 7.9e-5` — a **daily** rate — is subtracted from **tick** returns of order 1e-5. The risk-free term dominates the mean, so excess returns are negative regardless of performance.

Committed evidence, `experiments/baselines/baseline_metrics.json`: MACD reports **+27.62% return with Sharpe −1.97**. Return and Sharpe disagree in sign, which is only possible under a units error.

### M2 (Med) — Max drawdown double-scaled
`metrics.py:97` returns `max_dd * 100`; `run_baselines.py:209` stores `mdd * 100` again. Reported 986.77% and 6619.96% are really 9.87% and 66.20%. Drawdown outside [0, 100] is definitionally impossible for an unlevered curve — a two-line bounds check catches it.

### M3 (High) — Hardcoded metrics substitute for measurement
`scripts/run_cluster_training.py:215-219`:
```python
sharpe_ratio = compute_sharpe(...) if len(returns_series) > 10 else 2.14
max_dd       = compute_max_drawdown(...) if len(portfolio_values) > 1 else 6.8
var_95       = ... if len(returns_series) > 10 else 0.028
cvar_95      = ... if len(returns_series) > 10 else 0.041
```
Four plausible-looking constants stand in for measurements when a run is too short. Given D5 — runs that terminate with no trades — the fallback branch is reachable in exactly the conditions where it is most misleading. A Sharpe of 2.14 that was never computed is indistinguishable in the output JSON from one that was.

### M4 (Med) — Annualisation inconsistent across scripts
`metrics.py:46` defaults to `periods_per_year=252` (daily); `run_cluster_training.py:215` passes `252*24*60` (minutes). Neither matches FI-2010 tick spacing. Sharpe values are not comparable between scripts, and neither is on a defensible scale.

### M5 (Med) — No market benchmark exists
`grep -rn "buy_and_hold|BuyHold|benchmark"` over `src/` and `scripts/` returns **nothing**. Baselines are MACD, Bollinger, and a supervised LSTM. Without buy-and-hold, a zero-position control, and a random-action control, no result can distinguish skill from drift. This is precisely the correction the reference video made in its first iteration.

### M6 (Med) — Baselines reported from a single run
`baseline_metrics.json` contains one point estimate per strategy. `configs/default.yaml` defines five seeds `[42, 1, 100, 7, 2024]`; the baseline runner ignores them. No dispersion, no confidence band, no overlap test.

---

## S — Claim–evidence mismatch *(proposed Check 4)*

### S1 (Critical) — The ablation contains no Mamba arm

`scripts/run_ablations.py:82-113`:

| Config | Extractor | RL algo |
|---|---|---|
| `tcn_ppo` | `TCNFeatureExtractor` | PPO |
| `lstm_ppo` | `MambaFeatureExtractor(backend="lstm")` | PPO |
| `lstm_dsac` | `MambaFeatureExtractor(backend="lstm")` | TQC |
| `full` | `MambaFeatureExtractor(backend="lstm")` | TQC |

Every arm that instantiates `MambaFeatureExtractor` passes `backend="lstm"`. **No configuration runs Mamba.** The project's central claim — Mamba SSM beats TCN/LSTM for LOB modelling — is not tested by the experiment built to test it. The `full` model is an LSTM.

This is a distinct failure class from C1–C3: nothing leaks, nothing is contaminated, no component is dead. The experiment is internally valid and simply cannot bear on the claim.

**Detection signature (artifact-only):** given a config table plus a results table, check that arms differ in the dimension the claim names. If every arm shares a value on the claimed axis, the comparison is vacuous — computable from metadata alone, no source access required. **This is a real gap in the current three-check product and I would add it.**

### S2 (High) — `full` differs from `lstm_dsac` only by two dead inputs
Per `run_ablations.py:103-113`, `full` = `lstm_dsac` + constant regime (D2) + untrained alpha model (D4). The headline arm adds four constant dims and four random-projection dims to an otherwise identical model. It cannot show an effect; any measured difference is seed noise.

### S3 (High) — 500-timestep runs presented as results
`ablation_results.json` and `ood_results.json` record `"timesteps": 500` with `train_time_s` of 1.6–6.3 s. TQC's default `learning_starts` is 100. These are smoke tests. `configs/default.yaml` specifies `total_timesteps: 1000000` — three orders of magnitude more.

### S4 (Med) — Multi-horizon claim inert on live data
`fetch_live_market_data.py:152-157` writes the **same** label into all five horizon columns:
```python
for k in range(5):
    train_df[144 + k] = train_labels + 1
```
`horizon_idx` has no effect on live data. The `prediction_horizons: [10, 20, 50, 100]` config is decorative there.

### S5 (Med) — Ablation and baseline tables are not comparable
`run_ablations.py`, `run_ood_transfer.py`, and `run_robustness.py` all instantiate `ABIDESEnv` (the synthetic mock); `run_baselines.py` uses `HistoricalLOBEnv` on FI-2010. The two result families describe different worlds and are presented in the same report.

### S6 (Med) — Input dimension inconsistent
`run_ablations.py` builds extractors with `input_dim=40`; `data_loader.py` and `historical_lob_env.py` use 144; `train_dsac_trader.py:52` passes 40. Whichever is right, the ablation is not measuring the production configuration.

---

## R — Reproducibility *(proposed bundle)*

### R1 (High) — The venv is broken; no result can be regenerated
```
venv/bin/python3 → system python, now 3.14 (anaconda)
venv/lib/python3.13/site-packages/   ← orphaned
sys.path = ['', '/opt/anaconda/lib/python314.zip', '/opt/anaconda/lib/python3.14', ...]
```
`./venv/bin/python -c "import numpy"` → `ModuleNotFoundError`. A host Python upgrade orphaned the environment. **Every JSON in `experiments/` is currently unreproducible** — the results cannot be re-derived, only trusted. No pinned interpreter, no lockfile, no container.

### R2 (Med) — Silent dataset substitution
`data/live_market/FI2010_train.csv` and `FI2010_test.csv` contain Coinbase crypto LOB snapshots, not FI-2010 (verified: distinct md5, 4,001/1,001 rows vs 362,401/31,937). They load through `FI2010DataLoader` without complaint. A reader of `run_baselines.py` cannot tell from the call site which asset class produced a result.

### R3 (Med) — No version control
The project root is not a git repository. No result in `experiments/` can be tied to the code that produced it. Combined with R1, the experimental record has no provenance chain at all.

---

## What this means for the two tracks

**As research (`alpha-aware-hrl`):** L1, L2, X1, X2, D1, D2, and S1 each independently invalidate the current results. Nothing in `experiments/` should be cited. Repair order matters — fix the harness (L1–L4, X1–X3, M1–M6) *before* touching the model (D1, D4, D6, S1), because model changes measured on a leaking harness optimise the leak. This is the reference video's own sequence: every improvement its self-improvement loop accepted was an evaluation-validity fix, and it kept 1 proposal in 14.

**As product (`PRODUCT_PLAN.md`):** the three planned checks would catch 13 of 24. The metric bundle catches 4 more. **Seven findings — S1–S6 and R1–R3 — fall outside the current taxonomy**, and S1 is the most striking single finding in the repo: an ablation designed to prove Mamba beats LSTM, in which no arm runs Mamba. Both proposed categories are computable from artifacts and metadata alone, which keeps them inside the artifact-first constraint in §1.

Highest-value demo sequence, in order of how fast the point lands:
1. **X1** — four lines, test set is the train set, catastrophic. Under 30 seconds.
2. **S1** — the config table, one column, no Mamba anywhere.
3. **D2** — `n_unique = 1` across 362,400 rows, printed live.
4. **M2** — a 6,620% drawdown, bounds-checked in one line.

---

*No code was modified in producing this report.*
