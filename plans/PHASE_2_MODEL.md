# Phase 2 — Model Integrity

**Closes:** D1, D2, D3, D4, D5, D6, S1, S2, S3, S4, S5, S6
**Estimate:** 2–3 weeks
**Precondition:** Phase 1 gate passed — every claim below is measured on the calibrated harness
**Why now and not earlier:** each item here is a change whose value is a *number*. Measured
on the Phase 0 harness, every one of those numbers would be a measurement of the leak.

---

## Objective

Make every component the project claims to have actually exist: trained, varying, and
tested against an arm that could beat it.

Framing rule for the whole phase: **a component that cannot be shown to contribute gets
removed, not retained with a caveat.** A smaller honest system beats a larger decorative
one, and the removals are as publishable as the additions.

---

## Work items

### 2.1 — Put the feature extractor inside the policy (D1)

**Current state.** `hierarchical_wrapper.py:148-149` runs the extractor in a
`gym.ObservationWrapper` under `torch.no_grad()`, while `dsac_trader.py:76-90` builds TQC
with `"MlpPolicy"`, `net_arch=[256,256]`. The extractor is outside SB3's graph and never
receives a gradient. It is a frozen random projection, and `checkpoints/*_mamba.pt` stores
initialisation noise.

**Action:**
1. Subclass `stable_baselines3.common.torch_layers.BaseFeaturesExtractor`, wrapping the
   existing `MambaFeatureExtractor` body.
2. Pass via `policy_kwargs={"features_extractor_class": ..., "features_extractor_kwargs": {...}}`.
3. **Move the sliding window into the environment.** The wrapper currently maintains
   `state_buffer` (`hierarchical_wrapper.py:139-143`); once the extractor lives in the
   policy, the env must emit the window itself. Change the observation space to
   `Box(shape=(seq_len, n_features))` and stack in the env, or use `VecFrameStack`.
   Frame-stacking in the env is preferable — it keeps the observation honest and lets the
   replay buffer store what the policy actually consumed.
4. Delete `HierarchicalEnvWrapper`'s feature-extraction path. Keep the class only if the
   regime/alpha concatenation survives 2.3 and 2.4.

**Memory note.** A replay buffer of 1,000,000 × (100 × 144) float32 is ~5.7 TB and will not
fit. Reduce `buffer_size` to ~1e5 and `seq_len` to ~50, or store flat observations and stack
on sample. Compute this budget before the first run — with 6 GB of VRAM the naive
configuration fails immediately.

**Verification:** compare extractor weights before and after N steps — the distribution must
move off its initialisation. Gradient-norm logging on extractor parameters must be non-zero.
This is the same check the product's D-class detector performs.

---

### 2.2 — Add a real Mamba arm, or withdraw the claim (S1, S5, S6)

**Current state.** `run_ablations.py:82-113` — `tcn_ppo` uses `TCNEncoder`; `lstm_ppo`,
`lstm_dsac`, and `full` all pass `backend="lstm"`. **No arm runs Mamba.** The experiment
built to prove Mamba beats LSTM does not contain Mamba.

**Branches on the Phase 0 / 0.1 build outcome:**

- **`mamba-ssm` builds** → add `mamba_dsac` and `mamba_ppo` arms. The comparison grid becomes
  extractor {TCN, LSTM, Mamba} × algo {PPO, TQC}, all six cells, five seeds, identical data
  and budget.
- **`mamba-ssm` does not build** → remove Mamba from the project's framing: README, module
  names, `PROJECT_WALKTHROUGH.md`, and the config. The system becomes an LSTM/TCN
  comparison, which is a perfectly publishable result. Running LSTM under a Mamba-shaped
  name *is the defect*, and keeping the name while knowing it is the more serious version
  of it.

**Also fix in the same pass:**
- **S6** — `run_ablations.py` builds extractors with `input_dim=40`; `data_loader.py` and
  `historical_lob_env.py` use 144; `train_dsac_trader.py:52` passes 40. Resolve to one value
  sourced from `dataset.json` and assert at construction.
- **S5** — `run_ablations.py`/`run_ood_transfer.py`/`run_robustness.py` all use the synthetic
  `ABIDESEnv` while `run_baselines.py` uses real data. Either move everything to the Phase 1
  corpus or label the two result families as different experiments and never table them
  together.
- **S4** — `fetch_live_market_data.py:152-157` writes the same label into all five horizon
  columns, making `horizon_idx` inert. Resolved by 1.1a, which removes labels from the RL
  corpus entirely; confirm the supervised path derives genuinely distinct horizons.

**Verification:** the ablation config table has ≥2 distinct values on the axis the claim
names. This is exactly the product's proposed **Check 4** detector, run against your own
experiment.

---

### 2.3 — Make the regime signal vary, or delete the tier (D2, D3, S2)

**Measured state.** `train_regimes.npy`: `n_unique=1`, value `2`, all 362,400 rows.
`test_regimes.npy`: `n_unique=1`, all 31,937. `train_confidences.npy`: 8 distinct values in
50,000-row blocks — one LLM call per chunk, broadcast. Effective sample size of the "LLM
signal" over training is **8**. `hierarchical_agent.py:130-135` calls `analyze()` once and
reuses the result for every timestep; `run_ablations.py:163` injects a hardcoded
`RegimeSignal(regime=0, confidence=0.9)`.

Consequence for S2: `full` differs from `lstm_dsac` only by a constant regime vector and an
untrained alpha model. The headline arm cannot show an effect.

**Action — choose one, and be willing to choose (b):**

**(a) Make it real.** Regime must be a function of information available at each timestep:
join news to the LOB clock by timestamp, compute per-window, enforce a strict
as-of join so no article dated after *t* reaches the observation at *t*. Then assert
`n_unique > 1` and mutual information with the target > 0 before the feature is admitted.

**(b) Delete the tier.** Drop the LLM analyst, the regime dims, and the "alpha-aware"
framing. The system becomes an honest Mamba/LSTM + TQC execution agent.

**Recommendation: (b), unless the news corpus can be timestamp-joined to the crypto LOB
clock.** `data/news/Stock_News_Dataset.csv` is equity news; the Phase 1 RL corpus is
BTC/ETH/SOL. These do not join, and forcing them produces a feature that is either constant
(today's defect) or a lookahead vector (a worse one). Choosing (b) costs the project its
most distinctive-sounding component and gains it a result that survives scrutiny — and the
deletion, documented, is a better taxonomy case study than a repair would be.

**Verification:** whichever branch — no constant column reaches the observation. Add a
degeneracy assertion over every feature column at env construction: `n_unique == 1` raises.

---

### 2.4 — Train the alpha model or remove it (D4)

`hierarchical_agent.py:84-91` constructs `SimpleAlphaModel`, calls `.eval()`, never fits it.
Its four output dims are random projections of price history. `TimesFMWrapper` documents
itself as frozen — defensible for a pretrained foundation model, meaningless for a
randomly-initialised MLP.

**Action:** either fit it on the Phase 1 corpus as a supervised k-step return predictor with
its own purged split, reporting out-of-sample skill *before* it may enter the observation —
or delete it. Same rule as 2.3: an untrained component must not be an input.

**Verification:** the D-class weight-distribution check (2.1) applied to the alpha model.

---

### 2.5 — Make the observation Markov (D6)

`historical_lob_env.py:68-70` sets `observation_space = Box(shape=(144,))` — the raw LOB
snapshot only. The agent cannot see its own inventory, cash, or unrealised PnL, so reward
depends on state the policy cannot observe. No policy can be optimal.

**Add:** current position weight, cash ratio, unrealised PnL in bps, steps remaining in
episode, and realised turnover so far. Normalise each to roughly unit scale.

**Verification:** an oracle policy that must hold a target inventory should now be
learnable; it is not learnable under the current observation. Use this as a unit test.

---

### 2.6 — Fix the reward and the training budget (D5, S3)

**Measured state.** `ablation_results.json` and `ood_results.json` report `mean_reward: 0.0`,
`mean_portfolio: 100000.0` — bit-identical to starting cash — across nearly every config and
every volatility regime. The agent never trades. Two causes:

- **S3:** runs are 500 timesteps (`train_time_s` 1.6–6.3 s) against TQC's default
  `learning_starts=100`. `configs/default.yaml` specifies `total_timesteps: 1000000` — three
  orders of magnitude more. These were smoke tests presented as results.
- **D5:** reward is raw PnL delta, order 1e-2 against a 1e5 portfolio — numerically
  negligible as a learning signal.

**Action:**
1. Reward in **basis points of portfolio value**, not currency.
2. Subtract realised transaction cost explicitly so the agent optimises net, not gross.
3. Add a risk term — differential Sharpe, or a drawdown penalty. Without one, TQC's
   distributional critics have no risk to be aware of and the "risk-aware" framing in the
   project name is unearned.
4. Budget: minimum 1e5 steps for a screening run, 1e6 for anything reported. Any result
   below 1e5 is labelled a smoke test in its manifest and excluded from every table.

**Verification:** turnover > 0 and reward variance > 0 for every reported arm. A config with
zero turnover is a failed run, not a data point — enforce in the results writer.

---

## Acceptance gate

1. Extractor weights measurably move off initialisation; gradient norms non-zero.
2. The ablation grid has ≥2 distinct values on every claimed axis, or the claim is withdrawn
   from the README and config.
3. No constant column reaches the observation; degeneracy assertion active.
4. Every component in the observation is either trained with reported out-of-sample skill,
   or removed.
5. Observation includes inventory state; the target-inventory unit test passes.
6. Every reported arm shows non-zero turnover and reward variance at ≥1e5 steps.
7. Full grid rerun across five seeds with ±3σ bands and overlap verdicts, on the Phase 1
   harness, **holdout untouched** — ledger still empty.

**Kill criterion:** if after 2.1–2.6 no arm separates from buy-and-hold outside ±3σ, record
that as the phase's finding and proceed to Phase 3 anyway. A well-built system with no edge
is the expected outcome and is the honest headline — the reference video's model returned
30.4% against the market's 33.5% and published it as a loss.

---

## Documentation deliverable

**Research track:** `docs/phase-2-log.md` — extractor-integration before/after weight
distributions, the ablation grid with bands, and an explicit record of every component
*removed* under the 2.3/2.4 rule with the reasoning. The removals are the most credible part
of the writeup.

**Product track:** taxonomy entries `D1–D6` and `S1–S6`.

D-class maps to the plan's existing Check 3, now with two ground-truth fixtures — the
`n_unique=1` regime arrays and the never-trained extractor:
- **D1 signature:** component weight distribution vs a fresh init of the same class;
  identical after N steps ⇒ never trained.
- **D2/D3 signature:** per-column `n_unique`, constant-block detection, effective sample
  size vs row count. `fixtures/D2/` carries the exact arrays.
- **D5 signature:** results columns bit-identical to their initial value across all configs.

S-class is the **proposed Check 4** and does not exist in `PRODUCT_PLAN.md` today:
- **S1 signature:** config table × claim axis; if all arms share a value on the claimed axis,
  the comparison is vacuous. Metadata-only — no source access — so it respects the
  artifact-first constraint in §1.
- **S3 signature:** training budget in the manifest vs the budget in the committed config;
  order-of-magnitude gaps flagged.
- **S5 signature:** results tabled together whose manifests carry different `dataset_id`s.

S1 is the strongest single demo in the corpus: an ablation built to prove Mamba beats LSTM,
in which no arm runs Mamba, detected from the config table alone in under a second.
