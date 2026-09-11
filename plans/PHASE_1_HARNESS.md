# Phase 1 — Harness Validity

**Closes:** L1, L2, L3, L4, X1, X2, X3, M1, M2, M3, M4, M5, M6
**Estimate:** 1–2 weeks
**Precondition:** Phase 0 gate passed
**Why now:** this phase builds the instrument. Until it can distinguish a real edge from a
planted one, no model change is measurable and no number is worth reporting.

---

## Objective

An evaluation harness that is **capable of returning a negative result**, and that has been
demonstrated to do so on strategies with known ground truth.

The success condition is not "the model looks better." It is "when there is no edge, the
harness says so."

---

## The central decision: what FI-2010 can and cannot support

This determines everything else in the phase, so settle it first.

**Measured fact.** FI-2010 as committed is z-score normalized. Row 0 of
`data/fi2010/FI2010/FI2010_train.csv` reads `0.318116, -0.56461858, 0.31353946, ...` —
prices near 0.318, volumes negative. The normalization is per-feature against a rolling
prior-day window.

**Consequence.** A mid-price reconstructed as `(f[0] + f[2]) / 2` is in units of σ, not
currency. Differences across time are scaled by `1/σ_day`, which changes at every day
boundary, so a return series built from it is discontinuous, not in percent, and not
comparable across days. **You cannot compute a valid PnL or Sharpe from this dataset.**

This is precisely why `historical_lob_env.py:148` invents a price from the label — it is the
only way to get a PnL number out of a corpus that has none. L1 is therefore not a bug to
patch. It is a dataset/task mismatch, and the repair is to stop asking FI-2010 for
something it cannot give.

**Decision — split the two tasks across two corpora:**

| Task | Corpus | Valid metrics | Invalid |
|---|---|---|---|
| Supervised LOB direction | FI-2010 | accuracy, F1, MCC, per-class recall | any PnL, Sharpe, drawdown |
| RL execution / PnL | real-price corpus (crypto L2) | PnL, Sharpe, drawdown, turnover | — |

FI-2010 is a *classification* benchmark and is entirely sound used as one — DeepLOB and its
successors benchmark on it exactly this way. Keep `train_lob_classifier.py`; it is the most
scientifically defensible script in the repo. Delete the PnL path over FI-2010 entirely.

**Alternative considered and rejected:** obtain the `DecPre` (decimal-precision) FI-2010
variant, which preserves relative price structure. Rejected because it still yields no
tradeable price level and no spread in currency terms, so transaction costs remain
unmodellable — it converts an impossible backtest into a misleading one.

---

## Work items

### 1.1 — Rebuild the RL corpus with real prices (L1, L2, X1)

The crypto path is the right foundation — Coinbase L2 via `ccxt` is genuine market data —
but `fetch_live_market_data.py` currently destroys it three ways.

**1.1a — Stop deriving labels from an observed feature (L2).**
`fetch_live_market_data.py:110-118` sets the label by thresholding `feat_144d[42]`, the
order-book imbalance, which is *inside the observation*. Composed with L1 this makes
`price_delta = f(observation[42])` exactly. Remove label construction from the fetcher
entirely. The RL task needs a **price series**, not labels; labels are a supervised-task
artifact and have no place in the RL corpus.

**1.1b — Persist the real mid-price (L1).**
`fetch_live_market_data.py:102-108` computes `mid`, then destroys it:
```python
scale = 100.0 / (mid + 1e-6)
feat_144d[41] = 100.0   # normalized mid
```
Write the raw `mid` and raw `spread` to dedicated columns *before* scaling, and carry a
UTC timestamp per snapshot. The env's price path becomes this recorded series — an
observed quantity, never a function of the label or of any feature the agent sees.

**1.1c — Stop tiling before the split (X1).**
`fetch_live_market_data.py:135-147` tiles ~1,000 real snapshots to 5,000, adds σ=0.003
noise, *then* splits 80/20 — so every test row is a perturbed duplicate of a train row.
Delete the tiling block outright. If the corpus is too small, **collect for longer**; do not
manufacture rows. At the current 0.02 s poll across 3 symbols, a continuous session yields
roughly 150 k snapshots/hour, so a genuine multi-hundred-thousand-row corpus is a matter of
runtime, not code.

**1.1d — Rewrite the env's price path.**
`HistoricalLOBEnv.step()` reads the recorded mid at `data_idx`. Delete lines 148-154. Fees
apply against the recorded spread rather than a flat 1 bp where spread is available.

**Verification:** `corrcoef(price_delta[t], feature[42][t])` no longer degenerate; peak
lead-lag cross-correlation between the return series and every feature column occurs at
lag ≥ 0 (no non-causal peak); nearest-neighbour distance from each test row to its closest
train row is broadly distributed, not collapsed to the noise scale.

---

### 1.2 — Purge and embargo (L3, L4)

`data_loader.py:154-159` splits already-overlapping sliding windows chronologically, so
train window `n_train-1` and val window `n_train` share 99 of 100 timesteps; the k-step
label extends the contamination further still.

**Action:** add `split_with_embargo(n, seq_len, horizon, ratios)` returning index ranges
separated by `seq_len + horizon` dropped samples at every boundary. Route all three splits
through it. Log the number of samples purged — a purge that drops zero samples is a bug.

**Verification:** assert `max(train_idx) + seq_len + horizon < min(val_idx)`. Add a unit
test asserting empty timestep-set intersection between the windows of adjacent splits.

---

### 1.3 — Sealed holdout with a single-use ledger (X3, X2)

Nothing in the repo currently reserves data before experimentation. The FI-2010 test set has
been scored by every baseline, ablation arm, and robustness seed; `run_ablations.py:167,223`
trains and evaluates on the *same env instance*.

**Action:**
1. Seal the final ~20% of the RL corpus by time, **now**, before any Phase 2 work. Record
   `{start_ts, end_ts, n_rows, sha256}` in `holdout.lock.json` and commit it.
2. Add `src/utils/holdout.py` exposing one function that (a) refuses to load the holdout
   unless passed an explicit `i_understand_this_is_single_use=True`, (b) appends
   `{timestamp, git_sha, config_sha, purpose}` to `holdout_ledger.jsonl`, and (c) **raises
   if the ledger already contains an entry for a different config**.
3. All development uses train/val only.
4. Fix X2 independently: `run_ablations.py` must build a separate eval env from a disjoint
   split. Same wrapper class, different data range, different seed.

**Design note.** The ledger is append-only and committed, so a second scoring is visible in
git history even if someone deletes the file. This mirrors the reference video's discipline
— the holdout was sealed *before* the improvement loop began, and the final backtest ran
exactly once, by construction rather than by intention.

**Verification:** a second holdout call with a changed config raises; the ledger shows one
entry.

---

### 1.4 — Repair the metrics module (M1, M2, M3, M4)

`src/utils/metrics.py` is explicitly listed in `PRODUCT_PLAN.md` §5 as a component that
survives the pivot. Repair it properly; it becomes a product test fixture.

| ID | Defect | Repair |
|---|---|---|
| M1 | `rf_per_period = 0.02/252` (daily) subtracted from tick returns → sign inversion; evidence: MACD `+27.62%` return with Sharpe `−1.97` | Remove the default. Require explicit `periods_per_year`; derive it from the corpus timestamps and assert it matches the caller's value |
| M4 | `252` in `metrics.py:46` vs `252*24*60` in `run_cluster_training.py:215` | Single source: `periods_per_year` computed once from `dataset.json` sampling interval |
| M2 | `metrics.py:97` returns `max_dd*100`; `run_baselines.py:209` multiplies by 100 again → 986%, 6620% | Return **fractions** from every metric function. Format as percent only at print time. Assert `0 ≤ dd ≤ 1` |
| M3 | `run_cluster_training.py:215-219` substitutes `2.14`, `6.8`, `0.028`, `0.041` when a run is too short | Delete. Raise `InsufficientDataError`. A metric that could not be computed must be absent, never plausible |

**Add:** `assert_metric_bounds(metrics)` — drawdown in [0,1], win rate in [0,100], Sharpe
finite, return and Sharpe sign-consistent unless volatility is degenerate. Call it in every
results writer.

**Verification:** unit tests over the *fixtures* — feed `fixtures/M2/baseline_metrics.json`
and confirm the bounds check flags it. This is simultaneously the product's regression test.

---

### 1.5 — Build the baseline suite (M5)

`grep -rn "buy_and_hold|BuyHold|benchmark"` over `src/` and `scripts/` returns nothing. The
harness has no reference point, so it cannot separate skill from drift.

**Add four controls**, all through the identical env, fee model, and metric path:

1. **Buy-and-hold** — enter at t₀ at full weight, hold. The one that matters.
2. **Zero-position** — never trade. Isolates fee drag and confirms the accounting closes.
3. **Random action** — uniform in [−1,1], N seeds. The dispersion cloud.
4. **Random-sign momentum** — a deliberately mediocre reference with realistic turnover.

This is the reference video's first correction, and its most consequential: he moved from
single random stocks to 25-stock random portfolios so the benchmark actually tracked the
market, and only then could ask whether the strategy escaped the cloud.

**Verification:** zero-position returns exactly `starting_cash` and zero turnover. If it
does not, the accounting is wrong and every other number is too.

---

### 1.6 — Seed dispersion and the overlap test (M6)

`baseline_metrics.json` reports one point estimate per strategy; `configs/default.yaml`
defines five seeds `[42, 1, 100, 7, 2024]` that the baseline runner ignores.

**Action:**
1. Every evaluated strategy runs across all five seeds minimum.
2. Report mean ± 3σ, not a point estimate.
3. Add `bands_overlap(a, b) -> bool` and print the verdict **in the results table itself**,
   not in a footnote.
4. Any comparison whose bands overlap is reported as **"indistinguishable"** — not as the
   larger mean with a caveat.

This is the discipline that made the reference video's negative result legible: 23.8%/yr vs
0.4%/yr looked decisive until the ±3σ bands turned out to overlap almost completely.

---

### 1.7 — Calibrate the harness against known ground truth

**The distinguishing work item of this phase.** A harness is an instrument; an instrument
that has never been shown to detect a known signal, and to stay silent on known noise, is
not yet an instrument.

Build `scripts/calibrate_harness.py` running four fixtures:

| Fixture | Construction | Required verdict |
|---|---|---|
| **Known-alpha** | real price series + injected predictable component of known effect size | detected; measured Sharpe within tolerance of analytic |
| **Known-null** | phase-randomised surrogate preserving the spectrum, destroying predictability | **indistinguishable from buy-and-hold** |
| **Known-leak** | the `pre-audit-baseline` env (L1 intact) | leakage check fires; non-causal lag peak reported |
| **Known-contaminated** | the `fixtures/X1/` tiled corpus | contamination check fires; NN-distance collapse reported |

The known-null fixture is the important one. A harness that cannot say *"nothing here"* will
never say it about your own model either.

**Verification:** all four verdicts correct, reproducibly, across five seeds.

---

## Acceptance gate

1. Calibration (1.7) returns all four correct verdicts.
2. Zero-position control returns exactly `starting_cash`.
3. No non-causal lag peak between the return series and any feature column.
4. Purge/embargo asserted at every split boundary; purged counts logged and non-zero.
5. `holdout.lock.json` committed; ledger empty; a second call with a changed config raises.
6. Every metric function returns fractions; bounds assertions active; no hardcoded fallback
   survives `grep -rn "else 2.14\|else 6.8\|else 0.028"`.
7. Baseline table shows four controls × five seeds with ±3σ bands and overlap verdicts.

**Kill criterion:** if the re-collected corpus cannot produce a buy-and-hold series whose
own Sharpe is stable across seeds, stop. The data is too short or too noisy for any PnL
claim, and Phase 2 would be measuring nothing. Extend collection before proceeding.

---

## Documentation deliverable

**Research track:** `docs/phase-1-log.md` — the FI-2010 decision and its reasoning, corpus
collection parameters, the calibration table, and the first honest baseline table. Expect
the numbers to look far worse than `_invalidated_2026-09-01/`. That is the phase succeeding.

**Product track:** taxonomy entries `L1–L4`, `X1–X3`, `M1–M6` in `docs/taxonomy/`, each with
its artifact-only detection signature — these are the three planned checks plus the metric
bundle from `PRODUCT_PLAN.md` §3, now backed by ground-truth fixtures:

- **L1/L2 → Check 1.** Lead-lag cross-correlation per feature against the target; flag
  non-causal peaks. Fixture: `fixtures/L1/`, known peak at the label horizon.
- **X1 → Check 2.** Nearest-neighbour distance from test rows to train rows; flag collapse
  toward an injected noise scale. Fixture: `fixtures/X1/`, σ=0.003.
- **X2/X3 → Check 2.** Index-overlap between fit and eval sets; repeated scoring of one
  eval set across N configs.
- **M1–M6 → free bundle.** Annualisation consistency, percentage double-scaling, drawdown
  bounds, hardcoded-constant detection, missing-benchmark detection, absent dispersion.

Each entry records the **before/after delta** — the metric as reported at
`pre-audit-baseline` and after repair. That delta is the case study; it is what makes the
demo land, because it is measured on a system whose author has already admitted the defect.
