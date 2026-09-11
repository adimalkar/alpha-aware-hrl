# Remediation record

Maps every finding in [AUDIT_FINDINGS.md](AUDIT_FINDINGS.md) to its fix, the
commit that made it, and the test that stops it regressing.

**Legend** — **Fixed**: defect removed and guarded by a test. **Structural**:
the defect is now impossible to reintroduce silently (the code raises).
**Documented**: cannot be fixed from the data available; stated explicitly and
the affected component excluded from claims.

## C1 — Temporal leakage

| ID | Finding | Status | Fix | Test |
|---|---|---|---|---|
| L1 | Env price path computed from the forward-looking label | **Structural** | `HistoricalLOBEnv` requires an explicit price series and raises `PriceSourceError` rather than synthesising one (`2b86936`) | `test_price_path_is_independent_of_labels` runs identical seeds against different label streams and asserts price and reward are bit-identical |
| L2 | Live label was thresholded `feature[42]`, inside the observation | **Fixed** | Labels are now the realised k-step-ahead move computed from the real mid, marked diagnostic-only and never touching price or reward (`2b86936`) | `test_price_path_is_independent_of_labels` |
| L3 | No purge or embargo at the train/val boundary | **Fixed** | `temporal_split()` drops `purge` rows (default 5× horizon) at the boundary; loader applies the same gap for the val carve (`2b86936`, `a86f4b4`) | covered by split shape assertions |
| L4 | Label horizon extends past the boundary | **Fixed** | Same purge gap; trailing rows with no resolvable horizon are labelled −1 | — |

## C2 — Evaluation contamination

| ID | Finding | Status | Fix | Test |
|---|---|---|---|---|
| X1 | Live test set was the train set tiled + σ=0.003 noise, split afterwards | **Fixed** | No tiling anywhere. Strictly temporal split applied to what was actually collected (`2b86936`) | — |
| X2 | Ablation trained and evaluated on the same env object | **Fixed** | Separate train/eval envs from disjoint splits with different seeds (`50e153d`) | — |
| X3 | No sealed holdout existed | **Fixed** | Three-way split: `val` carved off train for model selection, `test` touched once after training. Pretraining also switched off `test` (`a86f4b4`, `687cdc2`) | — |

## C3 — Degenerate components

| ID | Finding | Status | Fix | Test |
|---|---|---|---|---|
| D1 | Feature extractor never received a gradient | **Fixed** | Encoder moved into the policy as `LEMFeaturesExtractor`; `SequenceWindowWrapper` emits raw windows. Old wrapper raises `DeprecationWarning` (`5d7929f`, `553908d`) | `test_lem_encoder_specifically_gets_gradient`, `test_extractor_receives_gradient`, `test_hierarchical_wrapper_is_deprecated` |
| D2 | Regime signal constant across all 362,400 rows | **Documented** | Env warns on a constant regime array; component excluded from training and from README claims (`553908d`) | `test_constant_regime_warns` |
| D3 | Confidence chunk-broadcast; effective sample size 8, not 362,400 | **Documented** | Recorded in `experiments/INVALIDATED/README.md`; component unused | — |
| D4 | Alpha model random-init and never trained | **Fixed** | `SimpleAlphaModel` tracks `is_trained` and warns on use until `mark_trained()` (`cfb4f20`) | — |
| D5 | The agent never traded | **Fixed** | Reward changed from absolute PnL delta (O(1e-2) on a 1e5 book) to log return; every result records `n_position_changes` and flags a constant position (`2b86936`, `50e153d`) | verified end-to-end: 198 position changes |
| D6 | Observation omitted position and cash | **Fixed** | Observation is 147-d: market features + position weight, cash fraction, episode progress (`2b86936`) | `test_observation_includes_agent_state`, `test_position_weight_is_reflected_in_observation` |

## M — Metric sanity

| ID | Finding | Status | Fix | Test |
|---|---|---|---|---|
| M1 | Daily risk-free rate against tick returns inverted every Sharpe | **Structural** | Annual rates converted by compounding; `periods_per_year=None` by default; a frequency mismatch raises `MetricUnitError` (`18b343a`) | `test_sharpe_rejects_frequency_mismatch`, `test_sharpe_sign_agrees_with_mean_return` |
| M2 | Max drawdown double-scaled (986%, 2180%, 6619%) | **Structural** | Asserted into [0, 100] at the source; caller's second ×100 removed (`18b343a`, `d8754c6`) | `test_drawdown_is_bounded_and_already_percent` |
| M3 | Hardcoded metric constants substituted for measurement | **Structural** | All five `else <constant>` branches deleted; an empty evaluation raises (`18b343a`, `50e153d`) | `test_all_metrics_refuses_empty_evaluation` |
| M4 | Annualisation inconsistent: 252 vs 252·24·60 | **Fixed** | Single documented contract; callers pass `None` for tick data (`18b343a`) | `test_sharpe_annualisation_scales_by_sqrt` |
| M5 | No market benchmark existed | **Fixed** | Buy-and-hold on the identical price window, plus excess return and `beats_buy_and_hold` (`a86f4b4`) | — |
| M6 | Baselines reported from a single run | **Fixed** | Ablation runs multiple seeds and reports mean ± 95% CI (`50e153d`) | — |

## S — Claim–evidence mismatch

| ID | Finding | Status | Fix |
|---|---|---|---|
| S1 | Mamba-vs-LSTM ablation contained no Mamba arm | **Fixed** | Arms renamed for what they run: `lem` / `lem_frozen` / `gru` / `mlp`. `lem_frozen` reproduces the original broken config as a control (`50e153d`) |
| S2 | `full` differed from `lstm_dsac` only by two dead inputs | **Fixed** | Superseded by the new arm set |
| S3 | 500-timestep runs presented as results | **Fixed** | `learning_starts` scaled to run length; every result records `eval_steps` and `n_return_periods` |
| S4 | Multi-horizon claim inert on live data | **Documented** | Alpha component excluded; README states it is untrained |
| S5 | Ablation and baseline tables not comparable | **Fixed** | Baselines run on the same data, price path and fee as the agent (`d8754c6`) |
| S6 | Input dimension inconsistent (40 vs 144) | **Fixed** | Dimension derives from the observation space, not a literal |

## R — Reproducibility

| ID | Finding | Status | Fix |
|---|---|---|---|
| R1 | The venv was broken | **Fixed** | Rebuilt on Python 3.14.6; `requirements.lock.txt` pins 72 packages (`f089e65`). **Note:** the audit understated this — the committed venv contained only torch, numpy and gymnasium, with no stable-baselines3, sb3-contrib, pandas or flask, so it could never have run the training scripts at all. The environment that produced `experiments/` was never recorded. |
| R2 | Silent dataset substitution | **Fixed** | `FI2010DataLoader` warns when features look like raw price levels rather than z-scored data (`553908d`) |
| R3 | No version control | **Corrected** | The audit stated the project is not a git repository. That is true of the *parent* directory but not of `alpha-aware-hrl/`, which has history going back 6 commits. Provenance is now recorded per-artefact regardless: commit SHA, dirty flag, interpreter, seed, and SHA-256 digests of input data (`f089e65`, `a86f4b4`) |

## Defects found during remediation, not in the original audit

| Finding | Where | Why it matters |
|---|---|---|
| Hawkes likelihood scored event *i* using the state at position *i* | `event_encoder.py` | The causal mask is `triu(diagonal=1)`, so position *i* attends to itself and the hidden state already contains event *i*'s own type embedding. The model could read the answer off its own input, making pretraining degenerate. Fixed by shifting intensities (`5d7929f`) |
| Inter-arrival times were synthesised | `event_pipeline.py` | `dt = base_step*exp(-abs(mid_diff)*5) + Exponential(...)` is a deterministic function of the price move plus noise. Fitting a temporal point process to it models the sampling grid, not the market (`5d7929f`) |
| Symbols interleaved into one series | `fetch_live_market_data.py` | BTC, ETH and SOL were appended round-robin, so consecutive "ticks" were different assets and any price path jumped between ~$100k, ~$3k and ~$200 (`2b86936`) |
| `reset()` used `np.random` instead of `self.np_random` | `historical_lob_env.py` | `seed=` had no effect on episode placement, so no run was reproducible (`2b86936`) |
| `reset()` mutated `self.episode_length` | `historical_lob_env.py` | Persistent state corruption across episodes (`2b86936`) |
| Misaligned regime array swallowed by `try/except` | `historical_lob_env.py` | A length mismatch silently became "no regime" (`2b86936`) |
| `round()` called on `compute_max_drawdown`'s 3-tuple | `run_cluster_training.py` | Would have crashed had the `hasattr` guard ever succeeded — the fallback path was the only one that ran |
| `EventDataset` emitted no padding mask | `event_pipeline.py` | Padded slots contributed a spurious survival term to the likelihood (`5d7929f`) |
| `policy.features_extractor` is `None` on TQC | training scripts | SB3 off-policy algorithms build separate actor/critic extractors (`518a0ae`) |
| Env split resolution sent every non-train split to test | `historical_lob_env.py` | `val` would silently have received test data (`a86f4b4`) |
