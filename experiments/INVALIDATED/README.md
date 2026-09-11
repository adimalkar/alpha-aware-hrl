# Invalidated results

Every artefact in this directory was produced by code with at least one defect
sufficient to void it. They are retained for provenance and for the audit
record. **Do not cite, serve, or compare against any number in here.**

| Directory | Void because |
|---|---|
| `real_data_lem_run/` | Price path derived from the FI-2010 k-step-ahead label (L1). `max_drawdown_pct: 6.8` is the hardcoded fallback constant, not a measurement. `final_portfolio` came from `100000*(1+sum(rewards)*0.001)`, an invented formula reached when portfolio tracking returned nothing. |
| `live_data_lem_run/` | Same fallback path. Additionally the underlying live dataset had its test split tiled from the train split before splitting, so the +6,260.91% return and Sharpe 365.38 measure memorisation of ~1,000 snapshots. Sharpe was computed on raw reward deltas scaled by 1e-4 and annualised by sqrt(252*24*60). |
| `baselines/` | Max drawdown double-scaled (986.77%, 2180.70%, 6619.96% — all definitionally impossible). Sharpe sign-inverted by a daily risk-free rate subtracted from tick returns; MACD reports +27.62% return with Sharpe −1.97. |
| `benchmarks/` | Sharpe/return/drawdown columns were hardcoded literals in `run_event_benchmark.py`; only the latency column was measured. |
| `test_ablations/` | Trained and evaluated on the same env object (X2). Three of four arms report `mean_reward: 0.0, mean_portfolio: 100000.0` — the agent never traded. No arm ran Mamba despite the study being presented as Mamba-vs-LSTM (S1). |
| `test_ood/` | All four volatility regimes report identical zeros; the agent held a constant position throughout. An OOD table whose every cell is the starting balance measures nothing. |
| `test_robust/` | Two seeds, on the same defective harness. |
| `cluster_run/` | Produced by the deprecated `run_cluster_training.py`. |
| `ablation/`, `ood_test/`, `seed_runs/` | Empty. |

The regime artefacts under `data/precomputed_regimes/` are likewise degenerate:
`train_regimes.npy` is the constant `2` across all 362,400 rows and
`test_regimes.npy` the constant `2` across 31,937; the live splits are constant
`0` with confidence constant `0.95`. `train_confidences.npy` holds 8 distinct
values in blocks of exactly 50,000 — one LLM call per chunk, broadcast — so the
effective sample size of the "LLM signal" over the training set is 8, not
362,400.
