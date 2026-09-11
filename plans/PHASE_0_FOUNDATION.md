# Phase 0 — Foundation & Provenance

**Closes:** R1, R2, R3
**Estimate:** 1–2 days
**Precondition:** none — this is the entry point
**Why first:** nothing in `experiments/` can currently be regenerated. Until a script can
run, every later phase is unverifiable, and "before/after metric delta" has no *before*.

---

## Objective

Make the project executable, versioned, and self-describing. At the end of this phase any
result carries enough metadata to be re-derived by someone else, and no result can be
produced without that metadata being written.

This phase changes **no science**. It is deliberately mechanical so that Phase 1's
measurements land on stable ground.

---

## Work items

### 0.1 — Rebuild the environment on a pinned interpreter (R1)

**Current state:** `venv/bin/python3` resolves to the system Python, which a host upgrade
moved to 3.14 (anaconda). `venv/lib/python3.13/site-packages/` is orphaned. `import numpy`
fails inside the venv.

**Constraint discovered:** the only system interpreter is 3.14, which is ahead of wheel
coverage for `torch`, `mamba-ssm`, and `sb3-contrib`. Do not build on 3.14.

**Action:**
1. `uv python install 3.12` — `uv` is already at `~/.local/bin/uv`. 3.12 has the broadest
   wheel coverage for this stack.
2. Recreate: `uv venv --python 3.12 .venv` (new directory name; leave the broken `venv/`
   in place until 0.5 archives it).
3. Install from `requirements.txt` with two amendments:
   - **Drop `timesfm`.** `TimesFMWrapper` is never used at runtime — `hierarchical_agent.py:84`
     instantiates `SimpleAlphaModel` instead. It is a heavyweight dependency carrying zero
     code paths. Removing it also removes a JAX/CUDA conflict surface.
   - **Split `mamba-ssm` and `causal-conv1d` into an optional extra.** They compile against
     CUDA and are the most likely install failure. The core stack must install without them.
4. Emit a lockfile: `uv pip freeze > requirements.lock.txt`. Commit it.

**Verification:**
```
.venv/bin/python -c "import torch, numpy, pandas, gymnasium, stable_baselines3, sb3_contrib; print('core ok')"
.venv/bin/python -c "import torch; print('cuda:', torch.cuda.is_available())"
.venv/bin/python -m pytest tests/ -q
```
The three existing test files (`test_event_pipeline`, `test_tpp_regime`, `test_event_encoder`)
must pass or be triaged. Record which.

**Decision point — Mamba availability.** Attempt `uv pip install mamba-ssm causal-conv1d`.
The GPU is an RTX 4050 Laptop (6 GB, driver 610.57.04), which is sufficient to *run*
`d_model=128, n_layers=4` at `seq_len=100`; the risk is the CUDA build, not capacity.
Record the outcome in the phase log, because **Phase 2 / S1 branches on it**:
- Builds → S1 is closed by adding a real Mamba ablation arm.
- Does not build → S1 is closed by *withdrawing the Mamba claim* from the project's
  framing. It is not closed by running LSTM under a Mamba-shaped name, which is the
  defect itself.

---

### 0.2 — Put the project under version control (R3)

**Action:**
1. `git init` at `alpha-aware-hrl/` (not the parent — the parent holds 615 MB of FI-2010
   and the ABIDES checkout).
2. `.gitignore`: `venv/`, `.venv/`, `data/`, `checkpoints/`, `experiments/`,
   `frontend/node_modules/`, `*.pyc`, `.pytest_cache/`, `__pycache__/`.
3. Commit the current state **before any repair**, tagged `pre-audit-baseline`. This tag is
   the fixture corpus for the product track — it must remain reachable forever.
4. Large artifacts stay out of git. Track them by content hash in the manifest (0.4).

**Verification:** `git log --oneline` shows the baseline commit; `git status` is clean.

---

### 0.3 — End the silent dataset substitution (R2)

**Current state:** `data/live_market/FI2010_train.csv` and `FI2010_test.csv` contain
Coinbase crypto L2 snapshots, not FI-2010 (verified: distinct md5, 4,001/1,001 rows vs
362,401/31,937). They load through `FI2010DataLoader` without complaint, so a reader of
`run_baselines.py` cannot tell which asset class produced a result.

**Action:**
1. Rename `data/live_market/FI2010_{train,test}.csv` → `crypto_lob_{train,test}.csv`.
2. Update `scripts/fetch_live_market_data.py:153,158` to write the new names.
3. Add a `dataset_id` argument to `FI2010DataLoader.__init__` and assert it against a
   sidecar `dataset.json` (`{"id": ..., "rows": ..., "sha256": ..., "normalization": ...}`)
   written next to each dataset. Mismatch raises.
4. Write `dataset.json` for all three corpora: `fi2010`, `crypto_lob`, and any future one.

**Note:** this is a *labelling* fix only. The crypto dataset's substantive defects — L2
(label derived from an observed feature) and X1 (tiling before the split) — are Phase 1
work. Do not repair them here; renaming the file must not be mistaken for fixing it.

**Verification:** loading `crypto_lob` with `dataset_id="fi2010"` raises.

---

### 0.4 — Run manifests

**Action:** add `src/utils/provenance.py` with a `write_manifest(run_dir, **kw)` helper.
Every experiment script calls it before writing results. Manifest contents:

| Field | Source |
|---|---|
| `git_sha`, `git_dirty` | `git rev-parse HEAD`, `git status --porcelain` |
| `dataset_id`, `dataset_sha256`, `n_rows` | `dataset.json` |
| `seed`, `n_seeds` | CLI args |
| `config_sha256` | hash of the resolved config dict |
| `python`, `torch`, `cuda`, key package versions | runtime introspection |
| `timesteps`, `wall_clock_s` | run |
| `started_at`, `finished_at` | UTC ISO-8601 |

**Rule:** a results JSON written without an adjacent `manifest.json` is not a result.
Enforce in the writer, not by convention.

**Verification:** run any script; confirm `manifest.json` appears and `git_dirty` correctly
reflects a deliberately dirtied tree.

---

### 0.5 — Quarantine the invalidated record

**Current state:** `experiments/` holds 11 result directories produced under L1, L2, X1,
X2, D1, D2, and S1. Every number in them is invalid. Left in place, they will be cited.

**Action:**
1. `git mv experiments/ experiments/_invalidated_2026-09-01/` (or plain `mv`, since
   `experiments/` is gitignored — but preserve the tree).
2. Write `experiments/_invalidated_2026-09-01/README.md`: the date, the audit reference,
   the seven critical defect IDs, and an explicit **"do not cite"**.
3. Copy — do not move — the four highest-value broken artifacts into `fixtures/`:
   - `fixtures/X1/` — the tiled crypto CSVs + the fetch script at `pre-audit-baseline`
   - `fixtures/D2/` — `data/precomputed_regimes/*.npy` (the `n_unique=1` arrays)
   - `fixtures/M2/` — `baseline_metrics.json` (the 6,619% drawdown)
   - `fixtures/S1/` — `run_ablations.py` at baseline + `ablation_results.json`

**Verification:** `grep -r "experiments/" scripts/ src/ api/` finds no live path pointing
at the quarantined tree.

---

## Acceptance gate

All five must hold:

1. `.venv/bin/python -m pytest tests/ -q` runs to completion (pass or triaged failure).
2. `git log` shows `pre-audit-baseline`; working tree clean.
3. Loading a dataset under the wrong `dataset_id` raises.
4. A script run produces `manifest.json` with a correct `git_sha`.
5. `fixtures/` holds the four snapshots, each with a one-paragraph `README.md`.

**Kill criterion:** none. This phase cannot fail, only take longer — if the CUDA build
resists, proceed on CPU and record the branch for Phase 2.

---

## Documentation deliverable

**Research track:** `docs/phase-0-log.md` — the interpreter decision and why 3.14 was
rejected, the Mamba build outcome, the test triage table.

**Product track:** `docs/taxonomy/R1.md`, `R2.md`, `R3.md`. R-class checks are the cheapest
in the whole taxonomy and among the most persuasive, because a prospect can verify them in
seconds against their own repo:

- **R1 detection signature:** interpreter version in the manifest vs the interpreter that
  can import the environment's own site-packages. A mismatch means the reported results
  cannot be re-derived on the machine that claims to have produced them.
- **R2 detection signature:** declared `dataset_id` vs content hash. Catches a corpus
  swapped underneath an unchanged filename.
- **R3 detection signature:** results with no reachable commit, or `git_dirty=true`.

These three plus the M-class bundle form the "free" tier of the product: trivial to
implement, impossible to argue with, and they establish credibility before the harder
C1/C2/C3 claims are made.
