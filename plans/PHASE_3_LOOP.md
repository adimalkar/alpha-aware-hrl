# Phase 3 — Self-Improvement Loop

**Closes:** no defect IDs — this phase adds a method, and is the only phase that produces
performance claims
**Estimate:** 1 week to build, then runtime
**Precondition:** Phase 2 gate passed. **Do not start early.** A loop that selects on a
leaking harness selects the leak, at machine speed and with a plausible audit trail.

---

## Objective

Reproduce the method from *"My AI Stock Predictor is Now Improving Itself"* (LosingLoonies,
2026-08-29) under the Phase 1 harness: an LLM proposes a change, the harness tests it on
validation, the change survives only if it beats the incumbent, repeat — with the holdout
sealed before the first iteration and scored exactly once at the end.

**What the reference run actually produced:** 14 autonomous iterations, **1 kept**. Every
accepted improvement across the whole video was an evaluation-validity fix — a horizon
ranking bug, a too-weak baseline, survivorship bias, signal staleness — not a model
improvement. Final holdout: strategy 30.4% vs market 33.5%. He published the loss.

Design the loop expecting that outcome. A ~7% acceptance rate is the realistic prior, and a
loop that accepts most of its proposals is reporting harness noise.

---

## Work items

### 3.1 — Proposal interface

Constrain what a proposal may touch. Unconstrained self-modification produces changes that
cannot be attributed and a diff nobody can review.

**Allowed:** hyperparameters, reward-term weights, feature selection from an existing pool,
architecture dims within declared bounds, episode/window lengths.
**Forbidden:** anything under `src/utils/metrics.py`, `src/utils/holdout.py`, the split
logic, the baseline suite, or the calibration fixtures. **The loop must not be able to edit
its own scorer.** Enforce with a path allowlist in the runner, not with instructions.

Each proposal emits: a rationale, a unified diff, a falsifiable prediction of its effect,
and the config hash it derives from.

---

### 3.2 — Accept/reject criterion

Reject unless **both** hold:
1. Validation mean improves, **and**
2. the ±3σ bands across five seeds do **not** overlap the incumbent's.

Criterion (2) is the whole discipline. Without it the loop accepts noise, and 14 iterations
of accepted noise is precisely the failure mode that produces a confident, worthless model.

Every proposal is logged whether accepted or rejected — rationale, diff, both metrics,
verdict. **The rejects are the more valuable artifact**, both scientifically and as product
inventory: the reference run's 13 discarded ideas are a catalogue of plausible-sounding
changes that did not survive contact with an honest harness.

---

### 3.3 — Runner

`scripts/self_improve.py`:
1. Load incumbent config + validation score.
2. Propose (Claude Opus 5 — `claude-opus-5`; see the `claude-api` skill for current
   parameters and pricing before wiring the calls).
3. Apply the diff to a scratch worktree; **never** to the working tree.
4. Train and evaluate across five seeds on train/val only.
5. Apply the 3.2 criterion; on accept, commit with the rationale in the message; on reject,
   log and discard.
6. Append to `docs/loop-journal.md`.
7. Repeat until N iterations or a wall-clock budget.

Checkpoint after every iteration so an interrupted run resumes without rescoring.

**Cost control.** Each iteration is 5 seeds × 1e5–1e6 steps. On a 6 GB laptop GPU this is
the binding constraint, not the API spend. Budget the compute first and size N to it —
20 iterations at 1e6 × 5 seeds is a multi-day run. Consider a two-tier screen: 1e5 steps ×
3 seeds to shortlist, then full budget × 5 seeds to confirm.

---

### 3.4 — The single holdout run

Once, at the end, after the loop has stopped and no further changes will be made:

1. Confirm `holdout_ledger.jsonl` is still empty.
2. Score the final config on the sealed holdout.
3. Score buy-and-hold, zero-position, and random-action on the same holdout.
4. Report all four with ±3σ bands and the overlap verdict.
5. The ledger now has one entry. **There is no second run.** If the result disappoints, that
   is the result — re-scoring after seeing it converts the holdout into a validation set and
   destroys the only unbiased estimate the project has.

Write the outcome before deciding how to feel about it.

---

## Acceptance gate

1. Loop runs ≥10 unattended iterations without manual intervention.
2. Journal records every proposal, accepted and rejected, with rationale and both metrics.
3. Path allowlist verified: an attempted edit to `metrics.py` or `holdout.py` is refused —
   test this deliberately.
4. Holdout scored exactly once; ledger has exactly one entry.
5. Final report states the verdict plainly, including "indistinguishable from buy-and-hold"
   if that is what the bands show.

**Kill criterion:** if >50% of proposals are accepted, stop and re-audit the harness. That
acceptance rate against a ~7% prior means criterion (2) is not binding, and the loop is
selecting noise.

---

## Documentation deliverable

**Research track:** `docs/loop-journal.md` (auto-appended) and `docs/phase-3-report.md` —
the final holdout table, the accept/reject ledger, and the honest verdict. If the model does
not beat buy-and-hold, that is the report. The reference video's value came from publishing
exactly that.

**Product track:** two taxonomy entries that only this phase can produce, because they are
failure modes *of automated improvement itself* — and they are the entries most likely to
matter commercially as more research teams point LLMs at their own strategies:

- **`docs/taxonomy/S7.md` — selection-on-noise.** An improvement loop that accepts on mean
  improvement without a dispersion test. **Detection signature:** N configs scored against
  one evaluation set, accepted deltas within the seed-noise envelope. Computable from a run
  log alone. Ground-truth fixture: run your own loop for 10 iterations with criterion (2)
  disabled and keep the output — a loop that accepts ~50% where an honest one accepts ~7%.
- **`docs/taxonomy/X4.md` — holdout burn.** Detection signature: a holdout ledger with more
  than one entry, or evaluation-set access timestamps interleaved with config changes.

Both fall under the proposed Check 4 (claim–evidence mismatch) and extend the taxonomy into
territory the current `PRODUCT_PLAN.md` §3 does not cover at all. Given how many teams are
now pointing LLMs at their own backtests, "your self-improvement loop overfit your
validation set and here is the evidence" is a wedge with a shorter shelf life than the
others — worth building while it is still novel.
