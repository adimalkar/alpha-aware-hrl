# Phase Plans — `alpha-aware-hrl`

Four self-contained phases. Each has its own plan, its own acceptance gate, and its own
documentation deliverable. Work them in order; each phase's gate is the next one's precondition.

Source of truth for defect IDs: [`../AUDIT_FINDINGS.md`](../AUDIT_FINDINGS.md) (24 defects, 2026-09-01).

| Phase | Plan | Closes | Gate | Est. |
|---|---|---|---|---|
| **0** | [Foundation & provenance](PHASE_0_FOUNDATION.md) | R1, R2, R3 | Any script runs; every result carries a manifest | 1–2 days |
| **1** | [Harness validity](PHASE_1_HARNESS.md) | L1–L4, X1–X3, M1–M6 | Harness proves a known-good and a known-bad strategy apart | 1–2 weeks |
| **2** | [Model integrity](PHASE_2_MODEL.md) | D1–D6, S1–S6 | Every claimed component is trained, varying, and tested | 2–3 weeks |
| **3** | [Self-improvement loop](PHASE_3_LOOP.md) | — (method, not defects) | Loop runs unattended; sealed holdout scored once | 1 week + runtime |

## The ordering constraint

Harness before model. This is not a preference — a model change measured on a leaking
harness optimises the leak. The reference video (LosingLoonies, 2026-08-29) is the
worked example: every improvement its self-improvement loop accepted was an
evaluation-validity fix, and the autonomous round kept **1 proposal in 14**.

Phase 3 is the only phase that produces performance claims. Phases 0–2 produce
*the right to make them*.

## Dual-track output

Per the 2026-09-01 decision, every phase serves both tracks:

- **Research track** — the repair itself, with a before/after metric delta.
- **Product track** — each defect closed becomes a taxonomy case study in
  `docs/taxonomy/<ID>.md`: the defect, the artifact-only detection signature, the
  ground-truth fixture, and the delta it produced. These are the regression suite for
  the validation product and the demo corpus described in `../../PRODUCT_PLAN.md` §5.

**Fixture preservation rule.** Never delete a defect — move it. Before repairing any
defect, snapshot the broken state into `fixtures/<ID>/` with its inputs and outputs.
The broken version is product inventory; the repo's highest-value asset is a set of
known-ground-truth failures, and a repair that erases the evidence destroys it.

## Status

- [ ] Phase 0 — not started
- [ ] Phase 1 — not started
- [ ] Phase 2 — not started
- [ ] Phase 3 — not started
