# C05 — The behavior-energy verdict depends on the read-out (confirmed)

**One-line result:** on real Qwen3.5-2B, switching the behavior-energy read-out from
*first-token* to *full-string* flips the manifold-vs-linear faithfulness verdict on **exactly the
two multi-token-risk concepts** (agreement, education) and on **none of the four controls** —
so first-token energy verdicts on multi-token concepts are tokenizer artifacts, not geometry.

Evidence: `reports/manifold_c05/behavior_readout_audit_2b.json`
(runner: `scripts/_c05_mlx_audit.py`, Modal twin: `manifold_behavior_readout_audit_2b` in
`modal_app.py`). Run 2026-06-09 on `mlx-community/Qwen3.5-2B-bf16`.

## Why the first-token read-out is at risk

The behavior manifold ℳ_y (see `docs/MANIFOLD.md` §4.5) is built from distributions over a
concept's value strings. The original read-out (`_output_distribution`) takes the **next-token**
distribution over each value's *first* token. That is exact for single-token values, but for
multi-token values it has two failure modes:

1. **Collisions** — "strongly agree" and "strongly disagree" share the first token; the read-out
   cannot tell them apart, so probability mass is silently merged.
2. **Truncation** — a value whose identity lives in its later tokens ("high school" vs
   "high court") is scored on a prefix that carries little of it.

The full-string read-out (`_value_string_distribution`) instead scores each value by its
**teacher-forced continuation log-probability** P(" " + value | prompt) under the same replace
hook, and softmax-normalizes over the concept's values. One forward pass per value; exact for any
tokenization.

## The audit

Six concepts, each with its standard layer, labeled by risk *before* running:

| concept | layer | risk | tokens/value | collisions | first-token gap | full-string gap | flip |
|---|---|---|---|---|---|---|---|
| agreement | 8 | multi-token | 1.40 | 1 | −0.0082 (linear) | **+0.0055 (manifold)** | **YES** |
| education | 8 | multi-token | 1.43 | 0 | +0.0062 (manifold) | **−0.0014 (linear)** | **YES** |
| rank | 20 | control | 1.13 | 0 | +0.0334 | +0.0310 | no |
| valence | 16 | control | 1.14 | 0 | +0.0020 | +0.0019 | no |
| size | 16 | control | 1.00 | 0 | −0.0024 | −0.0024 (bit-identical) | no |
| days_of_week | 14 | control | 1.00 | 0 | +0.0190 | +0.0190 (bit-identical) | no |

(gap = linear energy − manifold energy; positive ⇒ manifold more faithful. Energy is the minimum
Bhattacharyya distance to ℳ_y, as everywhere in the lab.)

The pattern is exactly the preregistered prediction: **2/2 at-risk concepts flip, 0/4 controls
flip**, and the two single-token controls are bit-identical under both read-outs (the read-outs
provably coincide there, so this doubles as a correctness check on the implementation).

The flips go **both ways**, which matters:

- **agreement** flips *toward* manifold — the first-token collision ("strongly …") was masking a
  real manifold win.
- **education** flips *away* from manifold — its apparent first-token manifold win was a
  tokenization artifact.

Isometry r is essentially unchanged in all cases (≤0.026 movement), so the geometric
activation↔behavior correspondence is read-out-robust; only the *energy verdict* was fragile.

## What this changes

- Any prior **manifold-vs-linear energy verdict on a multi-token concept** (in `docs/MANIFOLD.md`
  §5 that's education in the "3/7" row) should be treated as read-out-sensitive; single-token
  verdicts (days, size) stand as-is.
- The service now exposes the fix: `behavior_readout: "full_string"` on
  `POST /api/manifold/compare` (and `manifold_steer(..., behavior_readout=...)`). Default remains
  `first_token` for continuity with the existing ledger; use `full_string` whenever
  `mean_tokens_per_value > 1` or any collision exists.
- Future energy-based claims in this repo should report which read-out they used; for multi-token
  concepts, full-string is the right metric.

## Scope and honesty

Single seed, greedy/teacher-forced scoring, six concepts at one layer each, n=1 run per cell —
this audit establishes *read-out sensitivity*, not a new global manifold-vs-linear ledger. The
corrected per-concept verdicts above are point estimates; re-running the full §5 table under
`full_string` (all concepts × layers) is the follow-up that would replace the "3/7" row.
