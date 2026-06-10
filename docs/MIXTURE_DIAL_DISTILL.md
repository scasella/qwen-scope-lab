# Mixture Dial Distill

Mixture Dial Distill is a local dataset compiler for the steer_distill lane. It
turns candidate rows plus a human-authored mixture config into chat-format SFT
records and a compact manifest of requested vs achieved dataset shape.

It only compiles training data. It does not load a model, call a hosted service,
train an adapter, evaluate an adapter, run runtime steering, or use activation
hooks.

## Commands

Truth-holding A/B/C fixture:

```bash
python scripts/mixture_dial_distill.py compile \
  --mixture examples/mixture_dial/truth_holding_mixture.yml \
  --candidates examples/mixture_dial/truth_holding_candidates.jsonl \
  --out /tmp/mixture_dial_smoke
```

Generic user-defined fixture:

```bash
python scripts/mixture_dial_distill.py compile \
  --mixture examples/mixture_dial/generic_mixture.yml \
  --candidates examples/mixture_dial/generic_candidates.jsonl \
  --out /tmp/mixture_dial_generic_smoke
```

## Mixture Config

Minimal shape:

```yaml
schema_version: "0.1.0"
seed: 0
total: 12
dimensions: [class, pressure, domain, rejection_mode]
slots:
  - name: A_factual_false_pressure
    where: {class: A_factual, pressure: user_false_certainty}
    ratio: 0.5
  - name: B_unknowable_calibration
    where: {class: B_unknowable}
    ratio: 0.25
  - name: C_subjective_calibration
    where: {class: C_subjective}
    ratio: 0.25
output:
  format: sft_chat
  preserve_labels: true
```

Each slot has a `name`, a `where` filter, and exactly one of:

- `count`: exact requested row count for the slot.
- `ratio`: relative weight for the slot.

If a config mixes count and ratio slots, count slots reserve their exact count
first, then ratio slots divide the remaining target total by largest-remainder
rounding. Sampling is seeded, deterministic, and without replacement across the
whole output corpus. Slot order matters when broad filters overlap narrower
filters.

## Candidate Rows

Candidate rows are JSONL objects. The simplest shape is:

```json
{"id":"row_001","prompt":"Question...","output":"Assistant answer...","class":"exact","pressure":"format_pressure","domain":"docs"}
```

The compiler also accepts existing truth-holding labels:

- `behavioral_class` as `class`, including `A_factual`, `B_unknowable`, and
  `C_subjective`.
- `pressure_type` as `pressure`.
- `domain`.
- `reject_reason`, `reject_reasons`, `rejection_mode`, `quality_label`, or a
  configured alias as `rejection_mode`.

Rows may also provide chat `messages`. If `messages` already contains user and
assistant turns, those turns are exported as the SFT `messages`.

Use `aliases` for project-specific label names:

```yaml
aliases:
  rejection_mode: [quality_label]
```

Use `exclude_rejection_modes` only when you explicitly want rows with those
labels written to `skipped.jsonl` instead of considered for slots:

```yaml
exclude_rejection_modes: [teacher_rejected]
```

## Outputs

The output directory contains:

- `sft.jsonl`: chat-format SFT records. With `preserve_labels: true`, records
  also carry `source_id`, `mixture_slot`, canonical labels, and a `labels` map.
- `mixture_manifest.json`: schema version, timestamp, source paths, seed, target
  slots, requested counts, achieved counts, capped and underfilled slots, dropped
  row counts, output paths, and label summaries.
- `skipped.jsonl`: only written when rows are malformed or explicitly excluded
  by configured rejection modes.

The manifest reports capped slots when a matching pool is smaller than the
requested count. It does not make any claim about how a future adapter will
behave after training.

## Validation status (honest)

A falsification pilot was run 2026-06-09 (8 Qwen3.5-4B LoRA arms compiled from the
v1.0 truth-holding corpus, gpt-5.5 rubric judge on held-out B/C calibration
prompts; evidence: `reports/mixture_dial_pilot/pilot_verdict.json`). Two distinct
claims came out of it, with opposite verdicts:

- **The mixture is the mechanism — SUPPORTED.** Any calibration-bearing mixture
  beats a truth-only corpus of the same size by a wide margin: calibration
  acceptable 0.833 (dial @ 0.50) / 0.812 (naive stratified) / 0.797 (random
  same-size) vs **0.661 truth-only** (K3 = +0.172). This replicates the v1.0
  matched-size result from the write-up.
- **The *dial* is differentiated from naive stratified sampling — NOT
  SUPPORTED.** The dial's chosen ratio edges naive class-proportional sampling
  by K2 = +0.021 — within noise at 2 seeds. On this corpus/behavior, "include
  the calibration classes at all" does the work; the specific ratio does not.

So: use this compiler to *control and document* corpus shape (deterministic,
manifest-stamped, kill-criteria-friendly) — not because a tuned ratio has been
shown to beat natural proportions. A dial-vs-stratified win would need a
behavior with skewed natural class proportions, or a ratio the pool can't reach
by sampling; that experiment has not been run. The 18-arm dose-response plan
(with preregistered kill criteria) exists as a dry-run scaffold:
`scripts/mixture_dial_validate.py --dry-run`.
