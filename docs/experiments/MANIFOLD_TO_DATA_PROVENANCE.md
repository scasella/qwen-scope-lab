# Manifold-to-data provenance compiler (C09)

**Question.** Does compiling concept-manifold steering into training data — keeping only the
samples whose *geometry* clears a gate (behavior-energy ≤ the linear chord, and `recovered_r`
above a threshold where available) — produce SFT/preference data that transfers ordered-concept
behavior **more reliably** than equal-size data from the linear chord, ungated pullback, or a
prompt-only teacher?

This is the "product" direction from the manifold-research-directions study: the manifold geometry
is used as a **data-selection signal**, not just a runtime steering diagnostic. The moat is the
provenance + geometry gate, not the model call.

> **SOURCE-GAP (unverified).** This builds on **Goodfire**'s concept-manifold geometry and
> manifold-steering work (arXiv **2604.28119**, **2605.05115**). Novelty *beyond* those papers and
> adjacent activation-steering distillation work has **not** been checked against the literature
> (no web access at authoring time). Treat "novel" as an open prior-art check, not a claim.

---

## Status: first build step (schema / gate / export only)

What exists today (`qwen_scope_lab/experiments/manifold_distill.py`,
`scripts/manifold_to_data_distill.py`, `tests/test_manifold_distill.py`):

- a **torch-free compiler** that normalizes `manifold` / `linear` / `pullback` legs (the shape of
  `SteeringService.manifold_compare` / `manifold_pullback`) into one record per (leg, waypoint);
- a **geometry gate** (non-empty/non-collapsed text, hook fired, on-manifold `energy ≤ linear`,
  `recovered_r ≥ threshold` when present) with a **rejection ledger** (reasons retained, never
  silently dropped);
- **provenance-stamped exports**: `pairs_all/kept/rejected.jsonl`, `sft.jsonl`, `preference.jsonl`
  (chosen = on-manifold, rejected = linear chord at the same waypoint), `metrics.json`,
  `dataset_card.md`, `report.md`;
- **equal-size arm accounting** (`equal_size_n_per_arm`) so the eventual training comparison is
  balanced;
- a deterministic, model-free **synthetic smoke** (`synthetic-smoke --scenario win|fail`).

**It trains no model and makes no transfer claim.** Real behavior-energy / `recovered_r` come from
`qwen_scope_lab/service.py` and the `/api/manifold/*` endpoints; the tests use synthetic payloads.

This step **depends on C05** (`docs` + `_c05_mlx_audit.py`): the behavior-energy metric the gate
relies on flips the manifold-vs-linear verdict on multi-token concepts under the first-token
read-out, so the live compiler must run with `behavior_readout="full_string"` for any multi-token
concept (the field is threaded through `manifold_compare`).

---

## Live result (2B, MLX) — rank (the naive text-SFT product does NOT work at 2B)

`scripts/_c09_mlx_generate.py --concept rank` ran the **real** Qwen3.5-2B (MLX) over 7 source
values × 5 carrier templates (35 sweeps), compiling each per-prompt payload through the gate.
Evidence in `reports/manifold_c09/rank/{generation_summary,signal_analysis}.json`.

**The geometry gate works at the metric level:**
- mean behavior-energy gap (manifold − linear) = **+0.0148**; manifold is more faithful on
  **31 / 35** prompts. The on-manifold path genuinely beats the linear chord on ℳ_y energy.

**But the steered *text* carries none of it:**
- mean **distinct generated texts per 5-waypoint sweep = 1.00** — the greedy continuation is
  **identical** at every point from source→target, even though the within-sweep behavior-energy
  range is **0.0685** (the value distribution *does* move).
- **manifold text == linear text on 35 / 35 prompts.**

**Verdict (refutes the naive formulation).** On 2B, the paper-faithful replace intervention at the
geometry-optimal layer (L20) + greedy decoding shifts the **next-token value distribution** but
leaves **free-text generation invariant**. So `(prompt → steered_text)` SFT data is **textually
identical across the gated-manifold, ungated-manifold, and linear arms** — a LoRA trained on it
cannot distinguish them *by construction*, so the equal-size training comparison is uninformative
and was **not run** (training on identical inputs would be theater, not evidence). The provenance
compiler and the geometry gate are correct; the **vehicle is wrong**: the manifold signal lives in
the distribution, not in greedy text.

**Generalizes to the multi-token concept.** `--concept education --readout full_string` (layer **8**,
6 sources × 5 templates): energy gap **+0.0107**, manifold beats linear **24 / 30**; distinct
texts/sweep **1.667**, manifold==linear text on **70 %** of prompts
(`reports/manifold_c09/education/signal_analysis.json`). The same metric-signal / text-invariance
split holds. Notably education (early layer 8) moves text *slightly* more than rank (late layer 20,
distinct=1.00) — direct evidence for salvage #2 below: earlier-layer injection surfaces more of the
steer in text.

**Refined product/research direction (the salvage).** The compiler should target the **distribution**,
not free text:
1. **Distribution distillation** — train to match the manifold-induced next-token value distribution
   (KL / soft-label), where the +0.0685 within-sweep shift lives, instead of SFT on text.
2. **A regime where the steer moves text** — earlier-layer injection, temperature sampling, or
   path extrapolation; check whether distinct-texts-per-sweep rises above 1.0 before any SFT claim.
3. **A larger model** — the effect size here matches the repo's documented 2B nulls (manifold beats
   linear on energy for only some concepts; raw perplexity 0/7). Bigger models may surface the steer
   in text.

The held-out transfer harness (`scripts/_c09_mlx_eval.py`, adapter-aware MLX load) is built and
ready for any of those once the data carries a usable signal.

### Salvage #2 — injection-layer sweep (`scripts/_c09_layer_sweep.py`)

`reports/manifold_c09/layer_sweep.json`. Distinct greedy texts per 5-waypoint sweep, and how often
manifold text ≠ linear text, as a function of injection layer:

| layer | rank distinct/sweep | rank man≠lin | education distinct/sweep | education man≠lin |
|---|---|---|---|---|
| 4  | 1.00 | 0% | **2.33** | 50% |
| 8  | 1.00 | 0% | 1.67 | 33% |
| 12 | 1.00 | 0% | 1.67 | 50% |
| 16 | 1.00 | 0% | 1.50 | 50% |
| 20 | 1.00 | 0% | 1.33 | 0% |

**Partial salvage, concept-dependent.** Single-token **rank** is text-frozen at *every* layer.
Multi-token **education** moves more at **earlier** layers (best at L4) and decays with depth — so
earlier-layer injection surfaces more of the steer in text for multi-token concepts, but the effect
is modest (≈2.3 of 5 waypoints distinct) and absent for single-token concepts.

### The deeper structural finding (why text-based distillation can't carry the provenance)

The manifold and linear arms **coincide at the target waypoint by construction**: at t=1 the manifold
path's point is the spline evaluated at the target index, which *is* the target centroid — the same
point the linear chord ends at. So manifold/linear/gated arms share the *endpoint* behavior. The
provenance distinguishes the **intermediate path** (where on-manifold stays low-energy and the linear
chord cuts off-manifold), not the endpoint. But intermediate waypoints have a **constant carrier
prompt** with a varying intended value — not a learnable hook-free `prompt → output` map. Combined
with text-invariance, this means **no text objective (SFT / DPO / preference) converts the
manifold-vs-linear distinction into a hook-free behavioral difference at 2B.** The only residual
signal is the per-waypoint value distribution (energy range ≈0.06) — tiny, and not learnable from the
constant prompt. This is a stronger refutation than "the text is junk": even with clean text, the
arms would share endpoints.

### Salvage #1 — geometry-gated distillation, trained + evaluated (MLX LoRA, rank)

Three LoRA arms trained locally on the rank SFT data (`gated_manifold` / `ungated_manifold` /
`linear`, equal-size), evaluated on a **held-out carrier template** by the hook-free value
distribution (`reports/manifold_c09/rank/{eval_results,distill_verdict}.json`):

| arm | held-out expected position (0=source … 1=target) | Δ vs untrained base |
|---|---|---|
| base (untrained) | **0.751** | — |
| gated_manifold | 0.741 | −0.010 |
| ungated_manifold | 0.665 | −0.086 |
| linear | 0.665 | −0.086 |

**Falsification gate TRIPPED → REFUTED.** No arm beat base (every Δ ≤ 0). Two confirmations of the
structural finding: (1) `ungated_manifold` and `linear` produced **bit-identical** eval results —
their text-SFT data is the same, so geometry provenance is invisible to a text objective; (2)
training on steered text *degraded* the base model's pre-existing target lean, and gating merely
**avoided** that damage rather than adding transfer. (Caveat: "general" is high-frequency, so base
already sits at 0.75 — a frequency-confounded ceiling; but the arm-equivalence and no-improvement
results are robust to that.) Geometry-gated **text** distillation of manifold steering yields no
hook-free transfer benefit at 2B — as the text-invariance + endpoint-coincidence mechanism predicts.

### Salvage #3 — does scale revive the steer? (2B vs 4B, `reports/manifold_c09/scale_comparison_2b_vs_4b.json`)

The larger-model salvage tests whether manifold-steered **text** moves on a bigger model. Because
Tinker can't run steering hooks, generation was done locally on **Qwen3.5-4B** (MLX). Distinct greedy
texts per 5-waypoint sweep:

| concept | layer | 2B | 4B |
|---|---|---|---|
| rank | 4 / 12 / 20 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| education | 4 | 2.33 | 2.33 |
| education | 12 / 20 | 1.67 / 1.33 | 1.17 / 1.33 |

**REFUTED.** Doubling the model (2B→4B) leaves the pattern unchanged: rank frozen at every layer on
both; education ~2.3 at L4 on both. **Manifold-text-invariance is architectural, not a model-size
artifact** — so a larger *training* base (Tinker 9B) cannot help, because the bottleneck is the
generated *data*, which is upstream of training and scale-invariant.

### Tinker note

The Tinker key in `.env` was initially billing-blocked (HTTP 402); after the balance was added it
**works** (confirmed: a `forward_backward` + `optim_step` on Qwen3.5-4B completed). But there is **no
usable manifold-steered data to train on** — the data bottleneck (text-invariance) is unaffected by
training-model size — so a full Tinker run on this line would be uninformative. The local **MLX LoRA**
training above stands as the end-to-end confirmation. Tinker is, however, the right tool to scale the
repo's **proven feature/probe-steering distillation** (the truth-holding / sentiment v1.0 line, which
*does* produce text-varying data) to a 9B base — a separate, positive use of the now-live billing.

---

## Preregistration (the live experiment, not yet run)

**Primary concepts.** `rank` (private→general, L20) and `education` (kindergarten→doctorate, L8) —
the repo's documented graded concepts where pullback recovers ℳ_h. **Stress case:** `days_of_week`
(Monday→Thursday, L14) — the clean-ring identifiability failure; it should be *rejected* by the
`recovered_r` gate or show energy success without geometry recovery.

**Arms (equal size, matched prompts/waypoints).**
1. geometry-gated **manifold** records
2. geometry-gated **pullback** records
3. **linear-chord** records (baseline)
4. **ungated pullback** (text-quality only — isolates whether the *gate* matters)
5. **prompt-only** teacher (instruction toward the target value, no hook)
6. **random-PCA-subspace / shuffled-label** control

**Metrics.** Held-out monotonic forced-choice accuracy / order correlation across concept values
after equal-size LoRA training (primary); behavior-energy vs ℳ_y; `recovered_r`; data-quality
(kept rate, relevance, genericness, repetition, collapse) via `distill_quality.py`.

**Falsification gate.** Refute if, on `rank` and `education`, the geometry-gated manifold/pullback
dataset **fails to beat both** equal-size prompt-only **and** linear-chord datasets by **≥ +0.05**
macro forced-choice/order score while keeping mean energy ≤ linear and `recovered_r ≥ max(linear,
0.75)` and not regressing relevance (> 0.05) / genericness (> 0.10) / collapse. Also refute if the
gain appears only on **reused** carrier templates and vanishes on held-out semantic prompts, or if
`days` passes the same gate **without** a `recovered_r` explanation (the gate would not be measuring
identifiable geometry).

**Expected honest result.** A modest but real data-efficiency advantage for geometry-gated
pullback/manifold records on `rank`/`education`; `days` rejected or energy-only. Concept-dependent,
matching the repo's documented nulls.

---

## Next build steps

1. Live generation: extend `HttpGenerationBackend` with `/api/manifold/compare` + `/api/manifold/pullback`
   calls; a `manifold-generate` CLI subcommand (`--concept/--source/--target/--layer/--path/--n-waypoints/--energy-gate/--recovered-r-gate`).
2. Equal-size LoRA training (`scripts/steering_distill_train_tinker.py`) across the six arms, or
   `training_not_run` if only the compiler is wired.
3. Held-out evaluation on semantically varied prompts **not** in `concept_presets.py` templates.
4. Optional gated Modal probe `manifold_to_data_probe_2b` (summary metrics only).
