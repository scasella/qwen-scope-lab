# C09 distribution-distillation salvage — report

**Verdict: REFUTED** (clean negative; this repo publishes negatives).

Geometry-gated **distribution** (soft-label KL) distillation of concept-manifold steering yields **no
hook-free transfer advantage** over a no-geometry prompt-only teacher, a linear-chord baseline, **or a
shuffled-label negative control** at 2B. The transfer that *does* appear comes from the value-filled
carrier prompt, not from the manifold geometry. This closes the last live formulation of the
manifold-to-data compiler: text-SFT was structurally refuted (greedy-text invariance + endpoint
coincidence; see the parent doc), and the distribution objective — the one regime where the
within-sweep behavior-energy signal actually lives — does not survive the preregistered gate.

---

## Method

The text-SFT arm was refuted because the manifold replace intervention leaves greedy text invariant
and the manifold/linear arms coincide at the target endpoint. The only residual signal is the
**next-token value distribution**, which the intervention *does* shift (within-sweep behavior-energy
range ~0.06). So we test the distribution objective directly: train a hook-free student LoRA to match
the manifold-intervention teacher's next-token distribution.

**Teacher data** (`scripts/_c09_distill_generate.py`). For each concept, for every carrier template x
source value, we walk the source->target sweep (5 waypoints) and store, **per waypoint**, the teacher
next-token distribution under the position-replace hook (the same `_ReplaceLayer` intervention used by
`manifold_steer`/`manifold_compare`). The supervision **prompt is the carrier filled with the
waypoint's intended value label**, and the replace position is that value-label token — so
`(prompt -> teacher_dist)` is a learnable hook-free map. (Choice: a constant carrier with a varying
intended value would be contradictory supervision; encoding the value in the prompt is the simplest
faithful fix, and it is exactly what makes the arms separable by a hook-free student.)

Teacher distributions are stored **top-k truncated + renormalized, k=256** (full vocab is 248,320;
top-256 captures the head where the value-token mass lives — documented per the ground rules).

**Arms** (equal-size train cap across arms; carrier-template held-out split — train templates 0..n-3,
held-out test template n-1):

| arm | residual injected at each waypoint | role |
|---|---|---|
| `gated_manifold` | manifold-spline waypoint, kept iff sweep energy <= linear chord | primary (geometry-gated) |
| `ungated_manifold` | all manifold-spline waypoints | isolates whether the gate matters |
| `linear` | linear-chord (ambient straight-line) waypoint | baseline |
| `prompt_only` | **none** — teacher = base model dist on the value-filled carrier | no-geometry teacher |
| `shuffled_label` | gated_manifold teacher dists with their (prompt,value) labels permuted | negative control |

**Pullback arm dropped (documented).** The prereg lists pullback as arms 2/4. `manifold_pullback`
does not return the optimized residual vectors and runs on `concept.steer_prompt` (not the carrier
template), so a faithful per-template pullback teacher would require new service plumbing and an
Adam-through-the-model optimization per (template, source, waypoint) — well over the 3-hour budget
guard. Rather than ship a misleading approximation, the pullback arm was omitted. The gate's decisive
comparison (gated manifold vs prompt-only vs linear vs shuffled) is fully covered.

**`recovered_r` gate not applied (documented).** The prereg gate also requires
`recovered_r >= max(linear, 0.75)`. `recovered_r` is produced only by the pullback computation; with
pullback dropped, the geometry gate here is **energy-only** (`energy <= linear chord`). Consequence:
the days-specific clause ("days passes the same gate *without* a `recovered_r` explanation") cannot be
fully adjudicated — days is reported as a secondary energy-only stress check.

**Training** (`scripts/_c09_distill_train.py`). One LoRA per arm, custom KL loop
(`mlx.optimizers.Adam` + `mlx_lm.tuner.utils.linear_to_lora_layers`), loss =
`KL(teacher || student)` over the teacher's top-k support. Hyperparameters mirror the refuted text-SFT
run for an apples-to-apples comparison: **rank 8, scale 20, 8 LoRA layers, lr 1e-4, batch 4, 120
iters, max_seq 64**. ~30 s/LoRA on M4 Pro; 15 LoRAs total. Adapters saved in mlx_lm format
(`adapters.safetensors` + `adapter_config.json`), loadable by the held-out eval via `adapter_path`.

**Evaluation** (`scripts/_c09_distill_gate.py`). For base + each arm, on the **held-out carrier
template**, read the model's value distribution (hook-free) and compute `expected_position`
(0=source end ... 1=target end, averaged over source carriers). The transfer score is
`delta expected_position vs base`. Also Spearman `order_corr` between intended value order and the
model's position.

---

## Numbers (delta expected-position vs untrained base; higher = more target transfer)

### rank (private->general, L20, first_token)

| arm | delta vs base |
|---|---|
| base | 0.751 (abs) |
| `prompt_only` | **+0.142** |
| `shuffled_label` (control) | **+0.159** |
| `ungated_manifold` | +0.155 |
| `gated_manifold` (primary) | +0.128 |
| `linear` | +0.067 |

`gated_manifold - prompt_only = -0.013`, `gated_manifold - shuffled_label = -0.031`.

### education (kindergarten->doctorate, L8, full_string)

| arm | delta vs base |
|---|---|
| base | abs |
| `prompt_only` | **+0.180** |
| `shuffled_label` (control) | +0.136 |
| `linear` | +0.114 |
| `gated_manifold` (primary) | +0.099 |
| `ungated_manifold` | +0.051 |

`gated_manifold - prompt_only = -0.081`, `gated_manifold - linear = -0.014`,
`gated_manifold - shuffled_label = -0.037`.

### days_of_week (stress, cyclic, L14, first_token, energy-gate only)

All arms slightly **negative** (-0.03 ... -0.07) on the held-out cyclic carrier; no arm transfers and
geometry confers nothing. The energy gate kept 30/36 sweeps (manifold "better" by energy), i.e. it
**over-keeps without a `recovered_r` filter** — consistent with the prereg worry that an energy-only
gate is not measuring identifiable geometry, but the point is moot because no arm transfers here.

Generation-side geometry signal (sanity, matches the documented C09 metric result): mean energy gap
(linear - manifold) = **+0.0148** (rank), **+0.0107** (education), **+0.0093** (days); manifold beats
linear on **31/35**, **24/30**, **30/36** sweeps. The geometry signal is real at the metric level — it
just does not convert into hook-free transfer.

---

## Verdict — preregistered gate

> Refute if, on `rank` and `education`, the geometry-gated manifold dataset fails to beat **both**
> equal-size prompt-only **and** linear-chord by **>= +0.05** transfer score.

- **rank**: `gated_manifold` (+0.128) does **not** beat `prompt_only` (+0.142) — it is *below* it.
- **education**: `gated_manifold` (+0.099) does **not** beat `prompt_only` (+0.180) by +0.05.

Both primary concepts trip the gate -> **REFUTED**. (`distill_verdict.json` in this directory.)

### The decisive confirmation: the shuffled-label control wins

On **both** concepts the **shuffled-label negative control beats `gated_manifold`** (rank +0.159 vs
+0.128; education +0.136 vs +0.099). A negative control outperforming the treatment means the
transfer is **not geometry-driven** — it comes from the value-filled carrier prompt biasing the model
toward the target value, which every arm (including the shuffle and the hook-free `prompt_only`)
shares. The geometry provenance adds nothing on top.

This is the distribution-side analog of the text-side structural finding: text-SFT failed because the
arms coincide in *text*; distribution-distillation fails because the only thing the student can learn
from the value-filled carrier is the carrier's own value bias, which is identical across arms — the
per-waypoint distribution shift the manifold induces is too small (energy ~0.06) and too entangled
with the prompt to be isolated by a hook-free objective at 2B.

---

## Honesty & scope

- Clean negative across the two primary concepts + a null stress case; verdict robust to the two
  documented simplifications (pullback dropped, energy-only gate) because the **shuffled control**
  already beats the treatment — no geometry-specific arm can rescue the claim.
- The distribution distillation *did* improve target transfer over the untrained base (unlike text-SFT,
  which degraded it) — but that gain is a prompt/carrier effect, not a manifold-provenance effect.
- SOURCE-GAP: builds on Goodfire arXiv 2604.28119 / 2605.05115; novelty vs adjacent
  activation-steering distillation work remains an unchecked prior-art prior, not a claim.

## Artifacts
- `<concept>/<arm>/{train,valid,test}.jsonl` — teacher top-k distributions per waypoint.
- `<concept>/<arm>/adapter/` — trained KL-LoRA (mlx_lm format) + `train_log.json`.
- `<concept>/generation_summary.json` — energy gap + equal-size accounting.
- `<concept>/gate_eval.json` — per-arm held-out transfer.
- `distill_verdict.json` — machine-readable gate application (this directory).
- Scripts: `scripts/_c09_distill_generate.py`, `_c09_distill_train.py`, `_c09_distill_gate.py`.
