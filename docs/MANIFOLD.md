# Concept-manifold steering — technical reference

How the **Manifold** mode works, the science behind it, and the honest results — including
the dead ends. This is the deep-dive companion to the task-oriented `USER_GUIDE.md`.

> One-line version: a global "map of meaning" over SAE features does **not** exist on Qwen,
> but **individual concepts lie on clean low-dimensional manifolds in the residual stream**,
> and you can steer a generation by moving a concept's residual along that manifold.

---

## 1. Motivation

Two papers from **[Goodfire](https://goodfire.ai)** frame this work — their concept-manifold geometry and manifold-steering research, which the Manifold mode builds on:

- **arXiv 2604.28119 — "Do Sparse Autoencoders Capture Concept Manifolds?"** Concepts live on
  low-dimensional manifolds; an SAE *tiles* a manifold with many features rather than encoding
  it in one. Naive geometry (decoder cosine, marginal correlation) fails to recover the
  manifold ("dilution"); conditional (Ising) couplings do better.
- **arXiv 2605.05115 — "Manifold Steering…"** Steering is a choice of *geometry*. Interpolating
  in a concept's intrinsic coordinates stays on the activation manifold ℳ_h; there is an
  **isometry** between ℳ_h and the behavior manifold ℳ_y; and the inverse problem —
  **pullback**, optimizing the activation that *induces* a target behavior — recovers ℳ_h.

The original product idea was a "latent space map": a 2D/3D topography of SAE features where
location encodes meaning. We built it, tested the premise, and it failed. What survived the
test — and became the product — is per-concept manifold steering.

---

## 2. The dead end: a global SAE-feature map does not exist

We tested whether the SAE's feature geometry encodes a navigable map of meaning. Three methods,
all on the real model, all negative:

| Method | Probe (archived) | Result |
|---|---|---|
| decoder-cosine projection | `latent_map_2b` / `_27b` | silhouette ≈ **0.003** (2B L12), ≈ **0.0016** (27B L32) — decoder directions near-isotropic; co-activating features land no closer than random (concentration ≈ 0.97) |
| co-activation fingerprints | `coact_map_2b` | silhouette ≈ **0.024** at 154 prompts (a 30-prompt 0.095 was a small-sample mirage); communities organized by **syntax/position** (`.`, `the`, `of`), not meaning |
| conditional couplings (Ising / L1-logistic) | `manifold_probe_2b` | coupling silhouette ≈ **0.005 < the marginal control 0.064**; `go_no_go = DEAD` |

**Conclusion:** neither decoder geometry, co-activation, nor the paper's SOTA coupling method
gives this SAE a "geography of meaning." The map idea is dead. All of this code was removed
from the public tree but is preserved in git history.

---

## 3. The pivot: concept manifolds in the residual stream

The same investigation, run on the **residual stream** (not SAE features) and swept across
layers, flipped to strongly positive:

- Per-concept residual centroids (one per value, averaged over carrier templates) → PCA →
  the principal subspace recovers the concept's intrinsic order.
- **Number line:** Spearman 0.94–0.97 (2B), robust across layers, peak ~L8.
- **Day ring:** a *clean* ring (nearest-neighbor adjacency = **1.0**) at 2B **L14/L16** — the
  earlier "days weak" reading was a wrong-layer artifact (L12 only 0.57).
- **Replication:** the number line and day ring reproduce on **both** 2B and 27B; deeper 27B
  layers partially recover the hardest concept (months 0.33→0.67 with depth).

So the right product is steering **along a concept's residual-stream manifold**, not an
SAE-feature map. The Modal probe `residual_manifold_sweep_2b`/`_27b` finds the best layer per
concept; `manifold_atlas_2b`/`_27b` runs the full census.

---

## 4. Architecture (how it works in code)

All of this lives in `qwen_scope_lab/service.py` (the `manifold_*` methods), exposed by
`web_api.py` and rendered by `web/app.js` + `web/manifold3d.js` (Three.js).

### 4.1 Fitting the manifold — `manifold_fit` / `_build_manifold`
For a concept (`concept_presets.py`: an ordered set of values + carrier templates):
1. For each value, build several carrier sentences and capture the **last-token residual** at
   the chosen layer (`_capture_last_residual` via `hooks.register_capture_hook`).
2. Average per value → centroids; fit `PCA(≤64)` on all rows; project centroids into PCA space.
3. Fit a `scipy` `CubicSpline` through the centroids — **periodic** for cyclic concepts (days,
   months, compass), natural for ordinal ones — giving a continuous intrinsic coordinate `u`.
4. Project to 3D for the UI (`_u_to_3d`); report a quality metric (`ring_adjacency` for cyclic,
   `abs_spearman` for ordinal).

The fit layer defaults to the concept's atlas-derived `best_layer` (`concept_presets.Concept.best_layer`)
when the caller passes none and it is valid for the loaded model.

### 4.2 The intervention — replace, not add
Manifold steering does **not** add a feature vector. It **replaces** the concept token's residual
with a point on the fitted manifold (`hooks.register_replace_hook`): overwrite
`hidden[:, position, :]` at the concept token; the KV-cache propagates the change through the
rest of generation. This is the paper-faithful intervention.

### 4.3 Traversing — `manifold_steer` (`path = manifold | linear`)
Walk `n_waypoints` from source `u` to target `u`. **manifold**: follow the spline (the geodesic
in intrinsic coords, reconstructed to d_model via `pca.inverse_transform`). **linear**: the
straight chord between the two centroids in activation space. At each waypoint we replace + generate.

### 4.4 SAE coverage — `manifold_sae_coverage`
For each value's centroid, the top-k SAE features (the *tiling* of the manifold), with notebook
labels. The 3D view recolors points by their dominant tiling feature. This is the one place the
SAE and the manifold meet on the same model.

### 4.5 The behavior manifold ℳ_y + energy
`_output_distribution` runs a forward pass under the replace-hook and reads the next-token
distribution over the concept's value tokens. `_build_behavior_manifold` fits a spline through
these distributions in **√p (Hellinger) space**; `_behavior_energy` is the minimum Bhattacharyya
distance from an induced distribution to ℳ_y — i.e. *how on-manifold the behavior is*. This is the
**right** faithfulness metric (raw perplexity is not — see §5).

For **multi-token values** the first-token read-out is unreliable (collisions like
"strongly agree"/"strongly disagree"; identity carried by later tokens). Pass
`behavior_readout: "full_string"` to score values by teacher-forced P(" "+value | prompt)
instead — see `docs/experiments/BEHAVIOR_READOUT_C05.md` for the audit: the manifold-vs-linear
verdict flips on exactly the multi-token concepts and on none of the single-token controls.

### 4.6 Pullback — `manifold_pullback` / `_pullback_path`
The inverse problem: instead of choosing where to go in activation space, specify the **target
behavior** and optimize the activation intervention that induces it. We parameterize a point in
the PCA subspace `z`, reconstruct the ambient residual **differentiably** in torch
(`z @ pca.components_ + pca.mean_`), inject it via a grad-safe mask-combine replace hook, and run
`torch.optim.LBFGS` to minimize `Hellinger²(induced next-token dist, target ℳ_y point)` + an L2
anchor. **Autograd flows through the model** to `z`. `_recover_intrinsic_r` then projects the
optimized path onto ℳ_h and correlates the recovered intrinsic coordinate with the ideal sweep.

---

## 5. Findings (real Qwen3.5-2B — honest, including the negatives)

| Claim (from the papers) | Result on Qwen-2B |
|---|---|
| activation↔behavior **isometry** | **REPLICATES: r = 0.94–1.00** across 7 concepts (days 1.00, rank 0.98, size 0.97, agreement 0.97, education 0.96, valence 0.94; integers degenerate→NA). Matches the paper's 0.89–0.999. |
| manifold interp is **more natural** (lower perplexity) | **DOES NOT replicate on raw perplexity: manifold beats linear 0/7.** Qwen snaps off-manifold chord points to fluent tokens, so raw fluency doesn't separate them. |
| manifold is **more faithful** (right metric) | **behavior-manifold energy (full-string read-out): manifold beats linear 4/6** (days, rank, agreement, valence; integers degenerate→NA) — real but **concept-dependent**. This is the corrected ledger from the C05 follow-up, scored under the multi-token-faithful full-string read-out (full table: `reports/manifold_c05/full_table_full_string.json`, runner `scripts/_c05_full_table.py`). *Historical first-token row (C05-caveated):* the original write-up reported "manifold beats linear 3/7 (rank, education, days)" under the first-token read-out from the Modal naturalness probe; that read-out merges/truncates multi-token values, so the per-concept verdict flips on exactly the two multi-token concepts — **agreement gains a win, education loses one** — leaving the same count by coincidence. Single-token verdicts (days, size) are bit-identical across read-outs. See `docs/experiments/BEHAVIOR_READOUT_C05.md`. |
| concepts lie on clean manifolds | **15/17 form clean/partial manifolds** in the atlas census; only days forms a clean *ring*; months & compass are diffuse. |
| **pullback** induces on-manifold behavior | **YES — energy ≤ linear 3/3** (days/rank/education), lowest of all three each time; LBFGS loss decreases every run (autograd verified). |
| pullback **recovers ℳ_h** (bidirectional isometry) | **2/3** (rank/education r≈0.88 > linear) — **but days fails (r = −0.36)**: the ring behavior is so easily induced the optimization is underconstrained and finds an off-manifold path that still works. The bidirectional claim holds for graded concepts, breaks for the trivial clean-ring. |

Net: the papers' **geometric** claim (isometry) replicates strongly on Qwen; manifold steering's
*faithfulness* advantage is real but partial; pullback works as a capability and partly confirms
the bidirectional result. We report the nulls as prominently as the wins.

**Atlas census (2026-06-10):** the geometry is dense — a sweep of 20 *new* candidate concepts found
**12 clean manifolds** (order recovery + isometry both ≥ 0.9), 8 partial, 0 diffuse, all promoted into
`concept_presets.py::_ATLAS_EXTRA`. But re-running the routing test on the 11 clean ordinals, only
**2 route** (certainty, age_stage), weakly — and `priority`, the cleanest fit (iso r 0.998), does not
route at all. **Clean geometry ≠ clean control.** See `docs/experiments/MANIFOLD_ATLAS_CENSUS.md` and
the interactive 3D atlas `docs/writeups/manifold-atlas-3d.html`.

---

## 6. API reference

All under `web_api.py`; the dev backend (`dev_backend.py`) serves a synthetic ring so the whole
stack runs on CPU.

| Endpoint | Purpose |
|---|---|
| `GET  /api/manifold/presets` | list concepts (name, label, kind, n_items, `best_layer`) |
| `POST /api/manifold/fit` | fit a concept manifold → 3D points + curve + quality |
| `POST /api/manifold/steer` | traverse source→target (`path=manifold\|linear`) → waypoints + generations |
| `POST /api/manifold/compare` | manifold vs linear, with per-waypoint energy + perplexity |
| `POST /api/manifold/sae_coverage` | the SAE features that tile the manifold |
| `POST /api/manifold/pullback` | 3-leg manifold/linear/pullback, each with `mean_energy` + `recovered_r` |
| `POST /api/recipes` `{"kind":"manifold"}` | save the last pullback as a manifold recipe |

(The removed dead-map route `POST /api/layout` no longer exists.)

---

## 7. Manifold recipes (Library integration)

A manifold steer is a first-class, saveable recipe. After running a **pullback**, the backend
snapshots it (`web_api.build_manifold_recipe`) into a `FeatureRecipe` with `kind="manifold"` and a
`ManifoldSpec` (concept, source, target, layer, path, n_waypoints) — see `recipe_schema.py`. The
**manifold benchmark is the pullback 3-leg comparison**; the verdict
(`manifold_validation_decision`) is honest: `validated` only if on-manifold steering induces the
target behavior at least as faithfully as the linear chord (energy ≤ linear), else `benchmarked`.
Saved recipes appear in the Library with the energy legs + a **Load into Manifold** action.

---

## 8. Modal validation harness

Live probes in `modal_app.py` (real model; run with `modal run modal_app.py::<fn>`):
`residual_manifold_sweep_2b/_27b` (best layer per concept), `manifold_atlas_2b/_27b` (census),
`manifold_steer_demo_2b/_27b`, `manifold_vs_linear_2b`, `manifold_naturalness_probe_2b`
(isometry + energy), `manifold_pullback_probe_2b` (pullback). These gated probes are the way to
produce a recorded real-GPU result; stop the warm GPU with `modal app stop qwen-scope-lab`
when done. **The entire manifold mode (fit / steer / compare / pullback / SAE coverage) also runs
locally on the real 2B via the MLX backend** — `serve_web.py --mlx … --mlx-sae …` — no Modal/CUDA;
the pullback's L-BFGS optimisation becomes an `mx.value_and_grad` + Adam loop on-device (see `MLX.md`).
The 27B remains Modal-only. For maximal fidelity, run the **base** model the SAE was trained on with
`serve_web.py --mlx-base` (or `configs/qwen35_2b_base_l0_100.yaml` on CUDA): the residual centroids and
SAE-coverage tiling then come from the exact activations the SAE saw — see the base-vs-instruct note in
`MLX.md`.

The dead-end probes that produced §2's negatives were removed from the public tree but are
preserved in git history.

## 9. References

The concept-manifold geometry and manifold-steering research below is **[Goodfire](https://goodfire.ai)'s**;
the Manifold mode builds on it and replicates/tests it on Qwen.

- Goodfire — "Do Sparse Autoencoders Capture Concept Manifolds?" (arXiv 2604.28119)
- Goodfire — "Manifold Steering Reveals the Shared Geometry of Representation and Behavior" (arXiv 2605.05115)
