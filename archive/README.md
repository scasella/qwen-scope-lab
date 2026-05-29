# archive/ — the dead-end geometry investigation (preserved for provenance)

This directory holds code that is **no longer part of the product** but is kept as a
record of *why* the product is shaped the way it is. None of it is wired into the live
app, the web UI, or the test suite. It is preserved, not executable as-is.

## Why this exists

The project began with a compelling idea: render a **"latent space map"** — a 2D/3D
topography of the SAE's features where nearby points mean related concepts, so you could
*navigate meaning by location*. We built it, then tested the premise on the real model.
It failed, decisively and repeatedly:

| Probe | What it tested | Result |
|---|---|---|
| decoder-cosine map (`latent_map_*`) | do SAE decoder directions cluster by meaning? | **silhouette ≈ 0.003 (2B L12), ≈ 0.0016 (27B L32)** — near-isotropic, no geography. Co-activating features land no closer than random (concentration ≈ 0.97). |
| co-activation map (`coact_map_*`) | do features that fire together sit together? | **silhouette ≈ 0.024** at 154 prompts (a 30-prompt 0.095 was a small-sample mirage). Communities organized by **syntax/position** (`.`, `the`, `of`), not semantics. |
| Ising / conditional couplings (`manifold_probe_*`) | the SOTA method from arXiv 2604.28119 | **coupling silhouette ≈ 0.005 < the marginal control 0.064**. `go_no_go = DEAD`. |
| early residual probe (`residual_manifold_*`) | per-class residual centroids → PCA | the bridge result that, once swept across **layers**, became the real finding (below). |

**Conclusion: a global SAE-feature "map of meaning" does not exist.** But the residual-stream
follow-up revealed that *individual concepts* (number line, day ring, …) **do** lie on clean
low-dimensional manifolds, replicated across 2B and 27B. That is what the shipped **Manifold**
mode steers along. The full write-up — including the positive results (isometry r≈1, pullback)
— is in `docs/MANIFOLD.md` and the project memory.

## Files

- **`research_probes.py`** — the five dead-end Modal probe groups extracted verbatim from
  `modal_app.py` (decoder-cosine latent map, co-activation map, Ising/coupling concept probe,
  early residual-manifold probe, manifold-vs-linear probe). The `@app.function(...)`
  decorators are kept for readability but are **inert** here — the file is not imported by
  `modal_app.py` and references helpers (`_spearman`, `_manifold_metrics`, `_CONCEPTS_V2`) and
  service internals that live in `modal_app.py` / the package. Not runnable as-is.
- **`latent_map_engine.py`** — the `SteeringService` methods that backed the removed
  `POST /api/layout` route and the Explore→Map view (`latent_layout`, `_feature_norms`,
  `_project_2d`). `research_probes.py::_latent_map` called `latent_layout`; `_coactivation_map`
  called `_project_2d`.

## What replaced it (in the live product)

`qwen_scope_steering_gui/service.py` → `manifold_fit` / `manifold_steer` / `manifold_compare`
/ `manifold_sae_coverage` / `manifold_pullback`; the `/api/manifold/*` routes; the **Manifold**
mode in `web/`; and the kept Modal validation probes (`manifold_steer_demo_*`,
`manifold_vs_linear_2b`, `manifold_naturalness_probe_2b`, `manifold_pullback_probe_2b`,
`manifold_atlas_*`, `residual_manifold_sweep_*`).
