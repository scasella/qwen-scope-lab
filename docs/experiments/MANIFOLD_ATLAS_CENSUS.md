# Manifold atlas census — how many concepts form clean manifolds, and how many route?

**One-line result:** of 20 net-new candidate concepts, **12 form clean residual-stream
manifolds** on real Qwen3.5-2B (order recovery ≥ 0.9 AND isometry r ≥ 0.9), 8 partial, 0
diffuse — but re-running the routing test on the 11 clean ordinals, **only 2 clear the strict
centroid-routing gate** (age_stage, certainty), and weakly. Clean geometry is common and cheap;
stepwise control is rare and concept-dependent.

Evidence: `reports/manifold_census/census.json` (fit sweep), `reports/manifold_census/routing.json`
(routing + energy), `reports/manifold_census/geometry_3d.json` (the plotted geometry).
Runners: `scripts/_manifold_census.py`, `scripts/_census_routing.py`. Interactive 3D:
`docs/writeups/manifold-atlas-3d.html`. Run 2026-06-10 on `mlx-community/Qwen3.5-2B-bf16`.

## Method

Each candidate is an ordered value set with carrier templates (the item at the end). For each, we
fit the residual-stream manifold (`manifold_fit` → PCA + spline through per-value centroids) at
layers {6, 8, 12, 16, 20} and score it two ways:

- **order recovery** — `_manifold_quality`: abs-Spearman of the principal axis vs the value order
  (ordinal), or ring-adjacency (cyclic). "Did the geometry recover the ordering?"
- **isometry r** — the paper's activation↔behavior correspondence under the full-string read-out.

Best layer = the layer maximizing order recovery. Classification at the best layer:

| verdict | rule |
|---|---|
| **clean** | order ≥ 0.9 AND isometry r ≥ 0.9 |
| **partial** | order ≥ 0.75 OR isometry r ≥ 0.85 |
| **diffuse** | neither |

This is a geometry measurement. It makes no steering claim — that is the separate routing test.

## The census — 12 clean, 8 partial, 0 diffuse

The 12 clean (promoted into `concept_presets.py::_ATLAS_EXTRA`, so they load in the Manifold mode):

| concept | layer | order | isometry r |
|---|---|---|---|
| priority (trivial→critical) | 12 | 1.00 | 0.998 |
| age_stage (infant→elderly) | 16 | 1.00 | 0.982 |
| certainty (impossible→certain) | 12 | 0.90 | 0.979 |
| difficulty (trivial→impossible) | 12 | 1.00 | 0.965 |
| distance (adjacent→remote) | 8 | 0.90 | 0.965 |
| time_of_day (cyclic) | 6 | 1.00 | 0.965 |
| quantity (none→all) | 20 | 0.94 | 0.948 |
| wealth (destitute→wealthy) | 6 | 1.00 | 0.935 |
| hardness (mushy→rigid) | 6 | 0.90 | 0.932 |
| quality (terrible→excellent) | 16 | 1.00 | 0.919 |
| weight (weightless→massive) | 16 | 1.00 | 0.913 |
| formality (casual→ceremonial) | 6 | 0.90 | 0.902 |

The 8 partials: pain, politeness, planets, belt_rank (all isometry r ≥ 0.96 but imperfect order
recovery — clean shape, slightly tangled ordering), spiciness, wetness, and the two small-n cyclic
concepts seasons (4 points, ring-adjacency 0.5) and moon_phases (5 points, 0.8) — too few points
to close a ring, the same effect that left compass/months diffuse in the original atlas.

## The routing test — clean geometry, rarely clean control

For each of the 11 clean **ordinals** at its best layer we ran the same protocol as the emotion
experiment: energy (manifold vs linear chord, full-string read-out) and centroid-routing
(per-waypoint argmax agreement) against shuffled-order and random-direction controls. Gate: routing
requires manifold agreement > linear AND > shuffled.

| concept | iso r | energy gap | manifold | linear | shuffled | verdict |
|---|---|---|---|---|---|---|
| certainty | 0.979 | +0.018 | 0.50 | 0.33 | 0.33 | **centroid routing** |
| age_stage | 0.982 | +0.001 | 0.33 | 0.17 | 0.17 | **centroid routing** |
| wealth | 0.935 | −0.004 | 0.67 | 0.33 | 0.50 | routes, not faithful |
| difficulty | 0.965 | −0.013 | 0.50 | 0.33 | 1.00 | no win (shuffled wins) |
| distance | 0.965 | +0.009 | 0.50 | 0.17 | 0.67 | faithful, not routing |
| hardness | 0.932 | −0.013 | 0.50 | 0.33 | 0.50 | no win |
| formality | 0.902 | −0.041 | 0.50 | 0.33 | 0.33 | routes, not faithful |
| quantity | 0.948 | −0.004 | 0.33 | 0.00 | 0.00 | routes, not faithful |
| quality | 0.919 | −0.018 | 0.33 | 0.17 | 0.33 | no win |
| weight | 0.913 | −0.029 | 0.17 | 0.17 | 0.17 | no win |
| priority | 0.998 | +0.020 | 0.00 | 0.17 | 0.00 | no win |

**Only 2/11 clear the strict gate, and weakly (0.33–0.50) next to arousal's 1.00.** The single
sharpest data point: **priority has the cleanest geometry in the entire census (isometry r 0.998)
and routes at 0.00** — its manifold walk snaps to the endpoint like the linear chord. Clean
geometry is necessary but not sufficient for control.

Caveat that sharpens, not rescues: routing used the generic carrier prompts that fit the geometry,
not prompts tuned to elicit each value as a continuation (the way the emotion `steer_prompt` was).
So these routing numbers are a **floor**; prompt-tuning is one variable a follow-up would sweep.
The pattern — clean geometry ≫ routability — is the finding.

## Recovery diagnostic — does a softer bar rescue the non-routers?

Greedy-argmax is the harshest metric. We re-scored all 11 clean ordinals (plus arousal as a
known-router anchor and fear as a known-dead anchor) across a layer sweep, picking the
best-*control* layer (not best-fit), with a **graded** metric: the correlation between the induced
distribution's center of mass and the waypoint position (`com_corr`). A monotone slide through the
values scores high even when no single token flips. Runner: `scripts/_census_recovery.py`; data:
`reports/manifold_census/recovery.json`.

**The graded metric lights up — but so does the linear chord, and that kills the manifold story.**
The fear anchor (steers nothing) posts a *linear* com_corr of **0.97**: a straight interpolation
between two endpoints slides the distribution across the middle by construction. So a high graded
score is not evidence of routing — only **manifold minus linear** is, and that difference is ≈ 0
for all 11. Sorted by Δ(manifold − linear): only quality (+0.04) and difficulty (+0.03) edge
positive (single seed, 6 waypoints — treat as noise); quantity's +1.10 is an artifact of its linear
chord running *backwards*. Every other concept has linear ≥ manifold (priority −0.41, certainty
−0.21, age_stage −0.15, distance −0.13…). The anchors calibrate it cleanly: **arousal +0.10, fear
−0.77.**

| honest bucket | n | concepts |
|---|---|---|
| manifold-specific (manifold com ≥ 0.7, Δ > 0.02, beats shuffled) | 2 | quality, difficulty — marginal |
| graded-controllable (dial-able, but linear ≥ manifold) | 8 | priority, certainty, distance, age_stage, wealth, hardness, weight, formality |
| weak (no monotone routing at any granularity) | 1 | quantity |

**What this recovers, and what it doesn't:** the softer bar recovers something real — these
behaviors *are* graded-controllable, you can dial certainty or formality smoothly end-to-end — but
a **linear** steer delivers it as well or better for 10/11; the manifold isn't the lever. The
manifold-specific advantage (routing *through* intermediates better than a straight line) stays
singular to arousal. So the recovered usefulness of these geometries is (1) as linear control knobs
where graded steering suffices, and (2) as read-out coordinates (the monitor use-case). The
manifold-specific lane is narrow — but now it has a calibrated test: **manifold − linear com_corr**
separates the anchors (arousal +0.10 / fear −0.77), so it is the target variable a
transverse-stiffness predictor should forecast from geometry. Interactive: the "Can a softer bar
recover them?" section of `docs/writeups/manifold-atlas-3d.html`.

## Why this matters

The census turned a 3-concept result (arousal routes; valence, fear don't) into a 13-concept
labeled set of (geometry, routability) pairs. That is the training data for a **transverse-stiffness
predictor**: a geometric signature (transverse variance, midpoint curvature, sign-flip detector)
computed *before* steering that predicts whether a clean manifold will route — a go/no-go certificate
so a deployment doesn't ship a knob the geometry says will snap. The clean concepts also drop
straight into the Manifold mode for interactive walking.

## Scope and honesty

Single seed, single model; best layer chosen per concept by order recovery; routing uses six
waypoints and the full-string read-out. The 3D atlas plots the first three PCA coordinates of a
higher-dimensional fit (faithful projection, not the whole space). Promotion to `_ATLAS_EXTRA`
marks these as geometry-vetted atlas concepts, **not** validated steering presets — only the routing
test speaks to steerability, and it is mostly negative here.
