# Emotion-space manifold steering — what's claimable

**One-line result:** on real Qwen3.5-2B, ordered emotion concepts form **clean residual manifolds
(isometry r = 0.997–0.999, 3/3)** — as clean as rank/days — but the manifold path's *advantage* over
the linear chord (faithfulness + routing-through-intermediates) holds on **only the monotone arousal
axis** (numb→frantic) and **fails on the two affect lines** that must cross a valence sign flip.

Evidence: `reports/emotion_manifold/verdict.json` (+ `report.md`).
Runner: `scripts/_emotion_manifold.py`. Run 2026-06-09 on `mlx-community/Qwen3.5-2B-bf16`, MLX,
on-device, no SAE. Full-string behavior read-out (single-token values ⇒ first-token bit-identical).

## Gates (preregistered)

| Gate | Threshold | Verdict |
|---|---|---|
| (a) clean manifold | isometry r ≥ 0.9 | **PASS 3/3** (r 0.997–0.999) |
| (b) manifold faithful | full-string energy gap > 0 on a majority | **FAIL 1/3** (only `emotion_arousal`, +0.0185) |
| (c) centroid-routing | manifold argmax-agreement > linear AND > shuffled | **FAIL 1/3** (only `emotion_arousal`) |

## The claimable win: `emotion_arousal` (numb → frantic)

Walking the manifold induced **all six intermediate emotions in order as the next-token argmax**
(numb→bored→calm→alert→excited→frantic; p 0.70–0.99). The linear chord stayed pinned on "numb" then
jumped to "frantic" (agreement 0.33); the random-direction control (matched norm) sat on "numb"
throughout (0.17); the shuffled-order control scored 0.83 but below the manifold's 1.00. Energy
agrees: manifold 0.023 < linear 0.041. This is the centroid-routing hypothesis confirmed on all three
comparisons — the lab's Manifold mode now has its own emotion result.

## What is NOT claimable

- **Valence lines don't route.** `emotion_valence_intensity` (furious→euphoric) induces the first
  three steps then collapses back to "furious"/"angry" — it cannot cross the negative→positive affect
  sign flip with a single 1-D spline. Energy favors linear (−0.033).
- **Fear is steer-resistant at these layers.** `emotion_fear` argmaxes "terrified" at every waypoint
  on *every* path — the residual replacement never pulls the distribution off the source, so routing
  is undefined (all paths = 0.167). Energy favors linear (−0.008).
- No safety-coupling claim. This is geometry only; the refuted affection→compliance coupling
  (RESULTS.md) is not revisited.

## Caveats

Single seed, n=1/cell, 3 concepts, layers {8,12,16}, 6 waypoints. Isometry r is **saturated** (~0.99
at every layer), so it is a poor layer-selector; abs_spearman is the better manifold-quality
discriminator. Pullback was descoped (it runs on MLX, was not blocked) — the obvious follow-up is a
pullback leg on `emotion_arousal`. The honest read: emotion **manifolds are real**; manifold
*steering* beats the chord on a clean monotone affect axis and not on valence reversals.

The emotion concepts (`emotion_valence_intensity`, `emotion_fear`, `emotion_arousal`) are defined in
`scripts/_emotion_manifold.py`; they are **not** yet added to `concept_presets.py` (see note in the
return — the registry add is a clean drop-in but was left out of the public tree pending owner sign-off).
