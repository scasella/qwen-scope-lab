# Emotion-space manifold steering (real Qwen3.5-2B, MLX, 2026-06-09)

The owner's parked May-30 idea, executed: Goodfire-style concept-manifold steering pointed at
**ordered emotion spaces** (à la Anthropic's emotion-vectors work), testing the novel
**centroid-routing** hypothesis — does the manifold path route *through* intermediate emotions
where the linear chord skips them? Real Qwen3.5-2B on-device (MLX), no SAE, full-string behavior
read-out (single-token values, so first-token is bit-identical). 192 s wall-clock; 3 concepts;
layer sweep {8, 12, 16}; 6 waypoints; n=1/cell; single seed. Driver:
`scripts/_emotion_manifold.py`; every number: `verdict.json`.

## Preregistered gates → verdicts

| Gate | Threshold | Verdict |
|---|---|---|
| (a) clean manifold | isometry r ≥ 0.9 | **PASS 3/3** — r = 0.997–0.999, as clean as rank/days |
| (b) manifold more faithful | full-string energy gap > 0 on a majority | **FAIL 1/3** — only `emotion_arousal` (+0.0185); valence/fear favor linear (−0.033, −0.008) |
| (c) centroid-routing | manifold argmax-agreement > linear AND > shuffled control | **FAIL 1/3** — only `emotion_arousal` |

## The clean win: `emotion_arousal` (numb → frantic)

Walking the manifold induced **all six intermediate emotions in order** as the next-token argmax —
numb→bored→calm→alert→excited→frantic (p 0.70–0.99; agreement 1.00). The linear chord stayed
pinned on "numb" for four waypoints then jumped to "frantic" (agreement 0.33 — textbook
endpoint-snapping). Random-direction control at matched norm sat on "numb" throughout
(0.17 = chance); the shuffled-order control scored 0.83 — high, but **below** the manifold's 1.00.
Energy agrees (manifold 0.023 < linear 0.041). Centroid-routing confirmed on all three
comparisons. The Manifold mode has its first emotion result.

## The honest failures, with mechanisms

- **`emotion_valence_intensity`** (furious→euphoric): induces the first three steps then collapses
  back to "furious"/"angry" past the calm midpoint — a single 1-D spline **cannot cross the
  negative→positive valence sign flip**.
- **`emotion_fear`**: argmaxes "terrified" at *every* waypoint on *every* path (all agreements
  0.167) — the residual replacement never pulls the distribution off the source at these layers.
  Steer-resistant at 2B, so routing is undefined rather than refuted.

## Surprises / bounds

- **Isometry r is saturated** (~0.99 at every swept layer), so it is a poor layer-selector here; it
  picked L16 for valence-intensity (worst abs_spearman 0.71). abs_spearman is the better
  tiebreaker. Source-anchoring is layer-robust, so this does not rescue gates (b)/(c).
- The clean win is the **monotone arousal axis** — a single activation-level dimension with no
  affect reversal — exactly the geometry where a 1-D manifold should work. This is an honest,
  concept-dependent positive, consistent with `docs/MANIFOLD.md` §5's "faithfulness real but
  partial."
- Pullback was **descoped** (budget, not blockage — `pullback_optimize` runs on-device). The
  obvious follow-up is a pullback leg on `emotion_arousal`.
- This is geometry only — it does not re-litigate the refuted affection→compliance coupling.

n=1 per cell, single seed, three concepts, one model. The routing *pattern* (manifold 1.00 /
linear 0.33 / chance 0.17 with ordered intermediate induction at p 0.70–0.99) is far from any
plausible noise floor, but the exact agreement rates are small-n.
