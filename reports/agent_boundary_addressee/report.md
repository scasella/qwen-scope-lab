# The addressee probe — pre-registered verdict: NEGATIVE

**Run:** `addressee_eval` · layer 12 · `mlx-community/Qwen3.5-2B-bf16` on-device · seed 0 ·
inference-only · 700 docs / 2,336 windows · 141 s wall.
**Pre-registration:** `preregistration.md` (before the run). **Raw:** `verdict.json`, `scores.jsonl`.

## Headline

**The addressee distinction — "is this instruction for the model or for a human reader?" — is
NOT cheaply linearly readable at layer 12 the way instruction-presence was.** Trained directly on
24 content-matched pairs (the cleanest possible contrast: same instruction, different addressee,
same carrier), the logistic probe reaches only **0.78** on held-out pairs, **0.64** when
transferred to real injection payloads, and **stacking it onto the presence probe makes the
firewall worse, not better** (combined 0.758 vs presence-alone 0.790 against benign-imperative
hard negatives). Decision rule: **NOT MET** — second consecutive honest negative in this line,
and the more informative one.

## Prediction scorecard

| # | Prediction | Outcome |
|---|---|---|
| P1 | addressee_lg ≥ 0.85 on held-out matched pairs | **FAIL** — 0.779 overall. At the fit condition (start position) it does hit **0.914**; middle 0.654, end 0.718 — the direction exists where conditions match training and degrades away from them |
| P2 | transfers to real payloads ≥ 0.85, beating presence's 0.790 | **FAIL, decisively** — 0.644 vs hard negatives; the authored-pair direction does not carry to BIPIA/InjecAgent payload style |
| P3 | lexical ≥ 0.80 overall, drops ≥ 0.15 on implicit; probe drops ≤ 0.10 | **PARTIAL** — probe's implicit drop 0.051 ✓ and probe > lexical on implicit (0.754 vs 0.644) ✓; but lexical never reached 0.80 (0.689) and its implicit drop is 0.098, not ≥ 0.15 |
| P4 | matched pairs restore diff-means to within 0.05 of logistic | **FAIL** — gap 0.106 (0.673 vs 0.779); the diff-means deficit at the boundary is not (only) class heterogeneity |
| P5 | combined_min ≥ 0.85 vs hard-neg and ≥ 0.90 vs clean | **FAIL** — 0.758 vs hard-neg (**worse** than presence alone, 0.790) and 0.900 vs clean (below presence's 0.943). At this quality the addressee stage subtracts signal |

## Numbers

### A. Held-out matched pairs (windowed-max; n = 72 vs 72)
| arm | overall | explicit | implicit |
|---|---|---|---|
| addressee_lg | 0.779 | 0.805 | 0.754 |
| addressee_dm | 0.673 | 0.715 | 0.643 |
| lexical_matched | 0.689 | 0.742 | 0.644 |
| presence_lg (incumbent) | 0.627 | 0.582 | 0.670 |
| random (20 dirs) | 0.497 ± 0.019 | | |

addressee_lg by position: start **0.914** / middle 0.654 / end 0.718 (fit was start-only).

### B. Transfer to real payloads (parent corpus)
| arm | vs hard-neg | vs clean |
|---|---|---|
| presence_lg | **0.790** | **0.943** |
| combined_min | 0.758 | 0.900 |
| addressee_lg | 0.644 | 0.647 |
| addressee_dm | 0.615 | 0.612 |
| lexical_matched | 0.585 | 0.613 |

### C. Axis anatomy (the one clean positive)
Reader-directed instructions vs clean docs on the addressee axis: **0.496** — the direction is
genuinely orthogonal to instruction-presence (it does not fire on "there is an instruction here",
only — weakly — on "it is for the model"). Model-directed vs clean: 0.647.

## Honest read

1. **A weak addressee direction exists** (0.91 at the fit condition, random control 0.50,
   clean anatomy) — but it is **brittle to position** and **does not transfer** across payload
   styles from 24 authored pairs. Compare presence: 105 pairs → 0.94 on held-out *families*.
   Addressee is a genuinely harder property for the layer-12 residual at 2B, not a data-format
   accident — exactly what the parent experiment's P6 refutation implied, now measured directly.
2. **The two-stage firewall does not assemble from these parts.** min-stacking a 0.64-transfer
   probe onto a 0.79 incumbent loses ground. No version of the firewall claim survives 2026-06-11.
3. **What would move it** (not run, listed for pre-registration next time): more matched pairs
   (the 24-pair fit is the binding constraint and the 0.91 fit-condition number says signal is
   there), multi-position fit, a layer sweep (addressee may live later than 12, where role/speaker
   information consolidates), and a bigger model. Until one of those lands ≥ 0.85 transfer,
   **the deployable detection story remains presence-only, threshold-per-distribution** — and the
   addressee gap is the honest, sharply-bounded reason "agent-boundary firewall" stays unclaimed.

## Bounds (pre-stated)

24 fit / 24 eval pairs (small-n: each held-out cell is ±~0.05–0.08); all payloads author-written
(style monoculture — the transfer eval exists precisely to catch this, and did); single
model/layer/seed; start-only fit; one combination rule (pre-registered, no combiner search).
