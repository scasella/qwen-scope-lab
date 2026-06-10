# Jailbreak probe hardening — the in-the-wild corpus (real Qwen3.5-2B, MLX)

**Run:** `jailbreak_wild_eval` · layer 12 · `mlx-community/Qwen3.5-2B-bf16` on-device (Apple Silicon) · seed 0 · inference-only · no paid APIs · ~31 s wall.

This executes the hardening plan written down but never run in `RESULTS.md:382-418`. The clean-split
jailbreak probe returned AUC 1.00 / `deployable`; the author-written adversarial stress test found the
**direction robust, the threshold not transferable**, and recommended *a real in-the-wild corpus +
per-distribution threshold recalibration*. Here is that corpus and that recalibration.

## The probe under test

The **deployable direction**: difference-of-means over the layer-12 mean-pooled residual, fit on the
author clean set (`JAILBREAK_POS` 8 overt manipulation prompts vs `JAILBREAK_NEG` 8 ordinary requests)
— *exactly* the direction `service.jailbreak_detection` / `jailbreak_screen` (the `/demo`) ship, with
its F1-calibrated author threshold (**0.2904**). Not retrained before the wild eval (per the plan); only
retrained in the held-out-source section (step 4). The probe pools over the first 64 tokens (`MAX_SEQ`),
so long DAN constructions are scored on their opening window — the real deployment contract.

## The corpus (public, deterministic, seed 0)

| split | n | source |
|---|---|---|
| `itw_jailbreak` | 300 | **real** DAN-style jailbreaks scraped from Discord / Reddit / FlowGPT / jailbreak sites (TrustAIRLab/in-the-wild-jailbreak-prompts, 2023-12 snapshot; 12 sources, 11 communities) |
| `benign_ordinary` | 300 | 150 Dolly instruction prompts + 150 ITW-"regular" community prompts |
| `hard_negative` | 130 | 100 JailbreakBench benign behaviors (spicy-but-legit) + 30 author hard-negatives (security Qs, fiction-about-crime, roleplay) |
| `jbb_harmful` | 100 | JailbreakBench harmful behaviors — bare harmful **intent**, no jailbreak framing |

## Results

| split | AUC | TPR@1%FPR | TPR@5%FPR | control AUC | FPR @ author-thr | recall @ author-thr | recall @ recal-thr (5% FPR) |
|---|---|---|---|---|---|---|---|
| **A** wild jb vs ordinary benign | **0.925** | 0.31 | 0.57 | 0.46 ± .15 | **0.417** | 0.98 | 0.57 |
| **B** wild jb vs hard negatives | **0.999** | 0.98 | 0.997 | 0.47 ± .25 | 0.00 | 0.98 | 0.997 |
| **C** harmful intent vs hard negatives | **0.609** | 0.00 | 0.08 | 0.49 ± .08 | 0.00 | **0.00** | 0.08 |
| **D** held-out-source generalisation | **0.906** (in-dist 0.929) | — | 0.50 | 0.45 ± .14 | 0.047 @ id-thr | 0.54 @ id-thr | — |

Every split beats its random-direction control decisively (control AUC 0.45–0.49 across the board).

## What the numbers say

**The direction survives in-the-wild data — discrimination, not the operating point.** On real
DAN/persona jailbreaks scraped from the wild, the layer-12 residual direction ranks jailbreaks above
ordinary benign at **AUC 0.925** (A) and generalises across **entirely unseen jailbreak sources** at
**AUC 0.906**, in-dist 0.929 → held-out-source 0.906, **drop only 0.023** (D). It learned a real,
source-transferable manipulation-intent direction — not the eight author templates, and not the
specific communities. This is the robust positive the stress test predicted, now confirmed on real
adversarial data instead of author-written stand-ins. It is **down from the clean-split 1.00**: the
honest reading is AUC ~0.91–0.93 under genuine in-the-wild load, not a perfect score.

**The threshold does NOT transfer — and we can now say by exactly how much.** At the shipped author
threshold (0.2904), the A-split false-positive rate is **41.7%** — nearly half of "ordinary benign"
prompts fire. Recalibrating to a 5% FPR operating point on the target distribution restores control,
at the cost of recall (0.98 → 0.57). This is the quantified threshold-transfer failure: the *ranking*
is fine, the *shipped cut-point* is not.

**…but the 41.7% decomposes into an honest story (the key surprise).** Breaking A's negatives by
source: **Dolly ordinary instructions false-positive at 0%**; **ITW-"regular" community prompts
false-positive at 83%**. The "regular" prompts in jailbreak communities are themselves long,
persona-laden, instruction-heavy constructions (median ~835 chars) that *look* structurally like
jailbreaks. So the threshold-transfer failure is driven by **distribution shift in the negatives** —
the probe fires on elaborate community-style prompts, not on genuinely ordinary traffic. On clean
ordinary requests the shipped threshold is fine; the recalibration need is specifically for deployments
where benign traffic is itself elaborate.

**The probe detects jailbreak *framing*, not harmful *intent* (C).** Pointed at bare harmful requests
with the jailbreak wrapper stripped off (JBB harmful behaviors), the probe scores **AUC 0.609** and
fires on **0%** of them at the author threshold. This is a clean scope boundary, not a failure to hide:
the layer-12 residual direction is a **manipulation-framing detector** (DAN / persona / instruction-
override / elaborate role construction), *not* a harmful-content classifier. B (AUC 0.999) and C
(AUC 0.609) are the two faces of the same fact — the probe trivially separates wild jailbreaks from
short clean benign because it keys on the elaborate-manipulation structure, and for the same reason it
misses harmful intent expressed as a plain short request.

## Verdict

- **Does the probe survive in-the-wild data?** **Yes, as a discriminator** — AUC 0.925 on real wild
  jailbreaks vs ordinary benign, 0.906 across held-out sources (drop 0.023), all far above a 0.45–0.49
  random-direction control. The manipulation-intent direction is real and source-transferable.
- **Does the operating point survive?** **No** — 41.7% FPR at the shipped author threshold, driven by
  elaborate community-style benign prompts (Dolly 0%, ITW-regular 83%). Per-distribution recalibration
  is mandatory; at a 5%-FPR recalibrated cut-point recall is ~0.57 on the hardest negative mix.
- **New scope finding:** the probe is a **jailbreak-framing** detector (AUC 0.999 vs short clean benign),
  **not a harmful-intent** detector (AUC 0.609 on bare harmful requests). Claimable: "flags
  manipulation-shaped prompts." Not claimable: "flags harmful requests."

**Now claimable:** the layer-12 residual jailbreak probe generalises to real in-the-wild DAN/persona
jailbreaks and to held-out jailbreak sources (AUC ~0.91), beating a random-direction control on every
split. **Still not claimable:** a fixed shipped threshold (recalibrate per distribution), or detection
of harmful intent without jailbreak framing.

## Honest bounds

Single model (Qwen3.5-2B), single layer (12), single probe seed (random-control averaged over 20 seeds).
ITW corpus is the 2023-12 TrustAIRLab snapshot, deterministically sampled to 300; results are the
distribution's, not the full 1,364. Prompts clipped to 1,200 chars on disk and pooled over the first 64
tokens — long jailbreaks judged on their opening window (the deployment contract, but a longer pooling
window could move A/C). C uses JBB harmful *goals* (intent text), not full attack transcripts. No
adaptive red-team that targets *this* probe specifically. The robust **pattern** (direction holds, AUC
~0.91; threshold doesn't, FPR-shift is negative-distribution-driven; framing != intent) is the finding;
the exact percentages are this corpus's.

## Files

- `reports/jailbreak_wild/verdict.json` — machine-readable, every number incl. the per-source FPR breakdown.
- `scripts/jailbreak_wild_build_corpus.py` — deterministic corpus builder.
- `scripts/jailbreak_wild_eval.py` — the MLX evaluation.
- `data/experiments/jailbreak_wild/*.jsonl` — the eval splits.
- `docs/experiments/JAILBREAK_IN_THE_WILD.md` — short summary.
