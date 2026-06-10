# Jailbreak probe in the wild — what's now claimable

Hardening the layer-12 difference-of-means residual jailbreak probe (the lab's flagship deployable)
against a **real in-the-wild corpus**, executing the plan in `RESULTS.md:382-418`. Real Qwen3.5-2B on
MLX, layer 12, seed 0, inference-only, no paid APIs. Full forensic write-up: `reports/jailbreak_wild/`.

## Corpus
- **Real in-the-wild jailbreaks** (300): TrustAIRLab/in-the-wild-jailbreak-prompts — DAN/persona prompts
  scraped from Discord / Reddit / FlowGPT / jailbreak sites (12 sources, 11 communities).
- **Ordinary benign** (300): Dolly instructions + ITW-"regular" community prompts.
- **Hard negatives** (130): JailbreakBench benign behaviors + 30 author hard-negatives (security Qs,
  fiction-about-crime, roleplay).
- **Bare harmful intent** (100): JailbreakBench harmful behaviors, no jailbreak framing.

## Result, by split (vs a random-direction control of AUC ~0.45–0.49 everywhere)
| split | AUC | threshold transfer |
|---|---|---|
| wild jb vs ordinary benign | **0.925** | FAILS — 41.7% FPR at the shipped author threshold |
| wild jb vs hard negatives | **0.999** | holds (0% FPR) |
| bare harmful intent vs hard neg | **0.609** | probe doesn't fire (0% recall) |
| held-out **source** generalisation | **0.906** (in-dist 0.929, drop 0.023) | reasonable (4.7% FPR at id-thr) |

## What's NOW claimable
- The probe **survives in-the-wild jailbreaks as a discriminator**: AUC ~0.91–0.93 on real DAN/persona
  prompts, and it **generalises across held-out jailbreak sources** (drop only 0.023) — a genuine,
  source-transferable manipulation-intent direction, not template/community memorisation.
- It beats a random-direction control on **every** split.

## What's NOT claimable (the honest negatives)
- **A fixed shipped threshold.** At the author threshold, 41.7% of "ordinary benign" prompts
  false-positive. Decomposed: Dolly ordinary instructions **0%** FPR, ITW-"regular" elaborate community
  prompts **83%** FPR. The cut-point must be **recalibrated per deployment distribution** — and the need
  is acute when benign traffic is itself elaborate/persona-heavy. Recalibrating to 5% FPR costs recall
  (0.98 → 0.57 on the hardest mix).
- **Harmful-intent detection.** The probe is a **jailbreak-FRAMING** detector, not a harmful-content
  classifier: AUC 0.609 / 0% recall on bare harmful requests with the jailbreak wrapper removed. It
  flags manipulation-shaped prompts, not harmful ones.

## Bounds
Single model/layer/seed; 2023-12 ITW snapshot sampled to 300; probe pools the first 64 tokens (long
jailbreaks seen on their opening window); no adaptive red-team targeting this probe. The robust pattern
(direction holds ~0.91, threshold doesn't, framing != intent) is the finding; exact percentages are this
corpus's.
