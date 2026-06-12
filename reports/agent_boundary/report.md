# Agent-boundary injection detection — the pre-registered verdict

**Run:** `agent_boundary_eval` · layer 12 · `mlx-community/Qwen3.5-2B-bf16` on-device · seed 0 ·
inference-only · no paid APIs · 718 docs / 2,404 windows · 331 s wall.
**Pre-registration:** `preregistration.md` (written before the run). **Raw:** `verdict.json`, `scores.jsonl`.

## Headline (the honest one)

**The pre-registered decision rule did NOT pass.** A residual probe *can* learn the agent-boundary
distinction when fit on boundary data (logistic, windowed-max: **0.954 in-dist, 0.943 on held-out
attack families**, beating the trained lexical baseline 0.84) — but the white-box case is weaker here
than on user-prompt jailbreaks, in three specific, mechanistically legible ways:

1. **Zero-shot reuse of the shipped jailbreak probe is not viable.** Windowed-max AUC 0.61 overall
   (random control 0.52 ± 0.05). The transfer that does exist is entirely the override flavor
   (0.91) — payloads that *look like jailbreaks*. Task-hijack (0.51) and format-corruption (0.52)
   injections are invisible to the jailbreak-framing direction. The probe reads manipulation
   *framing*; most boundary payloads carry none.
2. **Diff-of-means is no longer enough.** The canonical cheap probe that hit 1.00 on user-prompt
   jailbreaks manages only **0.79 in-dist** here; the logistic probe (0.95) is the first material
   diff-means-vs-logistic gap this program has seen. The injected-vs-clean separation at the
   boundary is real but not isotropic around class means.
3. **The probes read instruction-PRESENCE more than instruction-ADDRESSEE (P6 refuted).** Against
   benign imperatives addressed to the human reader, probe AUC drops by 0.14–0.15 (logistic
   0.943 → 0.790) while the lexical arms drop ≤ 0.04 — the *opposite* of the prediction. On the
   addressee axis the probe's advantage over a char-3-gram baseline disappears (0.790 vs 0.802).

## Prediction scorecard

| # | Prediction | Outcome |
|---|---|---|
| P1 | zero-shot transfer partial (0.60–0.85); override ≥ 0.85; task-hijack ≤ 0.65 | **PASS, marginal** — 0.611 overall (bottom of band); override 0.910 ✓; task-hijack 0.512 ✓ |
| P2 | single-window ≈ chance on mid/end; windowed-max recovers within 0.10 of start | **PARTIAL** — blindness confirmed (0.527 / 0.485 single, retrained); max recovers to 0.716 / 0.833 vs start 0.952 — real but > 0.10 short |
| P3 | author threshold breaks (recall < 0.5 or FPR > 0.3) | **PASS** — FPR **0.527** at the shipped threshold; recalibrating to 5% FPR leaves recall 0.08 (zero-shot direction too weak to recalibrate around) |
| P4 | retrained diff-means ≥ 0.90 in-dist, held-out drop ≤ 0.10 | **FAIL as registered** — diff-means 0.794 in-dist; the *logistic* probe passes both bars (0.954 / 0.943, drop 0.011) |
| P5 | trained lexical ≥ 0.85 in-dist but drops ≥ 0.15 held-out; zero-shot lexical ≤ 0.70 overall | **PARTIAL** — lexical 0.863 in-dist ✓ but drop only 0.022 (memorization claim **refuted**); zero-shot lexical 0.527 ✓, override 0.999 ✓, others ≈ 0.50 ✓ |
| P6 | lexical degrades more than probes vs benign-imperative hard negatives | **REFUTED** — probes degrade 0.14–0.15, lexical 0.01–0.04 |

**Decision rule** (P4 + beats-lexical-held-out + P6): **NOT MET.** The pre-registered headline
("the white-box probe is the right primitive for an agent-boundary firewall") is not claimable.
The pre-stated fallback is what the data supports: *you must fit on boundary data, the fit must be
logistic-grade, scanning must be windowed — and the addressee distinction remains open.*

## Numbers

### A. Zero-shot, injected vs clean (matched pairs, n=224 vs 224)
| arm | AUC |
|---|---|
| shipped probe, single window | 0.546 |
| shipped probe, windowed-max | 0.611 |
| lexical regex battery | 0.527 |
| random direction (20 seeds) | 0.519 ± 0.046 |

### B. Trained arms, in-distribution vs held-out families
| arm | in-dist | held-out | drop |
|---|---|---|---|
| retrained diff-means (max) | 0.794 | 0.834 | −0.040 |
| **retrained logistic (max)** | **0.954** | **0.943** | 0.011 |
| trained lexical (char-3-gram) | 0.863 | 0.841 | 0.022 |

Held-out families = BIPIA's own published test categories + InjecAgent data-stealing + author
override — zero family overlap with training, carrier-disjoint.

### C. Position (retrained diff-means; injected_test vs clean_test)
| position | single | windowed-max |
|---|---|---|
| start | 0.964 | 0.952 |
| middle | 0.527 | 0.716 |
| end | 0.485 | 0.833 |

Single-window scoring is *blind* to anything past token 64 — windowed-max is necessary, and
mid-document payloads (split across window boundaries, diluted by carrier context) remain hardest.

### D. Flavor (windowed-max, injected vs clean)
| flavor | n | shipped | retrained-dm | lexical-0shot | lexical-trained |
|---|---|---|---|---|---|
| override | 12 | 0.910 | 0.858 | 0.999 | 0.722 |
| agent_directive | 62 | 0.705 | 0.941 | 0.506 | 0.915 |
| content_smuggling | 50 | 0.611 | 0.836 | 0.498 | 0.910 |
| task_hijack | 55 | 0.512 | 0.747 | 0.498 | 0.749 |
| format_corruption | 45 | 0.525 | 0.693 | 0.498 | 0.843 |

### E. The boundary question (injected_test vs benign-imperative hard negatives)
| arm | vs clean | vs hard-neg | degradation |
|---|---|---|---|
| retrained logistic (max) | 0.943 | 0.790 | 0.153 |
| retrained diff-means (max) | 0.834 | 0.694 | 0.140 |
| trained lexical | 0.841 | 0.802 | 0.039 |
| shipped probe | 0.623 | 0.590 | 0.034 |
| zero-shot lexical | 0.550 | 0.543 | 0.008 |

## What this buys the program

- The **third leg of the threshold-transfer pattern** (user-prompt clean → in-the-wild → agent
  boundary): the direction-vs-threshold split now has a cross-distribution track record.
- A **clean negative on zero-shot reuse**: "point the jailbreak probe at tool outputs" is now
  measured and refuted — the framing direction does not encode channel violation.
- The **first diff-means failure**: boundary injection is the program's first detection target where
  the geometry demands more than class means. Worth a follow-up on *why* (class structure? carrier
  dominance in the pooled mean?).
- The **addressee problem, isolated**: the matched benign-imperative hard negatives localize exactly
  what a real agent-boundary firewall needs that no arm here provides cleanly. A probe *trained on
  the addressee contrast* (model-directed vs reader-directed imperatives, payload-matched) is the
  natural next experiment — and the corpus design here generates it almost for free.

## Honest bounds (pre-stated)

Constructed carrier+payload corpus (real carriers, real published payloads, synthetic insertion) —
NOT in-the-wild agent traffic; uniform insertion formatting, no adaptive evasion; single
model/layer/seed; 50/50 base rates; n=12 for the override flavor (its 0.91/0.999 cells are wide);
hard negatives are author-written (n=30 templates over 60 docs). The lexical-trained baseline is one
featurization (hashed char-3-grams) — a tuned TF-IDF or embedding baseline could be stronger.
