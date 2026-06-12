# Agent-boundary injection detection — can the probe guard tool outputs?

Extends the jailbreak-probe line (`docs/experiments/JAILBREAK_IN_THE_WILD.md`) from user prompts to
the **agent boundary**: malicious instructions embedded in *tool outputs / retrieved documents*
(indirect prompt injection). Also adds the baseline battery's missing arm: **lexical baselines**
(zero-shot regex + trained char-3-gram logistic). Real Qwen3.5-2B on MLX, layer 12, seed 0,
inference-only, no paid APIs.

**Pre-registered** (predictions P1–P6 + decision rule written before the run):
`reports/agent_boundary/preregistration.md`. Verdict + full numbers: `reports/agent_boundary/`.

## Design (matched-pair, deterministic seed 0)

Real Dolly passages as carriers (two surface forms: raw document / web_search tool-result JSON);
each **injected** doc = carrier + ONE published payload (BIPIA, using its own train/test category
split; InjecAgent direct-harm / data-stealing; 12 author override lines) at a tagged position
(start/middle/end); each **clean** doc = the same carrier untouched. **Hard negatives** = carriers +
benign imperatives addressed to the *human reader* — isolating instruction-presence from
model-directed instruction. Fit/train/test regions are carrier-disjoint AND family-disjoint.
Build: `scripts/agent_boundary_build_corpus.py` → eval: `scripts/agent_boundary_eval.py`.

## Verdict: the pre-registered decision rule did NOT pass

| finding | number |
|---|---|
| zero-shot shipped jailbreak probe (windowed-max) | **0.61** overall — override flavor 0.91, task-hijack 0.51 (chance) |
| shipped author threshold on this distribution | **53% FPR** — third confirmation of "direction robust, threshold not" |
| retrained diff-means (the user-prompt champion) | **0.79** in-dist — first material diff-means failure in the program |
| **retrained logistic, windowed-max** | **0.954 in-dist / 0.943 held-out families** (drop 0.011), beats trained lexical (0.86/0.84) |
| single-window scoring on mid/end-position payloads | ≈ chance (0.53/0.49) — windowed-max recovers to 0.72/0.83 vs start 0.95 |
| vs benign-imperative hard negatives (P6 **refuted**) | probes degrade **0.14–0.15** (logistic 0.943→0.790); lexical ≤0.04 (0.841→0.802) |

What survives: a logistic residual probe **fit on boundary data** with **windowed scanning**
detects injected tool outputs at 0.94 on attack families it never saw, ~0.10 AUC ahead of a trained
lexical baseline. What does not: zero-shot reuse of the jailbreak direction (it reads manipulation
*framing*; most boundary payloads have none), diff-of-means fitting, single-window scoring, any
fixed threshold — and, most importantly, **the addressee distinction**: against instructions
addressed to the human reader rather than the model, the probe's advantage over a char-3-gram
baseline disappears. A "firewall" claim needs that distinction; this experiment localizes it as
the open problem and the natural next probe (train on the addressee contrast directly —
the corpus generator already produces matched pairs for it).

## Honest bounds

Constructed corpus (real carriers, real published payloads, synthetic insertion) — not in-the-wild
agent traffic; no adaptive evasion; uniform insertion formatting; single model/layer/seed; 50/50
base rates; override flavor is n=12; hard negatives are author-written; one lexical featurization.
