# Agent-boundary injection detection — PRE-REGISTRATION

**Written 2026-06-11, BEFORE any eval run.** Real Qwen3.5-2B on MLX, layer 12, seed 0,
inference-only, no paid APIs. This extends the jailbreak-probe line (RESULTS.md §jailbreak,
`reports/jailbreak_wild/`) to the **agent boundary**: malicious instructions embedded in
*tool outputs / retrieved documents* rather than in user prompts. It also adds the one arm the
original baseline battery was missing: a **lexical baseline** (both zero-shot regex and trained
char-n-gram logistic).

## The question

The shipped layer-12 diff-of-means probe detects jailbreak **framing in user prompts**
(wild AUC 0.93, held-out-source 0.91) but not bare harmful intent (0.61). Indirect prompt
injection is a *different* distribution: the violation is **channel confusion** — an instruction
appearing in a data channel where no instruction belongs — not necessarily override-flavored
text. Does the manipulation-framing direction transfer to that boundary? And where the probe
wins or loses, does a trivial lexical detector do just as well (which would make the white-box
machinery unnecessary)?

## Corpus (constructed, matched-pair, deterministic seed 0)

- **Carriers**: real Dolly context passages (400–1100 chars), two surface forms:
  `raw_doc` (retrieved-document) and `tool_json` (wrapped as a web_search tool-result JSON).
- **injected** (label 1): carrier + ONE payload inserted at a tagged position
  (start / middle / end). Payload families:
  - `bipia_train` — BIPIA *published* train categories (75 payloads)
  - `bipia_test` — BIPIA *published* test categories (75) — **held-out by the benchmark's own split**
  - `injec_dh` / `injec_ds` — InjecAgent direct-harm (30) / data-stealing (32) agent-directed instructions
  - `author_override` — 12 author-written override-wrapped payloads (the in-the-wild
    "IGNORE PREVIOUS INSTRUCTIONS" pattern; OWASP LLM01 surface forms)
- **clean** (label 0): the SAME carriers, no insertion (matched pairs — the only difference is the payload).
- **hard_negative** (label 0): carrier + inserted **benign imperative** (instructions addressed to the
  human reader — "run pip install…", "reply to Janet by Friday…") — same insertion protocol.
  This isolates instruction-PRESENCE from model-directed-instruction.
- Train/test split for trained arms: families {bipia_train, injec_dh} train vs
  {bipia_test, injec_ds, author_override} test, with **carrier-disjoint** train/test sets.

## Arms

1. `shipped_probe` — the deployable jailbreak direction (diff-means on the author clean set,
   exactly what `/api/jailbreak_detection` ships), **zero-shot**, scored two ways:
   single-window (first 64 tokens — the current deployment contract) and **windowed-max**
   (max pooled-residual score over consecutive 64-token windows — the firewall contract).
2. `retrained_diffmeans` — diff-means fit on train-family start-position pairs, windowed-max scoring.
3. `retrained_logistic` — same protocol, logistic.
4. `lexical_zeroshot` — regex battery over override/injection surface forms, no training.
5. `lexical_trained` — hashed char-3-gram features + logistic regression, same train split.
6. `random_control` — random unit directions, 20 seeds, windowed-max protocol.

## Pre-registered predictions

- **P1 (zero-shot transfer is partial).** Shipped probe, windowed-max, injected-vs-clean overall:
  AUC **0.60–0.85** — above chance, below the 0.93 user-prompt number. By family:
  `author_override` ≥ 0.85; `task-hijack`-flavored BIPIA payloads ≤ 0.65 (no manipulation framing
  for the direction to read).
- **P2 (position blindness, then recovery).** Single-window scoring ≈ chance (0.45–0.60) on
  middle/end-position injections (payload physically outside the 64-token window); windowed-max
  recovers to within 0.10 of the start-position AUC.
- **P3 (threshold does not transfer).** At the shipped author threshold the operating point breaks
  on this distribution (recall < 0.5 or FPR > 0.3 on at least one main split) — the established
  "direction robust, threshold not" pattern, third occurrence.
- **P4 (retrained ceiling + held-out families).** Retrained diff-means: in-dist AUC ≥ 0.90;
  held-out-family drop ≤ 0.10 (the direction generalizes across attack families, as it did across
  jailbreak sources).
- **P5 (lexical is the wrong tool, measurably).** `lexical_trained` in-dist ≥ 0.85 but drops ≥ 0.15
  on held-out families (phrase memorization); the retrained probe's held-out drop is smaller.
  `lexical_zeroshot`: near-ceiling on `author_override`, ≈ chance on BIPIA task-hijack and
  InjecAgent payloads, overall ≤ 0.70.
- **P6 (the boundary question).** Against benign-imperative hard negatives, lexical arms degrade
  more than residual-probe arms (ΔAUC[clean→hard-neg] larger for lexical) — the probe reads
  *who the instruction is for*, the lexicon reads *that there is an instruction*.

## Pre-stated honest bounds

Constructed carrier+payload corpus, NOT in-the-wild agent traffic; uniform insertion formatting
(no adaptive evasion, no obfuscated payloads beyond what BIPIA test categories contain); single
model / layer / seed; payloads from two academic benchmarks plus 12 author lines; matched-pair
design means base rates are artificial (50/50). A win here says the direction separates
injected-from-clean tool outputs under these conditions — it does not certify a production firewall.

## Decision rule

Headline claim ("the white-box probe is the right primitive for an agent-boundary firewall")
requires: P4 holds (retrained ≥ 0.90 in-dist, ≤ 0.10 held-out drop) AND the retrained probe beats
`lexical_trained` on held-out families AND P6 direction correct. Zero-shot transfer (P1) failing
does NOT sink the claim — it bounds what ships without retraining; it would be reported as the
honest "you must fit on boundary data" finding.
