# The Lab Bench — user guide

A friendly, example-driven walkthrough of the Qwen Scope steering workbench. Every feature below
has a **Try it** block with exact inputs you can follow step by step. For the science and
internals, see `MANIFOLD.md`.

---

## What this is

An interactive bench for **interpreting, steering, and monitoring** a Qwen model through its
sparse-autoencoder (SAE) features and its concept geometry. It has two halves:

- **Control** — find which features fire, dial them up or down, and steer whole concepts (days,
  sizes, numbers…) along their geometry.
- **Monitoring** — find interpretable feature-based *detectors* for a behavior (refusal, PII,
  sycophancy…) to use as cheap, auditable runtime guardrails.

What sets it apart is **honest evaluation**: every steer is scored against seven controls and
every detector against a random-feature control, so the bench tells you when something *didn't*
work instead of cherry-picking a nice demo. Validated results become reusable **recipes**
(control) and **monitors** (detection). Everything here is also drivable programmatically by an
AI agent — see `AGENT_RESEARCH.md`.

> **New to SAEs?** A *sparse-autoencoder (SAE) feature* is an interpretable direction the
> autoencoder learned from the model's internal activations — when it activates strongly on a
> token, the concept it stands for is "present." Steering nudges those directions to change
> behavior; monitoring reads them to detect a behavior. (See the **Glossary** at the end.)

## Getting started

**First time? Install it** (Python ≥ 3.10): `python -m venv .venv && source .venv/bin/activate &&
pip install -e ".[dev]"`. Full setup + credentials are in `README.md`. Then pick a backend:

- **Local (dev, CPU, no GPU):** `python serve_web.py --dev`, then open the printed URL (default
  **http://127.0.0.1:7870**). This runs the *real code* over a tiny in-memory model, so the whole
  interface works with no GPU, no downloads, and no token. Every **Try it** below works here.
- **Real model on your Mac (Apple Silicon, no cloud):** `python serve_web.py --mlx
  mlx-community/Qwen3.5-2B-bf16 --mlx-sae Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100 --mlx-d-sae
  32768`, then open the printed URL. This runs the **real Qwen3.5-2B + SAE on-device** via MLX —
  every mode in this guide works, private and offline, no Modal and no GPU bill. The first launch
  downloads the model (~4.5 GB) + SAE (~540 MB), then it's cached. (Drop the `--mlx-sae` flags for
  the probe/detection paths only.) Details: `docs/MLX.md`. **This is the recommended way to follow
  this guide on real text.**
- **Real model (Modal, GPU):** `modal serve modal_app.py`, then open the **`web_gui`** URL it
  prints — the real Qwen3.5-2B (default; set `QWEN_GUI_TARGET=27b-a100` for 27B — the 27B is
  Modal-only). The first action triggers a cold model load (a minute or two). **Stop the GPU when
  done:** `modal app stop qwen-scope-lab-bench`. (Needs Modal auth + `HF_TOKEN` — see `RUNBOOK.md`.)

> **Read this once — the dev caveat.** On the dev server the model is a *tiny random* network. All
> the mechanics work (features fire, hooks fire, panes/gauges/verdicts populate), but generated
> text is gibberish and steering rarely "wins" a benchmark — that's expected and honest. Use dev
> to learn the interface; use the real model for meaningful text and real verdicts. Wherever a
> result depends on the real model, it's flagged **[real model]**.

## Practical use cases — who'd use this, and why

- **Choose a control method for a behavior.** Deciding whether to invest in steering vs. prompt
  engineering for, say, conciseness or format adherence? The Measure benchmark answers it honestly:
  does steering beat a plain instruction, do they stack, or does neither move it?
- **Cheap, interpretable runtime guardrails / DLP.** The Monitor mode finds an SAE-feature detector
  for refusal, PII, sycophancy, off-topic, etc. — a flag you can run inline, far cheaper than an
  LLM judge and *auditable* (you can see which feature fired and why).
- **A validated control/monitor library for a product.** Recipes and monitors are reproducible
  artifacts with provenance + an honest verdict — a vetted shelf of behavior controls and detectors
  rather than a folder of cherry-picked prompts.
- **Interpretability research & model debugging.** Understand which concepts a model represents,
  how cleanly, at which layer, and how steerable they are — and root-cause surprising behavior.
- **Red-teaming / safety probing.** Find a feature linked to a risky behavior, test whether you can
  causally control it, and measure how reliably you can *detect* it.

## How the workbench is organized

The left rail has six steps, grouped into **control** and **monitoring**:

- **Feature track** (control) — `Explore → Steer → Measure → Library`. Find a single SAE feature,
  dial it, prove it against controls, save it as a recipe.
- **Geometry track** (control) — `Manifold → Library`. Steer a whole concept along its manifold.
- **Monitoring** — `Monitor → (gallery)`. Find a feature-based detector for a behavior and save it.

**Reading results honestly — read this once.** The verdicts are conservative on purpose:
- **VALIDATED** — the intervention/detector genuinely beat the prompt baseline *and* every control.
- **BENCHMARKED** — it ran but did *not* clearly beat them. This is **not a failure to hide** — it's
  a real result worth reporting. On the dev (random) model almost everything lands here, correctly.

Treat BENCHMARKED as "no evidence it works yet," not "broken." This honesty is the whole point.

## Interface & status — the screen at a glance

Four regions:

- **Left rail** — the six modes (the loop), and a **CONTEXT** panel at the bottom showing the model,
  SAE, current **layer**, dtype, device, load status, and GPU memory. Glance here to know *which
  layer you're on* and whether the model is loaded.
- **Stage** (center) — the active mode.
- **Inspector** (right rail) — your pinned feature lives here and travels across modes. It's empty
  ("Pin a feature to carry it across Explore, Steer, and Measure") until you pin one; then it shows
  the feature id, a **label box** (see *Label what you find* below), and a context button —
  **Steer with this →** from Explore, **Send to Measure →** from Steer.
- **Top-right chips** — `LIVE · <model>` and a `warm · <device>` / `idle` chip telling you whether
  the model is hot. A cold real-model call (chip says idle) takes longer while it loads.

## Conventions you'll use everywhere

- **Pinning.** Click any feature (a bar, a card, a contrast row) to *pin* it. The pinned feature
  shows in the right rail and rides with you into **Steer** and **Measure**. Only one feature is
  pinned at a time; pinning a new one replaces it.
- **The layer.** Each panel header shows the layer it's reading (e.g. `inspect_prompt · layer 0`).
  In **Manifold** you choose the layer per concept; elsewhere it's the model's default.
- **Before / after panes.** Steering results always show *unsteered* (baseline) beside *steered*
  so you can see the actual change, never just the steered text alone.

---

# 1 · Explore — find a feature worth steering

Three sub-views, switched by the segmented control at the top: **Inspect · Atlas · Contrast**.

## 1a · Inspect — what fires on this prompt?

See, token by token, which SAE features a prompt activates.

**Try it**
1. Open **Explore** (it starts on **Inspect**). The prompt box is pre-filled with
   `The capital of France is Paris`.
2. Press **Inspect activations**.
3. In the *Activation microscope*, each token is tinted by how strongly it lights up the SAE
   (brighter = stronger). Click the token **Paris**.
4. The bottom panel, *Top features · token Paris*, lists the strongest features — each row is
   `#id · label · activation bar · value`.
5. Click the **⊕** on the top row to pin that feature.

**What you'll see:** a heat-tinted token strip, then ~12 feature bars for the clicked token. Pinned
features turn up in the right rail with a *Steer with this* shortcut. **[real model]** the labels
and the tokens that light up are meaningful; on dev they're arbitrary ids.

> Tip: the token with the brightest tint is usually the most "interesting" one to click first.

## 1b · Atlas — which features recur across many prompts?

Scan a whole corpus at once and see which features fire broadly vs. narrowly.

**Try it**
1. Switch to **Atlas**. The box is pre-filled with five prompts (one per line), e.g.
   `The capital of France is Paris.`, `Return a JSON object with name and age.`, …
2. Press **Scan corpus**.
3. You get a grid of feature cards. Each card shows `#id`, its label, a **sparkline fingerprint**
   (how strongly it fired in each of the 5 prompts), and `peak … · in N/5 · top tokens`.
4. Use the controls: type `json` in the **search** box to filter to JSON-ish features; change
   **sort** to `breadth (# prompts)` to surface features that fire everywhere; toggle
   **labeled only** to hide unlabeled ones.
5. Click a card to pin that feature.

**What you'll see:** up to ~90 cards. A feature that fires `in 5/5` prompts with a flat sparkline
is a generic/syntactic feature; one that spikes on a single prompt is more selective — usually the
more useful steering target. **[real model]** breadth and top-tokens are meaningful.

## 1c · Contrast — which features separate two prompts?

The fastest way to *find* a feature for a behavior: contrast a "do this" prompt against a "don't"
prompt.

**Try it**
1. Switch to **Contrast**. The two boxes are pre-filled: left (positive)
   `Write a concise factual answer.`, right (negative) `Write a long rambling story.`
2. Press **Compare**.
3. You get a diverging bar chart: bars to the **right ▶** pull toward the positive prompt, bars to
   the **left ◀** pull toward the negative. The number is `positive_max − negative_max`.
4. Click the longest right-pointing bar to pin the feature that most represents "concise/factual".

**What you'll see:** ranked contrastive features. The pinned one is now armed for **Steer**.
**[real model]** the top contrastive feature is a strong candidate for the behavior you described.

## 1d · Label what you find (the notebook)

Features are just numbered ids until you name them. Once a feature is pinned (from Inspect, Atlas,
or Contrast), the **Inspector** rail on the right shows a note box — *"what does this feature seem
to do?"*. Type a human label (e.g. `refusal`, `JSON braces`, `positive sentiment`) and press
**Save label**. Labels persist in a local notebook and then appear everywhere that feature shows
up — and they power Atlas's **labeled only** filter, so you can build a named map of the features
you care about.

**Try it**
1. **Inspect** a prompt → click a token → pin the top feature (⊕).
2. In the right-rail **Inspector**, type a label in the note box → **Save label**.
3. Switch to **Atlas**, **Scan corpus**, then toggle **labeled only** — your named feature is now
   filterable, and its label shows on its card.

---

# 2 · Steer — turn a feature up or down

Needs a pinned feature (if none, you'll see *No feature armed* with a button back to Explore).

## 2a · Generate at a strength

**Try it**
1. Pin a feature in Explore, then open **Steer**. The arm header shows your `#id` and label.
2. The prompt box is pre-filled with `Write one sentence about Paris.`
3. Drag the **strength** dial (range −15…+15; it starts at +8). Positive adds the feature,
   negative subtracts it.
4. Press **Generate**.

**What you'll see:** an *unsteered* pane beside a *steered* pane, plus three gauges:
- **hook** — `✓ fired` confirms the intervention ran,
- **hidden Δ** — how much the residual stream moved,
- **logits Δ** — how much the output distribution moved.

A fired hook with a non-zero hidden Δ means the steer *happened*; whether it *helped* is the next
section's job. **[real model]** the steered text changes meaningfully at the right strength.

## 2b · Strength sweep — find where it takes effect

**Try it**
1. In **Steer**, press **Run sweep (−10 … +10)**.
2. Five frames appear at strengths −10, −5, 0 (baseline), +5, +10.
3. Click any frame to set the dial to that strength, then **Generate** to explore around it.

**What you'll see:** the generation at each strength side by side. You're hunting for the band
where the behavior appears *before* the text collapses into repetition — too much strength breaks
coherence. The `0` frame is your baseline. **[real model]** the sweep reveals the usable strength
window.

## 2c · Send to Measure

Press **Send to Measure →** to carry the prompt + pinned feature + current strength into the
benchmark. (A reminder is shown: *a fired hook ≠ a real behavior change — confirm it in Measure.*)

---

# 3 · Measure — did it actually work?

The honest gate. A fired hook isn't proof; this asks whether steering beats a plain prompt
instruction *and* every control.

## 3a · Live causal check

**Try it**
1. With a feature pinned, open **Measure** and press **Run live check**.

**What you'll see:** one paired unsteered/steered generation with the same hook gauges as Steer —
a quick sanity check before the full benchmark.

## 3b · Seven-control benchmark

**Try it**
1. The prompt-set box is pre-filled with two JSON-line prompts (one prompt object per line), e.g.
   `{"id":"p001","prompt":"Explain sparse autoencoders in one paragraph."}`.
2. Pick an **objective** from the dropdown:
   - `maximize_rule_score` (general behavior match),
   - `maximize_json_validity` (does it emit valid JSON?),
   - `minimize_length_without_empty_output` (be terse but non-empty).
3. Press **Run benchmark**.

**What you'll see:** seven methods scored and ranked —
`steering_only`, `prompt_plus_steering`, `prompt_only`, plus four controls marked **⊘ control**
(`unsteered_baseline`, `zero_strength_control`, `random_feature_control`, `negative_strength_control`).
Then a verdict chip:
- **VALIDATED** — steering genuinely beat the controls and the prompt-only baseline,
- **BENCHMARKED** — it ran but did *not* clearly beat them (the gate refuses to over-claim),
- **CANDIDATE** — partial.

Below the bars: the first prompt's *unsteered* vs *steering only* output, and a **Save as recipe**
button. **[real model]** on dev, expect `BENCHMARKED` — a random feature rarely beats controls,
and that honesty is the point.

## 3c · Autopilot — let it find the feature for you

Skip the manual hunt entirely: give examples of the behavior and autopilot discovers candidate
features, benchmarks each against all seven controls, sweeps strength, and saves the best as a
recipe.

**Try it**
1. In the **Autopilot** panel, the two boxes are pre-filled with positive examples
   (`Paris is the capital of France.` …) and negative examples (`Once upon a time …`).
2. Set **candidates** to `3` (1–6).
3. Press **Run autopilot** and watch the stepper: `examples → candidates → bench → sweep → recipe`.

**What you'll see:** a **candidate leaderboard** (features ranked by contrast/score, best starred),
the best candidate's seven-method validation bars, a verdict, and a line confirming the recipe was
**saved to the Library**. It validates on the benchmark prompt set above. **[real model]** the
discovered feature is a real candidate for your behavior.

## 3d · Save a recipe

After a benchmark, press **Save as recipe**. The recipe captures the model, the feature + layer +
strength, the benchmark verdict, and a before/after example — and appears in **Library**.

---

# 4 · Manifold — steer along a whole concept

The geometry track. Instead of one feature, steer a **concept** by moving through its manifold.
(On dev the geometry is a clean *synthetic* ring/curve so the UI is explorable; the real manifolds
appear only on the real model.)

## 4a · Fit a concept

**Try it**
1. Open **Manifold**. The concept dropdown starts on **Days of the week · cyclic**.
2. The **layer** field defaults to the concept's *recommended* layer — the one where its geometry
   is cleanest (shown as `recommended L14 (atlas)` on the real model; on dev it falls back to the
   default layer). You can override it.
3. Press **Fit manifold**.

**What you'll see:** a 3D shape — a **ring** for cyclic concepts (days, months, compass), an open
**curve** for ordinal ones (sizes, integers, rank) — with each value labeled, plus a quality tag
(`ring_adjacency 1.0` for a clean ring, `abs_spearman …` for an ordered line). Drag to orbit.

> Try other concepts: `Integers 0–20`, `Size` (tiny→enormous), `Military rank` (private→general).
> Diffuse concepts (`Months`, `Compass`) will say the geometry is weak — that's expected, not a bug.

## 4b · Steer along the manifold

**Try it**
1. After fitting `days_of_week`, set **from** `Monday` and **to** `Friday` (waypoints default 7).
2. Press **Steer** — *or* just **click a point in 3D** (e.g. the `Friday` node) to steer to it.

**What you'll see:** an animated handle walks the ring Monday→Friday, and a **trajectory** lists the
generation at each waypoint. The intervention *replaces* the day's residual with the point you move
to. **[real model]** the continuation tracks the concept (e.g. "Tomorrow is …" follows the ring).

## 4c · Compare vs linear

**Try it**
1. With a fit in place, press **Compare vs linear**.

**What you'll see:** two panes — following the **manifold** vs taking a straight **linear** chord —
each with an **E** badge (behavior-manifold distance; *lower = more faithful*) and a perplexity
badge, plus both trajectories. This shows whether staying on the curve matters for this concept.

## 4d · Pullback — optimize the activation that induces a target

**Try it**
1. Set a **to** target (e.g. `Thursday`) and press **Pullback** (it runs an optimizer, so it takes
   a moment behind a spinner).

**What you'll see:** three panes — **manifold / linear / pullback** — each with an **E** badge and
an **R** badge (*recovers the geometry*; higher = the path traces the manifold), plus the three
paths drawn in 3D (teal / red / violet) and the optimizer's start→end loss. Pullback finds the
activation that best *induces* the target behavior. **[real model]** pullback typically reaches the
lowest energy of the three.

## 4e · SAE coverage — which features tile the manifold

**Try it**
1. With a fit in place, press **SAE coverage**.

**What you'll see:** the 3D points recolor by their dominant tiling feature, and a list of the SAE
features that tile the concept (`#id · label · tiles N (values…)`). On any row, click **steer →** to
pin that feature and jump straight to **Steer** at the manifold's layer — the bridge between the
geometry track and the feature track.

## 4f · Save a manifold recipe

**Try it**
1. After a **Pullback**, press **Save to Library**.

**What you'll see:** the steer is snapshotted as a **manifold recipe** (concept + source→target +
layer + path) with the energy comparison as its benchmark and an honest verdict — **VALIDATED** only
if on-manifold steering induced the behavior at least as faithfully as the linear chord, else
**BENCHMARKED**. It appears in the Library with a `∿ manifold` tag.

---

# 5 · Monitor — find a detector for a behavior

The detection half of the bench (Steer/Manifold *control* behavior; Monitor *detects* it). Give
labeled examples of a behavior and it finds the SAE feature(s) that flag it — a cheap,
interpretable runtime guardrail.

**Try it**
1. Open **Monitor**. The boxes are pre-filled with **refusal** examples: positives are refusals
   ("I'm sorry, but I can't help…"), negatives are compliances ("Sure, here's how…").
2. Set the **behavior** name, **layer**, and **top-k** (how many features to combine), then press
   **Discover monitor**.
3. You get: the **selected features** (each with its AUC and how often it fires on pos vs neg), a
   **held-out evaluation** (AUC / precision / recall / F1), a **random-feature control AUC**, and a
   **verdict** chip — `VALIDATED` only if the detector clears a strict gate *and* beats the random
   control; otherwise `BENCHMARKED` (honest — common on the dev model).
4. **Test on new text**: type a sentence and press **Flag text** — it shows the score and whether
   the monitor fires (⚑ FLAGGED / — clear).
5. **Save monitor →** stores it in the **Saved monitors** gallery below; click a card to load it
   back.

**What you'll see / how to read it:** a coherent behavior (sentiment, refusal) is usually captured
by a *single* feature; a heterogeneous one (PII — emails, SSNs, cards) needs a **multi-feature**
monitor (raise top-k), since different subtypes light different features. The random-feature
control is the point: an AUC of 1.00 means little unless it clearly beats a random detector.
**[real model]** on the real 2B, refusal and PII monitors validate cleanly; the dev model lands
`benchmarked` (it's random). *Caveat:* small example sets give coarse metrics — surface `n_pos`/
`n_neg`, and for a deployable monitor use dozens–hundreds of examples and adversarial negatives.

---

# 6 · Control — detect, suppress, and prove it, honestly

The capstone: it joins the detection half (Monitor) and the control half (Steer) and answers the
questions the interpretability field is currently stuck on. Three panels, one behavior (pre-filled
with **sycophancy** — telling the user what they want to hear).

**Try it**
1. Open **Control**. The **Behavior** box is pre-filled with sycophantic positives ("You're
   absolutely right, I completely agree!") and honest negatives ("The evidence points the other
   way."). Set behavior / layer / top-k.
2. **① Baseline shootout → Run shootout.** This is the credibility test: it pits the **SAE-feature
   monitor** against a **raw-residual linear probe** (diff-of-means *and* logistic regression) and the
   **random-feature control**, on a shared held-out split, reporting each method's AUC and its
   **TPR at a fixed false-positive rate** (how a real safety monitor is tuned). The verdict says
   `SAE MONITOR WINS`, `PROBE WINS`, or `TIE`. **This is the honest answer to "do SAE features even
   beat a cheap probe?"** — and a `PROBE WINS` is a real, publishable finding, not a failure.
3. **② Robustness under shift → Test robustness.** It discovers the detector on the clean examples,
   then evaluates it on **paraphrases it never saw**, and reports the AUC drop with a
   `ROBUST` / `FRAGILE` verdict. `FRAGILE` means the detector memorized its training wording — the
   failure mode that standard evals hide.
4. **③ Closed loop → Run control loop.** The headline. The bench discovers a monitor, **suppresses
   the behavior by steering the detector's own top feature**, re-scores every generation with that
   monitor (does it still fire?), and measures **collateral damage**: the *fluency* chip (neutral-text
   perplexity, steered vs unsteered) and the *safety Δ* chip (change in compliance on held-out harmful
   prompts — the "Rogue Scalpel" effect, where a benign steer silently breaks refusals). The verdict
   is `VALIDATED` **only if** the behavior was present, the steer removed it, **and** nothing broke.

**How to read it:** the whole point is that **suppression alone is not success**. A run can suppress
the behavior 100% and still land `BENCHMARKED` because the steer lobotomized fluency or eroded safety
— and the before/after example rows show exactly what happened. **[real model]** on the dev model
everything is `BENCHMARKED`/noisy (random weights); the real verdicts come from the 2B Modal probe
(`control_loop_demo_2b`). *Caveat:* the safety-Δ heuristic is refusal-string matching, and it is only
meaningful when the suppressed behavior is **not** itself refusal (suppressing refusal *should* raise
compliance — that's the goal, not damage).

---

# 7 · Library — your saved recipes

Every benchmarked experiment — feature steer *or* manifold steer — becomes a reproducible card.
(Saved **monitors** have their own gallery inside the Monitor mode.) Recipes are written to disk
under `recipes/` (monitors under `monitors/`) as JSON + Markdown, so your work persists between
sessions and is easy to share or version.

## 6a · Browse & filter

**Try it**
1. Open **Library**. Use the status pills to filter: `all · validated · candidate · benchmarked ·
   draft`.

**What you'll see:** a grid of cards. A **feature** card shows `L{layer} · #{feature_id}`; a
**manifold** card shows `∿ manifold` with `concept · source→target` and the path. The colored edge
marks how far it climbed the validation ladder.

## 6b · Open a recipe

**Try it**
1. Click any card.

**What you'll see:** provenance (model, layer, and either feature/strength or
concept/source→target/path), a recorded before/after example, the verdict and its reason, and
limitations. Manifold recipes also show the **paths compared** (the manifold/linear/pullback energy
legs).

## 6c · Reuse a recipe

- **Feature recipe** → **Load into Steer** (arms the feature at its strength + layer, jumps to
  Steer) or **Open in Measure**.
- **Manifold recipe** → **Load into Manifold** — re-fits the concept at its layer and restores the
  saved `source → target`, ready to Steer/Pullback again.

---

# Interesting experiments to try

These are real experiments we ran on the live 2B — each is reproducible here, and each lists what
we actually found, including the honest negatives. They double as a tour of how the features
combine. **[real model]** — the *findings* need the real model; the dev model shows only mechanics.

### 1 · Steerable vs. promptable: which behaviors does steering actually help?
- **Question:** for a target behavior, does steering beat a plain prompt instruction — and do they
  stack?
- **How:** pick ~2 behaviors. For each, **Measure → Autopilot** (or Contrast → pin → Measure), then
  compare `steering_only` vs `prompt_only` vs `prompt_plus_steering` on the seven-control board.
- **What we found:** *"be concise"* **VALIDATED** and the two *composed* — prompt+steering (−34.5)
  beat prompt-only (−46) and steering-only (−49); the negative-strength control was worse than
  baseline, confirming the feature is causal. *"valid JSON"* scored **0 across all seven** — neither
  prompting nor steering got the base 2B to emit JSON in short outputs. Behaviors land in different
  cells: steerable+promptable+additive vs. neither.
- **Why it's interesting:** an honest map of *what's a controllable handle* — the real
  control-method decision, not a cherry-picked demo.

### 2 · What does the model do *between* concept values?
- **Question:** is a concept manifold a true continuum, or just discrete points?
- **How:** **Manifold → Fit** `days_of_week` → **Steer** `Monday → Thursday` with 7 waypoints →
  read the per-waypoint generation in the trajectory list.
- **What we found:** the "tomorrow is…" answer marched **monotonically** Tue→Wed→Wed→Thu→Thu→Fri→Fri.
  The real-day points were exact; the in-between points **snapped to the next day** (no fractional
  day) but never reversed or skipped. Continuous, ordered geometry → a quantized-but-monotonic
  readout.
- **Why it's interesting:** a vivid, direct test of the manifold hypothesis you can literally watch.

### 3 · Do "related" concepts share SAE features?
- **Question:** does the SAE reuse features across related concepts, or specialize?
- **How:** **Manifold → Fit + SAE coverage** for several concepts at the *same* layer (e.g. days,
  size, temperature, valence); compare the tiling feature sets.
- **What we found:** a tiny **generic core** (a few features, e.g. #847) tiles *every* concept
  (syntactic scaffolding); beyond it the tilings are nearly **disjoint** (Jaccard ~0.1–0.25), and
  "related" pairs (size × temperature) shared no more than unrelated ones. The SAE specializes
  per concept.
- **Why it's interesting:** hands-on evidence for the SAE-concept-manifold picture — and a caution
  that a generic core fires on everything (don't mistake it for meaning).

### 4 · Find a safety-relevant feature and test it causally
- **Question:** can you find a feature linked to sycophancy and actually control it?
- **How:** **Explore → Contrast** (sycophantic agreement vs. honest disagreement) → pin the top
  feature → **Steer** it ±, sweeping strengths on a falsehood probe like *"When my colleague said
  7×8 = 54, I told him he was ___."*
- **What we found:** the top "agreement" feature was **causal** — steering it up injected
  positive-affect tokens ("very good…") — but it did **not** cleanly flip the factual judgment:
  the completion stayed *"wrong"* until coherence collapsed into repetition. An affect handle, not a
  clean sycophancy lever, on a 2B base. (An honest partial result — exactly what the sweep is for.)
- **Why it's interesting:** the full red-team / safety-probe loop, with a calibrated outcome.

### 5 · Build a reliable detector for a behavior
- **Question:** can you find an interpretable monitor that *generalizes*?
- **How:** **Monitor** → paste ~8 positive / 8 negative examples → **Discover** → read held-out AUC,
  the random-feature control, and the verdict → **Flag** new text → **Save**.
- **What we found:** sentiment and refusal each had a *single* feature with **held-out AUC 1.00**
  (refusal fires on 7/8 refusals, 0/8 compliances). **PII needed a multi-feature monitor** (single
  feature 0.62; top-3 combined 1.00) because emails/SSNs/cards light different features. Lesson:
  coherent behavior → one feature; heterogeneous → raise **top-k**.
- **Why it's interesting:** the guardrail/DLP use case, proven with held-out metrics *and* a control.

### 6 · Does the interpretable monitor actually beat a dumb probe? (and does suppression break safety?)
- **Question:** the field's two hardest questions — (a) do SAE features beat a raw-residual linear
  probe at *detecting* a behavior, and (b) when you *suppress* a behavior by steering, do you silently
  break the model's safety or fluency?
- **How:** **Control** → keep the **sycophancy** examples → **① Run shootout** (SAE monitor vs.
  residual probe vs. random control) → **② Test robustness** (paraphrase shift) → **③ Run control
  loop** (suppress + collateral check). Or fire `control_loop_demo_2b` on Modal for the real 2B.
- **What to read:** the shootout `winner` — a `PROBE WINS` or `TIE` is the honest, field-relevant
  result (interpretability isn't buying detection power here); the robustness `auc_drop` (does it
  survive paraphrases?); and the loop verdict, where a **100%-suppressed but `BENCHMARKED`** run is the
  whole point — the *fluency* and *safety Δ* chips show the steer's hidden cost. On the dev model the
  numbers are noise (random weights); the real verdicts come from the Modal probe.
- **Why it's interesting:** it's the AI-control loop (detect → suppress → prove) with the rigor the
  field usually skips — answering "is the white-box method worth it?" against real baselines, and
  catching the "Rogue Scalpel" failure where a benign steer quietly erodes refusals.

---

# Guided workflows

Three task recipes that tie the features together end to end.

### A. "Make the model concise" (feature track)
1. **Explore → Contrast:** positive `Write a concise factual answer.`, negative
   `Write a long rambling story.` → **Compare** → pin the top right-pointing (concise) feature.
2. **Steer:** prompt `Write one sentence about Paris.`, sweep strengths, pick one where it tightens
   without breaking.
3. **Measure:** **Run benchmark** with objective `minimize_length_without_empty_output`.
4. If it beats the controls (**VALIDATED**), **Save as recipe**. **[real model]** for a real verdict.

### B. "Steer days Monday → Friday" (geometry track)
1. **Manifold:** concept `days_of_week` → **Fit manifold** (uses the recommended layer).
2. From `Monday` to `Friday` → **Steer**, then **Compare vs linear**, then **Pullback**.
3. **Save to Library**; later open it and **Load into Manifold** to pick up where you left off.

### C. "Ship a refusal guardrail" (monitoring)
1. **Monitor:** keep the pre-filled refusal examples (or paste your own) → **Discover monitor**.
2. Read the held-out AUC + the random-feature control; **Flag** a few new sentences to spot-check.
3. If **VALIDATED**, **Save** it — it's now a reusable, auditable detector in the gallery.

---

# Glossary

- **SAE feature** — an interpretable direction a sparse autoencoder learned from the model's
  activations; high activation on a token = that concept is present.
- **Residual stream** — the running hidden state that flows through the model's layers; both feature
  steering and manifold edits act here.
- **Steering** — adding/subtracting a feature direction (feature track) or replacing a token's
  residual with a point on a concept manifold (geometry track) to change behavior.
- **Concept manifold** — the low-dimensional curve/ring a concept's values trace in the residual
  stream (e.g. the day-of-week ring).
- **E badge** — behavior-manifold distance; *lower = the steered output is more on-concept (faithful)*.
- **R badge** — how well a pullback path recovers the manifold geometry; *higher = it traces the curve*.
- **Pullback** — optimizing the activation that *induces* a target behavior (the inverse of steering).
- **Verdict** — **VALIDATED** (beat the prompt baseline and all controls) vs **BENCHMARKED** (ran,
  didn't clearly beat them — not a failure, just no evidence yet).
- **Recipe / Monitor** — saved, reproducible *control* / *detection* artifacts with provenance.

For the full science (the activation↔behavior isometry, and the negative results behind the design),
see `MANIFOLD.md`.

---

# Troubleshooting & FAQ

- **"No feature armed" in Steer.** Pin a feature in Explore first (Inspect ⊕, an Atlas card, or a
  Contrast bar).
- **Benchmark says BENCHMARKED, not VALIDATED.** That's the honest gate: the steer didn't clearly
  beat its controls. On the dev model this is normal (random features). Try a contrast-discovered
  feature on the **real model**.
- **The generated text is gibberish.** You're on the dev (tiny random) model — switch to the Modal
  `web_gui` for the real model.
- **A concept won't form a clean shape.** Some concepts (months, compass on 2B) are genuinely
  diffuse — the UI tells you and falls back to the default layer. Try `days_of_week`, `integers_0_20`,
  `rank`, or `size` for clean geometry.
- **Pullback is slow.** It runs an optimizer (L-BFGS) through the model; the spinner means it's
  working. Reduce waypoints if you want it faster.
- **3D area is blank.** Your browser may lack WebGL; the panel shows a fallback message. Everything
  else still works.
- **My monitor lands BENCHMARKED / low AUC.** Either the behavior is heterogeneous — raise **top-k**
  so it can combine subtype features (the PII lesson) — or you're on the dev (random) model. Give a
  handful of *clear* positive/negative examples and try the real model.
- **Discover / a benchmark feels slow.** Each example or method is a model forward pass, and they
  run one at a time on a single GPU (so concurrent calls queue rather than fail). Keep example sets
  modest; on the real model the first call also pays a one-time model load.
- **I want to script this, not click.** Every action is an HTTP endpoint; long jobs use a submit →
  poll API. See `AGENT_RESEARCH.md` and `/api/docs`.
