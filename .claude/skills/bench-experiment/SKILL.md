---
name: bench-experiment
description: >
  Conduct a rigorous, HONEST interpretability experiment on the Qwen Scope "Lab Bench" in this repo
  — find or steer an SAE feature, run the seven-control benchmark, fit/steer a concept manifold, or
  discover a behavior monitor, and read the verdict honestly. Use this whenever the user wants to run
  an experiment, test a hypothesis about the model, find a feature for a behavior, check whether a
  steer or a detector actually works, compare steering vs prompting, probe a safety-relevant
  behavior, or "see what we can learn" from the bench — even if they never say the word
  "experiment". It encodes the explore → hypothesize → run → read-verdict → save loop, the async job
  API (so concurrent calls don't 500 the single GPU), and the honesty contract (BENCHMARKED is a
  real result, not a failure to hide; report negatives).
  Triggers on: "run an experiment on the bench", "find a feature for X", "does this steer actually
  work", "is it validated or just benchmarked", "build or test a detector/monitor", "compare
  steering vs prompting", "fit or steer the X concept manifold", "submit a pullback/autopilot/sweep/
  benchmark job and read the result", "probe how the model represents X (sentiment, refusal, …)".
---

# Running an interpretability experiment on the Lab Bench

This bench exists to answer *does this actually work?* about feature steering, concept-manifold
steering, and feature-based detection — with controls, not vibes. Your job is to run falsifiable
experiments and report what's true, including the nulls. The bench's whole value is that it will
tell you when something didn't work; don't paper over that.

Orient first by skimming `docs/AGENT_RESEARCH.md` (the agent contract + the API), `docs/USER_GUIDE.md`
(the human tour + a list of worked experiments with real findings), and `docs/MANIFOLD.md` (the
science). The live machine schema is at `/api/openapi.json`.

## The loop

1. **Explore / hypothesize.** Use `inspect` (features per token), `compare` (contrast two prompts to
   find a separating feature), or `atlas` (scan a corpus) to find a candidate; or pick a `manifold`
   concept. Write down a falsifiable claim, e.g. *"feature #X steers toward concise output and beats
   a plain instruction"* or *"a refusal detector is findable at layer 12"*.
2. **Run** the right tool (see *What you can run* below).
3. **Read the verdict honestly** (see *The honesty contract* — this is the point).
4. **Record.** Jobs auto-append to the experiment log (`GET /api/experiments`). Save results worth
   keeping as recipes (steers) or monitors (detectors). Note genuinely surprising findings to memory.

## The honesty contract (read this — it's the reason the bench exists)

Always read `validation_decision`:

- **VALIDATED** — the intervention/detector genuinely beat the prompt baseline *and* every control.
- **BENCHMARKED** — it ran but did **not** clearly beat them. This is a real result to report, not a
  failure to bury. On the dev (random) model almost everything lands here; that is correct.

The seven controls (`unsteered`, `prompt_only`, `zero_strength`, `random_feature`,
`negative_strength`, …) and the monitor's random-feature control exist to stop you from believing
your own steers. A `negative_strength_control` that does *worse* than baseline is positive evidence
the feature is causal — cite it. Specific commitments:

- **Report negatives as prominently as positives.** A clean negative is a finding.
- **Don't tune until something "passes" and then report only that.** The experiment log is the trail.
- **Positive-control first.** Before trusting a *negative* on a hard behavior, confirm your method
  on an easy one (e.g. sentiment is trivially detectable — if even that fails, your harness/metric is
  broken, not the model). This caught two methodological bugs in past sessions.
- **A fired hook with a non-zero `hidden_delta_norm` means the intervention happened — not that the
  behavior changed.** Only the benchmark verdict settles that. If steering breaks coherence
  (repetition collapse) before achieving an effect, say so and sweep for the usable band.

## Concurrency: one GPU → fire-and-poll, never fire-and-forget-in-parallel

There is a single GPU and all model-touching work is serialized behind one lock. **Submit heavy ops
as jobs and poll** — don't fire several synchronous heavy calls at once (they queue, and bypassing
the job API with parallel sync calls contends and returns 500s; that exact mistake cost a session a
confusing detour). Use the bundled client:

```bash
# point at the dev server (default) or a Modal web_gui URL via BENCH_URL
python .claude/skills/bench-experiment/scripts/bench_client.py status
python .claude/skills/bench-experiment/scripts/bench_client.py job benchmark \
  '{"prompt_set":"{\"id\":\"p1\",\"prompt\":\"What is the capital of France?\"}","feature_id":42,"strength":8,"layer":12,"objective":"minimize_length_without_empty_output","max_new_tokens":12}'
```

`run_job(op, params)` in that script submits and polls to completion. Read-only calls (`status`,
`recipes`, `experiments`, `manifold/presets`) are never blocked by a running job.

## What you can run (ops)

Job-able ops (submit via `/api/jobs` or call the sync `/api/<op>`): `inspect`, `compare`, `atlas`,
`steer`, `sweep`, `benchmark`, `autopilot`, `manifold_fit|steer|compare|sae_coverage|pullback`,
`monitor_discover`, `monitor_score`. Params mirror the matching request body — see
`/api/openapi.json`. Two sub-workflows worth knowing:

- **Steerable vs. promptable** (control): discover a feature (Contrast or Autopilot) → `benchmark` →
  compare `steering_only` / `prompt_only` / `prompt_plus_steering`. The honest map of what's a real
  controllable handle.
- **Build a detector** (monitoring): `monitor_discover` with ~8 labeled positive / 8 negative
  examples → read held-out AUC + the random-feature control. **Coherent behavior → a single feature;
  heterogeneous behavior (e.g. PII) → raise `top_k`** so it combines subtype features.

## Where to run

- **Dev** (`python serve_web.py --dev`, GPU-free, default `http://127.0.0.1:7870`) runs the real code
  paths on a tiny random model — perfect for wiring an experiment and checking plumbing. Expect
  `BENCHMARKED`/chance results; that's the model, not a bug.
- **Real 2B, locally on a Mac** (`python serve_web.py --mlx mlx-community/Qwen3.5-2B-bf16 [--mlx-sae
  <repo> --mlx-d-sae N]`, no Modal/CUDA) runs the real model + SAE on-device — the preferred path for
  real 2B findings on Apple Silicon. Point `BENCH_URL` at it like any other server. See `docs/MLX.md`.
- **Real model via GPU** (the **27B**, or a recorded real-GPU run) is Modal — see the **bench-on-modal**
  skill. Point `BENCH_URL` at the served `web_gui` URL and reuse `bench_client.py`. Real findings need a
  real model (MLX or Modal), not the dev backend.

## Reporting

State the hypothesis, what you ran, the verdict and the key numbers, and the honest conclusion
(including nulls and caveats like small sample sizes). Save validated artifacts; leave the trail in
the experiment log. Mirror the worked examples in `docs/USER_GUIDE.md` ("Interesting experiments to
try") for the level of rigor and honesty expected.
