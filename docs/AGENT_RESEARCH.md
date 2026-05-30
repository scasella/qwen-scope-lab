# Conducting interpretability research with an agent

This bench is drivable by an AI agent, not just a human. Every capability is a structured HTTP
endpoint, the seven-control benchmark gives you an *honest verdict* instead of raw outputs to
judge yourself, and an async job API + experiment log let you run a real research loop. This
guide is the contract: how to drive it, and — more importantly — how to do so *honestly*.

The full machine schema is always live at **`/api/openapi.json`** (`/api/docs` for a browser).

## The research loop

1. **Explore / hypothesize.** Use `inspect`, `compare` (contrast two prompts), or `atlas` (scan a
   corpus) to find a candidate feature, or pick a `manifold_*` concept. Form a falsifiable
   hypothesis ("feature #X steers toward concise output and beats a plain instruction").
2. **Submit a job.** `POST /api/jobs {op, params}` → `{job_id}`. Heavy ops run in the background.
3. **Poll.** `GET /api/jobs/{job_id}` until `status` is `done` or `error`. Read `result`.
4. **Read the verdict — honestly (see below).**
5. **Log is automatic.** Every job appends to the experiment trail (`GET /api/experiments`).
6. **Save what's real.** If a steer validated, persist it: `POST /api/recipes` (feature recipe
   after a `benchmark`; manifold recipe with `{"kind":"manifold"}` after a `manifold_pullback`).

## Async job API

```
POST /api/jobs           {"op": "<op>", "params": {...}}   -> {"job_id": "...", "status": "queued"}
GET  /api/jobs/{job_id}                                     -> {status, result|error, timestamps}
GET  /api/jobs                                              -> [{id, op, status, timestamps}, ...]
GET  /api/experiments?limit=50                              -> the research trail (newest last)
```

`op` is one of: `inspect`, `compare`, `atlas`, `steer`, `sweep`, `benchmark`, `autopilot`,
`manifold_fit`, `manifold_steer`, `manifold_compare`, `manifold_sae_coverage`, `manifold_pullback`,
`monitor_discover`, `monitor_score`, `monitor_shootout`, `monitor_robustness`, `collateral`,
`control_loop`, `probe_discover`, `probe_score`, `steer_direction`, `caa_vs_sae`, `method_atlas`,
`emotion_coupling`, `safety_geometry`, `monitor_stream`.
`params` mirror the matching `POST /api/<op>` request body (see `/api/openapi.json` for every
field). The quick ops (`inspect`/`compare`/`steer`) are also fine to call synchronously at
`POST /api/<op>`; the heavy/experiment ops (`benchmark`, `autopilot`, `manifold_*`, `atlas`,
`sweep`) are what you'll usually submit as jobs.

## Concurrency — one GPU, so jobs serialize

There is a single GPU. All model-touching work runs under one lock, so **fire-and-poll, don't
fire concurrently**: if you launch two heavy operations at once they queue (correct) — and if you
bypass the job API and hit two synchronous model endpoints in parallel you'll contend. Submit a
job, poll it to completion, then submit the next. Read-only calls (`status`, `manifold/presets`,
`recipes`, `experiments`) are never blocked by a running job.

## The honesty contract (this is the important part)

The seven controls exist to stop you from believing your own steers. **Always read
`validation_decision`:**

- `validated` — the steer genuinely beat the prompt-only baseline *and* every control.
- `benchmarked` — it ran but did **not** clearly beat them. **This is not a failure to hide; it is
  a result to report.** Most steers on a small base model land here.
- A `negative_strength_control` that does *worse* than baseline is evidence the feature is
  directional and causal — cite it.

Rules of conduct:
- Report negatives as prominently as positives. A clean negative is a finding.
- Don't tune until something "passes" and then report only that. The experiment log keeps the
  whole trail; treat it as your lab notebook.
- A fired hook with a non-zero `hidden_delta_norm` means the intervention *happened* — it does
  **not** mean the behavior changed. Only the benchmark verdict settles that.
- When steering breaks coherence (repetition collapse) before achieving an effect, say so; sweep
  for the usable strength band rather than reporting the broken extreme.

**Worked negatives from a real session (so you know what honest looks like):**
- "Valid JSON" steering scored **0 across all seven methods** — neither the instruction nor the
  feature produced valid JSON on a 2B base model. Reported as `benchmarked`, not spun.
- A contrast-discovered "sycophancy" feature was causal (injected positive-affect tokens) but
  **never cleanly flipped a factual judgment** before coherence collapsed — reported as an affect
  handle, not a sycophancy lever.
- "Be concise," by contrast, **validated** and *composed* with prompting (`prompt_plus_steering`
  beat both alone) — a real positive, earned against the controls.

## Experiments vs. recipes

- **Experiments** (`/api/experiments`) = the full trail of everything tried + its outcome,
  including negatives. Your research history.
- **Recipes** (`/api/recipes`) = saved, reproducible, *validated-or-benchmarked* artifacts you
  chose to keep. Your published results.

## Worked example (the loop end to end)

```bash
# 1. Find a feature for a behavior (contrast), synchronously:
curl -s localhost:7870/api/compare -H 'content-type: application/json' \
  -d '{"positive":"Write a concise factual answer.","negative":"Write a long rambling story.","layer":12}'
#  -> read positive_stronger[0].feature_id

# 2. Benchmark it as a job:
JOB=$(curl -s localhost:7870/api/jobs -H 'content-type: application/json' \
  -d '{"op":"benchmark","params":{"prompt_set":"{\"id\":\"p1\",\"prompt\":\"What is the capital of France?\"}","feature_id":<id>,"strength":8,"layer":12,"objective":"minimize_length_without_empty_output","max_new_tokens":14}}' | jq -r .job_id)

# 3. Poll:
curl -s localhost:7870/api/jobs/$JOB | jq '.status, .result.validation_decision'

# 4. Read the verdict honestly; 5. it's already in /api/experiments; 6. save if validated:
curl -s -X POST localhost:7870/api/recipes -H 'content-type: application/json' -d '{}'
```

`autopilot` automates steps 1–2 (it discovers candidates, benchmarks each against all seven
controls, sweeps strength, and saves the best recipe) — submit it as a job and poll.

## The detection half: behavior monitors

The bench also finds **monitors** — cheap, interpretable feature-based detectors for a behavior
(refusal, PII, sycophancy, off-topic…), the same honest-evaluation discipline applied to
*classification* instead of control. The loop:

```
POST /api/monitor/discover  {behavior, positive_examples, negative_examples, layer, top_k}
   -> {features, threshold, metrics{auc,precision,recall,f1,fpr,control_auc,...}, per_feature, validation_decision}
POST /api/monitor/score     {text, monitor_id | features+layer+threshold}  -> {score, fires}
POST /api/monitors          (save the last discovered monitor)   GET /api/monitors[/{id}]
```

Discovery ranks SAE features by how well their activation separates the labeled examples, combines
the top-k by max-activation (so heterogeneous behaviors like PII can use an OR of subtype
features), and reports **held-out** metrics plus a **random-feature control**. Read
`validation_decision` exactly as for steering: `validated` only if the detector clears a strict
gate *and* beats the random-feature control — else `benchmarked`. The control is your defense
against a chance separation looking real. Submit `monitor_discover` as a job for fire-and-poll.

## The control half: does white-box detect-and-suppress actually work?

The bench's newest layer answers the questions the interpretability field is currently stuck on —
all with the same honest-controls discipline. Every op runs on dev (CPU) and identically on the real
model.

```
POST /api/monitor/shootout   {behavior, positive_examples, negative_examples, layer, top_k, target_fpr}
   -> {methods{sae_monitor, residual_diffmeans, residual_logistic, random_control}, verdict{winner, margin, ...}}
POST /api/monitor/robustness {positive_examples, negative_examples, shift_positive_examples, shift_negative_examples, layer, top_k}
   -> {in_distribution, shifted, auc_drop, robustness{status: robust|fragile, reason}}
POST /api/collateral         {feature_id, strength, layer, ppl_bound, safety_tol}
   -> {perplexity_ratio, safety_regression, unsteered/steered_compliance_rate, verdict{status: clean|damaged}}
POST /api/control_loop       {positive_examples, negative_examples, test_prompts, layer, suppress_strength, ...}
   -> {fires{fire_rate_unsteered, suppression_rate, ...}, collateral, rows, verdict{status: validated|benchmarked}}
POST /api/safety_geometry    {layer, strength, max_new_tokens, use_judge}
   -> {rows[{behavior, cos_with_refusal, collateral_raw, collateral_orth, ppl_raw, ppl_orth}], predictor_corr, fix_reduces_collateral}
POST /api/monitor/stream     {prompt, probe_id | direction, bias, threshold, layer, max_new_tokens}
   -> {generation, trajectory[{step, score, fires, text}], flagged_at_step, final_fires}
```

- **`monitor_shootout`** is the credibility check: does the interpretable SAE-feature monitor beat a
  *raw-residual linear probe* (the baseline SAEs are accused of not beating, arXiv 2502.16681) and the
  random control? `winner` is `sae_monitor`, `residual_probe`, `tie`, or `inconclusive`. A `tie` /
  `residual_probe` win is an honest negative, not a bug — report it.
- **`monitor_robustness`** discovers on a clean set and evaluates on paraphrases: `fragile` means the
  detector memorised its training distribution (the "looks fine in standard evals, fails under shift"
  failure mode).
- **`collateral`** is the Rogue-Scalpel check (arXiv 2509.22067): a steer that achieves its goal can
  still erode refusals or fluency. `damaged` if compliance on held-out harmful prompts rose, or neutral
  perplexity blew up.
- **`control_loop`** ties it together: discover → suppress the detector's own top feature → re-score
  every generation → measure collateral → one verdict. `validated` requires the behavior to have been
  present, removed, **and** clean. Read `verdict.status`; a perfect-suppression-but-damaged run is
  honestly `benchmarked`.
- **`safety_geometry`** asks *why* a steer causes collateral and *how to avoid it*: it discovers a
  refusal probe and a probe for each behavior, and tests whether the **cosine between them predicts
  the safety regression** of steering that behavior (`predictor_corr`). Then it re-measures collateral
  with the steer **projected orthogonal to the refusal direction** — `fix_reduces_collateral` is the
  honest test of whether null-space steering actually helps here. The geometric predictor is the piece
  the null-space-steering literature (AlphaSteer / NullSteer) skips. Use `use_judge` for credible
  collateral (string-matching manufactured a false positive in the emotion arc — preflight the judge).
- **`monitor/stream`** runs a residual probe **token-by-token** over a single generation, returning the
  per-step score trajectory and the step at which it first crosses threshold — an online guardrail that
  flags mid-stream rather than after the fact.

## Notes
- This works on the dev backend (CPU, no GPU) for developing your loop, and identically on the
  real model via the Modal `web_gui` — same endpoints, same contract.
- The science scope is SAE-feature steering + concept-manifold steering on Qwen-Scope; see
  `MANIFOLD.md` for what the manifold ops mean and `USER_GUIDE.md` for the human-facing tour.
