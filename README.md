# Qwen Scope Steering GUI

Interactive SAE feature inspection, residual-stream steering, recipe cards, benchmarking, and autopilot search for Qwen Scope models.

The primary target is `Qwen/Qwen3.5-27B` with `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100`. The 27B SAE repo stores one `layer{n}.sae.pt` checkpoint per transformer layer, so this project loads only the requested layer. The cheap/dev path uses `Qwen/Qwen3.5-2B` with `Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100`.

## What It Does

- Inspect token-level SAE feature activations for one prompt.
- Compare positive and negative prompts using max activation contrast.
- Generate paired unsteered and steered text by adding `strength * W_dec[:, feature_id]` to a selected layer residual stream.
- Benchmark steering recipes against prompt-only and control baselines.
- Search candidate features from user-provided positive and negative behavior examples.
- Export reproducible recipe cards as JSON and Markdown.
- Keep a small local JSON feature notebook.
- Request optional speculative feature labels when a model API key is configured.
- Show loaded config, cache state, device, dtype, and GPU memory.

Core inspect, compare, steer, bench, autopilot, and recipe export flows do not require hosted model APIs.

## Getting Started

**Prerequisites:** Python ≥ 3.10 (macOS, Linux, or WSL). **No GPU is required** for the dev path below — it runs a tiny in-memory model on CPU, with no model downloads and no Hugging Face token.

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Run the Lab Bench — fastest path (no GPU, no downloads, no credentials)

```bash
python serve_web.py --dev
```

Open **http://127.0.0.1:7870**. This serves the full web workbench over a tiny CPU model, so the entire interface is explorable in seconds. New here? Follow **`docs/USER_GUIDE.md`** — every feature has a click-along example using the app's pre-filled inputs.

To verify the install at any time:

```bash
pytest
```

### 3. Credentials — only needed for real models or Modal

The dev path needs **no credentials**. To run the real Qwen models (locally with CUDA, or on Modal), create a local `.env`:

```bash
cp .env.example .env
# edit .env — at minimum set HF_TOKEN for model/SAE downloads
```

Supported variables:

```bash
HF_TOKEN=
HUGGINGFACE_HUB_TOKEN=
MODEL_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
MODAL_TOKEN_ID=
MODAL_TOKEN_SECRET=
```

The Hugging Face token is used for model and SAE downloads. Model API keys are optional (speculative feature labels only) and are not used for core steering or benchmark runs. `MODAL_TOKEN_*` are only needed for the Modal commands below.

### 4. Run a real model

- **Local (needs CUDA):** `python serve_web.py --config configs/qwen35_2b_dev_l0_100.yaml` (see the Lab Bench section for flags).
- **No local GPU?** Serve the real model on Modal instead — see the [Modal](#modal) section (`modal serve modal_app.py` → the `web_gui` endpoint).

## Classic GUI (Gradio)

The original tabbed Gradio app — an alternative to the Lab Bench web UI below. It opens on **http://127.0.0.1:7860** (flags: `--server-name`, `--server-port`, `--share`).

```bash
python app.py --config configs/qwen35_2b_dev_l0_100.yaml
```

The app exposes:

- `Inspect prompt features`
- `Compare prompts`
- `Steer generation`
- `Bench`
- `Autopilot`
- `Recipe Library`
- `Feature notebook`
- `System/status`

The real model/SAE path loads lazily when an action button is pressed. On a machine without CUDA, use the Modal smoke commands.

## Lab Bench (web UI)

`serve_web.py` serves a richer single-page workbench (`web/`) over a thin FastAPI layer (`qwen_scope_steering_gui/web_api.py`) wrapping the same `SteeringService`. It organizes work as one loop — **Explore → Steer → Measure → Manifold → Monitor → Control → Library** — with a token-activation microscope, a cross-prompt feature atlas (scan a prompt corpus; search/sort by peak or breadth, see saved labels), a contrast lens, a steering studio (dial, live before/after, strength sweep), a live seven-control benchmark verdict board (save passing runs as recipe cards), an autopilot discovery run (examples → candidate search → benchmark → saved recipe), and the recipe gallery. A feature you pin in Explore carries into Steer and Measure. The **Manifold** mode is a first-class workflow: **fit** a concept's residual-stream manifold (per-value residual centroids → PCA → spline; the fit layer defaults to the concept's atlas-derived `best_layer`) and render its 3D geometry (Three.js), **steer** along it by traversing the curve/ring via a paper-faithful *replace* intervention (arXiv 2604.28119 / 2605.05115), **compare** manifold vs. linear steering (with a perplexity/naturalness badge — manifold stays on-manifold and more fluent), run a **pullback** (optimize the activation that induces a target behavior), inspect **SAE coverage** (color the 3D points by the dominant SAE atom that tiles the manifold; click to pin), and **save the result as a recipe** (after a pullback, a manifold steer can be stored as a Library recipe just like a feature steer). The science behind this mode is documented in `docs/MANIFOLD.md`. The **Monitor** mode is the bench's detection half (control's complement): give labeled positive/negative examples of a behavior (refusal, PII, sycophancy…) and it finds the SAE feature(s) that best separate them, evaluates the detector on a **held-out split** with a **random-feature control** as the validity gate, lets you flag new text, and saves validated detectors to a monitor gallery — a cheap, interpretable runtime guardrail. The **Control** mode joins the two halves into an honest AI-control loop and answers the questions the field is actually stuck on: a **baseline shootout** (`POST /api/monitor/shootout`) — does the interpretable SAE-feature monitor actually beat a raw-residual linear probe (diff-of-means + logistic) and the random control? reported with a **TPR-at-fixed-FPR** operating point, the way a deployed monitor is tuned; a **robustness** check (`POST /api/monitor/robustness`) — does the detector survive a paraphrase shift, or did it memorize its training distribution?; and the **closed loop** (`POST /api/control_loop`) — discover a monitor, suppress the behavior by steering the detector's *own* top feature, re-score every generation, and measure **collateral damage** (`POST /api/collateral`): perplexity on neutral text *and* safety regression on held-out harmful prompts (the "Rogue Scalpel" check, arXiv 2509.22067). The loop reports `validated` only if the behavior was actually present, the steer removed it, **and** nothing broke — perfect suppression that lobotomizes the model or erodes its refusals is honestly marked `benchmarked`. Two further probes turn the post-mortem into prediction and prevention: **safety geometry** (`POST /api/safety_geometry`) tests whether the cosine between a behavior's probe and the refusal probe *predicts* the collateral of steering it, then re-measures that collateral with the steer projected **orthogonal to the refusal direction** — the geometric predictor the null-space-steering literature (AlphaSteer / NullSteer) skips; and the **streaming detector** (`POST /api/monitor/stream`) runs a residual probe **token-by-token** over a generation, reporting the step at which it first crosses threshold — an online guardrail that flags mid-stream rather than after the fact. And **jailbreak detection** (`POST /api/jailbreak_detection`) points the free probe at the industry's #1 detection target: it runs the shootout (probe vs SAE vs paid judge) on jailbreak / prompt-injection prompts, then tests whether a probe trained on one set of attack families (DAN, instruction-override, dev-mode) still flags **held-out families** it never saw (grandma exploit, base64 obfuscation, prefix-injection) — reporting `deployable` only if the probe detects, generalises, **and** matches the judge, the honest bar that separates a real guardrail from template memorisation.

```bash
# Fastest: GPU-free dev backend (tiny in-memory model, no downloads, no token):
python serve_web.py --dev

# Real model locally (needs CUDA):
python serve_web.py --config configs/qwen35_2b_dev_l0_100.yaml   # 2B
python serve_web.py --config configs/qwen35_27b_l0_100.yaml      # 27B

# Flags: --host 0.0.0.0   --port 8000   --recipes path/to/recipes
```

Then open **http://127.0.0.1:7870** (override with `--host` / `--port`). JSON endpoints live under `/api/*` (`/api/docs` for the schema). **No local GPU?** Serve the real model on Modal instead — see the [Modal](#modal) section's `web_gui` endpoint. The dev backend exercises the real activation/contrast/steering code paths on a fake CPU model, so it proves wiring without a GPU; switching to a real config is the only change needed. The classic Gradio app (`app.py`) is unchanged and still available.

## Documentation

- `docs/USER_GUIDE.md`: how to use the Lab Bench workbench (Explore → Steer → Measure → Manifold → Library).
- `docs/MANIFOLD.md`: technical deep-dive on concept-manifold steering, including the science (isometry, pullback, and the negative results behind the design).
- `docs/AGENT_RESEARCH.md`: how an **AI agent** conducts research here — the async job API (`POST /api/jobs` → poll), the experiment log (`/api/experiments`), and the honesty contract (read `validation_decision`; report negatives). Endpoints are serialized on the GPU so concurrent calls queue instead of failing.

## Recipe Cards

A `FeatureRecipe` captures the target behavior, model/SAE metadata, layer, feature id, strength, discovery examples, validation prompts, benchmark/control results, before/after examples, side effects, limitations, artifacts, and provenance.

Recipe statuses are:

```text
draft
candidate
benchmarked
validated
failed
blocked
```

`validated` is intentionally strict. A recipe can only reach it when held-out benchmark prompts and all required controls pass the conservative validation gates.

Recipes are stored locally:

```text
recipes/
  recipe_id/
    recipe.json
    recipe.md
    benchmark_results.json
    examples.jsonl
```

Useful commands:

```bash
python scripts/create_recipe.py --config configs/fake_test.yaml --target-behavior concise_answers --layer 1 --feature-id 2 --strength 4.0
python scripts/export_recipe_markdown.py --recipe recipes/concise_answers_l1_f2_v1/recipe.json
```

## Steering Bench

Bench answers whether a recipe improves a target behavior compared with baselines and controls. Every benchmark compares:

```text
unsteered_baseline
prompt_only
steering_only
prompt_plus_steering
zero_strength_control
random_feature_control
negative_strength_control
```

Prompt-only comparison is a first-class baseline. A prompt-only instruction such as:

```text
Answer concisely in no more than two sentences. Prompt: {prompt}
```

is applied without steering for `prompt_only`, and combined with steering for `prompt_plus_steering`.

Rule metrics include length, sentence count, repetition, distinct n-grams, JSON validity, required/forbidden terms, max/min length pass flags, generation errors, latency, hook firing, hidden/logit deltas, explicit control deltas, and coherence proxies. Model-judge scoring is disabled by default and remains optional.

## Autopilot

Autopilot takes a target behavior plus positive and negative examples, searches candidate features from activation contrast, ranks candidates, runs a small benchmark, runs a strength sweep that includes `0.0`, and writes a candidate recipe card with evidence and caveats.

Example local fake-backend smoke:

```bash
python scripts/autopilot_recipe.py \
  --config configs/fake_test.yaml \
  --target-name json_validity \
  --target-description "Produce strict valid JSON without markdown or prose." \
  --positive-examples data/examples/json_positive.txt \
  --negative-examples data/examples/json_negative.txt \
  --validation-prompts data/prompt_sets/json_validity_dev.jsonl \
  --candidate-layers 0,1 \
  --candidate-count 3 \
  --objective maximize_json_validity \
  --output-dir recipes/json_validity_fake_smoke \
  --fake-backend
```

## Local Checks

```bash
pytest
python -m compileall -q qwen_scope_steering_gui app.py modal_app.py scripts
```

## Modal

`modal_app.py` defines:

- `smoke_2b`: L4, short 2B activation and steering smoke.
- `smoke_27b`: H100, one-layer 27B activation and steering smoke.
- `bench_smoke_2b`: L4, compact real 2B benchmark recipe smoke.
- `autopilot_smoke_2b`: L4, compact real 2B autopilot smoke.
- `bench_smoke_27b`: H100, compact real 27B benchmark recipe smoke.
- `autopilot_smoke_27b`: H100, compact real 27B autopilot smoke.
- `gradio_gui`: the classic tabbed Gradio app. Defaults to the 2B dev config on L4.
- `web_gui`: the Lab Bench web UI over the same backend, same `QWEN_GUI_TARGET` selection.

Commands:

```bash
modal run modal_app.py::smoke_2b
modal run modal_app.py::smoke_27b
modal run modal_app.py::bench_smoke_2b
modal run modal_app.py::autopilot_smoke_2b
modal run modal_app.py::bench_smoke_27b
modal run modal_app.py::autopilot_smoke_27b
modal run modal_app.py::web_parity_2b
modal run modal_app.py::manifold_steer_demo_2b
modal run modal_app.py::manifold_vs_linear_2b
modal run modal_app.py::manifold_atlas_2b
modal serve modal_app.py
```

`web_parity_2b` is a bounded L4 job that exercises every Lab Bench web endpoint against the real 2B and exits (no warm GPU) — use it to confirm the `web_gui` reaches parity with the real model.

`manifold_steer_demo_2b` (L4) / `manifold_steer_demo_27b` (A100-80GB) power the **Manifold** mode: they fit a concept's activation manifold and steer along it on the real model, printing the behavioral trajectory. (`residual_manifold_sweep_2b/_27b` locate the layer where concept manifolds peak.) The dead-end geometry-investigation probes (`latent_map_*` / `coact_map_*` / `manifold_probe_*` / `residual_manifold_*` / `manifold_vs_linear_probe_*`) are no longer live — their source is archived in `archive/research_probes.py` with the full negative-result writeup in `archive/README.md`.

`modal serve modal_app.py` registers two GUI endpoints — `gradio_gui` and `web_gui` — sharing one warm container and the same target. The default is the cheap 2B L4 target. To choose another target, set `QWEN_GUI_TARGET` before serving:

```bash
QWEN_GUI_TARGET=27b-a100 modal serve modal_app.py
QWEN_GUI_TARGET=27b-h100 modal serve modal_app.py
```

Supported target names are `2b-l4`, `27b-a100`, and `27b-h100`. Aliases include `2b`, `27b`, `a100`, and `h100`; `27b` maps to the lower-cost A100 80GB path.

The GUI endpoint keeps one container warm for five idle minutes so model state survives normal tab-to-tab interaction. Stop the Modal app when you are done to shut the GPU down immediately.

The app mounts a persistent Modal Volume at `/cache` so Hugging Face artifacts can be reused.

## Failure Modes

- A real hook and hidden/logit delta does not guarantee a visible text change on short smoke prompts.
- Random-feature or prompt-only controls can match or beat steering. In that case, the recipe remains `benchmarked` or `candidate`, not `validated`.
- Cold 27B Modal GUI loads can take several minutes.
- Local 2B and 27B configs require adequate GPU memory. Use Modal when CUDA is unavailable locally.
- Never claim 27B support without a successful Modal run.

## License

Licensed under the Apache License 2.0 — see [LICENSE](LICENSE).
