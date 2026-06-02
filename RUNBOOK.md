# Runbook

## Orientation

This runbook is the operational command reference. See `README.md` for setup and the project tour, and `docs/AGENT_RESEARCH.md` for the API and research workflow (job API, experiment log, honesty contract).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"                 # GPU-free dev + the test suite
# On an Apple Silicon Mac, add the on-device backend: pip install -e ".[mlx]"
```

The base install is slim (torch + the web layer); the heavy/cloud deps are opt-in extras
(`.[mlx]` on a Mac, `.[cuda]` for a CUDA GPU, `.[cuda,modal]` to drive `modal_app.py`, `.[all]` for everything).

## Credentials

```bash
cp .env.example .env
# edit .env
```

Use `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` for Hugging Face downloads. Modal can read `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` from the same `.env`.

Never print `.env`, serialize secrets into recipe cards, or commit credentials.

## Unit Tests

```bash
pytest
python -m compileall -q qwen_scope_lab app.py modal_app.py scripts
```

## Local GUI

```bash
python app.py --config configs/qwen35_2b_dev_l0_100.yaml
```

Open the printed Gradio URL. The 2B model loads lazily on first model-backed action. If the machine has no suitable CUDA device, use the Modal commands below.

## Lab Web UI

The Lab is a richer single-page workbench (`web/`) served over FastAPI (`qwen_scope_lab/web_api.py`) wrapping the same `SteeringService`.

```bash
# GPU-free dev backend (tiny CPU model, no downloads) -- verify the UI/wiring locally:
python serve_web.py --dev

# Apple Silicon, local, no Modal/CUDA -- the FULL bench on the real 2B + its SAE via MLX (see docs/MLX.md):
python serve_web.py --mlx                       # bare: the default 2B (instruct) + its SAE, on-device
python serve_web.py --mlx-base                   # the base model the SAE was trained on (SAE/manifold fidelity)
python serve_web.py --mlx --mlx-sae none        # skip the SAE download (probe + steering + manifold only)
# (--mlx <repo> overrides the model; --mlx-layer N sets the probe/capture layer;
#  first run downloads model ~4.5GB + SAE ~540MB, then cached)

# real model paths (need CUDA):
python serve_web.py --config configs/qwen35_2b_dev_l0_100.yaml    # instruct 2B (the behavioral demos)
python serve_web.py --config configs/qwen35_2b_base_l0_100.yaml   # base 2B (matches the base-trained SAE)
python serve_web.py --config configs/qwen35_27b_l0_100.yaml
```

Defaults to `http://127.0.0.1:7870`; override with `--host/--port`. JSON API under `/api/*`, schema at `/api/docs`. The `--dev` backend (`qwen_scope_lab/dev_backend.py`) runs the real activation/contrast/steering code paths against a fake CPU model, so its generations are intentionally toy but the wiring is real. The `--mlx` backend (`qwen_scope_lab/mlx_backend.py`) runs the **real** 2B + SAE on-device on Apple Silicon — bf16 (not 4-bit) for fidelity; results replicate the Modal/CUDA findings qualitatively, not bit-for-bit. Web-API tests: `pytest tests/test_web_api.py`; MLX tests: `pytest tests/test_mlx_backend.py` (the real-model layer skips unless `mlx_lm` + a cached model are present).

## Core Scripts

```bash
python scripts/inspect_features.py --config configs/qwen35_2b_dev_l0_100.yaml --prompt "The capital of France is" --layer 12
python scripts/steer_once.py --config configs/qwen35_2b_dev_l0_100.yaml --prompt "Write one sentence about Paris." --layer 12 --auto-feature --strength 5.0 --max-new-tokens 16
python scripts/compare_prompts.py --config configs/qwen35_2b_dev_l0_100.yaml --positive-prompt "Write a concise answer." --negative-prompt "Write a long story." --layer 12
```

## Recipe Commands

Create a manual recipe:

```bash
python scripts/create_recipe.py \
  --config configs/qwen35_2b_dev_l0_100.yaml \
  --target-behavior concise_answers \
  --target-description "Make answers shorter and more direct while preserving correctness." \
  --layer 12 \
  --feature-id 963 \
  --strength 4.0
```

Benchmark a recipe:

```bash
python scripts/bench_recipe.py \
  --recipe recipes/concise_answers_l12_f963_v1/recipe.json \
  --prompt-set data/prompt_sets/concise_dev.jsonl \
  --prompt-only-instruction "Answer concisely in no more than two sentences. Prompt: {prompt}" \
  --max-new-tokens 16 \
  --temperature 0.0 \
  --objective minimize_length_without_empty_output \
  --save-recipe
```

Run autopilot:

```bash
python scripts/autopilot_recipe.py \
  --config configs/qwen35_2b_dev_l0_100.yaml \
  --target-name json_validity \
  --target-description "Produce strict valid JSON without markdown or prose." \
  --positive-examples data/examples/json_positive.txt \
  --negative-examples data/examples/json_negative.txt \
  --validation-prompts data/prompt_sets/json_validity_dev.jsonl \
  --candidate-layers 12 \
  --candidate-count 3 \
  --objective maximize_json_validity \
  --output-dir recipes/json_validity_2b_candidate \
  --prompt-only-instruction "Return only strict valid JSON. Do not include markdown or prose. Prompt: {prompt}" \
  --max-new-tokens 8 \
  --temperature 0.0
```

Export Markdown:

```bash
python scripts/export_recipe_markdown.py --recipe recipes/json_validity_2b_candidate/recipe.json
```

## Modal Cost Preflight

> **On an Apple Silicon Mac, the 2B no longer needs Modal.** Run `serve_web.py --mlx` (above /
> `docs/MLX.md`) for the full bench on-device — no GPU billing, no cold starts, offline. Modal
> stays the path for the **27B** (won't fit a laptop) and for a **shareable, always-on hosted
> demo** (`modal serve` → a public `web_gui` URL). Posture: 2B local-first, 27B + hosting on Modal.

Primitive: Modal Functions plus one Modal web server Function.

GPU choices:

- `smoke_2b`: `L4`, timeout 3600 seconds, retries 0, `/cache` Volume.
- `smoke_27b`: `H100`, timeout 5400 seconds, retries 0, `/cache` Volume.
- `bench_smoke_2b`: `L4`, timeout 3600 seconds, retries 0, `/cache` Volume.
- `autopilot_smoke_2b`: `L4`, timeout 3600 seconds, retries 0, `/cache` Volume.
- `bench_smoke_27b`: `H100`, timeout 5400 seconds, retries 0, `/cache` Volume.
- `autopilot_smoke_27b`: `H100`, timeout 5400 seconds, retries 0, `/cache` Volume.
- `gradio_gui` and `web_gui`: both selected by `QWEN_GUI_TARGET`, timeout 7200 seconds, `/cache` Volume, one Modal container with concurrent ASGI request handling, 300 second scaledown window. `gradio_gui` is the classic tabbed app; `web_gui` is the Lab web UI over the same backend. Default target is `2b-l4` on `L4`; explicit targets are `27b-a100` on `A100-80GB` and `27b-h100` on `H100`.

Each smoke is intentionally short, with tiny prompts, low `max_new_tokens`, and one active SAE layer. Stop unexpected long-running apps with the Modal dashboard or:

```bash
modal app stop qwen-scope-lab
```

## Modal Commands

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
modal run modal_app.py::residual_manifold_sweep_2b
modal run modal_app.py::monitor_demo_2b
modal run modal_app.py::control_loop_demo_2b
modal serve modal_app.py
```

`web_parity_2b` (L4, timeout 3600s, `/cache` Volume) loads the real 2B and drives every Lab web endpoint through a FastAPI `TestClient`, prints a JSON parity summary, and exits — a one-shot, so no app is left warm.

**Manifold steering** (the `Manifold` GUI mode) — LIVE probes, still in `modal_app.py`: `manifold_steer_demo_2b` (L4, layer 14) / `manifold_steer_demo_27b` (A100-80GB, layer 48) fit a concept's activation manifold (`SteeringService.manifold_fit`: residual centroids → PCA → spline) and steer along it (`manifold_steer`: replace the concept token's residual with manifold points across waypoints), printing the behavioral trajectory. `residual_manifold_sweep_2b/_27b` locate the layer where concept manifolds peak (number-line robust everywhere; day-ring cleanest at 2B L14–16 / 27B L48). `manifold_vs_linear_2b` prints manifold-path vs linear-path steering with perplexity (concept-dependent on raw perplexity; manifold wins on behavior-manifold energy). `manifold_naturalness_probe_2b` and `manifold_pullback_probe_2b` back the fluency and pullback findings. `manifold_atlas_2b/_27b` census ~17 candidate continuous concepts → per-concept best layer + metric + verdict (`/api/manifold/compare` and `/api/manifold/sae_coverage` back the GUI's compare + SAE-tiling views). One-shots; no warm GPU. A manifold steer can also be saved as a Library recipe: `POST /api/recipes` with body `{"kind":"manifold"}` after running a pullback (recipe `kind` is `"feature"` or `"manifold"`).

**Monitoring + Control** (the `Monitor` / `Control` GUI modes) — LIVE probes in `modal_app.py`: `monitor_demo_2b` (L4, layer 12) discovers feature-based detectors for sentiment/refusal/PII and prints held-out AUC/F1 + the random-feature control + verdict. `control_loop_demo_2b` (L4, layer 12) runs the whole honest-control story on **sycophancy** in one model load: (1) the **baseline shootout** (`monitor_shootout` — SAE-feature monitor vs. raw-residual linear probe vs. random control, with TPR@FPR), (2) **robustness** (`monitor_robustness` — clean → paraphrase shift, AUC drop), and (3) the **closed loop** across a suppression-strength sweep (`control_loop` — suppress the detector's own top feature, re-score, and measure collateral damage via `collateral`: neutral-text perplexity + safety regression on held-out harmful prompts, the "Rogue Scalpel" check, arXiv 2509.22067). Prints `any_clean_suppression` — whether any strength suppressed sycophancy *without* breaking fluency or safety. One-shots; no warm GPU. Read the verdicts honestly: a `PROBE WINS` shootout or a 100%-suppressed-but-`benchmarked` loop is a real finding, not a failure.

**Geometry-investigation harness** — REMOVED. The dead-end exploration that led to manifold steering (`latent_map_2b/_27b` decoder-cosine layout — DEAD; `coact_map_2b` co-activation layout — DEAD; `manifold_probe_2b` Ising conditional-coupling — DEAD; `residual_manifold_2b`; `manifold_vs_linear_probe_2b`) has been removed from `modal_app.py` and the public tree (preserved in git history). The full negative-result writeup is in `docs/MANIFOLD.md` §2.

## Modal GUI Choices

Start the default 2B L4 GUI:

```bash
modal serve modal_app.py
```

Two GUI web endpoints are registered per serve process — `gradio_gui` (classic tabbed app) and `web_gui` (Lab) — sharing one warm container and the same target. Choose the target explicitly with `QWEN_GUI_TARGET`:

```bash
QWEN_GUI_TARGET=27b-a100 modal serve modal_app.py
QWEN_GUI_TARGET=27b-h100 modal serve modal_app.py
```

Supported target names:

- `2b-l4`: cheapest GUI path, 2B model on L4. This is the default.
- `27b-a100`: lower-cost real 27B path, 27B model on A100 80GB.
- `27b-h100`: fastest/headroom real 27B path, 27B model on H100.

Aliases are accepted: `2b`, `l4`, `dev`, `27b`, `a100`, `a100-80gb`, and `h100`. The `27b` alias maps to `27b-a100`.

Closing the browser tab does not stop the Modal app. Stop the server with `Ctrl-C` in the `modal serve` terminal or:

```bash
modal app stop qwen-scope-lab
```

The GUI function intentionally uses a 300 second `scaledown_window` so the model stays loaded while moving between tabs during an active session. This means the selected GPU can remain allocated for up to about five idle minutes after the last request unless you stop the app explicitly.

The helper wrapper scripts call the same Modal functions:

```bash
python scripts/modal_bench_smoke_2b.py
python scripts/modal_autopilot_smoke_2b.py
python scripts/modal_bench_smoke_27b.py
python scripts/modal_autopilot_smoke_27b.py
```

Add `--dry-run` to any helper script to print the underlying `modal run` command without starting a GPU job.

## Expected Bench Proof

Benchmark JSON should show:

- All seven required methods.
- `prompt_only` and `prompt_plus_steering` using the configured instruction.
- `zero_strength_control`, `random_feature_control`, and `negative_strength_control`.
- Per-prompt outputs and aggregate metrics.
- A `validation_decision` explaining why the recipe is or is not validated.
- Hook evidence for steering methods: `hook_fired: true` and positive hidden delta.

## Expected 27B Proof

The compact 27B smoke JSON should show:

- Model `Qwen/Qwen3.5-27B`.
- SAE `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100`.
- One active layer, usually layer `32` for smoke runs.
- A selected feature id.
- `hook_fired: true`.
- `hidden_delta_norm > 0`.
- Recipe paths under `/root/recipes/...` for Modal recipe-card smokes.

## Blocked Handling

If Hugging Face auth, model access, Modal credentials, GPU availability, OOM, dependency install, or hook incompatibility blocks the run, record the failed command, traceback excerpt, completed work, and smallest next action.
