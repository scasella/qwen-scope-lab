# Running the Lab locally on Apple Silicon (MLX)

The whole bench — model **and** SAE, every GUI mode, the agent API — runs on an Apple
Silicon Mac via [MLX](https://github.com/ml-explore/mlx), with **no Modal and no CUDA**.
The 2B is local-first: private, offline, free, and instant to iterate on. Modal stays for
the 27B (which won't fit a laptop) and for shareable hosted demos.

This is the third backend behind the same `SteeringService`, alongside the **dev** backend
(tiny CPU stand-in, for wiring/tests) and the **CUDA/Modal** backend (real GPU). It is
selected with `serve_web.py --mlx` (or `mlx_backend.build_mlx_service(...)` in code).

---

## TL;DR

```bash
# The FULL bench — the real 2B + its SAE, on-device. No other args needed:
python serve_web.py --mlx

# The base model the SAE was actually trained on (most faithful for SAE-feature / manifold work):
python serve_web.py --mlx-base

# Probe-only (skips the SAE download): detection + steering + manifold, no SAE-feature path:
python serve_web.py --mlx --mlx-sae none
```

Open the printed URL. The header shows `LIVE · Qwen3.5-2B-bf16`. That's the complete
Explore → Steer → Measure → Manifold → Monitor → Control → Library loop, on-device.

Requirements: an Apple Silicon Mac and `pip install -e ".[mlx]"` — that adds `mlx` + `mlx-lm`
to the slim base install (torch ships in the base because the Qwen-Scope SAE is a torch `.pt`).
The first run downloads the model (~4.5 GB bf16) and the SAE (~540 MB per layer); both are
cached after that. (Pass `--mlx-sae none` to skip the SAE download.)

---

## What runs on MLX

**Everything the GUI exposes.** Confirmed working on the real Qwen3.5-2B, end to end:

| GUI mode | What runs on MLX |
| --- | --- |
| **Explore** | `inspect` (per-token SAE features), `atlas`, `compare` |
| **Steer** | SAE-feature steering (`steer`), CAA/direction steering (`steer_direction`), `sweep`, the logit-effect metric |
| **Measure** | the 7-control `benchmark`, `autopilot` (they orchestrate `steer` + `generate`) |
| **Manifold** | `fit` (residual PCA + spline), `steer`/`compare` (position-replace intervention), `pullback` (**gradient optimisation through the model**), `sae_coverage` |
| **Monitor** | `monitor_discover`/`score`/`shootout`/`robustness` |
| **Control** | `control_loop`, `collateral`, the full jailbreak suite (`detection`/`hardening`/`screen` + the `/demo` page) |
| **Library** | recipes (no model) |

The probe/detection paths (probes, the jailbreak suite, `/demo`) run **without** an SAE.
The SAE-feature paths (SAE monitor/atlas/coverage, SAE-feature steering, the SAE arm of the
shootout) need `--mlx-sae`.

### The one caveat: run bf16, not 4-bit

Use a **bf16** (or fp16) build — `mlx-community/Qwen3.5-2B-bf16` — not a quantized one.
Quantization shifts activations, which changes which SAE feature fires and where a probe's
threshold sits. For an interpretability bench that makes claims about *which feature*, that
fidelity matters. A 2B in bf16 is ~4 GB and fits comfortably on a 16 GB+ Mac.

---

## How it works (architecture)

The bench was built so a backend can be **injected** behind one interface. `dev_backend.py`
already does this: it builds a real `SteeringService` whose `ModelBundle` and SAE loader are
tiny CPU stand-ins, and "nothing else changes." `mlx_backend.build_mlx_service()` is the same
move with a real MLX model.

The one wrinkle versus the dev backend: the dev backend keeps the **torch runtime** (its fake
model is still a `torch.nn.Module`, so it reuses the torch forward hooks). MLX swaps the
**runtime**, so the model-touching primitives are reimplemented in MLX. Every one branches on
a single duck-typed flag — `is_mlx_runtime` — at exactly one point, so **`service.py` carries
no MLX import and the torch/CUDA path is byte-for-byte unchanged.**

```python
# the pattern, everywhere a primitive touches the model:
bundle = self.ensure_model()
if getattr(bundle.model, "is_mlx_runtime", False):
    return bundle.model.<mlx primitive>(...)
import torch
# ... the original torch path, untouched ...
```

`MlxModel` (in `mlx_backend.py`) sits in the `ModelBundle.model` slot and provides the
primitives the service calls:

| Primitive | Replaces (torch) | Used by |
| --- | --- | --- |
| `pooled_residual(text, layer)` | `register_capture_hook` + forward | probes, monitors, jailbreak, `/demo` |
| `last_residual(text, layer)` | last-token capture | the manifold fitter |
| `inspect(prompt, sae, …)` | `extract_prompt_features` | SAE monitor/atlas/coverage, shootout SAE arm |
| `generate(prompt, …)` | `generate_text` (HF `.generate`) | all generation; reuses **mlx-lm's** generate (handles the hybrid cache) |
| `install_steer(layer, vec, k, trace)` | `register_steering_hook` | CAA/direction steering, collateral, control |
| `install_replace(layer, vec, pos, trace)` | `register_replace_hook` | manifold steer/compare/pullback |
| `perplexity(prompt, cont, steer=…)` | `sequence_/steered_perplexity` | collateral fluency, naturalness |
| `logits_delta(prompt, layer, vec, k)` | `logits_delta_norm` | the Steer logit-effect metric |
| `last_logits(text, replace=…)` | last-token logits | manifold energy read-out |
| `vocab_size` | `model(…).logits.shape[-1]` | the behaviour manifold |
| `pullback_optimize(…)` | the L-BFGS `_pullback_path` | manifold pullback |

**Capture** is a manual pass over the decoder blocks that grabs (and optionally injects into)
the residual at a layer. **Steering** is a layer swap: `_SteerLayer` / `_ReplaceLayer` wrap a
decoder block to add or overwrite the residual — the exact equivalent of a torch forward hook,
and active automatically inside mlx-lm's own generation. **Pullback** differentiates *through
the model* via `mx.value_and_grad` (the gradient flows via the injection), with **Adam**
standing in for L-BFGS (mlx has no L-BFGS; Adam is scale-invariant, which matters because the
raw gradient is large).

The seam is small: `mlx_backend.py` plus one-line branches in `service.py`, `generation.py`,
and `hooks.py`. Above that line — the 30+ service methods, `baselines`/`probes`/`monitor`/
`manifold` math, `web_api`, the web SPA, the agent job API, the tests — nothing changes.

---

## Fidelity and numerics

MLX results **replicate the Modal/CUDA findings qualitatively, not bit-for-bit.** Different
kernels and accumulation order mean exact AUCs, feature indices, and thresholds differ
slightly. The bench's verdicts are relative and control-gated, so they reproduce — but you
should **re-run and recalibrate, not assume the recorded numbers.**

Worked example (jailbreak shootout, real 2B, via the live GUI): residual probe **AUC 1.00**
beats the SAE feature monitor **0.78** (Modal recorded 0.84); the *finding* — probe > SAE —
reproduces, the second decimal does not.

This dovetails with the bench's own jailbreak-hardening result: the **direction is robust,
the threshold is mobile.** When you switch runtime (or quantize), the discriminative
direction transfers; the operating point should be recalibrated on the target distribution.
A `--mlx` server re-discovers probes on MLX activations automatically, so probe thresholds are
already calibrated to the runtime.

---

## What stays on Modal

- **27B** — won't fit a laptop (bf16 ≈ 54 GB; even 4-bit ≈ 14 GB and slow). Keep it on Modal
  (`QWEN_GUI_TARGET=27b-a100`). The bench already treats 27B as optional/gated.
- **A shareable, always-on hosted demo** — a laptop isn't a public URL. `modal serve` stays
  the "send someone a link" path.

So the posture is **2B local-first on MLX, 27B + hosting on Modal.** See `bench-on-modal`
(skill) / the Modal section of `RUNBOOK.md` for the Modal side.

---

## Models and the SAE

- **Model:** any mlx-lm-supported Qwen build. Prefer a pre-converted bf16 repo
  (`mlx-community/Qwen3.5-2B-bf16`); `mlx-lm` also converts a raw HF repo on load. `build_mlx_service`
  reads `d_model` and `num_layers` from the loaded model, so smaller Qwens work for wiring/iteration
  (e.g. `mlx-community/Qwen2.5-0.5B-Instruct-4bit` for a fast, SAE-less smoke).
- **SAE:** the Qwen-Scope residual SAE, e.g. `Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100`
  (`--mlx-d-sae 32768`). It downloads as a per-layer `layerN.sae.pt`; the existing loader
  fetches and validates it, and `MlxModel` converts the encoder to MLX once and caches it.

### Base vs. instruct — match the model to the SAE

The Qwen-Scope 2B SAE is trained on the **base** model (its card: *"Base model: Qwen3.5-2B-Base"*),
so the most faithful SAE-feature and manifold work runs the base model:

```bash
python serve_web.py --mlx-base     # Qwen/Qwen3.5-2B-Base + the base-trained SAE — the faithful pairing
```

`--mlx` defaults to the **instruct** model (`mlx-community/Qwen3.5-2B-bf16`, a finetune of that base)
on purpose: the lab's headline behavioral demos — jailbreak detection, refusal suppression, the
detect→suppress→prove control loop — only mean something on a model that follows instructions and
refuses. A base model does neither.

Rule of thumb: **`--mlx-base` for SAE-feature / manifold fidelity**, **`--mlx` (instruct) for anything
behavioral.** Both use the same base-trained SAE; only the model changes. No pre-converted
`mlx-community` base build exists yet, so mlx-lm converts the Qwen bf16 weights on first load — same
`qwen3_5` architecture as the instruct build, so the backend's trunk-finding and hybrid masking are
identical. (On CUDA/Modal, the equivalent pairing is `configs/qwen35_2b_base_l0_100.yaml`.)

### How much does the base→instruct mismatch actually cost? (measured)

We measured it rather than assume it — running the **same** base-trained
SAE on both models at L12. The SAE is a **TopK-100** SAE (its true forward,
recovered by picking the convention that best reconstructs the base model: `topk(pre, 100)` with
`pre = x @ Wᵀ_enc + b_enc`, no `b_dec` pre-subtraction — matching `sae_math.py`). On 40 neutral texts:

| metric (L12) | base (faithful) | instruct (default) |
|---|---|---|
| reconstruction FVU ↓ | 0.198 | 0.205 |
| explained variance ↑ | 0.80 | 0.795 |
| reconstruction cosine ↑ | 0.919 | 0.921 |

So the **reconstruction cost of running the SAE on instruct is ~0.7 points of FVU — negligible.** The reason:
the instruct L12 residuals are **0.965 cosine** to the base residuals (fine-tuning moved this layer only
slightly). The lab uses the SAE as an *encoder* (top-k features), not to reconstruct, so the more relevant
number is **feature agreement**: per-token activation cosine **0.94**, and **top-20 feature Jaccard 0.72**
(≈28% of the *exact* top-20 feature identities shift between base and instruct). That 28% churn is the one
real cost — it only matters if you read an **individual** feature's activation on instruct and assume it is
the same feature the SAE learned on base, which is exactly the per-feature / manifold work `--mlx-base` is for.
The behavioral demos don't read individual base-feature identities, so the instruct default is fine for them.

We also re-ran the jailbreak **SAE-feature-vs-residual-probe shootout** on both models (same code path). On the
faithful base pairing the SAE feature improves (AUC **0.78 → 0.84**) and the residual probe's lead narrows
(**0.22 → 0.16**) — but the **raw-residual probe still wins (AUC 1.00 on both)**. So "a cheap probe beats the
SAE feature" was mildly *conservative* on instruct (the SAE was off-distribution), not wrong: it survives the
fair base-model test. (Banks are 8/8 → AUC granularity ~0.06; read the margins, not third decimals.)

### Hybrid architecture note

Qwen3.5 (`qwen3_5`) is a **hybrid**: most decoder blocks are linear-attention (GatedDeltaNet,
which need an SSM mask) and every Nth is full-attention (causal mask). The backend mirrors the
model's own forward and picks the right mask per layer. It also locates the decoder trunk **by
shape** (it lives under a `language_model.model` wrapper for `qwen3_5`, vs `model` for
`qwen2`), so the same code works across Qwen variants.

---

## Performance (M4 Pro, 24 GB, bf16)

- Model load: ~3 s (cached).
- Activation capture: ~16 ms per prompt on the 2B (`/demo` screens at this rate).
- The SAE-feature shootout (16 examples) and a monitor discovery: seconds.
- Manifold pullback: a gradient loop (waypoints × iters backward passes through the 2B) —
  the slowest GUI action, tens of seconds to a couple of minutes; it's a one-off.

---

## Troubleshooting

- **`'TokenizerWrapper' object is not callable`** — a code path called the tokenizer in the
  torch style, `tokenizer(text, return_tensors="pt")["input_ids"]`. The mlx-lm tokenizer isn't
  callable that way; use `tokenizer.encode(text)` (both backends support it) or branch the
  method to an `MlxModel` primitive. This was the failure mode the full-GUI walk caught in the
  manifold fitter.
- **SAE download fails with a symlink `FileNotFoundError`** — usually a broken partial cache
  from an interrupted download. `rm -rf ~/.cache/huggingface/hub/models--Qwen--SAE-Res-*` and
  retry. (`build_mlx_service` already uses the *resolved* `HF_HUB_CACHE`; an unexpanded
  `~/.cache/...` path is what originally caused this.)
- **Quantized/odd SAE results** — you're probably on a 4-bit model. Switch to a bf16 build.
- **Out of memory** — close other apps, or use a smaller model for iteration; a 2B bf16 needs
  ~4 GB resident plus the SAE (~0.5 GB when `--mlx-sae` is on).

---

## Extending the backend (for contributors)

If you add a **new model-touching method** to `service.py` (capture, generate, a hook), it
will NOT run on MLX until you branch it. The recipe:

1. Add the MLX primitive to `MlxModel` (capture-based primitives can reuse `_forward_capture`).
2. Branch the service/generation/hooks function on `is_mlx_runtime` at one point, delegating to
   the primitive; leave the torch path untouched below.
3. If it only touches the **tokenizer**, prefer switching to `.encode()` over branching — both
   backends support it (and `_DevTokenizer` now does too).
4. **The completeness check is a live GUI walk, not a unit test.** A per-method port can miss
   sibling helpers (the manifold fitter's `_capture_last_residual` / `_output_distribution`
   were missed until a Playwright walk of every mode on the real 2B surfaced the 500). Serve
   `--mlx`, click through every mode, and watch for `500`s and console errors.

## Tests

- `tests/test_mlx_backend.py` has a **CI-safe** layer (a stub with `is_mlx_runtime=True`
  exercises every service branch with no MLX installed) and a **real** layer (builds an MLX
  service and runs capture/generate/steer/inspect/manifold/pullback) that **skips** unless
  `mlx_lm` and a cached model are present — so it runs on Apple Silicon and is inert in CI / on
  Linux.
- The rest of the suite runs on the dev backend, unchanged.
