---
name: bench-on-modal
description: >
  Run the Qwen Scope "Lab Bench" in this repo, or its probes, on the REAL Qwen model via Modal —
  safely and cheaply. Use this whenever the user wants real-model results (not the dev backend),
  wants to serve the live web GUI, asks to run a Modal probe (manifold / monitor / benchmark on the
  real 2B or 27B), wants the `web_gui` / `gradio_gui` URL, or mentions Modal / GPU / "the real model"
  for this project. It encodes the project's hard operational rules — dev backend first; NEVER
  `modal deploy`; `modal run ::fn` for bounded probes vs `modal serve` for the GUIs; ALWAYS stop the
  `qwen-scope-steering-gui` app to halt billing; 2B-on-L4 by default; cost-preflight from RUNBOOK.
  NOTE: the real 2B now runs locally on Apple Silicon via MLX (`serve_web.py --mlx`, no Modal/CUDA) —
  prefer that for 2B work on a Mac; use Modal for the 27B, a hosted/shareable demo, a CUDA run, or a
  recorded gated probe (see `docs/MLX.md`).
  Consult it before issuing any `modal` command for this repo. (Defers generic Modal mechanics to
  the `modal-gpu` skill; this is the project-specific discipline.)
  Triggers on: "run on the real model / real 2B / 27B", "serve the web_gui (or gradio_gui)", "modal
  run <probe>", "run monitor_demo_2b / the pullback probe / web_parity on the real model", "real
  results, not the dev backend", "stop the modal GPU", "which modal function validates X".
---

# Running the bench on the real model (Modal — and MLX for the 2B)

The real **27B** is GPU-only and, in this environment, **Modal-only** — there is no local CUDA for it.
The real **2B**, however, now runs **locally on Apple Silicon via MLX** with no Modal at all. Reach
for Modal when you need the **27B**, a **shareable hosted demo** (`modal serve` → a public `web_gui`
URL), or a CUDA run. GPU time costs money, and the two ways to waste it are (a) leaving a container
warm and (b) deploying something persistent. This skill is the discipline that prevents both.
`RUNBOOK.md` is the authoritative command + cost reference; read it for the GPU/timeout per function.

## Dev backend first

Before reaching for Modal, ask whether the dev backend suffices. `python serve_web.py --dev` runs
the **real code paths** on a tiny in-memory CPU model — no GPU, no downloads, no token. Use it for
wiring, UI work, and tests. It shows *mechanics*, not real results (the model is random, so verdicts
land `BENCHMARKED`). Switch to a real backend only when you genuinely need real-model behavior.

## On a Mac, MLX before Modal (for the 2B)

If you need real **2B** results and you're on Apple Silicon, run it **locally via MLX** before
reaching for Modal: `python serve_web.py --mlx mlx-community/Qwen3.5-2B-bf16 --mlx-sae
Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100 --mlx-d-sae 32768`. The **whole bench** (every GUI mode + the
agent `/api/*`) runs on-device — no GPU billing, no cold load, offline. Use bf16 (not 4-bit) for
fidelity; results replicate the Modal findings qualitatively, not bit-for-bit (re-run + recalibrate).
Use Modal for the 2B only when you specifically need a CUDA run or a public hosted URL. The gated
Modal probes (`*_2b`) are still the way to produce a recorded, citable real-GPU result; MLX is for
interactive local work and the GUI. Full details: `docs/MLX.md`.

## The rules (the reason this skill exists)

- **NEVER `modal deploy`.** Persistent deployment is out of scope for this project and has been
  blocked before. Don't propose or run it.
- **`modal run modal_app.py::<fn>`** — a one-shot probe that self-terminates and leaves **no warm
  GPU**. This is the preferred way to get real-model evidence (e.g. `manifold_pullback_probe_2b`,
  `monitor_demo_2b`, `web_parity_2b`, `smoke_2b`). Bounded and cheap.
- **`modal serve modal_app.py`** — serves the live GUIs (`gradio_gui` + `web_gui`) sharing one warm
  container. It is a **long-running foreground process**: best launched by the *user* in their own
  terminal (it holds the terminal and streams logs). It prints two URLs — open the **`web_gui`** one
  for the Lab Bench. Don't launch it fire-and-forget from a tool call expecting it to return.
- **ALWAYS stop the GPU when done:** `modal app stop qwen-scope-steering-gui` (or Ctrl+C the serve).
  The container idles down after ~5 minutes, but stop it explicitly to halt billing immediately.

## Cost preflight & targets

Check `RUNBOOK.md` for the GPU type + timeout per function before running. Defaults and selection:

- **2B → L4** (cheap) is the default for everything; prefer it.
- **27B → A100 / H100** — only when 27B is explicitly required.
- Serving picks a target via `QWEN_GUI_TARGET` (`2b-l4` default, `27b-a100`, `27b-h100`); aliases
  `2b` / `27b` / `a100` / `h100`. e.g. `QWEN_GUI_TARGET=27b-a100 modal serve modal_app.py`.

The first real-model call is a **cold load** (model + SAE download/load, a minute or two); later
calls in the same warm container are fast.

## Auth & setup

Run from the repo root in the venv. Needs `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` and an `HF_TOKEN`
in `.env` (Hugging Face token gates the model/SAE download). A persistent Modal Volume is mounted at
`/cache` so artifacts are reused across runs.

## Which functions are live vs archived

Live probes in `modal_app.py` (run with `modal run modal_app.py::<fn>`): `smoke_2b`/`_27b`,
`web_parity_2b`, `bench_smoke_*`, `autopilot_smoke_*`, `manifold_steer_demo_*`, `manifold_vs_linear_2b`,
`manifold_naturalness_probe_2b`, `manifold_pullback_probe_2b`, `manifold_atlas_*`,
`residual_manifold_sweep_*`, `monitor_demo_2b`, plus the `gradio_gui` / `web_gui` ASGI endpoints.
The dead-end research probes live in `archive/research_probes.py` and are **preserved but not
runnable** — don't try to `modal run` them.

## Driving the served GUI's API from outside

Once `modal serve` is up, the same JSON API is at the `web_gui` URL. To run experiments against it,
point the bench client at it: `BENCH_URL=https://<...>-web-gui-dev.modal.run python
.claude/skills/bench-experiment/scripts/bench_client.py status` (it disables SSL verification so it
works against Modal dev URLs). Then submit jobs and poll exactly as for the dev server.

## Reporting discipline

Report findings honestly, including negatives. **Don't claim 27B support without a successful Modal
run** (or a clear note that it's untested). After any real-model session, confirm the GPU is stopped.
