# Qwen Scope Lab Bench

**A local, browser-based SAE interpretability lab. Inspect, steer, monitor, and control a real language model from a visual GUI — running entirely on your Mac via MLX. No GPU, no cloud, no notebooks.**

[![CI](https://github.com/scasella/qwen-scope-lab-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/scasella/qwen-scope-lab-bench/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

![The Lab Bench — token-level SAE feature inspection on the real Qwen3.5-2B, running on-device via MLX](docs/assets/lab-explore.png)

The Lab Bench is a comprehensive Sparse-Autoencoder interpretability workbench over [Qwen Scope](https://huggingface.co/Qwen) that you drive entirely from a **point-and-click web GUI** — no notebooks, no scripting. It runs the **whole pipeline on-device via [MLX](https://github.com/ml-explore/mlx)** — the real model **and** its SAE — on **any Apple-Silicon Mac (M1–M4)**, so you can inspect features, steer generation, fit concept manifolds in 3D, train behavior detectors, and run an honest detect→suppress→prove control loop just by clicking through your browser, all locally and offline. A GPU-free dev backend lets you explore the whole interface with no downloads; an optional Modal/CUDA path scales to the 27B and shareable hosted demos.

## Quickstart — the full lab on your Mac

```bash
pip install -e ".[mlx]"
python serve_web.py --mlx          # the real 2B + its SAE, entirely on-device
# → open http://127.0.0.1:7870 in your browser — the whole lab is a visual GUI
```

That one command runs the whole bench on the real `Qwen3.5-2B` and its Qwen-Scope SAE and serves it as a **browser-based GUI** — everything in this README is done by clicking, not coding. No Modal, no CUDA, no API key. First launch downloads the model (~4.5 GB, bf16) and SAE (~540 MB), then caches them; every run after is offline. (Pass `--mlx-sae none` to skip the SAE download and use the probe-only paths; pass `--mlx <repo>` to override the model.)

**Just want to explore the interface, with no downloads at all?**

```bash
pip install -e ".[dev]"
python serve_web.py --dev          # a tiny in-memory model — real code paths, toy outputs
```

New here? `docs/USER_GUIDE.md` is a click-along tour of every mode using the app's pre-filled inputs.

## What you can do — the loop

The GUI is one click-through loop — each step is a mode in the left-hand nav: **Explore → Steer → Measure → Manifold → Monitor → Control → Library.**

- 🔎 **Explore** — a token-level SAE feature microscope; atlas a whole prompt corpus by peak or breadth; contrast two prompts. Pin a feature to carry it into Steer and Measure.
- ↗ **Steer** — dial a feature up or down; live before/after generation; a strength sweep; the logit-effect metric.
- ▤ **Measure** — a **seven-control benchmark** that returns an honest `validated` / `benchmarked` verdict (it only validates if the steer beats prompt-only *and* every control).
- ∿ **Manifold** — fit a concept's residual-stream manifold, render its 3D geometry, steer *along* it with a paper-faithful replace intervention, and run a gradient **pullback** that optimizes the activation inducing a target behavior.
- ◉ **Monitor** — discover the SAE feature(s) or a linear probe that best detect a behavior (refusal, PII, sycophancy…); held-out eval gated by a **random-feature control**; save validated detectors as runtime guardrails.
- ⟳ **Control** — the honest detect→suppress→prove loop: a probe-vs-SAE-vs-judge shootout, robustness under paraphrase, collateral-damage measurement (the "Rogue Scalpel" safety check), CAA-vs-SAE steering, and the **jailbreak-detection suite**.
- ▦ **Library** — save and reuse steering & manifold **recipe cards**.

![Concept-manifold mode — the days-of-the-week ring fitted in the residual stream and rendered in 3D](docs/assets/lab-manifold.png)

## Why it's credible — honest controls

The bench's differentiator is its rigor. A steer "validates" only if it beats a prompt-only baseline **and** seven controls — including a **random-feature control** (inject a *different* feature at the same strength) and a negative-strength control. Detectors are scored against a raw-residual linear probe and a random-feature control, at a **TPR-at-fixed-FPR** operating point — the way a deployed monitor is actually tuned. Suppression only counts if the behavior was present, the steer removed it, **and** nothing else broke; perfect suppression that lobotomizes the model or erodes its refusals is honestly marked `benchmarked`. Negatives are reported, not hidden. (Driving the lab from an agent? Every op runs through a job API and an experiment log — see [`docs/AGENT_RESEARCH.md`](docs/AGENT_RESEARCH.md).)

## Highlight: a free jailbreak detector

A difference-of-means **residual probe** — one dot product on activations the model already computes — detects jailbreak / prompt-injection prompts as well as a paid AI judge, beats the SAE feature, and generalizes to attack families it never saw. Try the live single-message demo at **`/demo`**. Write-ups: [the residual probe (for researchers)](docs/writeups/jailbreak-detection-residual-probe.html) · [for a general audience](docs/writeups/jailbreak-detection-mainstream.html) · [white-box control on Qwen-2B](docs/writeups/white-box-control-qwen-2b.html).

## Scale up (optional)

The 2B is local-first on MLX. For the **27B** (won't fit a laptop) or a **shareable hosted demo**, use the Modal path:

```bash
pip install -e ".[cuda,modal]"
modal serve modal_app.py            # then open the printed web_gui URL
```

See [`docs/MLX.md`](docs/MLX.md) for the local↔cloud split and fidelity notes, and [`RUNBOOK.md`](RUNBOOK.md) for the Modal commands, GPU targets, and cost discipline. (You can also load the real model on a local CUDA GPU with `".[cuda]"` and `python serve_web.py --config configs/qwen35_2b_dev_l0_100.yaml`.)

## Install options

| Use case | Command |
| --- | --- |
| **Mac, on-device (recommended)** | `pip install -e ".[mlx]"` |
| Explore the UI, no downloads | `pip install -e ".[dev]"` |
| Real model on a CUDA GPU | `pip install -e ".[cuda]"` |
| Drive Modal probes / hosted serve | `pip install -e ".[cuda,modal]"` |
| Everything | `pip install -e ".[all]"` |

The base install is deliberately slim (torch + the web layer); the heavy and cloud dependencies are opt-in extras.

## Documentation

- [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) — a click-along tour of every mode.
- [`docs/MLX.md`](docs/MLX.md) — running the whole lab on Apple Silicon: architecture, fidelity caveats, what stays on Modal.
- [`docs/MANIFOLD.md`](docs/MANIFOLD.md) — the concept-manifold science, including the honest negatives behind the design.
- [`docs/AGENT_RESEARCH.md`](docs/AGENT_RESEARCH.md) — driving the lab programmatically: the job API, the experiment log, and the honesty contract.
- [`RUNBOOK.md`](RUNBOOK.md) — operational reference: the CLI scripts, recipe commands, and Modal/GPU runbook.

## Contributing & License

Contributions welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md). The house rules: keep new capabilities testable GPU-free on the dev backend, build in an honest control from the start, and add tests. Licensed under the Apache License 2.0 — see [`LICENSE`](LICENSE).

<sub>A legacy tabbed Gradio app (`app.py`, installed with `pip install -e ".[gradio]"`) predates the Lab Bench web UI and remains available as an alternative front end.</sub>
