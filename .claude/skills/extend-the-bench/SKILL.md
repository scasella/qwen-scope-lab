---
name: extend-the-bench
description: >
  Add a new capability to the Qwen Scope "Lab Bench" in this repo the project's way — a new
  analysis / steering / detection feature, end to end: pure logic module, a SteeringService method,
  a job-able API op + sync route, a saveable artifact, a GUI mode, a Modal probe, tests, and docs.
  Use this whenever the user wants to add a feature, a new endpoint or mode, a new kind of
  intervention/probe/metric/detector, or otherwise extend the workbench (touching service.py,
  web_api.py, web/, modal_app.py). It encodes the architecture conventions — thin FastAPI over
  SteeringService, the dev-backend mirror, the OPS job registry, recipe-style saveable artifacts,
  and an HONEST verdict against a control — so new features stay consistent and agent-drivable
  instead of drifting from the patterns. Reach for this before hand-rolling a new endpoint.
  Triggers on: "add a feature/endpoint/mode to the bench", "a new probe/metric/intervention/
  detector", "wire X into the workbench", "a new saveable artifact like recipes", "how should I
  structure a new bench feature", or any edit to service.py / web_api.py / web/ / modal_app.py that
  adds a capability.
---

# Extending the Lab Bench

The bench has been extended three times the same way (manifold steering → the async job layer →
feature-as-monitor). Follow that pattern so a new capability is consistent, testable GPU-free, and
drivable by both the GUI and an agent. The **monitor** stack is the cleanest, most complete template
— read `qwen_scope_steering_gui/monitor.py`, `monitor_schema.py`, `monitor_store.py`, the `op_monitor_*`
wiring in `web_api.py`, the Monitor mode in `web/app.js`, and `monitor_demo_2b` in `modal_app.py`,
then mirror them. The README's **Lab Bench** section and `docs/AGENT_RESEARCH.md` describe the architecture.

## Architecture in one breath

A thin FastAPI layer (`web_api.py`) wraps a single `SteeringService` (`service.py`). The same backend
serves the Gradio app, the `web/` SPA, and Modal. A **dev backend** (`dev_backend.build_dev_service`)
injects a tiny CPU model into a real `SteeringService`, so every code path runs GPU-free — that's
how tests and local UI work without a GPU. Heavy ops also run as **jobs** through an `OPS` registry
serialized on one `gpu_lock`. Build new features to fit this shape.

## The pattern (steps)

1. **Pure logic module** — `qwen_scope_steering_gui/<feature>.py`. Keep it GPU-free *given
   activations* where possible (operate on the dicts `inspect_prompt` returns), like `monitor.py`.
   Why: it's unit-testable without a model, and it forces a clean seam. If the feature makes a
   claim ("this works"), compute an **honest verdict** here — a `validation_decision` with a
   **control** baseline (mirror `monitor.discover`'s random-feature control or
   `web_api.manifold_validation_decision`). Never let a no-op read as `validated`.
2. **Saveable artifact** (only if it produces one) — a dataclass + store mirroring
   `recipe_schema.py` / `recipe_store.py` (reuse `TargetBehavior` / `ModelMetadata`), like
   `monitor_schema.BehaviorMonitor` / `MonitorStore`. Give it `create/compute_id/validate/
   to_dict/from_dict/to_markdown`; `validated` status must require evaluation evidence.
3. **Service method** on `SteeringService` — thin; uses `inspect_prompt` / `hooks` / `generation`.
   Resolve the layer with the existing fallback (`config.default_layer`, or a concept's `best_layer`).
4. **web_api wiring** — add `op_<feature>(p: dict)` to the **`OPS`** dict (so agents can submit it as
   a job), a Pydantic request model, and a sync route `POST /api/<feature>` that calls the op via
   **`_guard_gpu`** (the lock; concurrent model calls 500 without it). If it's saveable, stash
   `last_<feature>` and mirror the `save_recipe(Request)` save route + `GET` list/detail. Extend
   `_summarize` so the experiment log captures the salient result. One source of truth: the sync
   route and the job runner both call the same `op_<feature>`.
5. **Dev-backend mirror** — confirm the op runs under `build_dev_service()`. On the random dev model
   results are chance, so the verdict lands `BENCHMARKED`; that's expected. Tests assert *shape* and
   that the verdict is a valid value, not high scores. **If the op touches the model** (capture /
   generate / a hook / the tokenizer), it must ALSO run on the **MLX backend** (`mlx_backend.py`):
   model-touching primitives branch on the duck-typed `is_mlx_runtime` flag at one point; tokenizer
   calls should use `.encode()` (both backends support it). A per-method port can miss sibling
   helpers — the real completeness check is a live `serve_web.py --mlx` GUI walk (see `docs/MLX.md`).
6. **GUI mode** (if user-facing) — a nav item in `web/index.html`, a `view<Feature>()` dispatched
   from `renderStage` in `web/app.js`, handlers in the click switch, reusing existing UI
   (`.score` bars, `.verdict` chip, `.ba` panes, `.wrow`, `.rcard`, `withBusy`, `api()`). Match
   `viewManifold` / `viewMonitor`. Call heavy ops synchronously behind the busy spinner (like
   `runBenchmark`) — the lock makes that safe.
7. **Real-model validation** — on a Mac, run it locally via the MLX backend (`serve_web.py --mlx …`
   + the GUI / `/api/<feature>`) for fast, free, offline real-2B evidence. For a recorded, citable
   real-GPU result, add a `<feature>_demo_2b` in `modal_app.py`, mirroring `monitor_demo_2b` /
   `manifold_pullback_probe_2b`. (Running either is the **bench-on-modal** skill.)
8. **Tests** — `tests/`: a unit test of the pure module on synthetic inputs (deterministic), plus
   API round-trips in `test_web_api.py` (discover/save→list→detail, and one **job** round-trip with
   `TestClient` as a context manager so the async task runs). Pass `*_root=tmp_path` to stores. The
   existing suite must stay green.
9. **Docs** — update `README.md`, `docs/AGENT_RESEARCH.md` (endpoints + the new stack), and if it
   adds an op, `docs/AGENT_RESEARCH.md` (op list); if it adds a GUI mode, `docs/USER_GUIDE.md`. Note
   surprising decisions in memory.
10. **Verify** — `pytest` green; dev-server Playwright with **0 console errors** + a screenshot; no
    Modal re-run unless you actually need real-model evidence; leave no warm GPU.

## Conventions that matter (the why)

- **Honest negatives.** The bench's credibility is its controls. A feature that can't beat its
  control should report `benchmarked`, not be massaged into `validated`. Build the control in from
  the start.
- **Dev mirror, always.** If a path can't run on the dev backend, it can't be tested in CI and the
  GUI can't be demoed offline — keep the model-touching part behind the service so the tiny model
  exercises it.
- **Archive, don't delete.** Dead-end explorations move to `archive/` with a README explaining the
  negative result (see `archive/research_probes.py`) — provenance over erasure (this repo has no git
  safety net).
- **Recipes vs monitors vs experiments.** Recipes = validated *control* artifacts; monitors =
  validated *detection* artifacts; the experiment log = the full trail incl. negatives. Keep a new
  artifact type in its own lane rather than overloading recipes.
