"""Thin FastAPI layer over :class:`SteeringService`.

Every endpoint is a small JSON wrapper around a service method that already returns
JSON-serialisable dicts, so the same backend powers the dev (CPU) and real (GPU)
paths unchanged. Blocking torch work is pushed to a threadpool.
"""
from __future__ import annotations

import asyncio
import re
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .autopilot import run_autopilot
from .benchmark import ServiceGenerationBackend, attach_benchmark_to_recipe, recipe_from_manual, run_benchmark
from .benchmark_metrics import score_for_objective
from .experiment_log import ExperimentLog
from .monitor_schema import BehaviorMonitor
from .monitor_store import MonitorStore
from .prompt_sets import parse_prompt_text
from .recipe_schema import FeatureRecipe, ManifoldSpec, ModelMetadata, TargetBehavior
from .recipe_store import RecipeStore


def _slug(text: str, fallback: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return s or fallback

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def manifold_validation_decision(legs: dict[str, Any]) -> dict[str, Any]:
    """Honest, concept-dependent verdict for a manifold steer: PASS if on-manifold steering
    (manifold or pullback) induces the target behavior at least as faithfully as the linear
    chord (lower/equal behavior-manifold energy)."""
    man = (legs.get("manifold") or {}).get("mean_energy")
    lin = (legs.get("linear") or {}).get("mean_energy")
    pb = (legs.get("pullback") or {}).get("mean_energy")
    best = min([e for e in (man, pb) if e is not None], default=None)
    if lin is not None and best is not None and best <= lin + 1e-9:
        return {"status": "validated", "passed": True,
                "reason": f"on-manifold steering induces the target behavior at least as faithfully as the linear chord (energy {best:.3f} ≤ linear {lin:.3f})."}
    return {"status": "benchmarked", "passed": False,
            "reason": "ran the manifold/linear/pullback comparison; on-manifold steering did not beat the linear chord on behavior-energy for this concept (honest negative — the advantage is concept-dependent)."}


def build_manifold_recipe(service: Any, req: Any, result: dict[str, Any]) -> FeatureRecipe:
    """Snapshot a pullback comparison into a saveable manifold recipe."""
    cfg = service.config
    legs = {name: {"mean_energy": (result.get(name) or {}).get("mean_energy"),
                   "recovered_r": (result.get(name) or {}).get("recovered_r"),
                   "steered_text": (result.get(name) or {}).get("steered_text")}
            for name in ("manifold", "linear", "pullback") if name in result}
    decision = manifold_validation_decision(legs)
    behavior = TargetBehavior(
        name=f"{req.concept}: {req.source}→{req.target}",
        description=f"Steer the concept '{req.concept}' from {req.source} to {req.target} along its residual-stream manifold.",
    )
    model = ModelMetadata(model_id=cfg.model_id, sae_id=cfg.sae_id or "",
                          dtype=str(getattr(cfg, "dtype", "")), config_name=str(getattr(service, "config_path", "")))
    layer = int(result.get("layer", req.layer) if result.get("layer") is not None else (req.layer if req.layer is not None else cfg.default_layer))
    spec = ManifoldSpec(concept=req.concept, source=req.source, target=req.target, layer=layer,
                        path="pullback", n_waypoints=int(req.n_waypoints))
    recipe = FeatureRecipe.create_manifold(behavior, model, spec)
    recipe.benchmark.update({"status": decision["status"], "prompt_set_id": "manifold_pullback",
                             "methods_compared": list(legs.keys()), "legs": legs,
                             "validation_decision": decision, "summary": decision["reason"]})
    recipe.examples = [{"prompt": result.get("steer_prompt", ""),
                        "unsteered": result.get("unsteered_text", ""),
                        "steered": (result.get("pullback") or result.get("manifold") or {}).get("steered_text", "")}]
    recipe.status = decision["status"]
    return recipe


def build_monitor(service: Any, behavior_name: str, result: dict[str, Any], n_pos: int, n_neg: int) -> BehaviorMonitor:
    """Snapshot a monitor discovery into a saveable BehaviorMonitor."""
    cfg = service.config
    name = behavior_name or "monitored_behavior"
    behavior = TargetBehavior(name=name, description=f"Detect '{name}' in text via SAE features.")
    model = ModelMetadata(model_id=cfg.model_id, sae_id=cfg.sae_id or "",
                          dtype=str(getattr(cfg, "dtype", "")), config_name=str(getattr(service, "config_path", "")))
    evaluation = {**result.get("metrics", {}), "validation_decision": result.get("validation_decision"),
                  "per_feature": result.get("per_feature")}
    return BehaviorMonitor.create(behavior, model, layer=result.get("layer", cfg.default_layer),
                                  features=result.get("features", []), threshold=result.get("threshold", 0.0),
                                  top_k=result.get("top_k", 3), combine=result.get("combine", "max"),
                                  evaluation=evaluation, discovery={"n_pos": n_pos, "n_neg": n_neg},
                                  status=(result.get("validation_decision") or {}).get("status", "benchmarked"))


class InspectReq(BaseModel):
    prompt: str
    layer: int | None = None
    top_k: int | None = 12
    max_seq_len: int | None = 128


class CompareReq(BaseModel):
    positive: str
    negative: str
    layer: int | None = None
    limit: int = 24


class SteerReq(BaseModel):
    prompt: str
    feature_id: int
    strength: float
    layer: int | None = None
    max_new_tokens: int | None = None
    temperature: float = 0.0
    mode: str = "all_positions"


class SweepReq(BaseModel):
    prompt: str
    feature_id: int
    strengths: list[float]
    layer: int | None = None
    max_new_tokens: int | None = None
    temperature: float = 0.0
    mode: str = "all_positions"


class NoteReq(BaseModel):
    feature_id: int
    layer: int
    human_label: str = ""
    notes: str = ""
    example_prompts: list[str] = []
    observed_effects: str = ""
    failure_notes: str = ""


class BenchmarkReq(BaseModel):
    prompt_set: str
    feature_id: int
    strength: float
    layer: int | None = None
    target_behavior: str = "steered_behavior"
    target_description: str = ""
    prompt_only_instruction: str = "Answer concisely in no more than two sentences. Prompt: {prompt}"
    max_new_tokens: int = 24
    temperature: float = 0.0
    objective: str = "maximize_rule_score"


class AtlasReq(BaseModel):
    prompts: list[str]
    layer: int | None = None
    top_k: int = 12
    max_features: int = 90


class ManifoldFitReq(BaseModel):
    concept: str
    layer: int | None = None


class ManifoldSteerReq(BaseModel):
    concept: str
    target: str
    source: str | None = None
    layer: int | None = None
    prompt: str | None = None
    n_waypoints: int = 7
    max_new_tokens: int = 24
    temperature: float = 0.0
    path: str = "manifold"


class ManifoldCompareReq(BaseModel):
    concept: str
    target: str
    source: str | None = None
    layer: int | None = None
    prompt: str | None = None
    n_waypoints: int = 7
    max_new_tokens: int = 24
    temperature: float = 0.0


class ManifoldSaeReq(BaseModel):
    concept: str
    layer: int | None = None
    top_k: int = 5


class ManifoldPullbackReq(BaseModel):
    concept: str
    target: str
    source: str | None = None
    layer: int | None = None
    n_waypoints: int = 5
    max_new_tokens: int = 20
    lbfgs_iters: int = 25


class AutopilotReq(BaseModel):
    positive_examples: str
    negative_examples: str
    validation_prompts: str
    target_name: str = "discovered_behavior"
    target_description: str = ""
    candidate_layers: list[int] | None = None
    candidate_count: int = 3
    objective: str = "maximize_rule_score"
    prompt_only_instruction: str = "Answer concisely. Prompt: {prompt}"
    max_new_tokens: int = 16
    temperature: float = 0.0


class MonitorDiscoverReq(BaseModel):
    behavior: str = "monitored_behavior"
    positive_examples: str = ""
    negative_examples: str = ""
    layer: int | None = None
    top_k: int = 3


class MonitorScoreReq(BaseModel):
    text: str
    monitor_id: str | None = None
    features: list[int] = []
    layer: int | None = None
    threshold: float = 0.0


async def _guard(fn, *args, **kwargs):
    try:
        return await run_in_threadpool(fn, *args, **kwargs)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def create_app(service: Any, recipes_root: str | Path = "recipes",
               experiments_root: str | Path | None = None,
               monitors_root: str | Path = "monitors") -> FastAPI:
    app = FastAPI(title="Qwen Scope Steering", docs_url="/api/docs", openapi_url="/api/openapi.json")
    store = RecipeStore(recipes_root)
    monitor_store = MonitorStore(monitors_root)
    exp_log = ExperimentLog(experiments_root) if experiments_root else None
    last_bench: dict = {}
    last_manifold: dict = {}
    last_monitor: dict = {}
    gpu_lock = asyncio.Lock()  # one GPU: serialize model-touching work so concurrent calls queue, not 500
    jobs: dict[str, dict] = {}

    async def _guard_gpu(fn, *args, **kwargs):
        async with gpu_lock:
            return await _guard(fn, *args, **kwargs)

    # ---- op implementations: one source of truth for both sync routes and async jobs ----
    def op_inspect(p: dict) -> dict:
        return service.inspect_prompt(p["prompt"], p.get("layer"), p.get("top_k"), p.get("max_seq_len"))

    def op_compare(p: dict) -> dict:
        return service.compare_prompts(p["positive"], p["negative"], p.get("layer"), p.get("limit", 20))

    def op_atlas(p: dict) -> dict:
        prompts = [s for s in (p.get("prompts") or []) if s and s.strip()]
        if not prompts:
            raise ValueError("provide at least one prompt to scan")
        layer = p["layer"] if p.get("layer") is not None else service.config.default_layer
        top_k, max_features = p.get("top_k", 12), p.get("max_features", 90)
        n = len(prompts)
        feats: dict[int, dict[str, Any]] = {}
        prompt_meta = []
        for pi, prompt in enumerate(prompts):
            insp = service.inspect_prompt(prompt, layer=layer, top_k=top_k, max_seq_len=128)
            prompt_meta.append({"text": prompt, "n_tokens": len(insp["tokens"])})
            for row in insp["top_features_by_token"]:
                tok = row["token_text"]
                for f in row["features"]:
                    fid = int(f["feature_id"])
                    a = float(f["activation"])
                    e = feats.get(fid)
                    if e is None:
                        e = feats[fid] = {"feature_id": fid, "peak": a, "sum": 0.0, "count": 0,
                                          "prompts": set(), "tokens": {}, "per_prompt_peak": [0.0] * n}
                    e["peak"] = max(e["peak"], a)
                    e["sum"] += a
                    e["count"] += 1
                    e["prompts"].add(pi)
                    if a > e["per_prompt_peak"][pi]:
                        e["per_prompt_peak"][pi] = a
                    if a > e["tokens"].get(tok, -1e9):
                        e["tokens"][tok] = a
        features = []
        for e in feats.values():
            top_tokens = [t for t, _ in sorted(e["tokens"].items(), key=lambda kv: -kv[1])[:6]]
            features.append({"feature_id": e["feature_id"], "peak": e["peak"],
                             "mean": e["sum"] / max(e["count"], 1), "n_prompts": len(e["prompts"]),
                             "fingerprint": e["per_prompt_peak"], "top_tokens": top_tokens})
        features.sort(key=lambda x: -x["peak"])
        return {"layer": layer, "n_prompts": n, "prompts": prompt_meta, "features": features[:max_features]}

    def op_steer(p: dict) -> dict:
        return service.steer(p["prompt"], p.get("layer"), p["feature_id"], p["strength"],
                             p.get("max_new_tokens"), p.get("temperature", 0.7), p.get("mode", "all_positions"))

    def op_sweep(p: dict) -> dict:
        frames, unsteered = [], None
        for strength in p["strengths"]:
            result = service.steer(p["prompt"], p.get("layer"), p["feature_id"], float(strength),
                                   p.get("max_new_tokens"), p.get("temperature", 0.7), p.get("mode", "all_positions"),
                                   compute_logits_delta=False)
            if unsteered is None:
                unsteered = result["unsteered_text"]
            frames.append({"strength": float(strength), "text": result["steered_text"],
                           "hook_fired": result["hook_fired"], "hidden_delta_norm": result["hidden_delta_norm"]})
        return {"prompt": p["prompt"], "feature_id": p["feature_id"], "unsteered_text": unsteered, "frames": frames}

    def op_benchmark(p: dict) -> dict:
        prompts = parse_prompt_text(p["prompt_set"])
        if not prompts:
            raise ValueError("provide at least one prompt (one JSON object or plain line per row)")
        layer = p["layer"] if p.get("layer") is not None else service.config.default_layer
        objective = p.get("objective", "maximize_rule_score")
        recipe = recipe_from_manual(config=service.config, config_path=str(getattr(service, "config_path", "web")),
                                    target_behavior=p.get("target_behavior") or "steered_behavior",
                                    target_description=p.get("target_description", ""), layer=int(layer),
                                    feature_id=int(p["feature_id"]), strength=float(p["strength"]))
        result = run_benchmark(recipe, prompts, ServiceGenerationBackend(service),
                               prompt_only_instruction=p.get("prompt_only_instruction", ""),
                               max_new_tokens=int(p.get("max_new_tokens", 24)), temperature=float(p.get("temperature", 0.0)),
                               seed=0, objective=objective, prompt_set_id="web_inline")
        last_bench["recipe"], last_bench["result"] = recipe, result
        method_scores = {m: score_for_objective(result["aggregate_metrics"].get(m, {}), objective) for m in result["methods"]}
        return {"methods": result["methods"], "method_scores": method_scores,
                "validation_decision": result["validation_decision"], "objective": objective,
                "examples": [{"prompt": r["prompt"], "outputs": r["outputs"]} for r in result["per_prompt_results"][:3]],
                "config": result["config"]}

    def op_autopilot(p: dict) -> dict:
        positive = [line for line in p.get("positive_examples", "").splitlines() if line.strip()]
        negative = [line for line in p.get("negative_examples", "").splitlines() if line.strip()]
        validation = parse_prompt_text(p.get("validation_prompts", ""))
        if not positive or not negative:
            raise ValueError("provide at least one positive and one negative example")
        if not validation:
            raise ValueError("provide at least one validation prompt")
        objective = p.get("objective", "maximize_rule_score")
        layers = p.get("candidate_layers") or [service.config.default_layer]
        slug = _slug(p.get("target_name", ""), "discovered_behavior")
        description = p.get("target_description") or f"Steer toward {slug.replace('_', ' ')}."
        result = run_autopilot(config=service.config, config_path=str(getattr(service, "config_path", "web")),
                               target_name=slug, target_description=description, positive_examples=positive,
                               negative_examples=negative, validation_prompts=validation,
                               candidate_layers=[int(x) for x in layers], candidate_count=int(p.get("candidate_count", 3)),
                               objective=objective, backend=ServiceGenerationBackend(service), service=service,
                               output_dir=Path(recipes_root) / f"{slug}_autopilot",
                               prompt_only_instruction=p.get("prompt_only_instruction", ""),
                               max_new_tokens=int(p.get("max_new_tokens", 24)), temperature=float(p.get("temperature", 0.0)), seed=0)
        bench = result["benchmark"]
        method_scores = {m: score_for_objective(bench["aggregate_metrics"].get(m, {}), objective) for m in bench["methods"]}
        best_recipe = result["best_recipe"]
        return {"candidates": result["candidate_features"], "best_candidate": result["best_candidate"],
                "recipe_id": best_recipe.get("recipe_id"),
                "best_recipe": {"recipe_id": best_recipe.get("recipe_id"), "status": best_recipe.get("status"),
                                "interventions": best_recipe.get("interventions", [])},
                "methods": bench["methods"], "method_scores": method_scores,
                "validation_decision": bench["validation_decision"], "warning": result["warning"], "objective": objective}

    def op_manifold_fit(p: dict) -> dict:
        return service.manifold_fit(p["concept"], p.get("layer"))

    def op_manifold_steer(p: dict) -> dict:
        return service.manifold_steer(p["concept"], p["target"], p.get("layer"), p.get("source"), p.get("prompt"),
                                      p.get("n_waypoints", 5), p.get("max_new_tokens"), p.get("temperature", 0.7), p.get("path", "manifold"))

    def op_manifold_compare(p: dict) -> dict:
        return service.manifold_compare(p["concept"], p["target"], p.get("layer"), p.get("source"), p.get("prompt"),
                                        p.get("n_waypoints", 5), p.get("max_new_tokens"), p.get("temperature", 0.7))

    def op_manifold_sae_coverage(p: dict) -> dict:
        return service.manifold_sae_coverage(p["concept"], p.get("layer"), p.get("top_k", 6))

    def op_manifold_pullback(p: dict) -> dict:
        result = service.manifold_pullback(p["concept"], p["target"], p.get("layer"), p.get("source"),
                                           p.get("n_waypoints", 5), p.get("max_new_tokens", 20), p.get("lbfgs_iters", 25))
        try:  # snapshot a saveable manifold recipe (the pullback IS the manifold benchmark)
            last_manifold["recipe"] = build_manifold_recipe(service, SimpleNamespace(**p), result)
            last_manifold["result"] = result
        except Exception:
            last_manifold.clear()
        return result

    def _split_examples(v):
        return [s for s in v.splitlines() if s.strip()] if isinstance(v, str) else list(v or [])

    def op_monitor_discover(p: dict) -> dict:
        pos, neg = _split_examples(p.get("positive_examples")), _split_examples(p.get("negative_examples"))
        result = service.discover_monitor(pos, neg, p.get("layer"), p.get("top_k", 3))
        result["behavior"] = p.get("behavior") or "monitored_behavior"
        try:  # snapshot a saveable monitor (discovery IS the monitor benchmark)
            last_monitor["monitor"] = build_monitor(service, result["behavior"], result, len(pos), len(neg))
        except Exception:
            last_monitor.clear()
        return result

    def op_monitor_score(p: dict) -> dict:
        if p.get("monitor_id"):
            m = monitor_store.load(p["monitor_id"])
            features, layer, threshold = m.features, m.layer, m.threshold
        else:
            features, layer, threshold = p.get("features", []), p.get("layer"), p.get("threshold", 0.0)
        if not features:
            raise ValueError("provide a monitor_id or a non-empty features list")
        return service.score_monitor(p["text"], features, layer, threshold)

    OPS = {"inspect": op_inspect, "compare": op_compare, "atlas": op_atlas, "steer": op_steer, "sweep": op_sweep,
           "benchmark": op_benchmark, "autopilot": op_autopilot, "manifold_fit": op_manifold_fit,
           "manifold_steer": op_manifold_steer, "manifold_compare": op_manifold_compare,
           "manifold_sae_coverage": op_manifold_sae_coverage, "manifold_pullback": op_manifold_pullback,
           "monitor_discover": op_monitor_discover, "monitor_score": op_monitor_score}

    def _summarize(op: str, result: Any) -> dict:
        if not isinstance(result, dict):
            return {}
        if op in ("benchmark", "autopilot"):
            return {"validation_decision": result.get("validation_decision"), "method_scores": result.get("method_scores"),
                    "recipe_id": result.get("recipe_id"), "best_candidate": result.get("best_candidate")}
        if op in ("manifold_pullback", "manifold_compare"):
            return {"legs": {k: {"mean_energy": (result.get(k) or {}).get("mean_energy"),
                                 "recovered_r": (result.get(k) or {}).get("recovered_r")}
                             for k in ("manifold", "linear", "pullback") if k in result}}
        if op == "manifold_sae_coverage":
            return {"n_distinct_features": result.get("n_distinct_features"), "n_values": len(result.get("per_value", []))}
        if op == "manifold_fit":
            return {"kind": result.get("kind"), "layer": result.get("layer"), "quality": result.get("quality")}
        if op in ("steer", "manifold_steer"):
            return {"hook_fired": result.get("hook_fired"), "steered_text": (result.get("steered_text") or "")[:120]}
        if op == "atlas":
            return {"n_prompts": result.get("n_prompts"), "n_features": len(result.get("features", []))}
        if op == "monitor_discover":
            mx = result.get("metrics", {})
            return {"behavior": result.get("behavior"), "features": result.get("features"),
                    "auc": mx.get("auc"), "f1": mx.get("f1"), "control_auc": mx.get("control_auc"),
                    "validation_decision": result.get("validation_decision")}
        return {}

    def _log_experiment(op: str, params: dict, status: str, result: Any = None, error: str | None = None) -> None:
        if exp_log is None:
            return
        rec: dict = {"op": op, "status": status,
                     "params": {k: v for k, v in (params or {}).items() if k != "prompt_set"}}
        if error:
            rec["error"] = error
        if result is not None:
            rec["summary"] = _summarize(op, result)
        try:
            exp_log.append(rec)
        except Exception:
            pass

    async def _run_job(job_id: str, op: str, params: dict) -> None:
        job = jobs[job_id]
        job["status"], job["started_at"] = "running", time.time()
        try:
            async with gpu_lock:
                result = await run_in_threadpool(OPS[op], params)
            job["result"], job["status"] = result, "done"
            _log_experiment(op, params, "done", result=result)
        except Exception as exc:  # never let a job failure crash the worker
            job["error"], job["status"] = str(exc), "error"
            _log_experiment(op, params, "error", error=str(exc))
        finally:
            job["finished_at"] = time.time()

    @app.get("/api/status")
    async def status() -> dict:
        return await _guard(service.status)

    @app.post("/api/inspect")
    async def inspect(req: InspectReq) -> dict:
        return await _guard_gpu(op_inspect, req.model_dump())

    @app.post("/api/compare")
    async def compare(req: CompareReq) -> dict:
        return await _guard_gpu(op_compare, req.model_dump())

    @app.post("/api/atlas")
    async def atlas(req: AtlasReq) -> dict:
        return await _guard_gpu(op_atlas, req.model_dump())

    @app.get("/api/manifold/presets")
    async def manifold_presets() -> dict:
        return await _guard(service.manifold_presets)

    @app.post("/api/manifold/fit")
    async def manifold_fit(req: ManifoldFitReq) -> dict:
        return await _guard_gpu(op_manifold_fit, req.model_dump())

    @app.post("/api/manifold/steer")
    async def manifold_steer(req: ManifoldSteerReq) -> dict:
        return await _guard_gpu(op_manifold_steer, req.model_dump())

    @app.post("/api/manifold/compare")
    async def manifold_compare(req: ManifoldCompareReq) -> dict:
        return await _guard_gpu(op_manifold_compare, req.model_dump())

    @app.post("/api/manifold/sae_coverage")
    async def manifold_sae_coverage(req: ManifoldSaeReq) -> dict:
        return await _guard_gpu(op_manifold_sae_coverage, req.model_dump())

    @app.post("/api/manifold/pullback")
    async def manifold_pullback(req: ManifoldPullbackReq) -> dict:
        return await _guard_gpu(op_manifold_pullback, req.model_dump())

    @app.post("/api/monitor/discover")
    async def monitor_discover(req: MonitorDiscoverReq) -> dict:
        params = req.model_dump()
        result = await _guard_gpu(op_monitor_discover, params)
        _log_experiment("monitor_discover", params, "done", result=result)
        return result

    @app.post("/api/monitor/score")
    async def monitor_score(req: MonitorScoreReq) -> dict:
        return await _guard_gpu(op_monitor_score, req.model_dump())

    @app.post("/api/steer")
    async def steer(req: SteerReq) -> dict:
        return await _guard_gpu(op_steer, req.model_dump())

    @app.post("/api/sweep")
    async def sweep(req: SweepReq) -> dict:
        return await _guard_gpu(op_sweep, req.model_dump())

    @app.post("/api/benchmark")
    async def benchmark(req: BenchmarkReq) -> dict:
        params = req.model_dump()
        result = await _guard_gpu(op_benchmark, params)
        _log_experiment("benchmark", params, "done", result=result)
        return result

    @app.post("/api/autopilot")
    async def autopilot(req: AutopilotReq) -> dict:
        params = req.model_dump()
        result = await _guard_gpu(op_autopilot, params)
        _log_experiment("autopilot", params, "done", result=result)
        return result

    @app.post("/api/recipes")
    async def save_recipe(request: Request) -> dict:
        try:
            body = await request.json()
        except Exception:
            body = {}
        body = body or {}

        def _save() -> dict:
            if body.get("kind") == "manifold":
                if "recipe" not in last_manifold:
                    raise ValueError("run a pullback before saving a manifold recipe")
                recipe = last_manifold["recipe"]
                store.save(recipe, benchmark_results=last_manifold.get("result"), examples=recipe.examples)
                return {"recipe_id": recipe.recipe_id, "status": recipe.status}
            if "recipe" not in last_bench:
                raise ValueError("run a benchmark before saving a recipe")
            recipe = attach_benchmark_to_recipe(last_bench["recipe"], last_bench["result"])
            store.save(recipe, benchmark_results=last_bench["result"], examples=recipe.examples)
            return {"recipe_id": recipe.recipe_id, "status": recipe.status}

        return await _guard(_save)

    @app.get("/api/notebook")
    async def notebook() -> dict:
        return await _guard(service.notebook)

    @app.post("/api/notebook")
    async def save_note(req: NoteReq) -> dict:
        return await _guard(service.save_notebook_entry, req.model_dump())

    @app.get("/api/recipes")
    async def recipes() -> list[dict]:
        return await _guard(store.search, "", "all")

    @app.get("/api/recipes/{recipe_id}")
    async def recipe_detail(recipe_id: str) -> dict:
        def _load() -> dict:
            return store.load(recipe_id).to_dict()

        try:
            return await run_in_threadpool(_load)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"recipe {recipe_id} not found") from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ---- behavior monitors (the detection half) ----
    @app.post("/api/monitors")
    async def save_monitor() -> dict:
        def _save() -> dict:
            if "monitor" not in last_monitor:
                raise ValueError("discover a monitor before saving")
            m = last_monitor["monitor"]
            monitor_store.save(m)
            return {"monitor_id": m.monitor_id, "status": m.status}

        return await _guard(_save)

    @app.get("/api/monitors")
    async def monitors() -> list[dict]:
        return await _guard(monitor_store.search, "", "all")

    @app.get("/api/monitors/{monitor_id}")
    async def monitor_detail(monitor_id: str) -> dict:
        def _load() -> dict:
            return monitor_store.load(monitor_id).to_dict()

        try:
            return await run_in_threadpool(_load)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"monitor {monitor_id} not found") from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ---- async job API: submit a model op, poll for the result (serialized on the GPU) ----
    @app.post("/api/jobs")
    async def submit_job(request: Request) -> dict:
        try:
            body = await request.json()
        except Exception:
            body = {}
        body = body or {}
        op, params = body.get("op"), (body.get("params") or {})
        if op not in OPS:
            raise HTTPException(status_code=400, detail=f"unknown op {op!r}; choose from {sorted(OPS)}")
        job_id = uuid.uuid4().hex[:12]
        jobs[job_id] = {"id": job_id, "op": op, "params": params, "status": "queued",
                        "result": None, "error": None, "created_at": time.time(),
                        "started_at": None, "finished_at": None}
        asyncio.create_task(_run_job(job_id, op, params))
        return {"job_id": job_id, "status": "queued"}

    @app.get("/api/jobs")
    async def list_jobs() -> list[dict]:
        return [{k: j[k] for k in ("id", "op", "status", "created_at", "started_at", "finished_at")}
                for j in jobs.values()]

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"job {job_id} not found")
        return job

    @app.get("/api/experiments")
    async def experiments(limit: int = 50) -> list[dict]:
        return exp_log.tail(limit) if exp_log else []

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    return app
