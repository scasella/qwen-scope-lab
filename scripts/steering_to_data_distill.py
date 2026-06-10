"""Steering-to-data distillation CLI.

Compile a validated (or benchmarked) runtime steering recipe into ordinary training data —
SFT and preference pairs — so the steered behavior can later be learned without activation
hooks, and (optionally) evaluate a distilled checkpoint against baseline and runtime steering.

Subcommands
-----------
    generate         Generate unsteered/steered pairs, score+filter them, export the dataset.
    eval             Compare baseline / runtime-steering / distilled / prompt-only arms.
    synthetic-smoke  End-to-end on a tiny in-memory model + crafted pairs (no GPU, no network).

Examples
--------
    # From a saved recipe, against a running lab service:
    python scripts/steering_to_data_distill.py generate \\
        --url http://127.0.0.1:7870 \\
        --recipe recipes/concise_answers_l1_f2_v1/recipe.json \\
        --prompts data/experiments/steering_distill/prompts.jsonl \\
        --out reports/steering_distill/run_001

    # Explicit steering config:
    python scripts/steering_to_data_distill.py generate \\
        --url http://127.0.0.1:7870 \\
        --feature-id 1234 --layer 12 --strength 6 --target-name concise \\
        --prompts data/experiments/steering_distill/prompts.jsonl \\
        --out reports/steering_distill/run_002

    # No model required:
    python scripts/steering_to_data_distill.py synthetic-smoke --out reports/steering_distill/smoke

    # Evaluate a distilled adapter once it exists:
    python scripts/steering_to_data_distill.py eval \\
        --baseline-url http://127.0.0.1:7870 \\
        --distilled-url http://127.0.0.1:7871 \\
        --recipe recipes/concise_answers_l1_f2_v1/recipe.json \\
        --prompt-corpus data/experiments/steering_distill/eval_prompts.jsonl \\
        --target concise --out reports/steering_distill/eval_001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_scope_lab.experiments import distill_quality as dq
from qwen_scope_lab.experiments import steering_distill as sd
from qwen_scope_lab.prompt_sets import format_prompt_only, load_prompt_set


# --------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------


def make_backend(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    """Resolve a generation backend + its model metadata from --url / --config / --backend echo."""
    if getattr(args, "url", None):
        backend = sd.HttpGenerationBackend.connect(args.url, timeout=getattr(args, "timeout", 600.0))
        meta = {"model_id": backend.model_id, "sae_id": backend.sae_id, "num_layers": backend.num_layers, "d_sae": backend.d_sae}
        return backend, meta
    if getattr(args, "config", None):
        # In-process real model (CUDA/MLX/dev) — heavy imports, so do them lazily.
        from qwen_scope_lab.benchmark import ServiceGenerationBackend
        from qwen_scope_lab.service import SteeringService

        service = SteeringService.from_config_path(args.config)
        cfg = service.config
        backend = ServiceGenerationBackend(service)
        meta = {"model_id": cfg.model_id, "sae_id": cfg.sae_id, "num_layers": cfg.num_layers, "d_sae": cfg.d_sae}
        return backend, meta
    # Echo: a deterministic, model-free stand-in (used by smoke and for offline dev).
    from qwen_scope_lab.benchmark import EchoGenerationBackend

    backend = EchoGenerationBackend()
    meta = {"model_id": "synthetic/echo", "sae_id": "synthetic/echo-sae", "num_layers": 0, "d_sae": backend.d_sae}
    return backend, meta


# --------------------------------------------------------------------------------------
# Spec / config assembly
# --------------------------------------------------------------------------------------


def build_spec(args: argparse.Namespace, meta: dict[str, Any]) -> sd.SteerSpec:
    if getattr(args, "probe_id", None):
        missing = [n for n in ("layer", "strength", "target_name") if getattr(args, n, None) is None]
        if missing:
            raise SystemExit(
                "probe/direction mode requires --layer --strength --target-name "
                f"(missing: {', '.join('--' + m.replace('_', '-') for m in missing)})"
            )
        spec = sd.SteerSpec.from_probe(
            probe_id=args.probe_id,
            layer=args.layer,
            strength=args.strength,
            target_name=args.target_name,
            target_description=getattr(args, "target_description", "") or "",
            model_id=meta.get("model_id", ""),
        )
        _check_ranges(spec, meta)
        return spec
    if getattr(args, "recipe", None):
        spec = sd.SteerSpec.from_recipe(sd.load_recipe(args.recipe))
    else:
        missing = [name for name in ("feature_id", "layer", "strength", "target_name") if getattr(args, name, None) is None]
        if missing:
            raise SystemExit(
                "explicit steering mode requires --feature-id --layer --strength --target-name "
                f"(missing: {', '.join('--' + m.replace('_', '-') for m in missing)}); or pass --recipe <path>"
            )
        spec = sd.SteerSpec.explicit(
            feature_id=args.feature_id,
            layer=args.layer,
            strength=args.strength,
            target_name=args.target_name,
            target_description=getattr(args, "target_description", "") or "",
        )
    # Fill model metadata from the live backend when the spec doesn't carry it.
    spec.model_id = spec.model_id or meta.get("model_id", "")
    spec.sae_id = spec.sae_id or meta.get("sae_id", "")
    _check_ranges(spec, meta)
    return spec


def _check_ranges(spec: sd.SteerSpec, meta: dict[str, Any]) -> None:
    num_layers = int(meta.get("num_layers", 0) or 0)
    d_sae = int(meta.get("d_sae", 0) or 0)
    if num_layers and not (0 <= spec.layer < num_layers):
        raise SystemExit(f"layer {spec.layer} out of range for model with {num_layers} layers")
    if not spec.is_direction and d_sae and not (0 <= spec.feature_id < d_sae):
        raise SystemExit(f"feature_id {spec.feature_id} out of range for SAE with d_sae={d_sae}")


def build_distill_cfg(args: argparse.Namespace, spec: sd.SteerSpec | None) -> sd.DistillConfig:
    target = getattr(args, "target", None) or (spec.target_name if spec else None) or "concise"
    return sd.DistillConfig(
        target=target,
        min_delta=getattr(args, "min_delta", 0.0),
        max_length_ratio=getattr(args, "max_length_ratio", 1.0),
        min_content_overlap=getattr(args, "min_content_overlap", 0.5),
        concise_ref_tokens=getattr(args, "concise_ref_tokens", 80),
        score_command=getattr(args, "score_command", None),
    )


def _gen_params(args: argparse.Namespace) -> sd.GenParams:
    return sd.GenParams(
        max_new_tokens=getattr(args, "max_new_tokens", 64),
        temperature=getattr(args, "temperature", 0.0),
        seed=getattr(args, "seed", 0),
        prompt_only_instruction=getattr(args, "prompt_only_instruction", "") or "",
        include_random_control=getattr(args, "random_control", False),
    )


# --------------------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> dict[str, Any]:
    backend, meta = make_backend(args)
    spec = build_spec(args, meta)
    cfg = build_distill_cfg(args, spec)
    prompts = load_prompt_set(args.prompts)
    params = _gen_params(args)
    pairs = sd.generate_pairs(spec, prompts, backend, params)
    result = sd.distill_pairs(pairs, spec, cfg, params)
    paths = sd.write_outputs(args.out, spec, result, cfg)
    return {
        "out": str(args.out),
        "paths": paths,
        "metrics": result["metrics"],
        "recipe_status": spec.recipe_status,
        "validated": spec.validated,
    }


def cmd_synthetic_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Drive the whole pipeline with no model: crafted pairs (guaranteed keep/reject mix) plus a
    handful generated through the echo backend (proves the generation wiring), then distill+export."""
    from qwen_scope_lab.benchmark import EchoGenerationBackend

    target = getattr(args, "target", None) or "concise"
    cfg = sd.DistillConfig(target=target)
    spec = sd.SteerSpec.explicit(
        feature_id=1234, layer=6, strength=6.0, target_name=target, model_id="synthetic/echo", sae_id="synthetic/echo-sae"
    )
    spec.source = "synthetic"
    spec.recipe_id = "synthetic_smoke"
    spec.recipe_status = "benchmarked"

    crafted = sd.build_synthetic_pairs(target)
    smoke_prompts = [
        {"id": "echo_1", "prompt": "Explain what a sparse autoencoder does.", "metadata": {"task": target}},
        {"id": "echo_2", "prompt": "Write a concise factual answer about steering.", "metadata": {"task": target}},
    ]
    params = sd.GenParams(max_new_tokens=24, prompt_only_instruction="Answer in one short sentence.", include_random_control=True)
    echo_pairs = sd.generate_pairs(spec, smoke_prompts, EchoGenerationBackend(), params)
    result = sd.distill_pairs(crafted + echo_pairs, spec, cfg, params)
    paths = sd.write_outputs(args.out, spec, result, cfg)
    return {"out": str(args.out), "paths": paths, "metrics": result["metrics"]}


def _arm_rows(backend: Any, prompts: list[dict[str, Any]], params: sd.GenParams, *, steer_spec: sd.SteerSpec | None = None,
              instruction: str = "") -> list[dict[str, Any]]:
    rows = []
    direction_mode = bool(steer_spec and steer_spec.is_direction)
    iv = steer_spec.intervention() if (steer_spec and not direction_mode) else None
    for row in prompts:
        prompt = row["prompt"]
        if direction_mode:
            res = backend.steer_direction(
                prompt, layer=steer_spec.layer, strength=steer_spec.strength, probe_id=steer_spec.probe_id,
                direction=steer_spec.direction, max_new_tokens=params.max_new_tokens, temperature=params.temperature, seed=params.seed,
            )
            output = res.get("steered_text") or res.get("text") or ""
        elif iv is not None:
            res = backend.steer(prompt, iv, max_new_tokens=params.max_new_tokens, temperature=params.temperature, seed=params.seed)
            output = res.get("steered_text") or res.get("text") or ""
        else:
            effective = format_prompt_only(prompt, instruction) if instruction else prompt
            output = backend.generate(effective, max_new_tokens=params.max_new_tokens, temperature=params.temperature, seed=params.seed)["text"]
        rows.append({"id": row.get("id", ""), "prompt": prompt, "output": output, "metadata": row.get("metadata", {})})
    return rows


def cmd_eval(args: argparse.Namespace) -> dict[str, Any]:
    cfg = build_distill_cfg(args, None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        eval_result = sd.evaluate(sd.build_synthetic_eval_arms(), cfg)
    else:
        if not args.baseline_url:
            raise SystemExit("eval requires --baseline-url (or --synthetic)")
        params = _gen_params(args)
        prompts = load_prompt_set(args.prompt_corpus)
        baseline_backend = sd.HttpGenerationBackend.connect(args.baseline_url, timeout=args.timeout)
        spec = build_spec(args, {"model_id": baseline_backend.model_id, "sae_id": baseline_backend.sae_id,
                                 "num_layers": baseline_backend.num_layers, "d_sae": baseline_backend.d_sae}) \
            if (args.recipe or args.probe_id or args.feature_id is not None) else None

        arms: dict[str, list[dict[str, Any]]] = {"baseline": _arm_rows(baseline_backend, prompts, params)}
        if spec is not None:
            steered_backend = sd.HttpGenerationBackend.connect(args.steered_url, timeout=args.timeout) if args.steered_url else baseline_backend
            arms["runtime_steering"] = _arm_rows(steered_backend, prompts, params, steer_spec=spec)
        if args.distilled_url:
            distilled_backend = sd.HttpGenerationBackend.connect(args.distilled_url, timeout=args.timeout)
            arms["distilled"] = _arm_rows(distilled_backend, prompts, params)
        if args.prompt_only_instruction:
            arms["prompt_only"] = _arm_rows(baseline_backend, prompts, params, instruction=args.prompt_only_instruction)
        eval_result = sd.evaluate(arms, cfg)

    (out / "eval_metrics.json").write_text(json.dumps(eval_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "eval_report.md").write_text(sd.render_eval_report(eval_result), encoding="utf-8")
    return {
        "out": str(out),
        "paths": {"eval_metrics": str(out / "eval_metrics.json"), "eval_report": str(out / "eval_report.md")},
        "eval": eval_result,
    }


# --------------------------------------------------------------------------------------
# v0.2 warm-but-useful audit
# --------------------------------------------------------------------------------------


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _warmth_cfg(args: argparse.Namespace) -> "dq.WarmthConfig":
    return dq.WarmthConfig(
        min_relevance=args.min_relevance,
        max_repetition=args.max_repetition,
        max_genericness=args.max_genericness,
        max_stock_share=args.max_stock_share,
        max_unsupported_specifics=args.max_unsupported_specifics,
        min_sentiment=args.min_sentiment,
        reject_think=not args.allow_think,
    )


def cmd_audit(args: argparse.Namespace) -> dict[str, Any]:
    """Hardened v0.2 pass over existing distilled datasets and/or eval arms — no model calls."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = _warmth_cfg(args)
    result: dict[str, Any] = {"out": str(out), "paths": {}}

    pairs: list[dict[str, Any]] = []
    if args.synthetic and not args.pairs:
        pairs = dq.build_synthetic_warmth_pairs()
    for path in args.pairs or []:
        pairs += _read_jsonl(path)
    if pairs:
        audit = dq.audit_dataset(pairs, cfg)
        sd.write_jsonl(out / "pairs_kept_v2.jsonl", audit["kept"])
        sd.write_jsonl(out / "pairs_rejected_v2.jsonl", audit["rejected"])
        sd.write_jsonl(out / "sft_v2.jsonl", sd.to_sft_records(audit["kept"]))
        (out / "phrase_concentration.json").write_text(json.dumps(audit["phrase_concentration"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out / "audit_metrics.json").write_text(json.dumps(audit["metrics"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out / "dataset_audit.md").write_text(dq.render_dataset_audit(audit, title=args.title or "distilled dataset"), encoding="utf-8")
        result["dataset"] = {
            "n_pairs": audit["metrics"]["n_pairs"],
            "n_kept_v2": audit["metrics"]["n_kept_v2"],
            "keep_rate_v2": audit["metrics"]["keep_rate_v2"],
            "reject_reason_counts": audit["metrics"]["reject_reason_counts"],
            "warnings": audit["phrase_concentration"]["warnings"],
        }
        result["paths"].update({
            "dataset_audit": str(out / "dataset_audit.md"),
            "sft_v2": str(out / "sft_v2.jsonl"),
            "pairs_kept_v2": str(out / "pairs_kept_v2.jsonl"),
            "pairs_rejected_v2": str(out / "pairs_rejected_v2.jsonl"),
            "phrase_concentration": str(out / "phrase_concentration.json"),
        })

    arms: dict[str, Any] = {}
    if args.synthetic and not args.arms:
        arms = dq.build_synthetic_quality_arms()
    for path in args.arms or []:
        arms.update(json.loads(Path(path).read_text(encoding="utf-8")))
    if arms:
        judge = dq.make_command_judge(args.judge_command) if getattr(args, "judge_command", None) else None
        ev = dq.evaluate_quality_arms(arms, judge=judge)
        (out / "eval_quality.json").write_text(json.dumps(ev, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out / "eval_quality.md").write_text(dq.render_quality_eval(ev), encoding="utf-8")
        result["eval"] = {"verdict": ev["verdict"]["status"], "deltas_vs_baseline": ev["verdict"].get("deltas_vs_baseline", {})}
        result["paths"]["eval_quality"] = str(out / "eval_quality.md")

    if not pairs and not arms:
        raise SystemExit("audit needs --pairs and/or --arms (or --synthetic)")
    return result


# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------


def _add_backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--url", help="URL of a running lab service (e.g. http://127.0.0.1:7870)")
    p.add_argument("--config", help="Load the model in-process from a config YAML instead of over HTTP")
    p.add_argument("--backend", choices=["echo"], help="Use the model-free echo backend (offline)")
    p.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout in seconds")


def _add_spec_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--recipe", help="Path to a saved recipe.json (validated or benchmarked)")
    p.add_argument("--probe-id", help="CAA/direction steering: a saved probe id (residual direction)")
    p.add_argument("--feature-id", type=int, help="Explicit steering: SAE feature id")
    p.add_argument("--layer", type=int, help="Explicit steering: layer index")
    p.add_argument("--strength", type=float, help="Steering strength (SAE feature, or signed multiple of the unit direction)")
    p.add_argument("--target-name", help="Target behavior name (e.g. concise, json, calibrated, sentiment)")
    p.add_argument("--target-description", default="", help="Human description of the behavior")


def _add_filter_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--target", help="Scoring target: concise | json | calibrated | deference | generic (default: recipe's behavior)")
    p.add_argument("--min-delta", type=float, default=0.0, help="Keep only pairs whose steered target score beats unsteered by > this")
    p.add_argument("--max-length-ratio", type=float, default=1.0, help="concise: max steered/unsteered token ratio")
    p.add_argument("--min-content-overlap", type=float, default=0.5, help="concise: min fraction of steered content grounded in the baseline")
    p.add_argument("--concise-ref-tokens", type=int, default=80, help="concise: length scoring ~0 brevity")
    p.add_argument("--score-command", help="Custom scorer: a shell command fed candidate text on stdin, emitting a float on stdout")


def _add_gen_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt-only-instruction", default="", help="If set, also generate a prompt-only arm using this instruction")
    p.add_argument("--random-control", action="store_true", help="Also generate a random-feature control output per prompt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Steering-to-data distillation: turn a runtime steer into training data.")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Generate, score, filter, and export a steering-distilled dataset")
    _add_backend_args(g)
    _add_spec_args(g)
    _add_filter_args(g)
    _add_gen_args(g)
    g.add_argument("--prompts", required=True, help="Prompt corpus JSONL ({id, prompt, metadata})")
    g.add_argument("--out", required=True, help="Output directory for the dataset + report")
    g.set_defaults(func=cmd_generate)

    e = sub.add_parser("eval", help="Compare baseline / runtime-steering / distilled / prompt-only arms")
    e.add_argument("--baseline-url", help="URL of the baseline (unsteered) model service")
    e.add_argument("--steered-url", help="URL of a service to run runtime steering on (default: baseline-url)")
    e.add_argument("--distilled-url", help="URL of the distilled model service")
    e.add_argument("--prompt-corpus", help="Eval prompt corpus JSONL")
    e.add_argument("--synthetic", action="store_true", help="Use crafted arms (no network) — for testing the harness")
    _add_spec_args(e)
    _add_filter_args(e)
    _add_gen_args(e)
    e.add_argument("--out", required=True, help="Output directory for the eval report")
    e.set_defaults(func=cmd_eval)

    s = sub.add_parser("synthetic-smoke", help="Run the whole pipeline with no model (CI-safe)")
    s.add_argument("--target", default="concise", help="concise | json | calibrated")
    s.add_argument("--out", required=True, help="Output directory")
    s.set_defaults(func=cmd_synthetic_smoke)

    a = sub.add_parser("audit", help="v0.2 warm-but-useful hardened audit of a dataset and/or eval arms (no model)")
    a.add_argument("--pairs", nargs="+", help="pairs_*.jsonl file(s) to re-filter with hardened warmth gates")
    a.add_argument("--arms", nargs="+", help="eval arm JSON file(s) to re-score on quality metrics")
    a.add_argument("--synthetic", action="store_true", help="Use built-in crafted pairs/arms (no files needed)")
    a.add_argument("--title", default="", help="Title for the dataset audit report")
    a.add_argument("--out", required=True, help="Output directory")
    a.add_argument("--min-relevance", type=float, default=0.2)
    a.add_argument("--max-repetition", type=float, default=0.15)
    a.add_argument("--max-genericness", type=float, default=0.35)
    a.add_argument("--max-stock-share", type=float, default=0.25)
    a.add_argument("--max-unsupported-specifics", type=float, default=0.12)
    a.add_argument("--min-sentiment", type=float, default=0.55)
    a.add_argument("--allow-think", action="store_true", help="Do not reject outputs containing <think> tags")
    a.add_argument("--judge-command", help="Optional rubric judge: a command fed 'PROMPT…/OUTPUT…' on stdin, emitting a float (adds a non-gating judge score per arm)")
    a.set_defaults(func=cmd_audit)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = args.func(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
