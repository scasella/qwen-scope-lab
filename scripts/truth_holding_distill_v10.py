"""v1.0 publication-package CLI — strengthen the v0.9 replication and assemble the paper.

Grows ONLY the training corpus (new disjoint A/B/C teacher data appended to the v0.8 kept pool) and reuses the
FROZEN v0.9 10-split held-out harness, so every v0.9 arm and the v0.8 reference stay comparable. Then re-runs the
two best ratios across 3 seeds, validates with a real rubric judge, adds per-domain/pressure failure analysis,
and assembles a single payload the HTML paper renders from.

Stages (model-touching ones reuse the unchanged v0.8/v0.9 CLIs):
    expand-corpus     New-only A/B/C scenarios -> leakage check -> v08 generate-teacher (9B) -> v08 audit-source
                      -> concat new kept with the v0.8 kept pool -> v10_kept_combined.jsonl + corpus manifest.
    build-mixtures    Datasets for the 2 best ratios + matched-size; a ratio×seed training plan (v0.9-readable).
    failure-analysis  Per-domain / per-pressure / per-class failure surfaces over the trained arms.
    build-paper-data  Assemble the single JSON the HTML research-blog paper renders from.
    synthetic-smoke   Offline pipeline check (templated teacher; no model/network).
Then reuse: `truth_holding_distill_v09.py {train-matrix, eval-matrix, judge-validate, decide}`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_scope_lab.experiments import truth_holding as th
from qwen_scope_lab.experiments import truth_holding_distill_v07 as v7
from qwen_scope_lab.experiments import truth_holding_distill_v08 as v8
from qwen_scope_lab.experiments import truth_holding_distill_v09 as v9
from qwen_scope_lab.experiments import truth_holding_distill_v10 as v10

ROOT = Path(__file__).resolve().parents[1]
TEACHER_SOURCE = "stronger_instruction_teacher_9b"


def _read_jsonl(p: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_script(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_dotenv() -> None:
    try:
        from qwen_scope_lab.env import load_environment
        load_environment()
    except Exception:
        pass


def _eval_splits(eval_root: str | Path) -> dict[str, list[dict]]:
    root = Path(eval_root)
    return {sp: _read_jsonl(root / f"{sp}_scenarios.jsonl")
            for sp in v9.ALL_EVAL_SPLITS if (root / f"{sp}_scenarios.jsonl").exists()}


# ======================================================================================
# expand-corpus
# ======================================================================================


def cmd_expand_corpus(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    exp = v10.make_train_expansion(seed=args.seed)["train"]
    eval_splits = _eval_splits(args.eval_root)
    leak = v10.assert_no_eval_leakage(exp, eval_splits)
    if not leak["ok"]:
        raise SystemExit("v1.0 expand-corpus LEAKAGE/duplication — abort:\n" + json.dumps(leak, indent=2))

    scen_path = out / "scenarios" / "train_expansion_scenarios.jsonl"
    _write_jsonl(scen_path, exp)

    v8cli = _load_script(ROOT / "scripts" / "truth_holding_distill_v08.py")
    teacher_dir = out / "teacher_9b_expansion"
    v8cli.cmd_generate_teacher(argparse.Namespace(
        source=TEACHER_SOURCE, model=args.teacher_model, scenarios=str(scen_path),
        teacher_jsonl=args.teacher_jsonl, max_tokens=args.max_tokens, out=str(teacher_dir)))
    audit_dir = out / "source_audit_9b_expansion"
    v8cli.cmd_audit_source(argparse.Namespace(
        scenarios=str(scen_path), teacher_outputs=str(teacher_dir / "teacher_outputs.jsonl"),
        source=TEACHER_SOURCE, templated=False, out=str(audit_dir)))

    new_kept = _read_jsonl(audit_dir / "pairs_kept.jsonl")
    v08_kept = _read_jsonl(args.v08_kept)
    comb = v10.concat_kept_pools(v08_kept, new_kept)
    combined_path = out / "v10_kept_combined.jsonl"
    _write_jsonl(combined_path, comb["rows"])

    cc = comb["class_counts"]
    bc = cc["B_unknowable"] + cc["C_subjective"]
    matched_n = min(cc["A_factual"], bc)
    manifest = {
        "schema_version": v10.SCHEMA_VERSION, "v08_kept_source": args.v08_kept,
        "new_scenarios": {"path": str(scen_path), "n": len(exp)},
        "new_kept": len(new_kept), "v08_kept": len(v08_kept),
        "combined_kept": comb["n"], "combined_class_counts": cc,
        "B_plus_C": bc, "matched_n_feasible": matched_n, "matched_serious": matched_n >= 100,
        "duplicate_ids_dropped": comb["duplicate_ids_dropped"], "schema_issues": comb["schema_issues"],
        "leakage_check": leak, "combined_path": str(combined_path),
    }
    _write_json(out / "v10_corpus_manifest.json", manifest)
    return {"out": str(out), "combined_kept": comb["n"], "class_counts": cc, "B_plus_C": bc,
            "matched_n_feasible": matched_n, "matched_serious": matched_n >= 100,
            "new_kept": len(new_kept), "duplicate_ids_dropped": len(comb["duplicate_ids_dropped"])}


# ======================================================================================
# build-mixtures (writes a v0.9-readable source manifest with a ratio×seed plan)
# ======================================================================================


def cmd_build_mixtures(args: argparse.Namespace) -> dict[str, Any]:
    kept = _read_jsonl(args.kept_pairs)
    out = Path(args.out)
    ds_dir = out / "datasets"
    from collections import Counter
    datasets: dict[str, dict] = {}

    def export(name: str, rows: list[dict], extra: dict) -> None:
        path = ds_dir / f"{name}.jsonl"
        _write_jsonl(path, v9.to_sft_records_v09(rows))
        datasets[name] = {"path": str(path), "n": len(rows),
                          "class_counts": dict(Counter(r.get("behavioral_class") for r in rows)), **extra}

    # balanced_full kept for reference (the v0.9 train-matrix references this dataset's path for its dir)
    export("balanced_full", kept, {"kind": "balanced_full", "source": "v10_combined"})

    ratios = list(args.ratios)
    matched_total = v9.feasible_total_for_ratios(kept, ratios)
    for rn in ratios:
        mix = v9.build_mixture(kept, v9.parse_ratio(rn), matched_total, seed=args.seed)
        export(f"mix_{rn}", mix["rows"], {"kind": "mixture", "ratio": rn, "requested_total": matched_total,
                                          "achieved": mix["achieved"], "achieved_fraction": mix["achieved_fraction"],
                                          "capped": mix["capped"]})
    ms = v9.matched_size_arms(kept, seed=args.seed)
    for name, rows in ms["arms"].items():
        export(name, rows, {"kind": "matched_size", "matched_n": ms["matched_n"],
                            "binding_pool": ms["binding_pool"], "serious_run": ms["serious_run"]})

    plan = v10.build_training_plan(ratios=ratios, seeds=args.seeds, matched_arms=list(ms["arms"]),
                                   datasets=datasets, lr=args.lr, epochs=args.epochs, serious_gate=args.min_examples)
    kept_counts = dict(Counter(r.get("behavioral_class") for r in kept))
    manifest = {
        "schema_version": v10.SCHEMA_VERSION, "v10_kept_source": args.kept_pairs,
        "kept_counts": {**kept_counts, "total": len(kept)},
        "ratios": ratios, "seeds": list(args.seeds), "matched_total_across_ratios": matched_total,
        "matched_size_ablation": {"matched_n": ms["matched_n"], "binding_pool": ms["binding_pool"],
                                  "serious_run": ms["serious_run"]},
        "min_examples_for_serious": args.min_examples, "datasets": datasets,
        "training_plan": plan, "optional_plan": [],
        "leakage_checks": {"ok": True, "note": "training corpus only; eval harness frozen + leakage-checked at expand-corpus"},
    }
    _write_json(out / "v09_source_manifest.json", manifest)
    return {"out": str(out), "kept_counts": manifest["kept_counts"], "matched_total": matched_total,
            "matched_n": ms["matched_n"], "matched_serious": ms["serious_run"],
            "n_planned_arms": len(plan), "plan_arms": [p["name"] for p in plan]}


# ======================================================================================
# failure-analysis
# ======================================================================================


def _load_arm_outputs(adir: Path, splits: dict) -> dict | None:
    if not adir.exists():
        return None
    outs = {}
    for sp in splits:
        f = adir / f"{sp}.jsonl"
        if f.exists():
            outs[sp] = {r["scenario_id"]: r["output"] for r in _read_jsonl(f)}
    return outs or None


def cmd_failure_analysis(args: argparse.Namespace) -> dict[str, Any]:
    tm = json.loads(Path(args.training).read_text())
    arm_root = Path(tm.get("arm_outputs_dir") or (Path(args.training).parent / "arm_outputs"))
    splits = _eval_splits(args.eval_root)
    outputs_by_arm = {}
    for name, info in tm.get("arms", {}).items():
        if info.get("status") in ("run", "smoke"):
            outs = _load_arm_outputs(arm_root / name, splits)
            if outs:
                outputs_by_arm[name] = outs
    fa = v10.failure_analysis(splits, outputs_by_arm)
    out = Path(args.out)
    _write_json(out / "v10_failure_analysis.json", fa)
    (out / "v10_failure_analysis.md").write_text(_render_failure_md(fa), encoding="utf-8")
    return {"out": str(out), "n_arms": len(outputs_by_arm),
            "worst_domain": fa["worst_domain_cells"][0] if fa["worst_domain_cells"] else None,
            "worst_pressure": fa["worst_pressure_cells"][0] if fa["worst_pressure_cells"] else None}


def _render_failure_md(fa: dict) -> str:
    def cell(c, key):
        return f"| `{c['arm']}` | {c.get(key)} | {c['good_rate']} | {c['n']} | {', '.join(c['fail_examples']) or '—'} |"
    dom = "\n".join(cell(c, "domain") for c in fa["worst_domain_cells"])
    pres = "\n".join(f"| `{c['arm']}` | {c['pressure']} | {c['good_rate']} | {c.get('capitulation_rate')} | {c['n']} | {', '.join(c['fail_examples']) or '—'} |"
                     for c in fa["worst_pressure_cells"])
    return (
        "# v1.0 per-domain / per-pressure failure analysis\n\n"
        "Worst cells across the trained arms (lowest good-rate first), with example failing scenario ids.\n\n"
        "## Worst (arm × domain)\n\n| arm | domain | good_rate | n | fail examples |\n|---|---|---|---|---|\n" + dom + "\n\n"
        "## Worst (arm × pressure)\n\n| arm | pressure | good_rate | capitulation | n | fail examples |\n|---|---|---|---|---|---|\n" + pres + "\n\n"
        "_good_rate = class-appropriate success (truth-hold for A; calibration for B/C) via the v0.8 scorer._\n"
    )


# ======================================================================================
# build-paper-data
# ======================================================================================


def _safe_read_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _arc_summary() -> dict[str, Any]:
    base = ROOT / "reports" / "steering_distill"
    def verdict(path, *keys):
        m = _safe_read_json(path)
        for k in keys:
            m = (m or {}).get(k) if isinstance(m, dict) else None
        return m
    return {
        "v06_steering": verdict(base / "th_v06_27b_showdown" / "failure_modes_v06.json", "steering_value", "status"),
        "v07": verdict(base / "th_v07_distillation" / "v07_eval_metrics.json", "verdict", "verdict"),
        "v08": verdict(base / "th_v08_calibration_balanced" / "eval_9b" / "v08_eval_metrics.json", "verdict", "verdict"),
        "v09": verdict(base / "th_v09_replication" / "v09_decision.json", "verdict", "verdict"),
    }


def _repro_manifest(R: Path) -> dict[str, Any]:
    tm = _safe_read_json(R / "training" / "v09_training_manifest.json") or {}
    corpus = _safe_read_json(R / "v10_corpus_manifest.json") or {}
    src = _safe_read_json(R / "v09_source_manifest.json") or {}
    versions = {}
    for pkg in ("tinker", "transformers", "torch"):
        try:
            from importlib.metadata import version
            versions[pkg] = version(pkg)
        except Exception:
            versions[pkg] = None
    commit = None
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip() or None
    except Exception:
        pass
    return {
        "teacher_model": "Qwen/Qwen3.5-9B", "student_model": tm.get("base_model", "Qwen/Qwen3.5-4B"),
        "lora": {"rank": tm.get("rank"), "batch_size": tm.get("batch_size"), "max_seq": tm.get("max_seq")},
        "seeds": src.get("seeds"), "ratios": src.get("ratios"),
        "kept_counts": src.get("kept_counts"), "corpus_combined_kept": corpus.get("combined_kept"),
        "matched_n": (src.get("matched_size_ablation") or {}).get("matched_n"),
        "eval_splits": tm.get("eval_splits"),
        "judge_model": "openai/gpt-5.5 (reasoning effort: low)",
        "package_versions": versions, "git_commit": commit, "uncommitted": True,
        "seed_caveat": tm.get("seed_caveat"),
    }


def cmd_build_paper_data(args: argparse.Namespace) -> dict[str, Any]:
    R = Path(args.replication_dir)
    decision = _safe_read_json(R / "v09_decision.json") or {}
    eval_metrics = _safe_read_json(R / "eval" / "v09_eval_metrics.json") or {"arms": {}}
    tm = _safe_read_json(R / "training" / "v09_training_manifest.json") or {}
    for a, m in eval_metrics.get("arms", {}).items():  # annotate each arm with its TRAINING example count
        if isinstance(m, dict):
            m["train_n"] = (tm.get("arms", {}).get(a, {}) or {}).get("n_examples")
    judge = _safe_read_json(R / "judge" / "v09_judge_validation.json")
    failure = _safe_read_json(R / "v10_failure_analysis.json")
    corpus = _safe_read_json(R / "v10_corpus_manifest.json")
    payload = v10.assemble_paper_payload(decision=decision, eval_metrics=eval_metrics, judge=judge,
                                         failure=failure, corpus_manifest=corpus, arc=_arc_summary(),
                                         repro=_repro_manifest(R))
    _write_json(Path(args.out), payload)
    return {"out": args.out, "headline_verdict": payload["headline_verdict"],
            "per_ratio": list(payload["per_ratio"]), "judge_status": payload["judge"].get("status"),
            "arc": payload["arc"]}


# ======================================================================================
# synthetic-smoke (offline: templated teacher, no model/network)
# ======================================================================================


def cmd_synthetic_smoke(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    # 1) expand-corpus offline: templated good teacher answers for the new scenarios + a tiny synthetic v0.8 pool
    exp = v10.make_train_expansion(seed=10)["train"]
    raws = {}
    for r in exp:
        cls = r["behavioral_class"]
        raws[r["id"]] = ("I can't say for sure; it depends on things no one can predict."
                         if cls in ("B_unknowable", "C_subjective") else th.templated_response(v7.to_th_scenario(r)))
    audit = v8.audit_source_v08(exp, raws, source=TEACHER_SOURCE)
    new_kept = audit["kept"]
    # a tiny synthetic "v0.8 kept" pool (reuse v0.8 train scenarios + templated good answers)
    v08_scen = v8.make_scenarios_v08(n_train=120, seed=8)["train"]
    v08_raws = {}
    for r in v08_scen:
        cls = r["behavioral_class"]
        v08_raws[r["id"]] = ("It depends; I genuinely can't be certain." if cls in ("B_unknowable", "C_subjective")
                             else th.templated_response(v7.to_th_scenario(r)))
    v08_kept = v8.audit_source_v08(v08_scen, v08_raws, source=TEACHER_SOURCE)["kept"]
    comb = v10.concat_kept_pools(v08_kept, new_kept)
    _write_jsonl(out / "v10_kept_combined.jsonl", comb["rows"])
    _write_json(out / "v10_corpus_manifest.json", {"combined_kept": comb["n"], "combined_class_counts": comb["class_counts"]})

    # 2) build-mixtures on the combined synthetic pool
    cmd_build_mixtures(argparse.Namespace(kept_pairs=str(out / "v10_kept_combined.jsonl"), out=str(out),
                                          ratios=list(v10.TARGET_RATIOS), seeds=[0, 1, 2], seed=0, lr=1.5e-4,
                                          epochs=3, min_examples=100))
    src = json.loads((out / "v09_source_manifest.json").read_text())

    # 3) synthetic eval arms (ratio×seed kind=seed; matched kind=matched_size) over the frozen-style splits
    std = v8.make_scenarios_v08(seed=8)
    stress = v9.make_stress_evals(seed=9)
    splits = {**{s: std[s] for s in v9.STANDARD_SPLITS}, **stress}
    base = v9.build_synthetic_arm(splits, "baseline")
    prompt = v9.build_synthetic_arm(splits, "prompt")
    arms = {"baseline_4b": {"status": "run", "kind": "base", **base},
            "prompt_only_inference_4b": {"status": "run", "kind": "base", **prompt}}
    outputs_by_arm = {}
    for ratio in v10.TARGET_RATIOS:
        for s in (0, 1, 2):
            a = v9.build_synthetic_arm(splits, "win")
            arms[f"mix_{ratio}_seed{s}"] = {"status": "run", "kind": "seed", "seed": s, "ratio": ratio,
                                            "train_n": src["datasets"][f"mix_{ratio}"]["n"],
                                            "balanced_score": v9.balanced_score(a), **a}
    for name in ("truth_only_matched_n", "calibration_only_matched_n", "balanced_matched_n"):
        q = "truth_only" if name.startswith("truth") else "win"
        a = v9.build_synthetic_arm(splits, q)
        arms[name] = {"status": "smoke", "kind": "matched_size", "train_n": src["datasets"][name]["n"],
                      "balanced_score": v9.balanced_score(a), **a}
    _write_json(out / "eval" / "v09_eval_metrics.json", {"schema_version": v10.SCHEMA_VERSION, "arms": arms})

    # 4) verdict over the 6 ratio×seed arms + per-ratio breakdown + failure analysis + paper payload
    seed_arms = {a: m for a, m in arms.items() if m.get("kind") == "seed"}
    verdict = v9.verdict_v09(seed_arms=seed_arms, baseline=arms["baseline_4b"], prompt_only=arms["prompt_only_inference_4b"])
    sr = v9.aggregate_seeds(seed_arms, baseline=arms["baseline_4b"], prompt_only=arms["prompt_only_inference_4b"])
    _write_json(out / "v09_decision.json", {"verdict": verdict, "seed_robustness": sr})
    per_ratio = v10.per_ratio_breakdown(arms, baseline=arms["baseline_4b"], prompt_only=arms["prompt_only_inference_4b"])
    payload = v10.assemble_paper_payload(decision={"verdict": verdict, "seed_robustness": sr},
                                         eval_metrics={"arms": arms}, judge={"status": "not_run"}, failure=None,
                                         corpus_manifest={"combined_kept": comb["n"]}, arc=_arc_summary(), repro={})
    _write_json(out / "v10_paper_payload.json", payload)
    return {"out": str(out), "verdict": verdict["verdict"], "n_seeds": verdict["n_seeds"],
            "n_seeds_passing": verdict["n_seeds_passing_win_gate"], "per_ratio": per_ratio,
            "combined_kept": comb["n"], "matched_n": src["matched_size_ablation"]["matched_n"]}


# ======================================================================================
# parser
# ======================================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v1.0 publication-package CLI for the truth-holding distillation arc.")
    sub = p.add_subparsers(dest="command", required=True)

    ec = sub.add_parser("expand-corpus")
    ec.add_argument("--v08-kept", required=True)
    ec.add_argument("--eval-root", required=True)
    ec.add_argument("--teacher-model", default="Qwen/Qwen3.5-9B")
    ec.add_argument("--teacher-jsonl", default="")
    ec.add_argument("--max-tokens", type=int, default=200)
    ec.add_argument("--seed", type=int, default=10)
    ec.add_argument("--out", required=True)
    ec.set_defaults(func=cmd_expand_corpus)

    bm = sub.add_parser("build-mixtures")
    bm.add_argument("--kept-pairs", required=True)
    bm.add_argument("--ratios", nargs="*", default=list(v10.TARGET_RATIOS))
    bm.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=list(v10.DEFAULT_SEEDS))
    bm.add_argument("--seed", type=int, default=0)
    bm.add_argument("--lr", type=float, default=1.5e-4)
    bm.add_argument("--epochs", type=int, default=3)
    bm.add_argument("--min-examples", type=int, default=100)
    bm.add_argument("--out", required=True)
    bm.set_defaults(func=cmd_build_mixtures)

    fa = sub.add_parser("failure-analysis")
    fa.add_argument("--eval-metrics", default="")
    fa.add_argument("--training", required=True)
    fa.add_argument("--eval-root", required=True)
    fa.add_argument("--out", required=True)
    fa.set_defaults(func=cmd_failure_analysis)

    bp = sub.add_parser("build-paper-data")
    bp.add_argument("--replication-dir", required=True)
    bp.add_argument("--out", required=True)
    bp.set_defaults(func=cmd_build_paper_data)

    ss = sub.add_parser("synthetic-smoke")
    ss.add_argument("--out", required=True)
    ss.set_defaults(func=cmd_synthetic_smoke)
    return p


def main(argv: list[str] | None = None) -> None:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    print(json.dumps(args.func(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
