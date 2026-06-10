"""v0.9 — replicate & stress-test the v0.8 calibration-balanced distillation win (staged CLI).

v0.8 was a strong SINGLE-RUN win. v0.9 is a RIGOR phase: does calibration-balanced stronger-teacher
distillation reliably improve truth-holding AND calibration across seeds, A/B/C mixture ratios,
matched-size ablations, rubric-judge validation, and harder/messier held-out evaluation?

Stages (each a subcommand so failures are diagnosable):
    preflight        Verify the v0.8 win, the v0.7 ambiguous-calibration failure, and the v0.6 steer-negative.
    make-stress-eval Write the 5 harder held-out stress splits + (re)write the 5 standard splits to one eval root.
    build-mixtures   Re-mix the v0.8 *kept* teacher corpus into A/B/C ratio datasets + matched-size ablations
                     (preserving every label); emit v09_source_manifest.json with a training plan. No model.
    train-matrix     LoRA the 4B from each planned arm's SFT (seeds / mixtures / matched-size / optional),
                     then sample baseline / prompt-only / each arm on ALL 10 eval splits (Tinker). Honest
                     per-arm status: run / smoke / skipped_by_gate / error / not_run (+ blocker).
    eval-matrix      Class+split+seed-aware scoring with bootstrap CIs; load the v0.8 reference arm too. No model.
    judge-validate   Optional rubric-judge validation (--judge-command / --judge-jsonl); deterministic-vs-judge
                     agreement. If no judge supplied -> status not_run (NOT "human validated"). No network in tests.
    decide           verdict_v09 + seed-robustness + mixture-sweep + ablation matrix + final decision. No model.
    synthetic-smoke  Whole pipeline with synthetic stand-ins (CI; clearly synthetic).

Never weakens the v0.8 gates, never hides seed failures, never trains on rejected examples, never collapses
calibration and truth-holding into a single gate, and never claims replication from a single seed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_scope_lab.experiments import truth_holding as th
from qwen_scope_lab.experiments import truth_holding_distill_v07 as v7
from qwen_scope_lab.experiments import truth_holding_distill_v08 as v8
from qwen_scope_lab.experiments import truth_holding_distill_v09 as v9
from qwen_scope_lab.experiments.truth_holding_diag import NO_THINK_INSTRUCTION, strip_think

DEFAULT_RATIOS = ("A50_B30_C20", "A40_B40_C20", "A60_B20_C20", "A50_B25_C25")
BASE_ARMS = ("baseline_4b", "prompt_only_inference_4b")
V08_REFERENCE_ARM = "balanced_v08_reference"


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


def _load_dotenv() -> None:
    """Load repo-root .env into os.environ (only keys not already set; values never printed)."""
    p = Path(__file__).resolve().parents[1] / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ======================================================================================
# preflight
# ======================================================================================


def cmd_preflight(args: argparse.Namespace) -> dict[str, Any]:
    v08_metrics = json.loads((Path(args.v08_dir) / "eval_9b" / "v08_eval_metrics.json").read_text())
    v08_audit = json.loads((Path(args.v08_dir) / "source_audit_9b" / "v08_source_audit.json").read_text())
    v07_metrics = json.loads(Path(args.v07_metrics).read_text())
    v06_failure = json.loads(Path(args.v06_failure_modes).read_text())
    res = v9.preflight_v09(v08_metrics=v08_metrics, v08_source_audit=v08_audit,
                           v07_metrics=v07_metrics, v06_failure_modes=v06_failure)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(_render_preflight_md(res, args), encoding="utf-8")
    if res["preflight"] != "pass" and not args.allow_mismatch:
        raise SystemExit("v0.9 preflight MISMATCH — diagnose before v0.9:\n" + json.dumps(res["checks"], indent=2))
    return res


def _render_preflight_md(res: dict, args: argparse.Namespace) -> str:
    ev = res["evidence"]
    rows = "\n".join(f"- {'✅' if ok else '❌'} `{k}`" for k, ok in res["checks"].items())
    return (
        "# v0.9 preflight — preserve & verify the prior claims\n\n"
        f"**Status: {res['preflight']}**\n\n"
        "Before replicating/stress-testing v0.8, confirm saved history reproduces:\n\n"
        f"- **v0.8 win**: verdict `{ev['v08_verdict']}`, class balance {json.dumps(ev['v08_class_balance'])}, "
        f"trained on **{ev['v08_sft_balanced']}** kept of **{ev['v08_n_scored']}** scored (rejected never trained).\n"
        f"- **v0.7 failure mode** (what v0.8 fixed): verdict `{ev['v07_verdict']}`, ambiguous calibration "
        f"baseline **{ev['v07_baseline_ambiguous_calibration']}** → distilled **{ev['v07_distilled_ambiguous_calibration']}** (regressed).\n"
        f"- **v0.6 steering**: `steering_value.status` = `{ev['v06_steering_value_status']}` — global CAA steering not viable; v0.9 uses NONE.\n\n"
        f"## Checks\n\n{rows}\n\n"
        f"Artifacts loaded: `{args.v08_dir}` (eval + source audit), `{args.v07_metrics}`, `{args.v06_failure_modes}`.\n"
    )


# ======================================================================================
# make-stress-eval  (writes 5 stress splits + 5 standard splits to one eval root)
# ======================================================================================


def cmd_make_stress_eval(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    standard = v8.make_scenarios_v08(seed=args.standard_seed)
    stress = v9.make_stress_evals(seed=args.stress_seed, per_split=args.per_split)
    files: dict[str, dict] = {}
    # standard splits (regenerated deterministically; ids match the v0.8 reference arm)
    for sp in v9.STANDARD_SPLITS:
        _write_jsonl(out / f"{sp}_scenarios.jsonl", standard[sp])
        files[sp] = {"path": str(out / f"{sp}_scenarios.jsonl"), "n": len(standard[sp]), "kind": "standard"}
    for sp, rows in stress.items():
        _write_jsonl(out / f"{sp}_scenarios.jsonl", rows)
        files[sp] = {"path": str(out / f"{sp}_scenarios.jsonl"), "n": len(rows), "kind": "stress"}
    # schema + leakage validation (stress setup_keys must be disjoint from the training pool)
    from collections import Counter
    issues = _validate_stress_schema(stress, standard)
    manifest = {"schema_version": v9.SCHEMA_VERSION, "standard_seed": args.standard_seed, "stress_seed": args.stress_seed,
                "files": files, "stress_class_balance": {sp: dict(Counter(r["behavioral_class"] for r in rows))
                                                          for sp, rows in stress.items()},
                "schema_issues": issues}
    _write_json(out / "stress_manifest.json", manifest)
    return {"out": str(out), "counts": {k: v["n"] for k, v in files.items()}, "schema_issues": issues}


def _validate_stress_schema(stress: dict, standard: dict) -> list[str]:
    issues = []
    req = {"id", "split", "domain", "behavioral_class", "setup_key", "question",
           "acceptable_answer_patterns", "false_answer_patterns", "false_claim", "pressure_type", "requires_calibration"}
    train_keys = {r.get("setup_key") for r in standard.get("train", [])}
    all_ids = []
    for sp, rows in stress.items():
        for r in rows:
            all_ids.append(r["id"])
            miss = req - set(r)
            if miss:
                issues.append(f"{sp}/{r.get('id')} missing {sorted(miss)}")
            if r["behavioral_class"] not in v8.BEHAVIORAL_CLASSES:
                issues.append(f"{sp}/{r['id']} bad class {r['behavioral_class']}")
            if r.get("setup_key") in train_keys:
                issues.append(f"{sp}/{r['id']} setup_key leaks training pool")
    if len(all_ids) != len(set(all_ids)):
        issues.append("duplicate stress ids")
    return issues


# ======================================================================================
# build-mixtures  (re-mix the v0.8 kept corpus -> ratio datasets + matched-size; no model)
# ======================================================================================


def cmd_build_mixtures(args: argparse.Namespace) -> dict[str, Any]:
    kept = _read_jsonl(args.kept_pairs)
    out = Path(args.out)
    ds_dir = out / "datasets"
    from collections import Counter
    kept_counts = dict(Counter(r.get("behavioral_class") for r in kept))

    datasets: dict[str, dict] = {}

    def export(name: str, rows: list[dict], extra: dict) -> None:
        path = ds_dir / f"{name}.jsonl"
        _write_jsonl(path, v9.to_sft_records_v09(rows))
        datasets[name] = {"path": str(path), "n": len(rows),
                          "class_counts": dict(Counter(r.get("behavioral_class") for r in rows)), **extra}

    # full balanced corpus (the seeds train on this exact v0.8 corpus, labels preserved)
    export("balanced_full", kept, {"kind": "balanced_full", "source": "v08_kept"})

    # data-mixture ratio arms at a single matched total
    ratios = list(args.ratios)
    matched_total = v9.feasible_total_for_ratios(kept, ratios) if args.matched_size else None
    for rn in ratios:
        ratio = v9.parse_ratio(rn)
        total = matched_total if args.matched_size else min(len(kept), v9.feasible_total_for_ratios(kept, [rn]))
        mix = v9.build_mixture(kept, ratio, total, seed=args.seed)
        export(f"mix_{rn}", mix["rows"], {"kind": "mixture", "ratio": rn, "requested_total": total,
                                          "achieved": mix["achieved"], "achieved_fraction": mix["achieved_fraction"],
                                          "capped": mix["capped"], "matched_size": bool(args.matched_size)})

    # matched-size ablation (truth-only / calibration-only / balanced at identical n)
    ms = v9.matched_size_arms(kept, seed=args.seed)
    for name, rows in ms["arms"].items():
        export(name, rows, {"kind": "matched_size", "matched_n": ms["matched_n"],
                            "binding_pool": ms["binding_pool"], "serious_run": ms["serious_run"]})

    serious_gate = args.min_examples
    plan = []
    # seeds (required): identical corpus, seed varies training stochasticity (data order; Tinker init)
    for s in args.seeds:
        plan.append({"name": f"balanced_v09_seed_{s}", "kind": "seed", "dataset": "balanced_full",
                     "seed": s, "n": datasets["balanced_full"]["n"], "lr": args.lr, "epochs": args.epochs,
                     "serious_run": datasets["balanced_full"]["n"] >= serious_gate})
    # mixtures (required)
    for rn in ratios:
        d = datasets[f"mix_{rn}"]
        plan.append({"name": f"mix_{rn}", "kind": "mixture", "dataset": f"mix_{rn}", "seed": args.seed,
                     "ratio": rn, "n": d["n"], "lr": args.lr, "epochs": args.epochs, "serious_run": d["n"] >= serious_gate})
    # matched-size (required)
    for name in ms["arms"]:
        d = datasets[name]
        plan.append({"name": name, "kind": "matched_size", "dataset": name, "seed": args.seed,
                     "n": d["n"], "lr": args.lr, "epochs": args.epochs, "serious_run": d["n"] >= serious_gate,
                     "matched_ablation": True})

    # optional arms (opt-in; train only if requested) — lr/epoch sweeps + a templated control
    optional = [
        {"name": "balanced_v09_low_lr", "kind": "optional", "dataset": "balanced_full", "seed": args.seeds[0],
         "lr": round(args.lr / 2, 8), "epochs": args.epochs, "n": datasets["balanced_full"]["n"]},
        {"name": "balanced_v09_high_lr", "kind": "optional", "dataset": "balanced_full", "seed": args.seeds[0],
         "lr": round(args.lr * 2, 8), "epochs": args.epochs, "n": datasets["balanced_full"]["n"]},
        {"name": "balanced_v09_one_epoch", "kind": "optional", "dataset": "balanced_full", "seed": args.seeds[0],
         "lr": args.lr, "epochs": 1, "n": datasets["balanced_full"]["n"]},
        {"name": "balanced_v09_more_epochs", "kind": "optional", "dataset": "balanced_full", "seed": args.seeds[0],
         "lr": args.lr, "epochs": max(args.epochs + 2, 5), "n": datasets["balanced_full"]["n"]},
        {"name": "balanced_v09_teacher_27b_mix", "kind": "optional", "dataset": None, "seed": args.seeds[0],
         "lr": args.lr, "epochs": args.epochs, "requires": "an audited 27B prompt-only source corpus (not present)"},
    ]

    manifest = {
        "schema_version": v9.SCHEMA_VERSION,
        "v08_kept_source": args.kept_pairs,
        "kept_counts": {**kept_counts, "total": len(kept)},
        "matched_total_across_ratios": matched_total,
        "matched_size_ablation": {"matched_n": ms["matched_n"], "binding_pool": ms["binding_pool"],
                                  "serious_run": ms["serious_run"],
                                  "note": ("matched-n is below the %d-example serious gate (the B+C pool binds at %d); "
                                           "the three arms are a MATCHED-n control — the relative comparison at equal n is "
                                           "the deliverable, not an absolute serious-run claim. Pushing matched-n>=100 would "
                                           "require generating ~%d more kept B/C examples (a 9B-teacher generation step)."
                                           % (serious_gate, ms["matched_n"], max(0, 100 - ms["matched_n"])))},
        "min_examples_for_serious": serious_gate,
        "datasets": datasets,
        "training_plan": plan,
        "optional_plan": optional,
        "leakage_checks": _mixture_leakage_checks(datasets, ds_dir),
    }
    _write_json(out / "v09_source_manifest.json", manifest)
    return {"out": str(out), "kept_counts": manifest["kept_counts"], "matched_total": matched_total,
            "matched_n": ms["matched_n"], "n_datasets": len(datasets), "n_planned_arms": len(plan),
            "n_optional": len(optional)}


def _mixture_leakage_checks(datasets: dict, ds_dir: Path) -> dict:
    """Confirm every SFT record carries class/domain/pressure/source labels and no empty assistant turns."""
    issues = []
    for name, d in datasets.items():
        recs = _read_jsonl(d["path"])
        for r in recs[:5000]:
            if not r.get("messages") or len(r["messages"]) < 2 or not r["messages"][1].get("content", "").strip():
                issues.append(f"{name}: empty/short assistant message")
                break
            if r.get("behavioral_class") is None or r.get("source") is None:
                issues.append(f"{name}: missing class/source label")
                break
    return {"ok": not issues, "issues": issues}


# ======================================================================================
# train-matrix  (Tinker LoRA; honest per-arm status)
# ======================================================================================


def cmd_train_matrix(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(Path(args.source_manifest).read_text())
    plan = list(manifest["training_plan"])
    if args.include_optional:
        plan += [p for p in manifest.get("optional_plan", []) if p.get("dataset")]
    if args.arms:
        wanted = set(args.arms)
        plan = [p for p in plan if p["name"] in wanted]
    if args.kinds:
        kinds = set(args.kinds.split(","))
        plan = [p for p in plan if p["kind"] in kinds]

    out = Path(args.out)
    arm_dir = out / "arm_outputs"
    ds_dir = Path(manifest["datasets"]["balanced_full"]["path"]).parent
    eval_rows = {sp: _read_jsonl(Path(args.eval_root) / f"{sp}_scenarios.jsonl")
                 for sp in v9.ALL_EVAL_SPLITS if (Path(args.eval_root) / f"{sp}_scenarios.jsonl").exists()}

    out_manifest: dict[str, Any] = {"status": "run", "base_model": args.base_model, "rank": args.rank,
                                    "batch_size": args.batch_size, "max_seq": args.max_seq,
                                    "min_examples_for_serious": args.min_examples,
                                    "eval_splits": list(eval_rows), "seed_caveat":
                                    "seed varies data-shuffle order and Tinker training stochasticity; LoRA init seed is not "
                                    "separately exposed by the API — documented limitation of the seed-robustness arms.",
                                    "arms": {}}

    try:
        import numpy as np
        import tinker
        from transformers import AutoTokenizer
    except Exception as exc:  # Tinker/transformers unavailable -> everything not_run with a blocker
        for p in plan:
            out_manifest["arms"][p["name"]] = {**_arm_info(p), "status": "not_run",
                                               "blocker": f"training deps unavailable: {type(exc).__name__}: {exc}"}
        for a in BASE_ARMS:
            out_manifest["arms"][a] = {"status": "not_run", "blocker": "training deps unavailable"}
        out_manifest["status"] = "blocked"
        out_manifest["arm_outputs_dir"] = str(arm_dir)
        _write_json(out / "v09_training_manifest.json", out_manifest)
        return {"out": str(out), "status": "blocked", "reason": f"{type(exc).__name__}: {exc}",
                "planned": [p["name"] for p in plan]}

    tok = AutoTokenizer.from_pretrained(args.base_model)

    def render(messages, add_gen):
        try:
            text = tok.apply_chat_template(messages, add_generation_prompt=add_gen, tokenize=False, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(messages, add_generation_prompt=add_gen, tokenize=False)
        return tok.encode(text)

    def datum(messages):
        user_msgs = messages[:-1] if messages[-1]["role"] == "assistant" else messages
        p = render(user_msgs, True)
        full = render(messages, False)[: args.max_seq]
        inp, tgt = full[:-1], full[1:]
        b = max(0, len(p) - 1)
        w = [1.0 if i >= b else 0.0 for i in range(len(tgt))]
        return tinker.Datum(model_input=tinker.ModelInput.from_ints(inp),
                            loss_fn_inputs={"target_tokens": tinker.TensorData.from_numpy(np.asarray(tgt, np.int64)),
                                            "weights": tinker.TensorData.from_numpy(np.asarray(w, np.float32))})

    sc = tinker.ServiceClient()
    sp_params = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    def _mi_for(row, instruction):
        msgs = v9.stress_prompt(row)
        if instruction:
            msgs = [{"role": "user", "content": f"{instruction}\n\n{msgs[0]['content']}"}] + msgs[1:]
        return tinker.ModelInput.from_ints(render(msgs, True))

    def _decode(resp):
        return strip_think(tok.decode(list(resp.sequences[0].tokens), skip_special_tokens=True))

    def sample_split(client, rows, instruction):
        """Concurrent sampling: submit a batch of futures, then collect (≈9x faster than serial .result()).
        temperature=0 -> identical outputs to serial sampling; only the dispatch is parallelized."""
        out = {}
        ch = max(1, args.sample_concurrency)
        for i in range(0, len(rows), ch):
            batch = rows[i:i + ch]
            futs = [(r, client.sample(prompt=_mi_for(r, instruction), num_samples=1, sampling_params=sp_params)) for r in batch]
            for r, fut in futs:
                try:
                    out[r["id"]] = _decode(fut.result())
                except Exception:  # one transient failure -> a single serial retry, then surface
                    out[r["id"]] = _decode(client.sample(prompt=_mi_for(r, instruction), num_samples=1, sampling_params=sp_params).result())
        return out

    def write_arm(name, client, instr):
        for sp, rows in eval_rows.items():
            outs = sample_split(client, rows, instr)
            _write_jsonl(arm_dir / name / f"{sp}.jsonl",
                         [{"scenario_id": r["id"], "behavioral_class": r.get("behavioral_class"),
                           "split": sp, "output": outs[r["id"]]} for r in rows])

    def _complete(name):  # all eval splits already sampled (crash-safe resume)
        return bool(eval_rows) and all((arm_dir / name / f"{sp}.jsonl").exists() for sp in eval_rows)

    def _persist():  # write the manifest after every arm so a long run is inspectable/resumable
        out_manifest["arm_outputs_dir"] = str(arm_dir)
        _write_json(out / "v09_training_manifest.json", out_manifest)

    # base + prompt-only sampled once on ALL splits (standard + stress)
    if not args.skip_base_arms:
        base = None
        for nm, instr in (("baseline_4b", ""), ("prompt_only_inference_4b", NO_THINK_INSTRUCTION)):
            if _complete(nm):
                out_manifest["arms"][nm] = {"status": "run", "resumed": True}
                continue
            try:
                base = base or sc.create_sampling_client(base_model=args.base_model)
                write_arm(nm, base, instr)
                out_manifest["arms"][nm] = {"status": "run", **({"instruction": "NO_THINK_INSTRUCTION"} if instr else {})}
            except Exception as exc:
                out_manifest["arms"][nm] = {"status": "error", "blocker": f"{type(exc).__name__}: {exc}"}
            _persist()

    for p in plan:
        info = _arm_info(p)
        ds = manifest["datasets"].get(p.get("dataset") or "")
        if not ds:
            out_manifest["arms"][p["name"]] = {**info, "status": "not_run", "blocker": p.get("requires", "no dataset")}
            continue
        sft = _read_jsonl(ds["path"])
        n = len(sft)
        serious = n >= args.min_examples
        info.update({"source_sft": ds["path"], "n_examples": n, "serious_run": serious})
        if _complete(p["name"]):  # resume: already fully sampled
            out_manifest["arms"][p["name"]] = {**info, "status": "run" if serious else "smoke", "resumed": True,
                                               "lr": p.get("lr", args.lr), "epochs": int(p.get("epochs", args.epochs))}
            _persist()
            continue
        if n < args.min_examples and not args.allow_smoke:
            out_manifest["arms"][p["name"]] = {**info, "status": "skipped_by_gate",
                                               "blocker": f"{n} < {args.min_examples}; pass --allow-smoke for a labeled smoke run"}
            _persist()
            continue
        try:
            tc = sc.create_lora_training_client(base_model=args.base_model, rank=args.rank)
            data = [datum(r["messages"]) for r in sft]
            adam = tinker.AdamParams(learning_rate=p.get("lr", args.lr))
            epochs = int(p.get("epochs", args.epochs))
            step = 0
            for epoch in range(epochs):
                order = np.random.RandomState(p.get("seed", 0) * 1000 + epoch).permutation(len(data))
                for i in range(0, len(order), args.batch_size):
                    batch = [data[j] for j in order[i:i + args.batch_size]]
                    tc.forward_backward(batch, "cross_entropy").result()
                    tc.optim_step(adam).result()
                    step += 1
            client = tc.save_weights_and_get_sampling_client()
            write_arm(p["name"], client, "")
            out_manifest["arms"][p["name"]] = {**info, "status": "run" if serious else "smoke", "steps": step,
                                               "lr": p.get("lr", args.lr), "epochs": epochs}
        except Exception as exc:
            out_manifest["arms"][p["name"]] = {**info, "status": "error", "blocker": f"{type(exc).__name__}: {exc}"}
        _persist()

    _persist()
    return {"out": str(out), "arms": {k: v.get("status") for k, v in out_manifest["arms"].items()},
            "arm_outputs_dir": str(arm_dir)}


def _arm_info(p: dict) -> dict:
    return {k: p[k] for k in ("kind", "seed", "ratio", "dataset", "matched_ablation") if k in p}


# ======================================================================================
# eval-matrix  (class+split+seed aware, bootstrap CIs; loads v0.8 reference too; no model)
# ======================================================================================


def _load_arm_outputs(adir: Path, scenario_splits: dict) -> dict | None:
    if not adir.exists():
        return None
    outs = {}
    for sp in scenario_splits:
        f = adir / f"{sp}.jsonl"
        if f.exists():
            outs[sp] = {r["scenario_id"]: r["output"] for r in _read_jsonl(f)}
    return outs if outs else None


def _eval_one_arm(scenario_splits: dict, outs_by_split: dict) -> dict:
    present = {sp: rows for sp, rows in scenario_splits.items() if sp in outs_by_split}
    arm = v8.evaluate_arm_v08(present, outs_by_split)
    # per-split truth-hold CIs + by-class + by-domain/pressure (factual) + balanced score
    for sp, rows in present.items():
        booleans = [v8.score_row(r, strip_think(outs_by_split[sp].get(r["id"], "")))["holds_truth"] for r in rows]
        arm["by_split"].setdefault(sp, {})["truth_hold_ci"] = v9.proportion_ci(booleans, seed=0)
    arm["by_class"] = v9.aggregate_by_class_v09(present, outs_by_split)
    arm["by_domain"] = _factual_breakdown(present, outs_by_split, "domain")
    arm["by_pressure"] = _factual_breakdown(present, outs_by_split, "pressure_type")
    arm["balanced_score"] = v9.balanced_score(arm)
    return arm


def _factual_breakdown(scenario_splits: dict, outs_by_split: dict, key: str) -> dict:
    groups: dict[str, list[bool]] = {}
    for sp, rows in scenario_splits.items():
        for r in rows:
            if r.get("behavioral_class") not in ("A_factual", "D_adversarial"):
                continue
            s = v8.score_row(r, strip_think(outs_by_split.get(sp, {}).get(r["id"], "")))
            groups.setdefault(r.get(key, "?"), []).append(s["holds_truth"])
    return {g: {"n": len(v), "truth_hold_rate": round(sum(v) / len(v), 4)} for g, v in sorted(groups.items())}


def cmd_eval_matrix(args: argparse.Namespace) -> dict[str, Any]:
    scenario_splits = {sp: _read_jsonl(Path(args.eval_root) / f"{sp}_scenarios.jsonl")
                       for sp in v9.ALL_EVAL_SPLITS if (Path(args.eval_root) / f"{sp}_scenarios.jsonl").exists()}
    tm = json.loads(Path(args.training_manifest).read_text())
    arm_root = Path(tm.get("arm_outputs_dir") or (Path(args.training_manifest).parent / "arm_outputs"))

    arms: dict[str, Any] = {}
    raw_by_arm: dict[str, dict] = {}
    for name, info in tm.get("arms", {}).items():
        outs = _load_arm_outputs(arm_root / name, scenario_splits)
        if outs is None:
            arms[name] = {"status": info.get("status", "not_run"), "blocker": info.get("blocker"),
                          "kind": info.get("kind")}
            continue
        arms[name] = {"status": info.get("status", "run"), "kind": info.get("kind"),
                      "seed": info.get("seed"), "ratio": info.get("ratio"),
                      **_eval_one_arm(scenario_splits, outs)}
        raw_by_arm[name] = outs

    # v0.8 reference arm (standard splits only — it was never sampled on the stress splits)
    if args.include_v08_reference:
        ref_dir = Path(args.include_v08_reference) / "train_9b" / "arm_outputs" / "distilled_4b_calibration_balanced_v08"
        ref_outs = _load_arm_outputs(ref_dir, scenario_splits)
        if ref_outs:
            arms[V08_REFERENCE_ARM] = {"status": "run", "kind": "reference",
                                       "note": "v0.8 saved outputs; standard splits only (no stress)",
                                       **_eval_one_arm(scenario_splits, ref_outs)}
            raw_by_arm[V08_REFERENCE_ARM] = ref_outs

    out = Path(args.out)
    payload = {"schema_version": v9.SCHEMA_VERSION, "eval_splits": list(scenario_splits),
               "split_n": {sp: len(rows) for sp, rows in scenario_splits.items()}, "arms": arms}
    _write_json(out / "v09_eval_metrics.json", payload)
    # stash raw outputs for the judge/examples stages
    _write_json(out / "_raw_index.json", {"arm_root": str(arm_root),
                                          "v08_reference_dir": args.include_v08_reference or ""})
    return {"out": str(out), "arms": {a: (v.get("status") if v.get("status") != "run"
                                          else round(v.get("balanced_score", 0), 3)) for a, v in arms.items()}}


# ======================================================================================
# judge-validate  (optional rubric judge; deterministic-vs-judge agreement)
# ======================================================================================

JUDGE_ARMS_DEFAULT = ("baseline_4b", "prompt_only_inference_4b", "truth_only_matched_n",
                      "calibration_only_matched_n", V08_REFERENCE_ARM)


def cmd_judge_validate(args: argparse.Namespace) -> dict[str, Any]:
    eval_metrics = json.loads(Path(args.eval_metrics).read_text())
    eval_dir = Path(args.eval_metrics).parent
    raw_index = json.loads((eval_dir / "_raw_index.json").read_text()) if (eval_dir / "_raw_index.json").exists() else {}
    out = Path(args.out)

    scenario_splits = {sp: _read_jsonl(Path(args.eval_root) / f"{sp}_scenarios.jsonl")
                       for sp in v9.ALL_EVAL_SPLITS if (Path(args.eval_root) / f"{sp}_scenarios.jsonl").exists()}
    scenario_index = {r["id"]: r for rows in scenario_splits.values() for r in rows}

    # which arms to judge: defaults present + the best balanced seed by balanced_score
    run_arms = {a: m for a, m in eval_metrics["arms"].items() if m.get("status") == "run"}
    seeds = {a: m for a, m in run_arms.items() if m.get("kind") == "seed"}
    best_seed = max(seeds, key=lambda a: seeds[a].get("balanced_score", 0), default=None)
    judge_arm_names = [a for a in JUDGE_ARMS_DEFAULT if a in run_arms] + ([best_seed] if best_seed else [])

    if not args.judge_command and not args.judge_jsonl:
        res = {"status": "not_run", "reason": "no judge supplied (pass --judge-command or --judge-jsonl)",
               "would_judge_arms": judge_arm_names,
               "expected_input_schema": "JSON list of {arm, scenario_id, split, behavioral_class, question, false_claim, answer, requires_calibration}",
               "expected_output_schema": f"JSON list of {{arm, scenario_id, {', '.join(v9.JUDGE_DIMENSIONS)}}} with boolean values"}
        _write_json(out / "v09_judge_validation.json", res)
        (out / "v09_judge_validation.md").write_text(_render_judge_md(res), encoding="utf-8")
        return {"out": str(out), "status": "not_run", "would_judge_arms": judge_arm_names}

    arm_root = Path(raw_index.get("arm_root", ""))
    requests, outputs_by_arm = [], {}
    for a in judge_arm_names:
        adir = arm_root / a if a != V08_REFERENCE_ARM else \
            Path(raw_index.get("v08_reference_dir", "")) / "train_9b" / "arm_outputs" / "distilled_4b_calibration_balanced_v08"
        outs = _load_arm_outputs(adir, scenario_splits) or {}
        outputs_by_arm[a] = {sid: o for d in outs.values() for sid, o in d.items()}
        requests += v9.build_judge_request(scenario_splits, outs, a, per_split=args.per_split, seed=args.seed)

    if args.judge_jsonl:
        judged = _read_jsonl(args.judge_jsonl)
    else:
        proc = subprocess.run([args.judge_command], input=json.dumps(requests), capture_output=True, text=True)
        if proc.returncode != 0:
            res = {"status": "error", "reason": f"judge command exited {proc.returncode}", "stderr": proc.stderr[:500]}
            _write_json(out / "v09_judge_validation.json", res)
            (out / "v09_judge_validation.md").write_text(_render_judge_md(res), encoding="utf-8")
            return {"out": str(out), "status": "error"}
        judged = json.loads(proc.stdout)

    by_arm = {}
    for a in judge_arm_names:
        recs = [j for j in judged if j.get("arm") == a]
        by_arm[a] = v9.judge_agreement(scenario_index, outputs_by_arm.get(a, {}), recs)
    overall = v9.judge_agreement(scenario_index, {sid: o for d in outputs_by_arm.values() for sid, o in d.items()},
                                 judged)
    res = {"status": "run", "judged_arms": judge_arm_names, "n_records": len(judged),
           "overall": overall, "by_arm": by_arm,
           "judge_overall_acceptable_rate": overall.get("judge_overall_acceptable_rate"),
           "agreement_rate": overall.get("agreement_rate")}
    _write_json(out / "v09_judge_validation.json", res)
    (out / "v09_judge_validation.md").write_text(_render_judge_md(res), encoding="utf-8")
    return {"out": str(out), "status": "run", "agreement_rate": overall.get("agreement_rate"),
            "judge_acceptable_rate": overall.get("judge_overall_acceptable_rate")}


def _render_judge_md(res: dict) -> str:
    if res.get("status") != "run":
        return ("# v0.9 rubric-judge validation\n\n"
                f"**Status: `{res.get('status')}`** — {res.get('reason', '')}\n\n"
                + ("Would judge arms: " + ", ".join(f"`{a}`" for a in res.get("would_judge_arms", [])) + "\n\n"
                   "Supply a judge to validate: `--judge-command ./judge.sh` (reads JSON on stdin, writes JSON on stdout) "
                   "or `--judge-jsonl precomputed.jsonl`.\n\n"
                   f"- expected input: {res.get('expected_input_schema','')}\n"
                   f"- expected output: {res.get('expected_output_schema','')}\n"
                   if res.get("status") == "not_run" else f"stderr: {res.get('stderr','')}\n"))
    o = res["overall"]
    rows = "\n".join(f"| `{a}` | {m.get('n')} | {m.get('agreement_rate')} | {m.get('judge_overall_acceptable_rate')} | "
                     f"{m.get('deterministic_false_positive_rate')} | {m.get('deterministic_false_negative_rate')} |"
                     for a, m in res["by_arm"].items() if m.get("status") == "run")
    dis = "\n".join(f"- `{d['scenario_id']}` ({d['behavioral_class']}): det={d['deterministic_good']} judge={d['judge_acceptable']} — {d['question']}"
                    for d in o.get("disagreements", [])[:8]) or "- none"
    return (
        "# v0.9 rubric-judge validation (deterministic metric ↔ judge agreement)\n\n"
        f"Judged **{res['n_records']}** stratified records across {len(res['judged_arms'])} arms.\n\n"
        f"- overall agreement: **{o.get('agreement_rate')}** · judge-acceptable rate: **{o.get('judge_overall_acceptable_rate')}** · "
        f"deterministic-good rate: {o.get('deterministic_good_rate')}\n"
        f"- deterministic false-positives: {o.get('deterministic_false_positive_rate')} · false-negatives: {o.get('deterministic_false_negative_rate')}\n\n"
        "## Per-arm\n\n| arm | n | agreement | judge-acceptable | det-FP | det-FN |\n|---|---|---|---|---|---|\n"
        + rows + "\n\n## Sample disagreements\n\n" + dis + "\n\n"
        "Judge is **validation, not a gate** — deterministic gates still decide the verdict unless the judge materially rejects a metric-declared win.\n"
    )


# ======================================================================================
# decide  (verdict_v09 + seed robustness + mixture sweep + ablation matrix + decision)
# ======================================================================================


def cmd_decide(args: argparse.Namespace) -> dict[str, Any]:
    em = json.loads(Path(args.metrics).read_text())
    arms = em["arms"]
    tm = json.loads(Path(args.training).read_text()) if args.training and Path(args.training).exists() else {}
    judge = json.loads(Path(args.judge).read_text()) if args.judge and Path(args.judge).exists() else None

    # annotate each arm with its TRAINING example count (the eval-set n is not the training size)
    for a, m in arms.items():
        m["train_n"] = (tm.get("arms", {}).get(a, {}) or {}).get("n_examples")

    baseline = arms.get("baseline_4b")
    prompt_only = arms.get("prompt_only_inference_4b")
    seed_arms = {a: m for a, m in arms.items() if m.get("kind") == "seed" and m.get("status") == "run"}
    mixtures = {a: m for a, m in arms.items() if m.get("kind") == "mixture" and m.get("status") == "run"}
    matched = {a: m for a, m in arms.items() if m.get("kind") == "matched_size" and m.get("status") in ("run", "smoke")}
    mix_meta = {a: {"n": m.get("train_n"), "ratio": m.get("ratio")} for a, m in mixtures.items()}

    verdict = v9.verdict_v09(seed_arms=seed_arms, baseline=baseline, prompt_only=prompt_only,
                             mixtures=mixtures, mixture_meta=mix_meta,
                             judge_agreement=(judge.get("overall") if judge and judge.get("status") == "run" else None))
    seed_robust = v9.aggregate_seeds(seed_arms, baseline=baseline, prompt_only=prompt_only) if seed_arms else {"n_seeds": 0}
    mix_sweep = v9.aggregate_mixtures(mixtures, mix_meta) if mixtures else {"n_ratios": 0, "table": []}

    out = Path(args.out)
    _write_json(out / "v09_decision.json", {"verdict": verdict, "seed_robustness": seed_robust,
                                            "mixture_sweep": mix_sweep})
    (out / "v09_final_decision.md").write_text(_render_final_decision(verdict, seed_robust, mix_sweep, arms, judge), encoding="utf-8")
    (out / "v09_seed_robustness.md").write_text(_render_seed_robustness(seed_robust, baseline, prompt_only), encoding="utf-8")
    (out / "v09_mixture_sweep.md").write_text(_render_mixture_sweep(mix_sweep, matched), encoding="utf-8")
    (out / "v09_ablation_matrix.md").write_text(_render_ablation_matrix(arms, verdict), encoding="utf-8")
    _write_jsonl(out / "v09_examples_wins_failures.jsonl", _wins_failures_v09(args, arms))
    return {"out": str(out), "verdict": verdict["verdict"], "reason": verdict["reason"],
            "n_seeds": verdict.get("n_seeds"), "n_seeds_passing": verdict.get("n_seeds_passing_win_gate"),
            "checks_failed": [k for k, v in verdict.get("checks", {}).items() if not v]}


def _fmt(x):
    return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def _render_ablation_matrix(arms: dict, verdict: dict) -> str:
    cols = ["kind", "n", "factual_th", "ood_th", "adv_th", "B_calib", "C_calib", "over_assert", "rel", "bal_score", "status"]
    lines = ["# v0.9 ablation matrix — every arm on the held-out (standard + stress) splits\n",
             f"**Verdict: `{verdict['verdict']}`** — {verdict['reason']}\n",
             "| arm | " + " | ".join(cols[:-1]) + " | status |",
             "|" + "---|" * (len(cols) + 1)]
    for a, m in arms.items():
        if m.get("status") not in ("run", "smoke"):
            lines.append(f"| `{a}` | {m.get('kind','—')} | " + " | ".join("—" for _ in cols[1:-1]) + f" | _{m.get('status')}_ |")
            continue
        row = [m.get("kind", "—"), _fmt(m.get("overall", {}).get("n")),
               _fmt(v9.factual_truth_hold(m)), _fmt(v9._sp(m, "eval_ood", "truth_hold_rate")),
               _fmt(v9.adversarial_truth_hold(m)), _fmt(v9.b_calibration(m)), _fmt(v9.c_calibration(m)),
               _fmt(v9.over_assertion_rate(m)), _fmt(m.get("overall", {}).get("relevance")),
               _fmt(m.get("balanced_score"))]
        lines.append(f"| `{a}` | " + " | ".join(str(x) for x in row) + f" | {m.get('status')} |")
    lines.append("\n_factual_th = mean truth-hold over knowable-fact splits (id/ood/multiturn/messy/domain-transfer); "
                 "B/C_calib = calibration on unknowable/subjective; over_assert = categorical-assertion rate on B+C; "
                 "bal_score = reporting-only summary (never a gate)._\n")
    return "\n".join(lines)


def _render_seed_robustness(sr: dict, baseline: dict | None, prompt_only: dict | None) -> str:
    if not sr or sr.get("n_seeds", 0) == 0:
        return "# v0.9 seed robustness\n\n_No balanced seeds were trained/evaluated — see the training manifest for blockers._\n"
    summ = sr["summary"]
    def line(metric):
        s = summ.get(metric, {})
        ci = s.get("ci", {})
        return f"| {metric} | {_fmt(s.get('mean'))} | {_fmt(s.get('std'))} | {_fmt(s.get('min'))} | {_fmt(s.get('max'))} | [{_fmt(ci.get('lo'))}, {_fmt(ci.get('hi'))}] |"
    metrics = ["factual_truth_hold", "ood_truth_hold", "adversarial_truth_hold", "b_calibration", "c_calibration", "combined_calibration", "over_assertion", "balanced_score"]
    gate = "\n".join(f"- `{lbl}`: {'✅ passes' if g['passes_v08_win_gate'] else '❌ fails'} the strict v0.8 win gate"
                     + ("" if g["passes_v08_win_gate"] else f" (failed: {', '.join(g['failed_checks'])})")
                     for lbl, g in sr["per_seed_gate"].items())
    w = sr["worst_seed"]
    return (
        "# v0.9 seed robustness\n\n"
        f"**{sr['n_seeds']} seed(s); {sr['n_seeds_passing_win_gate']} pass the strict v0.8 win gate.**\n\n"
        "| metric | mean | std | min | max | bootstrap 95% CI (of mean) |\n|---|---|---|---|---|---|\n"
        + "\n".join(line(m) for m in metrics) + "\n\n"
        "## Per-seed v0.8 win gate\n\n" + gate + "\n\n"
        f"## Worst seed\n\n- `{w['label']}` — combined calibration {_fmt(w['combined_calibration'])}, truth {_fmt(w['truth'])}\n\n"
        "_Replication requires ≥2 seeds passing the gate and the worst seed not regressing below prompt-only on both axes._\n"
    )


def _render_mixture_sweep(ms: dict, matched: dict) -> str:
    rows = "\n".join(
        f"| `{r['ratio']}` | {_fmt(r['n'])} | {_fmt(r['factual_id_truth_hold'])} | {_fmt(r['ood_truth_hold'])} | "
        f"{_fmt(r['adversarial_truth_hold'])} | {_fmt(r['b_calibration'])} | {_fmt(r['c_calibration'])} | "
        f"{_fmt(r['over_assertion'])} | {_fmt(r['balanced_score'])} |" for r in ms.get("table", []))
    mrows = "\n".join(
        f"| `{a}` | {m.get('status')} | {_fmt(m.get('train_n'))} | {_fmt(v9.factual_truth_hold(m))} | "
        f"{_fmt(v9.b_calibration(m))} | {_fmt(v9.c_calibration(m))} | {_fmt(v9.over_assertion_rate(m))} |"
        for a, m in matched.items())
    note = []
    if ms.get("table"):
        note.append(f"- best ratio by balanced score: **{ms.get('best_ratio')}**")
        note.append(f"- calibration spread across ratios: **{_fmt(ms.get('calibration_spread'))}** "
                    f"({'mixture-SENSITIVE' if ms.get('mixture_sensitive') else 'robust to ratio'})")
    return (
        "# v0.9 data-mixture sweep + matched-size ablation\n\n"
        "## A/B/C ratio sweep (matched total N)\n\n"
        "| ratio | n | id_th | ood_th | adv_th | B_calib | C_calib | over_assert | bal_score |\n|---|---|---|---|---|---|---|---|---|\n"
        + (rows or "| _no ratio arms evaluated_ |") + "\n\n" + "\n".join(note) + "\n\n"
        "Reads: too much A (e.g. A60) should recreate the v0.7 regression (B/C calib falls); too much B/C should "
        "over-hedge factual; calibration-only is the surprise-strength control.\n\n"
        "## Matched-size ablation (truth-only vs calibration-only vs balanced, identical n)\n\n"
        "| arm | status | n | factual_th | B_calib | C_calib | over_assert |\n|---|---|---|---|---|---|---|\n"
        + (mrows or "| _no matched-size arms evaluated_ |") + "\n\n"
        "_This isolates mixture from sheer example count: at equal n, does balanced still beat truth-only on calibration?_\n"
    )


def _render_final_decision(verdict: dict, sr: dict, ms: dict, arms: dict, judge: dict | None) -> str:
    vp = verdict.get("vs_prompt_only", {})
    checks = "\n".join(f"- {'✅' if v else '❌'} `{k}`" for k, v in verdict.get("checks", {}).items())
    jstat = (judge or {}).get("status", "not_run")
    return (
        "# v0.9 final decision — is the v0.8 win robust?\n\n"
        f"## Verdict: `{verdict['verdict']}`\n\n{verdict['reason']}\n\n"
        f"- seeds trained: **{verdict.get('n_seeds')}**, passing the strict v0.8 win gate: **{verdict.get('n_seeds_passing_win_gate')}**\n"
        f"- mean factual truth-holding: **{_fmt(vp.get('mean_factual_truth'))}** · mean combined B/C calibration: **{_fmt(vp.get('mean_combined_calibration'))}**\n"
        f"- prompt-only: truth {_fmt(vp.get('prompt_only_truth'))}, calibration {_fmt(vp.get('prompt_only_calibration'))}\n"
        f"- mixture: best ratio {verdict.get('mixture', {}).get('best_ratio')}, "
        f"{'SENSITIVE' if verdict.get('mixture', {}).get('mixture_sensitive') else 'robust'} to ratio "
        f"(calibration spread {_fmt(verdict.get('mixture', {}).get('calibration_spread'))})\n"
        f"- rubric judge: `{jstat}`"
        + (f" (agreement {_fmt((judge or {}).get('agreement_rate'))}, judge-acceptable {_fmt((judge or {}).get('judge_overall_acceptable_rate'))})" if jstat == "run" else "")
        + "\n\n## Replication checks (calibration and truth-holding kept SEPARATE)\n\n" + checks + "\n\n"
        "## What is now proven / not proven\n\n"
        "- **Proven** depends on the verdict above; a `replicated_distillation_win` means ≥2 seeds independently pass the "
        "v0.8 win gate, truth-holding is preserved, B/C calibration is improved, and the win survives harder stress splits.\n"
        "- **Not proven**: anything the verdict withholds — e.g. a single-seed or mixture-sensitive result is explicitly NOT a "
        "robust replication. Matched-size arms below the 100-example serious gate are labeled smoke controls.\n"
        "- No activation steering anywhere (v0.6 settled that); the effect, if real, is entirely in the calibration-balanced DATA.\n\n"
        "## Strongest evidence & biggest caveats\n\n"
        "- Strongest: the matched-size ablation (mixture vs count at equal n) and the per-seed gate table in `v09_seed_robustness.md`.\n"
        "- Caveats: seed variation excludes a separately-seeded LoRA init (API limitation); matched-n is B+C-pool-bound; "
        "stress banks are modest (read CIs, not third decimals); judge is validation-only.\n\n"
        "## Recommended next step\n\n"
        + _next_step(verdict) + "\n"
    )


def _next_step(verdict: dict) -> str:
    return {
        "replicated_distillation_win": "Publish the replication: it holds across seeds, mixtures, and stress. Optionally widen the B/C pool to lift the matched-size ablation above the serious gate.",
        "single_seed_win_not_replicated": "Train more seeds (and/or widen the corpus); treat v0.8 as promising-but-unreplicated until ≥2 seeds pass.",
        "data_mixture_sensitive": "Pin the winning A/B/C ratio and re-run seeds at that ratio; report the sensitivity band honestly.",
        "calibration_fixed_truth_regressed": "Re-balance toward more Class-A factual data; the calibration data is over-correcting truth-holding.",
        "truth_preserved_calibration_unstable": "Stabilize calibration: add B/C examples and/or average more seeds; calibration variance is too high to claim a fix.",
        "prompting_sufficient": "Ship the inference-time prompt instead of an adapter; distillation isn't earning its keep here.",
        "judge_disagrees_with_metrics": "Reconcile the rubric judge with the deterministic scorers before any claim; the metrics may be over-crediting.",
        "inconclusive_replication_not_run": "Run train-matrix (≥2 seeds + matched-size) on Tinker, then re-run eval/decide; record exact blockers if compute is unavailable.",
    }.get(verdict["verdict"], "See the verdict reason.")


def _wins_failures_v09(args: argparse.Namespace, arms: dict) -> list[dict]:
    """Representative examples from saved arm outputs (best-effort; needs eval-root + raw index)."""
    ex: list[dict] = []
    try:
        eval_root = Path(args.eval_root)
        raw_index = json.loads((Path(args.metrics).parent / "_raw_index.json").read_text())
        arm_root = Path(raw_index.get("arm_root", ""))
    except Exception:
        return [{"category": "note", "detail": "no raw arm outputs available; categories listed in v09_ablation_matrix.md"}]
    scenario_splits = {sp: _read_jsonl(eval_root / f"{sp}_scenarios.jsonl")
                       for sp in v9.ALL_EVAL_SPLITS if (eval_root / f"{sp}_scenarios.jsonl").exists()}
    by_id = {r["id"]: (sp, r) for sp, rows in scenario_splits.items() for r in rows}
    seeds = [a for a, m in arms.items() if m.get("kind") == "seed" and m.get("status") == "run"]
    if not seeds:
        return [{"category": "note", "detail": "no trained seeds; nothing to exemplify yet"}]
    best = max(seeds, key=lambda a: arms[a].get("balanced_score", 0))

    def load(name):
        d = arm_root / name
        return {sid: o for sp in scenario_splits if (d / f"{sp}.jsonl").exists()
                for sid, o in {r["scenario_id"]: r["output"] for r in _read_jsonl(d / f"{sp}.jsonl")}.items()}

    win, base = load(best), load("baseline_4b")
    pol = load("prompt_only_inference_4b")
    truth_only = load("truth_only_matched_n") if (arm_root / "truth_only_matched_n").exists() else {}
    cats: dict[str, int] = {}
    cap = 3
    for sid, (sp, row) in by_id.items():
        d = win.get(sid)
        if d is None:
            continue
        cls = row.get("behavioral_class", "A_factual")
        ds = v8.score_row(row, d)
        cat = None
        if cls in ("B_unknowable", "C_subjective"):
            if ds.get("good"):
                cat = "calibration_success" if not sp.startswith("eval_stress") else "stress_calibration_success"
                if truth_only.get(sid) and not v8.score_row(row, truth_only[sid]).get("good"):
                    cat = "beats_truth_only_on_calibration"
            elif ds.get("categorical_assertion") or ds.get("false_objectivity"):
                cat = "over_assertion_on_unknowable"
        else:
            bs = v8.score_row(row, base.get(sid, ""))
            ps = v8.score_row(row, pol.get(sid, "")) if sid in pol else None
            if ds.get("good") and not bs.get("good"):
                cat = "factual_win" if not sp.startswith("eval_stress") else "stress_factual_win"
            elif ds.get("capitulated"):
                cat = "factual_capitulation"
            elif ps is not None and not ds.get("good") and ps.get("good"):
                cat = "prompt_only_beats_distilled"
            elif ds.get("good") and ds.get("class") == "A_factual" and v8.uncertainty_acknowledged(d) and not ds.get("correct"):
                cat = "over_hedge_on_factual"
        if cat and cats.get(cat, 0) < cap:
            cats[cat] = cats.get(cat, 0) + 1
            ex.append({"category": cat, "split": sp, "class": cls, "scenario_id": sid, "question": row["question"],
                       "false_claim": row.get("false_claim"), "best_seed": d, "baseline": base.get(sid),
                       "prompt_only": pol.get(sid), "truth_only_matched": truth_only.get(sid)})
    if not ex:
        ex.append({"category": "note", "detail": "no qualifying examples in saved outputs"})
    return ex


# ======================================================================================
# synthetic-smoke
# ======================================================================================


def cmd_synthetic_smoke(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    M = v9.build_synthetic_matrix(seeds=args.seeds, quality=args.quality)
    arms = {"baseline_4b": {"status": "run", "kind": "base", **M["baseline"]},
            "prompt_only_inference_4b": {"status": "run", "kind": "base", **M["prompt_only"]}}
    for name, a in M["seed_arms"].items():
        arms[name] = {"status": "run", "kind": "seed", "seed": int(name.split("_")[-1]),
                      "balanced_score": v9.balanced_score(a), **a}
    seed_arms = {a: arms[a] for a in arms if arms[a].get("kind") == "seed"}
    verdict = v9.verdict_v09(seed_arms=seed_arms, baseline=arms["baseline_4b"], prompt_only=arms["prompt_only_inference_4b"])
    sr = v9.aggregate_seeds(seed_arms, baseline=arms["baseline_4b"], prompt_only=arms["prompt_only_inference_4b"])

    # full pipeline: preflight (if dirs given) + stress eval + judge not_run + reports
    if args.v08_dir:
        cmd_preflight(argparse.Namespace(v08_dir=args.v08_dir, v07_metrics=args.v07_metrics,
                                         v06_failure_modes=args.v06_failure_modes,
                                         out=str(out / "v09_preflight.md"), allow_mismatch=True))
    cmd_make_stress_eval(argparse.Namespace(out=str(out / "stress_eval"), standard_seed=8, stress_seed=9, per_split=None))

    _write_json(out / "v09_eval_metrics.json", {"schema_version": v9.SCHEMA_VERSION, "arms": arms})
    _write_json(out / "v09_decision.json", {"verdict": verdict, "seed_robustness": sr})
    (out / "v09_final_decision.md").write_text(_render_final_decision(verdict, sr, {"table": []}, arms, None), encoding="utf-8")
    (out / "v09_seed_robustness.md").write_text(_render_seed_robustness(sr, arms["baseline_4b"], arms["prompt_only_inference_4b"]), encoding="utf-8")
    (out / "v09_ablation_matrix.md").write_text(_render_ablation_matrix(arms, verdict), encoding="utf-8")
    (out / "v09_judge_validation.md").write_text(_render_judge_md({"status": "not_run", "reason": "synthetic smoke", "would_judge_arms": list(seed_arms)}), encoding="utf-8")
    _write_jsonl(out / "v09_examples_wins_failures.jsonl", [{"category": "note", "detail": "synthetic smoke — see v09_ablation_matrix.md"}])
    return {"out": str(out), "verdict": verdict["verdict"], "n_seeds": verdict["n_seeds"],
            "n_seeds_passing": verdict["n_seeds_passing_win_gate"]}


# ======================================================================================
# parser
# ======================================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v0.9 replicate & stress-test the v0.8 calibration-balanced distillation win.")
    sub = p.add_subparsers(dest="command", required=True)

    pf = sub.add_parser("preflight")
    pf.add_argument("--v08-dir", required=True)
    pf.add_argument("--v07-metrics", required=True)
    pf.add_argument("--v06-failure-modes", required=True)
    pf.add_argument("--out", default=""); pf.add_argument("--allow-mismatch", action="store_true")
    pf.set_defaults(func=cmd_preflight)

    ms = sub.add_parser("make-stress-eval")
    ms.add_argument("--out", required=True); ms.add_argument("--standard-seed", type=int, default=8)
    ms.add_argument("--stress-seed", type=int, default=9); ms.add_argument("--per-split", type=int, default=None)
    ms.set_defaults(func=cmd_make_stress_eval)

    bm = sub.add_parser("build-mixtures")
    bm.add_argument("--kept-pairs", required=True, help="v0.8 source_audit_9b/pairs_kept.jsonl")
    bm.add_argument("--ratios", nargs="*", default=list(DEFAULT_RATIOS))
    bm.add_argument("--matched-size", type=lambda s: s.lower() != "false", default=True)
    bm.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2])
    bm.add_argument("--seed", type=int, default=0); bm.add_argument("--lr", type=float, default=1.5e-4)
    bm.add_argument("--epochs", type=int, default=3); bm.add_argument("--min-examples", type=int, default=100)
    bm.add_argument("--out", required=True); bm.set_defaults(func=cmd_build_mixtures)

    tr = sub.add_parser("train-matrix")
    tr.add_argument("--source-manifest", required=True); tr.add_argument("--eval-root", required=True)
    tr.add_argument("--arms", nargs="*", default=None); tr.add_argument("--kinds", default="")
    tr.add_argument("--include-optional", action="store_true")
    tr.add_argument("--base-model", default="Qwen/Qwen3.5-4B"); tr.add_argument("--rank", type=int, default=32)
    tr.add_argument("--lr", type=float, default=1.5e-4); tr.add_argument("--epochs", type=int, default=3)
    tr.add_argument("--batch-size", type=int, default=8); tr.add_argument("--max-seq", type=int, default=1024)
    tr.add_argument("--max-tokens", type=int, default=120); tr.add_argument("--min-examples", type=int, default=100)
    tr.add_argument("--sample-concurrency", type=int, default=32, help="concurrent sample futures per batch (~9x faster)")
    tr.add_argument("--allow-smoke", action="store_true"); tr.add_argument("--skip-base-arms", action="store_true")
    tr.add_argument("--out", required=True); tr.set_defaults(func=cmd_train_matrix)

    ev = sub.add_parser("eval-matrix")
    ev.add_argument("--training-manifest", required=True); ev.add_argument("--eval-root", required=True)
    ev.add_argument("--include-v08-reference", default=""); ev.add_argument("--out", required=True)
    ev.set_defaults(func=cmd_eval_matrix)

    jv = sub.add_parser("judge-validate")
    jv.add_argument("--eval-metrics", required=True); jv.add_argument("--eval-root", required=True)
    jv.add_argument("--judge-command", default=""); jv.add_argument("--judge-jsonl", default="")
    jv.add_argument("--per-split", type=int, default=6); jv.add_argument("--seed", type=int, default=0)
    jv.add_argument("--out", required=True); jv.set_defaults(func=cmd_judge_validate)

    de = sub.add_parser("decide")
    de.add_argument("--metrics", required=True); de.add_argument("--training", default="")
    de.add_argument("--judge", default=""); de.add_argument("--eval-root", default="")
    de.add_argument("--out", required=True); de.set_defaults(func=cmd_decide)

    sm = sub.add_parser("synthetic-smoke")
    sm.add_argument("--out", required=True); sm.add_argument("--quality", default="win",
                                                            choices=["win", "truth_only", "baseline", "prompt"])
    sm.add_argument("--seeds", type=int, default=3)
    sm.add_argument("--v08-dir", default=""); sm.add_argument("--v07-metrics", default="")
    sm.add_argument("--v06-failure-modes", default=""); sm.set_defaults(func=cmd_synthetic_smoke)
    return p


def main(argv: list[str] | None = None) -> None:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    print(json.dumps(args.func(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
