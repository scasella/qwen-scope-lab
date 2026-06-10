"""Dry-run validation scaffold for Mixture Dial Distiller.

This script pre-registers and compiles a dose-response plus matched-size arm
matrix using the existing offline mixture compiler. By default it never trains:
it writes the exact SFT corpora and a real-run command plan that can later be
executed with ``--execute``.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_scope_lab.experiments import mixture_dial_distill as md


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "0.1.0"

REAL_CANDIDATE_PATHS = (
    ROOT / "reports" / "steering_distill" / "th_v10_publication" / "v10_kept_combined.jsonl",
    ROOT / "reports" / "steering_distill" / "th_v10_publication" / "source_audit_9b_expansion" / "pairs_kept.jsonl",
    ROOT / "reports" / "steering_distill" / "th_v08_calibration_balanced" / "source_audit_9b" / "pairs_kept.jsonl",
)
V10_CORPUS_MANIFEST = ROOT / "reports" / "steering_distill" / "th_v10_publication" / "v10_corpus_manifest.json"
FIXTURE_CANDIDATES = ROOT / "examples" / "mixture_dial" / "truth_holding_candidates.jsonl"
DEFAULT_EVAL_ROOT = ROOT / "data" / "experiments" / "steering_distill" / "truth_holding_v09"

DEFAULT_CLASSES = ("A_factual", "B_unknowable", "C_subjective")
TRUTH_CLASSES = ("A_factual", "D_adversarial")
DEFAULT_CALIB_FRACS = (0.0, 0.25, 0.5, 0.75)
DEFAULT_SEEDS = (0, 1, 2)

TRAIN_SCRIPT = ROOT / "scripts" / "steering_distill_train_tinker.py"
EVAL_SCRIPT = ROOT / "scripts" / "steering_distill_eval_report.py"
TRAIN_MAX_LEN_DEFAULT = 1024


@dataclass(frozen=True)
class CandidateSource:
    path: Path
    kind: str
    note: str
    fixture_based: bool = False


@dataclass(frozen=True)
class ArmDefinition:
    name: str
    rung: str
    kind: str
    description: str
    target_total: int
    slots: list[dict[str, Any]]
    target_class_weights: dict[str, float] | None = None
    calib_frac: float | None = None
    candidates_path: Path | None = None
    requested_class_counts: dict[str, int] | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_candidate_source(candidates: str | Path | None) -> CandidateSource:
    if candidates:
        path = Path(candidates)
        if not path.exists():
            raise FileNotFoundError(f"candidate corpus not found: {path}")
        fixture = path.resolve() == FIXTURE_CANDIDATES.resolve()
        return CandidateSource(
            path=path,
            kind="fixture" if fixture else "explicit",
            note="explicit --candidates path",
            fixture_based=fixture,
        )

    for path in REAL_CANDIDATE_PATHS:
        if path.exists():
            return CandidateSource(path=path, kind="real", note="auto-discovered truth-holding kept corpus")

    if FIXTURE_CANDIDATES.exists():
        return CandidateSource(
            path=FIXTURE_CANDIDATES,
            kind="fixture",
            note="FIXTURE-BASED fallback; TODO: pass --candidates pointing at the real kept corpus",
            fixture_based=True,
        )
    raise FileNotFoundError("no real kept corpus or fixture candidate JSONL found")


def base_load_spec() -> md.MixtureSpec:
    return md.normalize_mixture_spec(
        {
            "schema_version": md.SCHEMA_VERSION,
            "seed": 0,
            "total": 1,
            "dimensions": list(md.DEFAULT_DIMENSIONS),
            "slots": [{"name": "all", "where": {}, "count": 1}],
            "output": {"format": "sft_chat", "preserve_labels": True},
        }
    )


def load_valid_candidates(path: str | Path) -> list[dict[str, Any]]:
    return md.load_candidates(path, base_load_spec()).valid


def canonical_class(value: Any) -> str:
    text = str(value)
    if text in TRUTH_CLASSES:
        return "A_factual"
    return text


def summarize_vocabulary(valid: list[dict[str, Any]]) -> dict[str, Any]:
    labels_by_dim: dict[str, Counter[str]] = {dim: Counter() for dim in md.DEFAULT_DIMENSIONS}
    class_counts: Counter[str] = Counter()
    for row in valid:
        labels = row.get("labels") or {}
        cls = canonical_class(labels.get("class"))
        if cls not in ("None", ""):
            class_counts[cls] += 1
        for dim in md.DEFAULT_DIMENSIONS:
            vals = labels.get(dim)
            for val in vals if isinstance(vals, list) else [vals]:
                if val not in (None, ""):
                    labels_by_dim[dim][str(val)] += 1
    return {
        "n_valid": len(valid),
        "class_counts": dict(sorted(class_counts.items())),
        "label_vocabulary": {dim: dict(sorted(counts.items())) for dim, counts in labels_by_dim.items()},
    }


def largest_remainder_counts(weights: dict[str, float], total: int) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if not weights:
        return {}
    positive = {name: float(weight) for name, weight in weights.items() if float(weight) > 0.0}
    if not positive:
        return {name: 0 for name in weights}
    weight_sum = sum(positive.values())
    raw = {name: positive[name] / weight_sum * total for name in positive}
    counts = {name: int(raw[name]) for name in positive}
    remainder = total - sum(counts.values())
    order = sorted(positive, key=lambda name: (raw[name] - counts[name], -list(positive).index(name)), reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return {name: counts.get(name, 0) for name in weights}


def calibration_split(class_counts: dict[str, int]) -> dict[str, float]:
    b = class_counts.get("B_unknowable", 0)
    c = class_counts.get("C_subjective", 0)
    total = b + c
    if total <= 0:
        return {"B_unknowable": 0.5, "C_subjective": 0.5}
    return {"B_unknowable": b / total, "C_subjective": c / total}


def class_weights_for_calib_frac(calib_frac: float, calib_weights: dict[str, float]) -> dict[str, float]:
    frac = float(calib_frac)
    if frac < 0.0 or frac > 1.0:
        raise ValueError(f"calibration fraction must be in [0, 1], got {frac}")
    return {
        "A_factual": 1.0 - frac,
        "B_unknowable": frac * calib_weights.get("B_unknowable", 0.5),
        "C_subjective": frac * calib_weights.get("C_subjective", 0.5),
    }


def naive_class_weights(class_counts: dict[str, int]) -> dict[str, float]:
    total = sum(class_counts.get(cls, 0) for cls in DEFAULT_CLASSES)
    if total <= 0:
        raise ValueError("candidate corpus has no A/B/C class labels")
    return {cls: class_counts.get(cls, 0) / total for cls in DEFAULT_CLASSES}


def class_where(cls: str, observed_classes: set[str]) -> Any:
    if cls == "A_factual" and "D_adversarial" in observed_classes:
        return list(TRUTH_CLASSES)
    return cls


def class_slots(weights: dict[str, float], *, observed_classes: set[str]) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for cls in DEFAULT_CLASSES:
        weight = weights.get(cls, 0.0)
        if weight <= 0.0:
            continue
        slots.append(
            {
                "name": cls,
                "where": {"class": class_where(cls, observed_classes)},
                "ratio": weight,
            }
        )
    if not slots:
        raise ValueError("arm has no positive class slots")
    return slots


def arm_requested_class_counts(arm: ArmDefinition) -> dict[str, int]:
    if arm.requested_class_counts is not None:
        return dict(arm.requested_class_counts)
    if arm.target_class_weights is None:
        return {"<unconstrained>": arm.target_total}
    return {
        cls: count
        for cls, count in largest_remainder_counts(arm.target_class_weights, arm.target_total).items()
        if count > 0
    }


def build_arm_definitions(
    *,
    n: int,
    class_counts: dict[str, int],
    observed_classes: set[str],
    calib_fracs: tuple[float, ...],
    prompt_only_candidates: Path | None = None,
) -> tuple[list[ArmDefinition], dict[str, Any]]:
    calib_weights = calibration_split(class_counts)
    arms: list[ArmDefinition] = []

    for frac in calib_fracs:
        weights = class_weights_for_calib_frac(frac, calib_weights)
        name = "truth_only" if frac == 0.0 else f"calib_frac_{frac:.2f}"
        arms.append(
            ArmDefinition(
                name=name,
                rung="dose_response" if frac != 0.0 else "dose_response+matched_size_baseline",
                kind="truth_only" if frac == 0.0 else "calibration_dial",
                description=(
                    "truth-only arm; calibration fraction 0.0"
                    if frac == 0.0
                    else f"dose-response arm with calibration fraction {frac:.2f}"
                ),
                target_total=n,
                slots=class_slots(weights, observed_classes=observed_classes),
                target_class_weights=weights,
                calib_frac=frac,
            )
        )

    natural = naive_class_weights(class_counts)
    arms.extend(
        [
            ArmDefinition(
                name="random_same_size",
                rung="matched_size_baseline",
                kind="random_same_size",
                description="unconstrained random sample from the kept corpus at the same N",
                target_total=n,
                slots=[{"name": "random_all", "where": {}, "count": n}],
            ),
            ArmDefinition(
                name="naive_stratified",
                rung="matched_size_baseline",
                kind="naive_stratified",
                description="class-stratified sample using natural A/B/C frequencies at the same N",
                target_total=n,
                slots=class_slots(natural, observed_classes=observed_classes),
                target_class_weights=natural,
            ),
        ]
    )

    prompt_status: dict[str, Any]
    if prompt_only_candidates:
        arms.append(
            ArmDefinition(
                name="prompt_only",
                rung="matched_size_baseline",
                kind="prompt_only",
                description="prompt-only teacher candidate data supplied by --prompt-only-candidates",
                target_total=n,
                slots=[{"name": "prompt_only_all", "where": {}, "count": n}],
                candidates_path=prompt_only_candidates,
            )
        )
        prompt_status = {"available": True, "path": str(prompt_only_candidates), "note": "included as supplied data"}
    else:
        prompt_status = {
            "available": False,
            "path": None,
            "note": (
                "No prompt-only candidate SFT corpus was found or supplied. Existing prompt_only_inference_4b "
                "artifacts are eval outputs, not teacher data; this plan does not fabricate a prompt-only SFT arm."
            ),
        }

    return arms, {"calibration_split": calib_weights, "prompt_only": prompt_status}


def mixture_spec_for_arm(arm: ArmDefinition, seed: int) -> dict[str, Any]:
    return {
        "schema_version": md.SCHEMA_VERSION,
        "seed": int(seed),
        "total": int(arm.target_total),
        "dimensions": list(md.DEFAULT_DIMENSIONS),
        "slots": arm.slots,
        "output": {"format": "sft_chat", "preserve_labels": True},
    }


def class_counts_from_sft(path: str | Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in read_jsonl(path):
        cls = row.get("behavioral_class") or (row.get("labels") or {}).get("class")
        counts[canonical_class(cls)] += 1
    return dict(sorted(counts.items()))


def count_eval_prompts(eval_prompts: str | Path | None, eval_root: str | Path | None) -> dict[str, Any]:
    if eval_prompts:
        path = Path(eval_prompts)
        if path.exists():
            return {"count": len(read_jsonl(path)), "source": str(path), "kind": "eval_prompts"}
        return {"count": 0, "source": str(path), "kind": "missing_eval_prompts"}

    root = Path(eval_root or DEFAULT_EVAL_ROOT)
    files = sorted(root.glob("eval_*_scenarios.jsonl"))
    if files:
        count = sum(len(read_jsonl(path)) for path in files)
        return {"count": count, "source": str(root), "kind": "truth_holding_v09_scenarios"}
    return {"count": 0, "source": str(root), "kind": "missing_eval_root"}


def default_n_for_source(source: CandidateSource, vocabulary: dict[str, Any]) -> tuple[int, str]:
    if source.path.resolve() == (ROOT / "reports" / "steering_distill" / "th_v10_publication" / "v10_kept_combined.jsonl").resolve():
        if V10_CORPUS_MANIFEST.exists():
            manifest = json.loads(V10_CORPUS_MANIFEST.read_text(encoding="utf-8"))
            n = int(manifest.get("matched_n_feasible") or 0)
            if n > 0:
                return n, f"matched_n_feasible from {V10_CORPUS_MANIFEST.relative_to(ROOT)}"
    return int(vocabulary["n_valid"]), "valid candidate count"


def train_command_for_entry(entry: dict[str, Any], args: argparse.Namespace) -> list[str]:
    eval_out = Path(entry["arm_dir"]) / "eval_arms.json"
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--sft",
        str(Path(entry["arm_dir"]) / "sft.jsonl"),
        "--base-model",
        args.base_model,
        "--rank",
        str(args.rank),
        "--lr",
        str(args.lr),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--max-tokens",
        str(args.max_tokens),
        "--eval-prompts",
        str(args.eval_prompts or "<required --eval-prompts>"),
        "--eval-out",
        str(eval_out),
        "--name",
        entry["run_id"],
    ]
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    return cmd


def eval_command_for_entry(entry: dict[str, Any], args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(EVAL_SCRIPT),
        "--arms",
        str(Path(entry["arm_dir"]) / "eval_arms.json"),
        "--target",
        args.eval_target,
        "--out",
        str(Path(entry["arm_dir"]) / "eval_report"),
    ]


def compute_plan(matrix: list[dict[str, Any]], args: argparse.Namespace, eval_prompt_info: dict[str, Any]) -> dict[str, Any]:
    train_runs = len(matrix)
    train_steps = sum(math.ceil(entry["achieved_total"] / args.batch_size) * args.epochs for entry in matrix)
    train_token_upper_bound = sum(entry["achieved_total"] * args.epochs * TRAIN_MAX_LEN_DEFAULT for entry in matrix)
    eval_prompt_count = int(eval_prompt_info.get("count") or 0)
    eval_sampling_calls = train_runs * eval_prompt_count * 2
    return {
        "estimate": True,
        "estimate_note": (
            "Local scripts do not expose Tinker dollar pricing. The bill is expressed as LoRA train runs, "
            "optimizer steps, token upper-bound slots, eval sample calls, and eval report invocations."
        ),
        "train_script": str(TRAIN_SCRIPT.relative_to(ROOT)),
        "eval_report_script": str(EVAL_SCRIPT.relative_to(ROOT)),
        "base_model": args.base_model,
        "rank": args.rank,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_eval_tokens": args.max_tokens,
        "train_max_len_assumed": TRAIN_MAX_LEN_DEFAULT,
        "lora_train_runs": train_runs,
        "eval_report_runs": train_runs,
        "model_sampling_arms_per_train_run": 2,
        "eval_prompt_info": eval_prompt_info,
        "estimated_tinker_train_steps": train_steps,
        "estimated_training_token_upper_bound": train_token_upper_bound,
        "estimated_eval_sample_calls": eval_sampling_calls,
        "estimated_dollars": None,
    }


def render_plan_md(plan: dict[str, Any]) -> str:
    arms = plan["arms"]
    rows = "\n".join(
        "| `{name}` | {rung} | {n} | {calib} | {requested} |".format(
            name=arm["name"],
            rung=arm["rung"],
            n=arm["target_total"],
            calib="-" if arm.get("calib_frac") is None else arm["calib_frac"],
            requested=json.dumps(arm["requested_class_counts"], sort_keys=True),
        )
        for arm in arms
    )
    compute = plan["compute_plan"]
    prompt_note = plan["prompt_only"]["note"]
    source_label = "FIXTURE-BASED" if plan["candidate_source"]["fixture_based"] else "REAL-CORPUS"
    return (
        "# Mixture Dial Distiller dry-run pre-registration\n\n"
        f"Status: **{source_label} DRY RUN**. No Tinker, Modal, CUDA, hosted API, model weights, training, or eval "
        "execution is performed by this plan.\n\n"
        "## Hypothesis\n\n"
        "Mixture Dial Distiller should behave like a real product control, not a thin wrapper: increasing the "
        "calibration fraction in the SFT corpus should causally and differentiably change held-out model behavior "
        "after fine-tuning, improving calibration on unknowable/subjective prompts while preserving truth-holding "
        "on factual false-pressure prompts.\n\n"
        "## Data and compiler\n\n"
        f"- Candidate corpus: `{plan['candidate_source']['path']}` ({plan['candidate_source']['kind']}).\n"
        f"- N per arm: {plan['n']} ({plan['n_source']}).\n"
        f"- Seeds: {', '.join(str(s) for s in plan['seeds'])}.\n"
        f"- Compiler: `qwen_scope_lab/experiments/mixture_dial_distill.py::compile_to_dir`.\n"
        f"- Prompt-only teacher data: {prompt_note}\n\n"
        "## Arms\n\n"
        "| arm | rung | N | calibration fraction | requested class counts |\n"
        "|---|---:|---:|---:|---|\n"
        f"{rows}\n\n"
        "## Metrics for the later real run\n\n"
        "The real run will train one LoRA per arm x seed via `scripts/steering_distill_train_tinker.py`, then score "
        "held-out outputs via `scripts/steering_distill_eval_report.py --target deference` (truth-holding alias). "
        "Primary reads: calibration on B/C held-out prompts, truth-holding on A factual false-pressure prompts, "
        "dose-response monotonicity across calibration fractions, and matched-size comparisons against truth-only "
        "and naive-stratified baselines.\n\n"
        "## KILL CRITERIA\n\n"
        "- K1: dose-response is flat, non-monotone, or within seed noise on calibration/truth-holding behavior.\n"
        "- K2: dials are approximately equal to naive_stratified at matched N, indicating thin-wrapper failure.\n"
        "- K3: the win does not survive matched-size comparison against truth_only.\n\n"
        "## PASS criteria\n\n"
        "A pass requires a visible monotone or near-monotone dose-response from truth_only through higher "
        "calibration fractions, the 0.50 dial beating truth_only on B/C calibration without unacceptable A-class "
        "truth-holding loss, and the dialed arms beating random_same_size and naive_stratified at the same N.\n\n"
        "## Estimated real-run compute\n\n"
        f"- LoRA train runs: {compute['lora_train_runs']}.\n"
        f"- Eval report runs: {compute['eval_report_runs']}.\n"
        f"- Estimated optimizer steps: {compute['estimated_tinker_train_steps']}.\n"
        f"- Estimated train token upper-bound slots: {compute['estimated_training_token_upper_bound']}.\n"
        f"- Estimated eval sample calls: {compute['estimated_eval_sample_calls']} "
        f"({compute['eval_prompt_info']['count']} prompts x 2 sampled arms x train runs).\n"
        "- Dollar estimate: unavailable from local scripts; see `plan.json` for the unit bill.\n"
    )


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    source = resolve_candidate_source(args.candidates)
    prompt_only_path = Path(args.prompt_only_candidates) if args.prompt_only_candidates else None
    if prompt_only_path and not prompt_only_path.exists():
        raise FileNotFoundError(f"prompt-only candidates not found: {prompt_only_path}")

    valid = load_valid_candidates(source.path)
    vocabulary = summarize_vocabulary(valid)
    if not valid:
        raise ValueError(f"no valid candidates in {source.path}")

    n_default, n_source = default_n_for_source(source, vocabulary)
    n = int(args.n if args.n is not None else n_default)
    if n <= 0:
        raise ValueError("N must be positive")

    observed_classes = set(vocabulary["label_vocabulary"]["class"])
    arms, extra = build_arm_definitions(
        n=n,
        class_counts=vocabulary["class_counts"],
        observed_classes=observed_classes,
        calib_fracs=tuple(args.calib_fracs),
        prompt_only_candidates=prompt_only_path,
    )

    out = Path(args.out)
    matrix: list[dict[str, Any]] = []
    arms_json: list[dict[str, Any]] = []

    for arm in arms:
        candidates_path = arm.candidates_path or source.path
        arms_json.append(
            {
                "name": arm.name,
                "rung": arm.rung,
                "kind": arm.kind,
                "description": arm.description,
                "target_total": arm.target_total,
                "calib_frac": arm.calib_frac,
                "slots": arm.slots,
                "target_class_weights": arm.target_class_weights,
                "requested_class_counts": arm_requested_class_counts(arm),
                "candidates_path": str(candidates_path),
            }
        )
        for seed in args.seeds:
            arm_dir = out / "arms" / arm.name / str(seed)
            spec_path = arm_dir / "mixture.json"
            arm_dir.mkdir(parents=True, exist_ok=True)
            write_json(spec_path, mixture_spec_for_arm(arm, seed))
            summary = md.compile_to_dir(spec_path, candidates_path, arm_dir)
            manifest_path = Path(summary["paths"]["manifest"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            achieved_class_counts = class_counts_from_sft(summary["paths"]["sft"])
            run_id = f"{arm.name}_seed{seed}"
            entry = {
                "run_id": run_id,
                "arm": arm.name,
                "seed": int(seed),
                "rung": arm.rung,
                "kind": arm.kind,
                "target_total": arm.target_total,
                "calib_frac": arm.calib_frac,
                "arm_dir": str(arm_dir),
                "mixture": str(spec_path),
                "candidates": str(candidates_path),
                "sft": summary["paths"]["sft"],
                "manifest": summary["paths"]["manifest"],
                "requested_slot_counts": summary["requested_counts"],
                "achieved_slot_counts": summary["achieved_counts"],
                "requested_class_counts": arm_requested_class_counts(arm),
                "achieved_class_counts": achieved_class_counts,
                "achieved_total": manifest["achieved_total"],
                "underfilled_slots": summary["underfilled_slots"],
                "train_command": shlex.join(train_command_for_entry({"arm_dir": str(arm_dir), "run_id": run_id}, args)),
                "eval_command": shlex.join(eval_command_for_entry({"arm_dir": str(arm_dir)}, args)),
                "status": "compiled_dry_run",
            }
            matrix.append(entry)

    eval_prompt_info = count_eval_prompts(args.eval_prompts, args.eval_root)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "dry_run": not args.execute,
        "execute_requested": bool(args.execute),
        "candidate_source": {
            "path": str(source.path),
            "kind": source.kind,
            "note": source.note,
            "fixture_based": source.fixture_based,
        },
        "todo": (
            ["Point --candidates at the real truth-holding kept corpus before treating this as product evidence."]
            if source.fixture_based
            else []
        ),
        "n": n,
        "n_source": args.n_source or n_source,
        "seeds": [int(s) for s in args.seeds],
        "calib_fracs": [float(f) for f in args.calib_fracs],
        "candidate_vocabulary": vocabulary,
        "calibration_split": extra["calibration_split"],
        "prompt_only": extra["prompt_only"],
        "arms": arms_json,
        "matrix": matrix,
        "arm_count": len(arms_json),
        "arm_seed_count": len(matrix),
        "kill_criteria": {
            "K1": "dose-response is flat, non-monotone, or within seed noise",
            "K2": "dials approximately equal naive_stratified at matched N",
            "K3": "win does not survive matched-size vs truth_only",
        },
        "pass_criteria": (
            "Monotone or near-monotone calibration dose-response; 0.50 dial improves B/C calibration without "
            "unacceptable A-class truth-holding loss; dialed arms beat random_same_size and naive_stratified at N."
        ),
        "compute_plan": compute_plan(matrix, args, eval_prompt_info),
    }
    write_json(out / "plan.json", plan)
    (out / "plan.md").write_text(render_plan_md(plan), encoding="utf-8")

    if args.execute:
        execute_real_run(plan, args)
    return plan


def execute_real_run(plan: dict[str, Any], args: argparse.Namespace) -> None:
    if not args.eval_prompts:
        raise SystemExit("--execute requires --eval-prompts pointing at a JSONL with {id,prompt,metadata}")
    if not Path(args.eval_prompts).exists():
        raise SystemExit(f"--eval-prompts not found: {args.eval_prompts}")
    executed = []
    for entry in plan["matrix"]:
        train_cmd = train_command_for_entry(entry, args)
        eval_cmd = eval_command_for_entry(entry, args)
        eval_arms = Path(entry["arm_dir"]) / "eval_arms.json"
        if eval_arms.exists():  # idempotent/resumable: skip arms already sampled (and reuse a prior run)
            executed.append({"run_id": entry["run_id"], "skipped": "eval_arms.json exists"})
            continue
        subprocess.run(train_cmd, cwd=ROOT, check=True)
        subprocess.run(eval_cmd, cwd=ROOT, check=True)
        executed.append({"run_id": entry["run_id"], "train_command": shlex.join(train_cmd), "eval_command": shlex.join(eval_cmd)})
    write_json(Path(args.out) / "execute_manifest.json", {"executed_at": utc_now_iso(), "runs": executed})


def print_summary(plan: dict[str, Any]) -> None:
    source_label = "FIXTURE-BASED" if plan["candidate_source"]["fixture_based"] else "REAL-CORPUS"
    print("Mixture Dial Validate dry-run")
    print(f"source: {source_label} {plan['candidate_source']['path']}")
    print(f"N: {plan['n']} ({plan['n_source']})")
    print(f"arms: {plan['arm_count']}  seeds: {','.join(str(s) for s in plan['seeds'])}  arm x seed: {plan['arm_seed_count']}")
    print("\nArm matrix:")
    for entry in plan["matrix"]:
        print(
            f"- {entry['arm']} seed={entry['seed']} requested={entry['requested_class_counts']} "
            f"achieved={entry['achieved_class_counts']} total={entry['achieved_total']}"
        )
    compute = plan["compute_plan"]
    print("\nCompute estimate:")
    print(f"- LoRA train runs: {compute['lora_train_runs']}")
    print(f"- Eval report runs: {compute['eval_report_runs']}")
    print(f"- Estimated Tinker train steps: {compute['estimated_tinker_train_steps']}")
    print(f"- Estimated train token upper-bound slots: {compute['estimated_training_token_upper_bound']}")
    print(f"- Estimated eval sample calls: {compute['estimated_eval_sample_calls']}")
    underfilled = [entry for entry in plan["matrix"] if entry["underfilled_slots"]]
    print("\nUnderfilled slots:")
    if not underfilled:
        print("- none")
    else:
        for entry in underfilled:
            print(f"- WARNING {entry['run_id']}: {entry['underfilled_slots']}")
    print(f"\nWrote: {Path(plan['matrix'][0]['arm_dir']).parents[2] / 'plan.json'}")


def parse_csv_floats(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def parse_csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile and pre-register a dry-run Mixture Dial Distiller validation matrix."
    )
    parser.add_argument("--out", required=True, help="Output directory for plan.json, plan.md, and arm corpora")
    parser.add_argument("--candidates", help="Candidate JSONL. Defaults to auto-discovered real kept corpus, then fixture")
    parser.add_argument("--prompt-only-candidates", help="Optional prompt-only teacher candidate JSONL; never fabricated")
    parser.add_argument("--n", type=int, help="Fixed total N per arm. Defaults to real matched_n_feasible when available")
    parser.add_argument("--n-source", default="", help="Optional note overriding the plan's N source text")
    parser.add_argument("--seeds", type=parse_csv_ints, default=DEFAULT_SEEDS, help="Comma-separated seeds, default 0,1,2")
    parser.add_argument(
        "--calib-fracs",
        type=parse_csv_floats,
        default=DEFAULT_CALIB_FRACS,
        help="Comma-separated calibration fractions, default 0.0,0.25,0.50,0.75",
    )
    parser.add_argument("--dry-run", action="store_true", default=True, help="Default mode; compile only, no training")
    parser.add_argument("--execute", action="store_true", help="Actually run Tinker train/eval commands after compiling")
    parser.add_argument("--eval-prompts", help="Required for --execute; optional for dry-run estimate")
    parser.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT), help="Held-out truth-holding scenario root for estimates")
    parser.add_argument("--eval-target", default="deference", help="Target passed to steering_distill_eval_report.py")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=56)
    parser.add_argument("--limit", type=int, default=0, help="Optional training-example cap forwarded only under --execute")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    plan = run_validation(args)
    print_summary(plan)


if __name__ == "__main__":
    main()
