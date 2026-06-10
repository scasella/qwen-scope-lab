"""Manifold-to-data provenance compiler (C09 — first build step).

Compile concept-manifold steering payloads (the ``manifold`` / ``linear`` / ``pullback`` legs of
``SteeringService.manifold_compare`` / ``manifold_pullback``, each carrying behavior-energy and —
for pullback — ``recovered_r``) into **provenance-stamped** SFT / preference records, keeping only
the on-manifold samples whose *geometry* clears a gate. Rejected records are retained with reasons.

This is the offline, **torch-free** half of the C09 experiment: it consumes payloads shaped like
the service's manifold result dicts and never touches a model. The real behavior-energy and
``recovered_r`` come from :mod:`qwen_scope_lab.service` (``manifold_compare`` / ``manifold_pullback``)
and the ``/api/manifold/*`` endpoints in :mod:`qwen_scope_lab.web_api`; tests here use synthetic
payloads only.

FIRST BUILD STEP — schema / gating / export only. It proves the data schema, manifold-recipe
loading, equal-size arm accounting, the rejection ledger, and the report shape. It makes **no claim**
that geometry-gated data trains a better model than text-only steering data — that is C09's open,
preregistered question with a hard falsification gate (see
``docs/experiments/MANIFOLD_TO_DATA_PROVENANCE.md``). SOURCE-GAP: novelty beyond Goodfire
arXiv 2604.28119 / 2605.05115 and adjacent activation-steering distillation work is unverified.

The companion feature-steer distiller is :mod:`qwen_scope_lab.experiments.steering_distill`; this
module is its concept-manifold sibling and reuses its collapse/empty checks and JSONL writer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..recipe_schema import FeatureRecipe, utc_now_iso
from .steering_distill import SCHEMA_VERSION as STEER_SCHEMA_VERSION
from .steering_distill import is_collapsed, is_empty, write_jsonl

SCHEMA_VERSION = "0.1.0"

ON_MANIFOLD_PATHS = ("manifold", "pullback")  # the arms gated against the linear chord
ALL_PATHS = ("manifold", "linear", "pullback")

__all__ = [
    "SCHEMA_VERSION",
    "ON_MANIFOLD_PATHS",
    "ManifoldDataSpec",
    "GateConfig",
    "load_manifold_recipe",
    "spec_from_recipe",
    "records_from_payload",
    "gate_records",
    "compile_payload",
    "compute_metrics",
    "to_sft_records",
    "to_preference_records",
    "render_dataset_card",
    "render_report",
    "write_outputs",
    "build_synthetic_payload",
]


# --------------------------------------------------------------------------------------
# 1. Spec — the normalized input (a saved manifold recipe OR an explicit config)
# --------------------------------------------------------------------------------------


@dataclass
class ManifoldDataSpec:
    """A normalized concept-manifold data-generation config.

    ``recipe_status`` is carried through to every record, the dataset card, and the report so a
    consumer always knows whether the source steer was ``validated`` (on-manifold energy ≤ the linear
    chord on the manifold benchmark) or merely ``benchmarked`` / ``candidate``.
    """

    concept: str
    source: str
    target: str
    layer: int
    path: str = "manifold"
    n_waypoints: int = 5
    behavior_readout: str = "first_token"
    # provenance
    source_kind: str = "explicit"  # "recipe" | "explicit" | "synthetic"
    recipe_id: str = ""
    recipe_status: str = "candidate"  # validated | benchmarked | candidate | draft | explicit
    model_id: str = ""
    target_name: str = ""
    target_description: str = ""
    validation_decision: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

    @property
    def validated(self) -> bool:
        return self.recipe_status == "validated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_recipe(cls, recipe: FeatureRecipe, *, behavior_readout: str = "first_token") -> ManifoldDataSpec:
        if recipe.kind != "manifold" or not recipe.manifold:
            raise ValueError(
                "manifold-to-data distillation requires a manifold recipe "
                f"(got kind={recipe.kind!r}). Feature-steer recipes go through "
                "qwen_scope_lab.experiments.steering_distill instead."
            )
        m = recipe.manifold
        return cls(
            concept=m.concept,
            source=m.source,
            target=m.target,
            layer=int(m.layer),
            path=m.path,
            n_waypoints=int(m.n_waypoints),
            source_kind="recipe",
            recipe_id=recipe.recipe_id,
            recipe_status=recipe.status,
            model_id=recipe.model.model_id,
            target_name=recipe.target_behavior.name,
            target_description=recipe.target_behavior.description,
            validation_decision=dict(recipe.benchmark.get("validation_decision") or {}),
            limitations=list(recipe.limitations),
        )

    @classmethod
    def explicit(cls, *, concept: str, source: str, target: str, layer: int, path: str = "manifold",
                 n_waypoints: int = 5, behavior_readout: str = "first_token", model_id: str = "") -> ManifoldDataSpec:
        return cls(
            concept=concept, source=source, target=target, layer=int(layer), path=path,
            n_waypoints=int(n_waypoints), behavior_readout=behavior_readout,
            source_kind="explicit", recipe_status="candidate", model_id=model_id,
            target_name=f"manifold_{concept}", target_description=f"Steer '{concept}' {source}→{target} along its residual manifold.",
        )


def load_manifold_recipe(path: str | Path) -> FeatureRecipe:
    """Load a saved manifold recipe JSON and assert it is a manifold recipe."""
    recipe = FeatureRecipe.from_json(Path(path).read_text(encoding="utf-8"))
    if recipe.kind != "manifold" or not recipe.manifold:
        raise ValueError(f"{path}: not a manifold recipe (kind={recipe.kind!r})")
    return recipe


def spec_from_recipe(recipe: FeatureRecipe, *, behavior_readout: str = "first_token") -> ManifoldDataSpec:
    return ManifoldDataSpec.from_recipe(recipe, behavior_readout=behavior_readout)


# --------------------------------------------------------------------------------------
# 2. Gate config + record normalization
# --------------------------------------------------------------------------------------


@dataclass
class GateConfig:
    """Geometry gate for keeping a record. The on-manifold arms (manifold/pullback) must induce the
    target behavior at least as faithfully as the linear chord (``energy ≤ linear + margin``) and,
    when available, recover the manifold (``recovered_r ≥ min_recovered_r``)."""

    require_hook_fired: bool = True
    require_beats_linear: bool = True
    energy_margin: float = 0.0          # allow energy ≤ linear_energy + margin
    min_recovered_r: float = 0.0        # applied only when recovered_r is present
    reject_collapsed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _leg_records(spec: ManifoldDataSpec, payload: dict[str, Any], path: str) -> list[dict[str, Any]]:
    leg = payload.get(path)
    if not isinstance(leg, dict):
        return []
    prompt = leg.get("prompt") or payload.get("prompt") or ""
    mean_energy = leg.get("mean_energy")
    recovered_r = leg.get("recovered_r")
    leg_hook = leg.get("hook_fired")
    waypoints = leg.get("waypoints") or []
    rows = []
    for wi, wp in enumerate(waypoints):
        hook = wp.get("hook_fired")
        if hook is None:
            hook = leg_hook
        rows.append({
            "id": f"{spec.concept}_{path}_w{wi:02d}",
            "concept": spec.concept,
            "source": leg.get("source", spec.source),
            "target": leg.get("target", spec.target),
            "layer": leg.get("layer", spec.layer),
            "path": path,
            "waypoint_index": wi,
            "value": wp.get("value"),
            "prompt": prompt,
            "steered_text": wp.get("text", ""),
            "perplexity": wp.get("perplexity"),
            "waypoint_energy": wp.get("energy"),
            "mean_energy": mean_energy,
            "recovered_r": recovered_r,
            "hook_fired": bool(hook) if hook is not None else None,
            "behavior_readout": payload.get("behavior_readout") or spec.behavior_readout,
            "recipe_id": spec.recipe_id,
            "recipe_status": spec.recipe_status,
            "model_id": spec.model_id,
            "validation_decision": spec.validation_decision,
        })
    return rows


def records_from_payload(payload: dict[str, Any], spec: ManifoldDataSpec) -> list[dict[str, Any]]:
    """Flatten a manifold comparison payload into one record per (leg, waypoint).

    ``payload`` is shaped like ``service.manifold_compare`` / ``manifold_pullback`` output: a dict
    with ``manifold`` / ``linear`` / (optional) ``pullback`` legs, each a manifold_steer result.
    """
    records: list[dict[str, Any]] = []
    for path in ALL_PATHS:
        records.extend(_leg_records(spec, payload, path))
    return records


def _linear_energy(records: list[dict[str, Any]]) -> float | None:
    for r in records:
        if r["path"] == "linear" and r.get("mean_energy") is not None:
            return float(r["mean_energy"])
    return None


def _reasons_for(record: dict[str, Any], gate: GateConfig, linear_energy: float | None) -> list[str]:
    reasons: list[str] = []
    text = record.get("steered_text", "")
    if is_empty(text):
        reasons.append("empty")
    elif gate.reject_collapsed and is_collapsed(text)[0]:
        reasons.append("collapsed")
    if gate.require_hook_fired and record.get("hook_fired") is False:
        reasons.append("hook_not_fired")
    on_manifold = record["path"] in ON_MANIFOLD_PATHS
    if on_manifold and gate.require_beats_linear and linear_energy is not None and record.get("mean_energy") is not None:
        if float(record["mean_energy"]) > linear_energy + gate.energy_margin:
            reasons.append("energy_above_linear")
    if on_manifold and record.get("recovered_r") is not None:
        if float(record["recovered_r"]) < gate.min_recovered_r:
            reasons.append("recovered_r_below_threshold")
    # de-dupe, preserve order
    seen: set[str] = set()
    return [r for r in reasons if not (r in seen or seen.add(r))]


def gate_records(records: list[dict[str, Any]], gate: GateConfig) -> dict[str, list[dict[str, Any]]]:
    """Apply the geometry gate to every record; split into kept/rejected (rejected keep reasons)."""
    linear_energy = _linear_energy(records)
    graded = []
    for r in records:
        reasons = _reasons_for(r, gate, linear_energy)
        graded.append({**r, "linear_energy": linear_energy, "keep": not reasons, "reject_reasons": reasons})
    return {
        "all": graded,
        "kept": [r for r in graded if r["keep"]],
        "rejected": [r for r in graded if not r["keep"]],
        "linear_energy": linear_energy,
    }


# --------------------------------------------------------------------------------------
# 3. Metrics, exports, report
# --------------------------------------------------------------------------------------


def compute_metrics(graded: dict[str, list[dict[str, Any]]], spec: ManifoldDataSpec, gate: GateConfig) -> dict[str, Any]:
    rows = graded["all"]
    per_arm: dict[str, dict[str, Any]] = {}
    for path in ALL_PATHS:
        arm = [r for r in rows if r["path"] == path]
        if not arm:
            continue
        kept = [r for r in arm if r["keep"]]
        energies = [r["mean_energy"] for r in arm if r.get("mean_energy") is not None]
        rr = [r["recovered_r"] for r in arm if r.get("recovered_r") is not None]
        per_arm[path] = {
            "n_total": len(arm),
            "n_kept": len(kept),
            "n_rejected": len(arm) - len(kept),
            "mean_energy": round(energies[0], 4) if energies else None,
            "recovered_r": round(rr[0], 4) if rr else None,
        }
    reason_counts: dict[str, int] = {}
    for r in graded["rejected"]:
        for reason in r["reject_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    # equal-size training: the largest balanced N across the on-manifold arms + the linear baseline
    balanced_arms = [p for p in (*ON_MANIFOLD_PATHS, "linear") if p in per_arm]
    min_kept = min((per_arm[p]["n_kept"] for p in balanced_arms), default=0)
    n = len(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "steering_distill_schema": STEER_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "concept": spec.concept,
        "source": spec.source,
        "target": spec.target,
        "layer": spec.layer,
        "behavior_readout": spec.behavior_readout,
        "n_records": n,
        "n_kept": len(graded["kept"]),
        "n_rejected": len(graded["rejected"]),
        "keep_rate": round(len(graded["kept"]) / n, 4) if n else 0.0,
        "linear_energy": graded.get("linear_energy"),
        "per_arm": per_arm,
        "equal_size_n_per_arm": min_kept,  # truncate each arm to this for a balanced training comparison
        "reject_reason_counts": dict(sorted(reason_counts.items())),
        "n_sft": len(to_sft_records(graded["kept"])),
        "n_preference": len(to_preference_records(graded)),
        "recipe_status": spec.recipe_status,
        "validated": spec.validated,
        "source_kind": spec.source_kind,
        "gate": gate.to_dict(),
        "spec": spec.to_dict(),
        "source_gap": ("First build step: schema/export only. No model-training claim. Novelty beyond "
                       "Goodfire arXiv 2604.28119 / 2605.05115 and adjacent activation-steering "
                       "distillation work is UNVERIFIED."),
    }


def to_sft_records(kept: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """SFT JSONL from the kept on-manifold records: the steered output is the desired completion.

    Each record carries provenance so a downstream filter can re-gate without re-running the model.
    """
    out = []
    for r in kept:
        if r["path"] not in ON_MANIFOLD_PATHS:
            continue
        out.append({
            "messages": [{"role": "user", "content": r["prompt"]},
                         {"role": "assistant", "content": r["steered_text"]}],
            "provenance": {"concept": r["concept"], "path": r["path"], "value": r["value"],
                           "waypoint_index": r["waypoint_index"], "layer": r["layer"],
                           "mean_energy": r.get("mean_energy"), "recovered_r": r.get("recovered_r"),
                           "linear_energy": r.get("linear_energy"), "behavior_readout": r.get("behavior_readout"),
                           "recipe_id": r.get("recipe_id"), "recipe_status": r.get("recipe_status")},
        })
    return out


def to_preference_records(graded: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Preference JSONL: chosen = a KEPT on-manifold record, rejected = the linear chord at the same
    waypoint. Only emitted where both exist — the geometry-gated provenance IS the preference signal."""
    rows = graded["all"]
    linear_by_wp = {r["waypoint_index"]: r for r in rows if r["path"] == "linear"}
    out = []
    for r in rows:
        if r["path"] not in ON_MANIFOLD_PATHS or not r["keep"]:
            continue
        lin = linear_by_wp.get(r["waypoint_index"])
        if lin is None or is_empty(lin.get("steered_text", "")):
            continue
        out.append({
            "prompt": r["prompt"],
            "chosen": r["steered_text"],
            "rejected": lin["steered_text"],
            "provenance": {"concept": r["concept"], "path": r["path"], "waypoint_index": r["waypoint_index"],
                           "value": r["value"], "chosen_energy": r.get("mean_energy"),
                           "linear_energy": lin.get("mean_energy"), "recovered_r": r.get("recovered_r")},
        })
    return out


def compile_payload(payload: dict[str, Any], spec: ManifoldDataSpec, gate: GateConfig | None = None) -> dict[str, Any]:
    """End-to-end (payload → graded records + metrics). Pure; no I/O."""
    gate = gate or GateConfig()
    records = records_from_payload(payload, spec)
    graded = gate_records(records, gate)
    graded["metrics"] = compute_metrics(graded, spec, gate)
    return graded


def _validation_banner(spec: ManifoldDataSpec) -> str:
    if spec.validated:
        return ("> **Source steer: `validated`.** On-manifold steering induced the target behavior at "
                "least as faithfully as the linear chord (energy ≤ linear). This does *not* guarantee a "
                "model trained on this data reproduces the behavior — only that the runtime steer was real.")
    return (f"> ⚠️ **Source steer: `{spec.recipe_status}` (NOT validated).** Compiled from a manifold steer "
            f"that did not clearly beat the linear chord on behavior-energy. Treat the dataset as exploratory.")


def render_dataset_card(spec: ManifoldDataSpec, metrics: dict[str, Any], gate: GateConfig) -> str:
    m = metrics
    arms = "\n".join(
        f"| `{p}` | {a['n_total']} | {a['n_kept']} | {a.get('mean_energy')} | {a.get('recovered_r')} |"
        for p, a in m["per_arm"].items()) or "| — | 0 | 0 | — | — |"
    return (
        f"# Dataset card: manifold-distilled `{spec.concept}` ({spec.source}→{spec.target})\n\n"
        f"{_validation_banner(spec)}\n\n"
        f"Compiled by the **manifold-to-data provenance compiler** (C09) from concept-manifold steering "
        f"payloads. Each kept record's steered output is the desired behavior; geometry provenance "
        f"(behavior-energy, recovered_r) gates what is kept.\n\n"
        f"## Source steer\n\n"
        f"| field | value |\n|---|---|\n"
        f"| source | `{spec.source_kind}` |\n| recipe_id | `{spec.recipe_id or '—'}` |\n"
        f"| recipe_status | `{spec.recipe_status}` |\n| model_id | `{spec.model_id or '—'}` |\n"
        f"| concept | `{spec.concept}` |\n| source→target | `{spec.source}`→`{spec.target}` |\n"
        f"| layer | {spec.layer} |\n| behavior_readout | `{spec.behavior_readout}` |\n\n"
        f"## Arms\n\n| arm | records | kept | mean_energy | recovered_r |\n|---|---|---|---|---|\n{arms}\n\n"
        f"Linear-chord energy reference: **{m.get('linear_energy')}**. Equal-size N per arm for a balanced "
        f"training comparison: **{m['equal_size_n_per_arm']}**.\n\n"
        f"## Contents\n\n"
        f"- `sft.jsonl` — {m['n_sft']} SFT records (on-manifold kept; provenance attached).\n"
        f"- `preference.jsonl` — {m['n_preference']} pairs (`chosen`=on-manifold, `rejected`=linear chord).\n"
        f"- `pairs_kept.jsonl` / `pairs_rejected.jsonl` / `pairs_all.jsonl` — full per-record provenance.\n\n"
        f"## Gate\n\n"
        f"On-manifold arms ({', '.join(ON_MANIFOLD_PATHS)}) kept iff: non-empty/non-collapsed text"
        + (", hook fired" if gate.require_hook_fired else "")
        + (f", energy ≤ linear + {gate.energy_margin}" if gate.require_beats_linear else "")
        + f", and recovered_r ≥ {gate.min_recovered_r} when present. Rejected records retained with reasons "
        f"({json.dumps(m['reject_reason_counts']) if m['reject_reason_counts'] else 'none'}).\n\n"
        f"## Honesty\n\n"
        f"- This is **training data**, not a trained model. First build step: **schema/export only** — no "
        f"claim that geometry-gated data trains a better model (C09's open question; see "
        f"`docs/experiments/MANIFOLD_TO_DATA_PROVENANCE.md`).\n"
        f"- SOURCE-GAP: novelty beyond Goodfire arXiv 2604.28119 / 2605.05115 is unverified.\n\n"
        f"_Generated {m['generated_at']} · schema {m['schema_version']}._\n"
    )


def render_report(spec: ManifoldDataSpec, graded: dict[str, Any]) -> str:
    m = graded["metrics"]
    reasons = "\n".join(f"- `{r}`: {c}" for r, c in m["reject_reason_counts"].items()) or "- None."
    arms = "\n".join(
        f"- `{p}`: {a['n_kept']}/{a['n_total']} kept · energy={a.get('mean_energy')} · recovered_r={a.get('recovered_r')}"
        for p, a in m["per_arm"].items()) or "- (no arms)"
    return (
        f"# Manifold-to-data provenance compiler report — `{spec.concept}`\n\n"
        f"{_validation_banner(spec)}\n\n"
        f"## Config\n\n"
        f"- concept **{spec.concept}** `{spec.source}`→`{spec.target}` at layer {spec.layer}, "
        f"readout `{spec.behavior_readout}`\n- source `{spec.source_kind}`"
        + (f" · recipe `{spec.recipe_id}` (`{spec.recipe_status}`)" if spec.recipe_id else "") + "\n\n"
        f"## Results\n\n"
        f"- records: **{m['n_records']}** · kept: **{m['n_kept']}** ({m['keep_rate']:.0%}) · "
        f"rejected: **{m['n_rejected']}**\n- linear-chord energy: **{m.get('linear_energy')}** · "
        f"equal-size N/arm: **{m['equal_size_n_per_arm']}**\n- exported: **{m['n_sft']}** SFT, "
        f"**{m['n_preference']}** preference pairs\n\n"
        f"### Arms\n\n{arms}\n\n### Reject reasons\n\n{reasons}\n\n"
        f"## Honesty & falsification\n\n"
        f"- First build step: **schema/export only**; no model is trained here.\n"
        f"- The C09 claim — geometry-gated manifold data beats equal-size linear-chord and prompt-only "
        f"data on held-out ordered-concept transfer — is **preregistered with a hard falsification gate** "
        f"in `docs/experiments/MANIFOLD_TO_DATA_PROVENANCE.md`.\n"
        f"- SOURCE-GAP: novelty beyond Goodfire arXiv 2604.28119 / 2605.05115 unverified.\n\n"
        f"_Generated {m['generated_at']} · schema {m['schema_version']}._\n"
    )


def write_outputs(out_dir: str | Path, spec: ManifoldDataSpec, graded: dict[str, Any], gate: GateConfig) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = graded["metrics"]
    paths = {
        "pairs_all": write_jsonl(out / "pairs_all.jsonl", graded["all"]),
        "pairs_kept": write_jsonl(out / "pairs_kept.jsonl", graded["kept"]),
        "pairs_rejected": write_jsonl(out / "pairs_rejected.jsonl", graded["rejected"]),
        "sft": write_jsonl(out / "sft.jsonl", to_sft_records(graded["kept"])),
        "preference": write_jsonl(out / "preference.jsonl", to_preference_records(graded)),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "dataset_card.md").write_text(render_dataset_card(spec, metrics, gate), encoding="utf-8")
    (out / "report.md").write_text(render_report(spec, graded), encoding="utf-8")
    paths["metrics"] = str(out / "metrics.json")
    paths["dataset_card"] = str(out / "dataset_card.md")
    paths["report"] = str(out / "report.md")
    return paths


# --------------------------------------------------------------------------------------
# 4. Synthetic fixtures — drive the whole compiler with no model (CI / smoke)
# --------------------------------------------------------------------------------------


def _leg(path: str, source: str, target: str, layer: int, mean_energy: float | None,
         recovered_r: float | None, texts: list[str], values: list[str]) -> dict[str, Any]:
    return {
        "path": path, "source": source, "target": target, "layer": layer,
        "mean_energy": mean_energy, "recovered_r": recovered_r, "hook_fired": True,
        "prompt": f"The {target} one is",
        "waypoints": [{"value": v, "text": t, "perplexity": 12.0, "hook_fired": True, "energy": mean_energy}
                      for v, t in zip(values, texts)],
    }


def build_synthetic_payload(scenario: str = "win") -> dict[str, Any]:
    """A deterministic manifold-comparison payload (manifold + linear + pullback legs) for tests/CLI.

    ``win``  — on-manifold beats the linear chord (lower energy) with high recovered_r → kept.
    ``fail`` — on-manifold energy worse than linear and low recovered_r → rejected on geometry
                (plus one collapsed text to exercise the text gate).
    """
    src, tgt, layer = "private", "general", 20
    vals = ["private", "corporal", "sergeant", "captain", "general"]
    man_texts = ["a private", "a corporal", "a sergeant", "a captain", "a general officer in command"]
    lin_texts = ["a private soldier", "a low rank", "a mid rank", "a higher rank", "a general"]
    pb_texts = ["a private", "a junior", "a senior", "an officer", "a commanding general"]
    if scenario == "fail":
        payload = {
            "concept": "rank", "source": src, "target": tgt, "layer": layer, "behavior_readout": "first_token",
            "manifold": _leg("manifold", src, tgt, layer, 0.42, None,
                             ["xx xx xx xx xx", *man_texts[1:]], vals),     # one collapsed text
            "linear": _leg("linear", src, tgt, layer, 0.18, None, lin_texts, vals),
            "pullback": _leg("pullback", src, tgt, layer, 0.40, 0.10, pb_texts, vals),
        }
        return payload
    # win
    return {
        "concept": "rank", "source": src, "target": tgt, "layer": layer, "behavior_readout": "first_token",
        "manifold": _leg("manifold", src, tgt, layer, 0.12, None, man_texts, vals),
        "linear": _leg("linear", src, tgt, layer, 0.21, None, lin_texts, vals),
        "pullback": _leg("pullback", src, tgt, layer, 0.10, 0.88, pb_texts, vals),
    }
