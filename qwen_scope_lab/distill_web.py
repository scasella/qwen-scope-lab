"""Web-facing glue for the Mixture Dial Distill compiler.

The compiler in :mod:`qwen_scope_lab.experiments.mixture_dial_distill` is pure,
deterministic Python that turns candidate rows plus a mixture config into an SFT
corpus and a manifest. It only compiles training data — it never loads a model,
calls a service, trains, or evaluates an adapter. These helpers expose that same
compiler over the browser GUI: load candidates from pasted/uploaded text, summarize
the pool, and compile a mixture in-memory (no temp files) for the Distill mode.

Because nothing here touches the model, the Distill routes work identically on the
dev (CPU) and MLX backends.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .experiments.mixture_dial_distill import (
    CandidateLoad,
    MixtureSpec,
    build_manifest,
    build_mixture,
    normalize_candidate,
    normalize_mixture_spec,
    summarize_labels,
    to_sft_records,
)

# Examples discoverable from the GUI: the shipped fixtures plus the published v1.0 corpus.
# Paths are relative to the repo root (parent of this package).
_REPO_ROOT = Path(__file__).resolve().parent.parent

EXAMPLE_LIBRARY: list[dict[str, Any]] = [
    {
        "id": "truth_holding",
        "name": "Truth-holding A/B/C",
        "description": "The fixture from the write-up: A_factual (hold under false pressure), "
        "B_unknowable + C_subjective (calibration). Small, hand-built.",
        "candidates": "examples/mixture_dial/truth_holding_candidates.jsonl",
        "mixture": "examples/mixture_dial/truth_holding_mixture.yml",
    },
    {
        "id": "generic",
        "name": "Generic support / format / boundary",
        "description": "A user-defined fixture showing custom labels and a "
        "quality_label→rejection_mode alias with count + ratio slots.",
        "candidates": "examples/mixture_dial/generic_candidates.jsonl",
        "mixture": "examples/mixture_dial/generic_mixture.yml",
    },
    {
        "id": "v10_publication",
        "name": "v1.0 publication corpus (377 rows)",
        "description": "The full kept corpus behind the polite truth-holding write-up — "
        "real teacher-distilled rows across A/B/C classes and many domains.",
        "candidates": "reports/steering_distill/th_v10_publication/v10_kept_combined.jsonl",
        "mixture": None,
    },
]


def list_examples() -> dict[str, Any]:
    """Catalogue of one-click example candidates/mixtures, marking which files exist on disk."""
    out = []
    for ex in EXAMPLE_LIBRARY:
        cand = _REPO_ROOT / ex["candidates"]
        entry = {
            "id": ex["id"],
            "name": ex["name"],
            "description": ex["description"],
            "available": cand.is_file(),
            "has_mixture": bool(ex["mixture"]) and (_REPO_ROOT / ex["mixture"]).is_file(),
        }
        out.append(entry)
    return {"examples": out}


def load_example(example_id: str) -> dict[str, Any]:
    """Return the raw candidate JSONL text (and mixture YAML text, if any) for an example id."""
    ex = next((e for e in EXAMPLE_LIBRARY if e["id"] == example_id), None)
    if ex is None:
        raise ValueError(f"unknown example {example_id!r}; choose from {[e['id'] for e in EXAMPLE_LIBRARY]}")
    cand = _REPO_ROOT / ex["candidates"]
    if not cand.is_file():
        raise ValueError(f"example {example_id!r} candidate file is not available: {ex['candidates']}")
    result: dict[str, Any] = {
        "id": ex["id"],
        "name": ex["name"],
        "candidates_text": cand.read_text(encoding="utf-8"),
        "mixture_text": None,
    }
    if ex["mixture"]:
        mix = _REPO_ROOT / ex["mixture"]
        if mix.is_file():
            result["mixture_text"] = mix.read_text(encoding="utf-8")
    return result


def load_candidates_text(text: str, spec: MixtureSpec) -> CandidateLoad:
    """In-memory twin of :func:`mixture_dial_distill.load_candidates` — parse JSONL text
    (rather than a file path) so the GUI can compile pasted/uploaded candidates without temp files."""
    valid: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    n_lines = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        n_lines += 1
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            skipped.append({"line_number": line_number, "reason": "malformed_json",
                            "error": str(exc), "raw_line": line})
            continue
        normalized, skip = normalize_candidate(raw, line_number=line_number, spec=spec)
        if skip:
            skipped.append(skip)
            continue
        assert normalized is not None
        modes = {str(v) for v in _as_list((normalized.get("labels") or {}).get("rejection_mode"))}
        if spec.exclude_rejection_modes and (modes & set(spec.exclude_rejection_modes)):
            skipped.append({"line_number": line_number, "source_id": normalized["source_id"],
                            "reason": "excluded_rejection_mode", "labels": normalized["labels"], "raw": raw})
            continue
        valid.append(normalized)
    return CandidateLoad(valid=valid, skipped=skipped, n_lines=n_lines)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


# A minimal default mixture used only to *summarize* a freshly loaded candidate pool
# (one catch-all slot so every dimension is read; the GUI then proposes real slots).
def summarize_only_spec() -> MixtureSpec:
    return normalize_mixture_spec({
        "schema_version": "0.1.0",
        "seed": 0,
        "total": 0,
        "slots": [{"name": "_all", "where": {}, "count": 0}],
    })


def summarize_candidates(candidates_text: str) -> dict[str, Any]:
    """Counts by class/pressure/domain/rejection_mode over a candidate pool, for the dial preview."""
    spec = summarize_only_spec()
    loaded = load_candidates_text(candidates_text, spec)
    return {
        "candidate_rows": {
            "jsonl_lines": loaded.n_lines,
            "valid": len(loaded.valid),
            "skipped": len(loaded.skipped),
        },
        "label_summaries": summarize_labels(loaded.valid, list(spec.dimensions)),
        "skipped_reasons": _count_reasons(loaded.skipped),
    }


def _count_reasons(skipped: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in skipped:
        reason = str(row.get("reason", "unknown"))
        out[reason] = out.get(reason, 0) + 1
    return dict(sorted(out.items()))


def compile_mixture(candidates_text: str, mixture: dict[str, Any], *,
                    sample_limit: int = 6) -> dict[str, Any]:
    """Compile a mixture against pasted candidates and return everything the GUI renders:
    the manifest, achieved-vs-requested slot rows, a handful of sample SFT records, and the
    full artifacts (sft.jsonl rows + manifest) for download. No files are written."""
    if not isinstance(mixture, dict):
        raise ValueError("mixture config must be a JSON/YAML mapping")
    spec = normalize_mixture_spec(mixture)
    loaded = load_candidates_text(candidates_text, spec)
    if not loaded.valid:
        raise ValueError("no valid candidate rows found — paste JSONL with prompt/output or chat messages")
    built = build_mixture(spec, loaded.valid)
    sft = to_sft_records(built["rows"], preserve_labels=spec.preserve_labels)
    manifest = build_manifest(
        spec=spec,
        mixture_path="<web:inline>",
        candidates_path="<web:inline>",
        loaded=loaded,
        built=built,
        output_paths={"sft": "sft.jsonl", "manifest": "mixture_manifest.json"},
    )
    samples = [_sample_record(rec) for rec in sft[:sample_limit]]
    return {
        "manifest": manifest,
        "n_sft": len(sft),
        "slot_details": built["slot_details"],
        "requested_counts": built["requested_counts"],
        "achieved_counts": built["achieved_counts"],
        "capped_slots": manifest["capped_slots"],
        "underfilled_slots": manifest["underfilled_slots"],
        "label_summaries": manifest["label_summaries"],
        "samples": samples,
        "artifacts": {
            "sft_jsonl": "\n".join(json.dumps(rec, ensure_ascii=False) for rec in sft),
            "manifest_json": json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False),
        },
    }


def _sample_record(rec: dict[str, Any]) -> dict[str, Any]:
    """A compact, render-friendly view of one SFT record (chat turns + its labels/slot)."""
    return {
        "messages": rec.get("messages", []),
        "mixture_slot": rec.get("mixture_slot", ""),
        "source_id": rec.get("source_id", ""),
        "labels": rec.get("labels", {}),
    }
