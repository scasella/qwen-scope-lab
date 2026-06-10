"""Offline mixture-first SFT corpus compiler.

Mixture Dial Distill takes a candidate JSONL file plus a human-authored mixture
config and exports a train-ready chat SFT JSONL. It does not load models, call a
service, train adapters, or use runtime steering hooks.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .steering_distill import write_jsonl

SCHEMA_VERSION = "0.1.0"

DEFAULT_DIMENSIONS = ("class", "pressure", "domain", "rejection_mode")

DEFAULT_ALIASES: dict[str, tuple[str, ...]] = {
    "class": ("class", "behavioral_class", "class_label", "label", "category"),
    "pressure": ("pressure", "pressure_type", "pressure_label", "user_pressure"),
    "domain": ("domain", "topic", "subject_domain"),
    "rejection_mode": (
        "rejection_mode",
        "reject_reason",
        "reject_reasons",
        "rejection_reason",
        "rejection_reasons",
        "quality_label",
        "drop_reason",
        "skip_reason",
    ),
}


@dataclass(frozen=True)
class SlotSpec:
    name: str
    where: dict[str, Any]
    count: int | None = None
    ratio: float | None = None

    def target_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name, "where": self.where}
        if self.count is not None:
            data["count"] = self.count
        if self.ratio is not None:
            data["ratio"] = self.ratio
        return data


@dataclass(frozen=True)
class MixtureSpec:
    schema_version: str
    seed: int
    total: int
    dimensions: list[str]
    slots: list[SlotSpec]
    preserve_labels: bool = True
    output_format: str = "sft_chat"
    aliases: dict[str, list[str]] = field(default_factory=dict)
    exclude_rejection_modes: list[str] = field(default_factory=list)


@dataclass
class CandidateLoad:
    valid: list[dict[str, Any]]
    skipped: list[dict[str, Any]]
    n_lines: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_mixture_spec(path: str | Path) -> MixtureSpec:
    raw = _load_mapping(path)
    if not isinstance(raw, dict):
        raise ValueError(f"mixture config must be a mapping, got {type(raw).__name__}")
    return normalize_mixture_spec(raw)


def normalize_mixture_spec(raw: dict[str, Any]) -> MixtureSpec:
    alias_map = _merge_aliases(raw.get("aliases") or raw.get("label_aliases") or {})
    dimensions = [_canonical_dim(str(d), alias_map) for d in raw.get("dimensions") or DEFAULT_DIMENSIONS]

    slots_raw = raw.get("slots")
    if not isinstance(slots_raw, list) or not slots_raw:
        raise ValueError("mixture config requires a non-empty slots list")
    slots = [_normalize_slot(i, item, alias_map) for i, item in enumerate(slots_raw)]

    for slot in slots:
        for dim in slot.where:
            if dim not in dimensions:
                dimensions.append(dim)

    explicit_total = raw.get("total")
    if explicit_total is None:
        if any(slot.ratio is not None for slot in slots):
            raise ValueError("mixture config with ratio slots requires total")
        total = sum(int(slot.count or 0) for slot in slots)
    else:
        total = int(explicit_total)
    if total < 0:
        raise ValueError("total must be non-negative")

    output = raw.get("output") or {}
    if not isinstance(output, dict):
        raise ValueError("output must be a mapping when provided")
    output_format = str(output.get("format") or "sft_chat")
    if output_format != "sft_chat":
        raise ValueError(f"unsupported output.format {output_format!r}; only sft_chat is supported")

    excluded = raw.get("exclude_rejection_modes")
    if excluded is None and isinstance(raw.get("reject"), dict):
        excluded = raw["reject"].get("exclude_modes")

    return MixtureSpec(
        schema_version=str(raw.get("schema_version") or SCHEMA_VERSION),
        seed=int(raw.get("seed", 0)),
        total=total,
        dimensions=_unique(dimensions),
        slots=slots,
        preserve_labels=bool(output.get("preserve_labels", True)),
        output_format=output_format,
        aliases=alias_map,
        exclude_rejection_modes=[str(v) for v in _as_list(excluded)],
    )


def requested_counts_for_slots(slots: list[SlotSpec], total: int) -> dict[str, int]:
    fixed = {slot.name: int(slot.count) for slot in slots if slot.count is not None}
    fixed_total = sum(fixed.values())
    if fixed_total > total:
        raise ValueError(f"count slots request {fixed_total}, exceeding total {total}")

    ratio_slots = [slot for slot in slots if slot.ratio is not None]
    if not ratio_slots:
        if fixed_total != total:
            raise ValueError(f"count slots request {fixed_total}, but total is {total}")
        return fixed

    budget = total - fixed_total
    weights = {slot.name: float(slot.ratio or 0.0) for slot in ratio_slots}
    if any(v <= 0 for v in weights.values()):
        raise ValueError("ratio slots must have positive ratios")
    ratio_counts = _largest_remainder_counts(weights, budget)
    return {slot.name: fixed.get(slot.name, ratio_counts.get(slot.name, 0)) for slot in slots}


def load_candidates(path: str | Path, spec: MixtureSpec) -> CandidateLoad:
    valid: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    p = Path(path)
    n_lines = 0
    for line_number, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        n_lines += 1
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            skipped.append(
                {
                    "line_number": line_number,
                    "reason": "malformed_json",
                    "error": str(exc),
                    "raw_line": line,
                }
            )
            continue
        normalized, skip = normalize_candidate(raw, line_number=line_number, spec=spec)
        if skip:
            skipped.append(skip)
            continue
        assert normalized is not None
        if _excluded_by_rejection_mode(normalized, spec.exclude_rejection_modes):
            skipped.append(
                {
                    "line_number": line_number,
                    "source_id": normalized["source_id"],
                    "reason": "excluded_rejection_mode",
                    "labels": normalized["labels"],
                    "raw": raw,
                }
            )
            continue
        valid.append(normalized)
    return CandidateLoad(valid=valid, skipped=skipped, n_lines=n_lines)


def normalize_candidate(raw: Any, *, line_number: int, spec: MixtureSpec) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not isinstance(raw, dict):
        return None, {"line_number": line_number, "reason": "malformed_row", "error": "JSON value is not an object", "raw": raw}

    messages = _extract_messages(raw)
    if not messages:
        return None, {
            "line_number": line_number,
            "source_id": _source_id(raw, line_number),
            "reason": "malformed_row",
            "error": "row needs chat messages or prompt/question plus output/response",
            "raw": raw,
        }

    labels = _extract_labels(raw, spec)
    prompt = _last_content(messages, "user")
    output = _last_content(messages, "assistant")
    return (
        {
            "row_key": f"line:{line_number}",
            "line_number": line_number,
            "source_id": _source_id(raw, line_number),
            "prompt": prompt,
            "output": output,
            "messages": messages,
            "labels": labels,
            "metadata": raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
            "raw": raw,
        },
        None,
    )


def build_mixture(spec: MixtureSpec, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    requested = requested_counts_for_slots(spec.slots, spec.total)
    rng = random.Random(spec.seed)
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    slot_details: list[dict[str, Any]] = []

    for slot in spec.slots:
        pool = [row for row in candidates if row["row_key"] not in selected_keys and matches_slot(row, slot)]
        shuffled = list(pool)
        rng.shuffle(shuffled)
        need = requested[slot.name]
        taken = shuffled[: min(need, len(shuffled))]
        for row in taken:
            selected_keys.add(row["row_key"])
            selected.append({**row, "mixture_slot": slot.name})
        detail = {
            "name": slot.name,
            "where": slot.where,
            "requested": need,
            "available": len(pool),
            "achieved": len(taken),
            "capped": len(taken) < need,
            "underfilled_by": max(0, need - len(taken)),
        }
        if slot.count is not None:
            detail["count"] = slot.count
        if slot.ratio is not None:
            detail["ratio"] = slot.ratio
        slot_details.append(detail)

    rng.shuffle(selected)
    achieved = Counter(row["mixture_slot"] for row in selected)
    return {
        "rows": selected,
        "requested_counts": requested,
        "achieved_counts": {slot.name: int(achieved.get(slot.name, 0)) for slot in spec.slots},
        "slot_details": slot_details,
    }


def matches_slot(row: dict[str, Any], slot: SlotSpec) -> bool:
    labels = row.get("labels") or {}
    raw = row.get("raw") or {}
    for dim, desired in slot.where.items():
        actual = labels.get(dim)
        if actual is None and dim in raw:
            actual = raw.get(dim)
        if not _value_matches(actual, desired):
            return False
    return True


def to_sft_records(rows: list[dict[str, Any]], *, preserve_labels: bool = True) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        rec: dict[str, Any] = {"messages": row["messages"]}
        if preserve_labels:
            labels = {k: v for k, v in (row.get("labels") or {}).items() if v not in (None, [], "")}
            rec["source_id"] = row.get("source_id", "")
            rec["mixture_slot"] = row.get("mixture_slot", "")
            rec["labels"] = labels
            if labels.get("class") is not None:
                rec["class"] = labels["class"]
                rec["behavioral_class"] = labels["class"]
            if labels.get("pressure") is not None:
                rec["pressure"] = labels["pressure"]
                rec["pressure_type"] = labels["pressure"]
            if labels.get("domain") is not None:
                rec["domain"] = labels["domain"]
            if labels.get("rejection_mode") is not None:
                rec["rejection_mode"] = labels["rejection_mode"]
            if row.get("metadata"):
                rec["metadata"] = row["metadata"]
        records.append(rec)
    return records


def compile_to_dir(mixture: str | Path, candidates: str | Path, out: str | Path) -> dict[str, Any]:
    spec = load_mixture_spec(mixture)
    loaded = load_candidates(candidates, spec)
    built = build_mixture(spec, loaded.valid)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sft_path = out_dir / "sft.jsonl"
    manifest_path = out_dir / "mixture_manifest.json"
    skipped_path = out_dir / "skipped.jsonl"

    sft = to_sft_records(built["rows"], preserve_labels=spec.preserve_labels)
    write_jsonl(sft_path, sft)
    output_paths = {"sft": str(sft_path), "manifest": str(manifest_path)}
    if loaded.skipped:
        write_jsonl(skipped_path, loaded.skipped)
        output_paths["skipped"] = str(skipped_path)

    manifest = build_manifest(
        spec=spec,
        mixture_path=mixture,
        candidates_path=candidates,
        loaded=loaded,
        built=built,
        output_paths=output_paths,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "out": str(out_dir),
        "n_sft": len(sft),
        "paths": output_paths,
        "requested_counts": built["requested_counts"],
        "achieved_counts": built["achieved_counts"],
        "underfilled_slots": manifest["underfilled_slots"],
    }


def build_manifest(
    *,
    spec: MixtureSpec,
    mixture_path: str | Path,
    candidates_path: str | Path,
    loaded: CandidateLoad,
    built: dict[str, Any],
    output_paths: dict[str, str],
) -> dict[str, Any]:
    rows = built["rows"]
    selected_keys = {row["row_key"] for row in rows}
    matching_any = {row["row_key"] for row in loaded.valid if any(matches_slot(row, slot) for slot in spec.slots)}
    valid_not_selected = len([row for row in loaded.valid if row["row_key"] not in selected_keys])
    skipped_reasons = Counter(row.get("reason", "unknown") for row in loaded.skipped)
    capped = {
        d["name"]: {"requested": d["requested"], "available": d["available"]}
        for d in built["slot_details"]
        if d["capped"]
    }
    underfilled = {
        d["name"]: {"requested": d["requested"], "achieved": d["achieved"], "underfilled_by": d["underfilled_by"]}
        for d in built["slot_details"]
        if d["underfilled_by"]
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "source_paths": {"mixture": str(mixture_path), "candidates": str(candidates_path)},
        "seed": spec.seed,
        "target_total": spec.total,
        "dimensions": spec.dimensions,
        "target_slots": [slot.target_dict() for slot in spec.slots],
        "requested_counts": built["requested_counts"],
        "achieved_counts": built["achieved_counts"],
        "achieved_total": len(rows),
        "capped_slots": capped,
        "underfilled_slots": underfilled,
        "dropped_rows": {
            "total": len(loaded.skipped) + valid_not_selected,
            "malformed": skipped_reasons.get("malformed_json", 0) + skipped_reasons.get("malformed_row", 0),
            "excluded_by_rejection_mode": skipped_reasons.get("excluded_rejection_mode", 0),
            "valid_not_selected": valid_not_selected,
            "valid_not_matching_any_slot": len([row for row in loaded.valid if row["row_key"] not in matching_any]),
        },
        "candidate_rows": {
            "jsonl_lines": loaded.n_lines,
            "valid": len(loaded.valid),
            "skipped": len(loaded.skipped),
        },
        "slot_details": built["slot_details"],
        "output": {"format": spec.output_format, "preserve_labels": spec.preserve_labels},
        "output_paths": output_paths,
        "label_summaries": {
            "valid_candidates": summarize_labels(loaded.valid, spec.dimensions),
            "selected": summarize_labels(rows, spec.dimensions),
            "skipped": summarize_skipped_labels(loaded.skipped, spec.dimensions),
        },
    }


def summarize_labels(rows: list[dict[str, Any]], dimensions: list[str]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for dim in dimensions:
        counts: Counter[str] = Counter()
        for row in rows:
            value = (row.get("labels") or {}).get(dim)
            vals = _as_list(value) or ["<missing>"]
            for item in vals:
                counts[str(item)] += 1
        out[dim] = dict(sorted(counts.items()))
    return out


def summarize_skipped_labels(rows: list[dict[str, Any]], dimensions: list[str]) -> dict[str, dict[str, int]]:
    normalized = [{"labels": row.get("labels") or {}} for row in rows]
    return summarize_labels(normalized, dimensions)


def _load_mapping(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError:
        return json.loads(text)
    data = yaml.safe_load(text)
    return data or {}


def _normalize_slot(index: int, raw: Any, alias_map: dict[str, list[str]]) -> SlotSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"slot #{index + 1} must be a mapping")
    name = str(raw.get("name") or f"slot_{index + 1:03d}")
    where_raw = raw.get("where") or {}
    if not isinstance(where_raw, dict):
        raise ValueError(f"slot {name!r} where must be a mapping")
    where = {_canonical_dim(str(k), alias_map): v for k, v in where_raw.items()}
    has_count = raw.get("count") is not None
    has_ratio = raw.get("ratio") is not None
    if has_count == has_ratio:
        raise ValueError(f"slot {name!r} must set exactly one of count or ratio")
    count = int(raw["count"]) if has_count else None
    if count is not None and count < 0:
        raise ValueError(f"slot {name!r} count must be non-negative")
    ratio = parse_ratio_value(raw["ratio"]) if has_ratio else None
    return SlotSpec(name=name, where=where, count=count, ratio=ratio)


def parse_ratio_value(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("ratio cannot be a boolean")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("%"):
            return float(text[:-1]) / 100.0
        if "/" in text:
            num, den = text.split("/", 1)
            return float(num) / float(den)
        return float(text)
    raise ValueError(f"unsupported ratio value {value!r}")


def _largest_remainder_counts(weights: dict[str, float], total: int) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if total == 0:
        return {name: 0 for name in weights}
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("ratio weights must sum above zero")
    raw = {name: (weight / weight_sum) * total for name, weight in weights.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(weights, key=lambda name: (raw[name] - counts[name], -list(weights).index(name)), reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def _merge_aliases(extra: Any) -> dict[str, list[str]]:
    out = {key: list(values) for key, values in DEFAULT_ALIASES.items()}
    if not extra:
        return out
    if not isinstance(extra, dict):
        raise ValueError("aliases must be a mapping")
    for raw_dim, raw_aliases in extra.items():
        dim = _canonical_dim(str(raw_dim), out)
        merged = out.setdefault(dim, [dim])
        for alias in _as_list(raw_aliases):
            text = str(alias)
            if text not in merged:
                merged.append(text)
    return out


def _canonical_dim(key: str, alias_map: dict[str, list[str]]) -> str:
    for dim, aliases in alias_map.items():
        if key == dim or key in aliases:
            return dim
    return key


def _extract_labels(raw: dict[str, Any], spec: MixtureSpec) -> dict[str, Any]:
    labels: dict[str, Any] = {}
    for dim in spec.dimensions:
        aliases = spec.aliases.get(dim) or [dim]
        labels[dim] = _field_value(raw, aliases, combine=(dim == "rejection_mode"))
    return labels


def _field_value(raw: dict[str, Any], aliases: list[str], *, combine: bool = False) -> Any:
    values: list[Any] = []
    for key in aliases:
        if key not in raw:
            continue
        value = raw.get(key)
        if value in (None, ""):
            continue
        if combine:
            values.extend(_as_list(value))
        else:
            vals = _as_list(value)
            return vals[0] if vals else None
    if combine:
        return _unique([str(v) for v in values if v not in (None, "")])
    return None


def _extract_messages(raw: dict[str, Any]) -> list[dict[str, str]]:
    messages = raw.get("messages")
    if isinstance(messages, list):
        cleaned = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if role and content:
                cleaned.append({"role": role, "content": content})
        assistant = _first_text(raw, ("output", "assistant", "response", "completion", "answer"))
        if cleaned and any(m["role"] == "user" for m in cleaned) and assistant:
            return cleaned + [{"role": "assistant", "content": assistant}]
        if cleaned and any(m["role"] == "user" for m in cleaned) and any(m["role"] == "assistant" for m in cleaned):
            if cleaned[-1]["role"] == "assistant":
                return cleaned
            last_assistant = max(i for i, msg in enumerate(cleaned) if msg["role"] == "assistant")
            return cleaned[: last_assistant + 1]

    prompt = _first_text(raw, ("prompt", "question", "instruction", "input"))
    if prompt and "prompt" not in raw and raw.get("false_claim"):
        prompt = f"{prompt}\n\nUser pressure: {raw['false_claim']}"
    assistant = _first_text(raw, ("output", "assistant", "response", "completion", "answer"))
    if not prompt or not assistant:
        return []
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": assistant}]


def _first_text(raw: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _last_content(messages: list[dict[str, str]], role: str) -> str:
    for msg in reversed(messages):
        if msg.get("role") == role:
            return msg.get("content", "")
    return ""


def _source_id(raw: dict[str, Any], line_number: int) -> str:
    for key in ("id", "scenario_id", "source_id"):
        value = raw.get(key)
        if value not in (None, ""):
            return str(value)
    return f"line_{line_number:06d}"


def _excluded_by_rejection_mode(row: dict[str, Any], excluded: list[str]) -> bool:
    if not excluded:
        return False
    modes = {str(v) for v in _as_list((row.get("labels") or {}).get("rejection_mode"))}
    return bool(modes & set(excluded))


def _value_matches(actual: Any, desired: Any) -> bool:
    actual_values = _as_list(actual)
    desired_values = _as_list(desired)
    if not desired_values:
        return True
    if not actual_values:
        return False
    return any(str(a) == str(d) for a in actual_values for d in desired_values)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _unique(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


__all__ = [
    "SCHEMA_VERSION",
    "SlotSpec",
    "MixtureSpec",
    "CandidateLoad",
    "load_mixture_spec",
    "normalize_mixture_spec",
    "requested_counts_for_slots",
    "load_candidates",
    "normalize_candidate",
    "build_mixture",
    "matches_slot",
    "to_sft_records",
    "compile_to_dir",
    "build_manifest",
    "summarize_labels",
    "parse_ratio_value",
]
