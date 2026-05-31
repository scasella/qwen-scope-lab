from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"\S+", text or "")


def sentence_count(text: str) -> int:
    return len([part for part in re.split(r"[.!?]+", text or "") if part.strip()])


def distinct_ngram(text: str, n: int) -> float:
    tokens = tokenize_text(text)
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / max(len(ngrams), 1)


def repeated_ngram_rate(text: str, n: int = 3) -> float:
    tokens = tokenize_text(text)
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(len(ngrams), 1)


def json_validity(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except Exception as exc:
        return {"json_validity": 0.0, "json_error": str(exc)}
    return {"json_validity": 1.0, "json_error": "", "json_type": type(parsed).__name__}


def required_forbidden_terms(text: str, required_terms: list[str] | None = None, forbidden_terms: list[str] | None = None) -> dict[str, Any]:
    lower = (text or "").lower()
    required_terms = required_terms or []
    forbidden_terms = forbidden_terms or []
    required_hits = [term for term in required_terms if term.lower() in lower]
    forbidden_hits = [term for term in forbidden_terms if term.lower() in lower]
    return {
        "contains_required_terms": len(required_hits) == len(required_terms),
        "required_term_hits": required_hits,
        "excludes_forbidden_terms": not forbidden_hits,
        "forbidden_term_hits": forbidden_hits,
    }


def text_metrics(
    text: str,
    *,
    required_terms: list[str] | None = None,
    forbidden_terms: list[str] | None = None,
    max_length_chars: int | None = None,
    min_length_chars: int | None = None,
) -> dict[str, Any]:
    tokens = tokenize_text(text)
    repetition = repeated_ngram_rate(text)
    invalid_unicode = "\ufffd" in (text or "")
    metrics: dict[str, Any] = {
        "output_length_chars": len(text or ""),
        "output_length_tokens": len(tokens),
        "sentence_count": sentence_count(text),
        "repetition_score": repetition,
        "distinct_1": distinct_ngram(text, 1),
        "distinct_2": distinct_ngram(text, 2),
        "finish_reason": "length_or_stop",
        "generation_error": "",
        "empty_output": not bool((text or "").strip()),
        "truncated_output": False,
        "repeated_ngram_rate": repetition,
        "invalid_unicode": invalid_unicode,
        "excessive_repetition_flag": repetition > 0.25,
        "max_length_pass": True if max_length_chars is None else len(text or "") <= max_length_chars,
        "min_length_pass": True if min_length_chars is None else len(text or "") >= min_length_chars,
    }
    metrics.update(json_validity(text))
    metrics.update(required_forbidden_terms(text, required_terms, forbidden_terms))
    return metrics


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    aggregated: dict[str, Any] = {"count": len(rows)}
    for key in keys:
        values = [row[key] for row in rows if key in row]
        numeric = [float(value) for value in values if isinstance(value, (int, float, bool)) and not isinstance(value, str)]
        if numeric:
            aggregated[key] = sum(numeric) / len(numeric)
        elif values and all(isinstance(value, str) for value in values):
            aggregated[key] = values[0] if len(set(values)) == 1 else ""
    return aggregated


def score_for_objective(metrics: dict[str, Any], objective: str) -> float:
    objective = objective or "maximize_rule_score"
    if objective == "json_validity":
        return float(metrics.get("json_validity", 0.0))
    if objective == "maximize_json_validity":
        return float(metrics.get("json_validity", 0.0))
    if objective == "minimize_length_without_empty_output":
        if metrics.get("empty_output"):
            return -1_000_000.0
        return -float(metrics.get("output_length_chars", 0.0))
    if objective == "maximize_required_terms_without_forbidden_terms":
        return float(metrics.get("contains_required_terms", 0.0)) + float(metrics.get("excludes_forbidden_terms", 0.0))
    if objective == "maximize_rule_score":
        score = 0.0
        score += float(metrics.get("json_validity", 0.0))
        score += float(metrics.get("max_length_pass", 0.0))
        score += float(metrics.get("excludes_forbidden_terms", 0.0))
        score -= float(metrics.get("empty_output", 0.0))
        score -= float(metrics.get("excessive_repetition_flag", 0.0))
        return score
    return float(metrics.get(objective, 0.0)) if math.isfinite(float(metrics.get(objective, 0.0) or 0.0)) else 0.0
