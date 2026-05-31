from __future__ import annotations

import math
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CandidateFeature:
    layer: int
    feature_id: int
    positive_score: float
    negative_score: float
    contrast: float
    ratio: float
    frequency_positive: float
    frequency_negative: float
    combined_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "feature_id": self.feature_id,
            "positive_score": self.positive_score,
            "negative_score": self.negative_score,
            "contrast": self.contrast,
            "ratio": self.ratio,
            "frequency_positive": self.frequency_positive,
            "frequency_negative": self.frequency_negative,
            "combined_score": self.combined_score,
        }


def _summarize_inspections(inspections: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    per_feature: dict[int, dict[str, float]] = defaultdict(lambda: {"max_sum": 0.0, "count": 0.0, "active_examples": 0.0})
    for inspection in inspections:
        active = set()
        max_by_feature: dict[int, float] = {}
        for row in inspection.get("top_features_by_token", []):
            for feature in row.get("features", []):
                feature_id = int(feature["feature_id"])
                activation = float(feature["activation"])
                max_by_feature[feature_id] = max(max_by_feature.get(feature_id, 0.0), activation)
        for feature_id, activation in max_by_feature.items():
            active.add(feature_id)
            per_feature[feature_id]["max_sum"] += activation
            per_feature[feature_id]["count"] += 1.0
        for feature_id in active:
            per_feature[feature_id]["active_examples"] += 1.0
    return per_feature


def rank_candidates_from_inspections(
    *,
    layer: int,
    positive_inspections: list[dict[str, Any]],
    negative_inspections: list[dict[str, Any]],
    limit: int = 10,
    epsilon: float = 1e-6,
) -> list[CandidateFeature]:
    positive = _summarize_inspections(positive_inspections)
    negative = _summarize_inspections(negative_inspections)
    feature_ids = set(positive) | set(negative)
    candidates = []
    pos_count = max(len(positive_inspections), 1)
    neg_count = max(len(negative_inspections), 1)
    for feature_id in feature_ids:
        pos = positive.get(feature_id, {"max_sum": 0.0, "count": 0.0, "active_examples": 0.0})
        neg = negative.get(feature_id, {"max_sum": 0.0, "count": 0.0, "active_examples": 0.0})
        positive_score = pos["max_sum"] / max(pos["count"], 1.0)
        negative_score = neg["max_sum"] / max(neg["count"], 1.0)
        contrast = positive_score - negative_score
        ratio = positive_score / (epsilon + negative_score)
        frequency_positive = pos["active_examples"] / pos_count
        frequency_negative = neg["active_examples"] / neg_count
        combined = contrast * math.log(1.0 + frequency_positive / (epsilon + frequency_negative))
        candidates.append(
            CandidateFeature(
                layer=layer,
                feature_id=feature_id,
                positive_score=positive_score,
                negative_score=negative_score,
                contrast=contrast,
                ratio=ratio,
                frequency_positive=frequency_positive,
                frequency_negative=frequency_negative,
                combined_score=combined,
            )
        )
    return sorted(candidates, key=lambda item: (-item.combined_score, item.feature_id))[:limit]


def fake_inspection(prompt: str, layer: int, positive: bool) -> dict[str, Any]:
    digest = hashlib.sha256(f"{layer}:{prompt}".encode("utf-8")).hexdigest()
    base = int(digest[:8], 16) % 1000
    feature = base if positive else base + 10_000
    shared = 42
    return {
        "layer": layer,
        "top_features_by_token": [
            {
                "token_index": 0,
                "token_text": prompt[:8],
                "features": [
                    {"feature_id": feature, "activation": 10.0 if positive else 8.0},
                    {"feature_id": shared, "activation": 3.0},
                ],
            }
        ],
    }
