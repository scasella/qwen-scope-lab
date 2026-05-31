from __future__ import annotations

from typing import Any, TypedDict


class FeatureActivation(TypedDict):
    feature_id: int
    activation: float


class TokenFeatures(TypedDict):
    token_index: int
    token_text: str
    features: list[FeatureActivation]


JsonDict = dict[str, Any]
