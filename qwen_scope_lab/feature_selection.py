from __future__ import annotations

from typing import Any


def select_active_feature(inspection: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for row in inspection.get("top_features_by_token", []):
        for feature in row.get("features", []):
            candidate = {
                "feature_id": int(feature["feature_id"]),
                "activation": float(feature["activation"]),
                "token_index": int(row["token_index"]),
                "token_text": row["token_text"],
            }
            if best is None or candidate["activation"] > best["activation"]:
                best = candidate
    if best is None:
        raise ValueError("inspection has no active features to select")
    return best
