from __future__ import annotations

from typing import Any


def summarize_prompt_features(feature_json: dict[str, Any]) -> dict[int, dict[str, Any]]:
    summary: dict[int, dict[str, Any]] = {}
    for row in feature_json["top_features_by_token"]:
        token = row["token_text"]
        for feature in row["features"]:
            feature_id = int(feature["feature_id"])
            activation = float(feature["activation"])
            entry = summary.setdefault(feature_id, {"max": float("-inf"), "mean_sum": 0.0, "count": 0, "tokens": []})
            entry["max"] = max(entry["max"], activation)
            entry["mean_sum"] += activation
            entry["count"] += 1
            if len(entry["tokens"]) < 5:
                entry["tokens"].append(token)
    for entry in summary.values():
        entry["mean"] = entry["mean_sum"] / max(entry["count"], 1)
    return summary


def token_activation_summary(feature_json: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in feature_json["top_features_by_token"]:
        features = row["features"]
        activations = [float(feature["activation"]) for feature in features]
        rows.append(
            {
                "token_index": row["token_index"],
                "token_text": row["token_text"],
                "max_activation": max(activations) if activations else 0.0,
                "mean_top_activation": sum(activations) / len(activations) if activations else 0.0,
                "top_feature_ids": [int(feature["feature_id"]) for feature in features[:8]],
            }
        )
    return rows


def contrast_features(positive: dict[str, Any], negative: dict[str, Any], limit: int) -> dict[str, Any]:
    pos = summarize_prompt_features(positive)
    neg = summarize_prompt_features(negative)
    feature_ids = set(pos) | set(neg)
    rows = []
    for feature_id in feature_ids:
        p = pos.get(feature_id, {"max": 0.0, "mean": 0.0, "tokens": []})
        n = neg.get(feature_id, {"max": 0.0, "mean": 0.0, "tokens": []})
        pos_max = float(p["max"])
        neg_max = float(n["max"])
        diff = pos_max - neg_max
        ratio = (pos_max + 1e-6) / (neg_max + 1e-6)
        rows.append(
            {
                "feature_id": feature_id,
                "positive_max": pos_max,
                "negative_max": neg_max,
                "difference": diff,
                "ratio": ratio,
                "positive_tokens": p.get("tokens", []),
                "negative_tokens": n.get("tokens", []),
            }
        )
    positive_rows = sorted(rows, key=lambda row: row["difference"], reverse=True)[:limit]
    negative_rows = sorted(rows, key=lambda row: row["difference"])[:limit]
    return {
        "method": "max activation per prompt; difference = positive_max - negative_max; ratio = (positive_max+1e-6)/(negative_max+1e-6)",
        "positive_stronger": positive_rows,
        "negative_stronger": negative_rows,
        "positive_token_summary": token_activation_summary(positive),
        "negative_token_summary": token_activation_summary(negative),
    }
