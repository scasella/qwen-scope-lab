"""Feature-as-monitor: discover an SAE-feature detector for a behavior and evaluate it honestly.

The detection counterpart to steering. Given labeled positive/negative example activations, pick
the SAE feature(s) whose activation best separates them, combine by max-activation (so a
heterogeneous behavior — e.g. PII across emails/SSNs/cards — can use an OR of subtype features),
and report **held-out** metrics plus a **random-feature control** as the validity gate. This
module is pure/GPU-free: it operates on activation maps produced by ``service.inspect_prompt``.
"""
from __future__ import annotations

import random
import statistics as st
from typing import Any


def activation_map(inspection: dict[str, Any]) -> dict[int, float]:
    """Per-feature max activation across an inspected prompt's tokens (0 if a feature never fires)."""
    out: dict[int, float] = {}
    for row in inspection.get("top_features_by_token", []):
        for f in row.get("features", []):
            fid = int(f["feature_id"])
            a = float(f["activation"])
            if a > out.get(fid, 0.0):
                out[fid] = a
    return out


def auc(pos: list[float], neg: list[float]) -> float:
    """Probability a random positive scores above a random negative (ties count half)."""
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _combined(features: list[int], amap: dict[int, float]) -> float:
    return max((amap.get(f, 0.0) for f in features), default=0.0)


def score(features: list[int], threshold: float, amap: dict[int, float]) -> dict[str, Any]:
    s = _combined(features, amap)
    return {"score": round(s, 4), "fires": bool(s >= threshold)}


def _metrics_at(features: list[int], thr: float, pos_maps: list[dict], neg_maps: list[dict]) -> dict[str, float]:
    ps = [_combined(features, m) for m in pos_maps]
    ns = [_combined(features, m) for m in neg_maps]
    tp = sum(p >= thr for p in ps)
    fn = len(ps) - tp
    fp = sum(n >= thr for n in ns)
    tn = len(ns) - fp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / (len(ps) + len(ns)) if (ps or ns) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"auc": round(auc(ps, ns), 4), "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "accuracy": round(acc, 4), "fpr": round(fpr, 4)}


def _best_threshold(features: list[int], pos_maps: list[dict], neg_maps: list[dict]) -> float:
    """Threshold on the combined score that maximizes F1 on the given (train) set."""
    vals = sorted({_combined(features, m) for m in pos_maps + neg_maps})
    cands = [0.0]
    for i, v in enumerate(vals):
        cands.append(v)
        if i + 1 < len(vals):
            cands.append((v + vals[i + 1]) / 2)
    best, best_f1 = 0.0, -1.0
    for t in cands:
        f1 = _metrics_at(features, t, pos_maps, neg_maps)["f1"]
        if f1 > best_f1:
            best_f1, best = f1, t
    return best


def discover(pos_maps: list[dict], neg_maps: list[dict], top_k: int = 3,
             fire_thr: float = 0.5, d_sae: int | None = None, seed: int = 0) -> dict[str, Any]:
    """Select the top-k differential features (on a train split), choose a threshold, and report
    held-out metrics + a random-feature control. Verdict is `validated` only if the detector clears
    a strict gate AND beats the random control — so a no-op lands `benchmarked`."""
    if not pos_maps or not neg_maps:
        raise ValueError("monitor discovery needs both positive and negative examples")
    rng = random.Random(seed)
    all_maps = pos_maps + neg_maps
    counts: dict[int, int] = {}
    for m in all_maps:
        for fid in m:
            counts[fid] = counts.get(fid, 0) + 1
    feats = [f for f, c in counts.items() if c >= min(3, len(all_maps))] or list(counts)

    # interleaved train/test split for honest held-out metrics (fall back to train on tiny sets)
    tr_p, te_p = pos_maps[0::2], pos_maps[1::2] or pos_maps[0::2]
    tr_n, te_n = neg_maps[0::2], neg_maps[1::2] or neg_maps[0::2]

    def diff(fid: int) -> float:
        mp = st.mean([m.get(fid, 0.0) for m in tr_p]) if tr_p else 0.0
        mn = st.mean([m.get(fid, 0.0) for m in tr_n]) if tr_n else 0.0
        return mp - mn

    selected = sorted(feats, key=lambda f: -diff(f))[:max(1, top_k)]
    threshold = _best_threshold(selected, tr_p, tr_n)
    test_m = _metrics_at(selected, threshold, te_p, te_n)
    full_m = _metrics_at(selected, threshold, pos_maps, neg_maps)
    train_auc = _metrics_at(selected, threshold, tr_p, tr_n)["auc"]

    # random-feature control: same k random ids, fit + eval the same way; averaged over trials
    universe = list(range(d_sae)) if d_sae else list(counts)
    ctrl = []
    for _ in range(5):
        rand = rng.sample(universe, min(top_k, len(universe))) if universe else []
        rt = _best_threshold(rand, tr_p, tr_n)
        ctrl.append(_metrics_at(rand, rt, te_p, te_n)["auc"])
    control_auc = round(sum(ctrl) / len(ctrl), 4) if ctrl else float("nan")

    per_feature = []
    for fid in selected:
        ps = [m.get(fid, 0.0) for m in pos_maps]
        ns = [m.get(fid, 0.0) for m in neg_maps]
        per_feature.append({"feature_id": fid, "auc": round(auc(ps, ns), 4),
                            "fires_pos": int(sum(p >= fire_thr for p in ps)), "n_pos": len(ps),
                            "fires_neg": int(sum(n >= fire_thr for n in ns)), "n_neg": len(ns)})

    test_auc, f1 = test_m["auc"], test_m["f1"]
    passed = (test_auc >= 0.8) and (f1 >= 0.7) and (test_auc >= control_auc + 0.15)
    decision = {
        "status": "validated" if passed else "benchmarked",
        "passed": bool(passed),
        "reason": (f"held-out AUC {test_auc:.2f}, F1 {f1:.2f}; beats the random-feature control ({control_auc:.2f})."
                   if passed else
                   f"held-out AUC {test_auc:.2f}, F1 {f1:.2f} vs random-feature control {control_auc:.2f} — did not clear the gate (AUC≥0.80, F1≥0.70, beat control by ≥0.15)."),
    }
    metrics = {**test_m, "train_auc": train_auc, "full_auc": full_m["auc"], "control_auc": control_auc,
               "n_pos": len(pos_maps), "n_neg": len(neg_maps), "n_test_pos": len(te_p), "n_test_neg": len(te_n)}
    return {"features": [int(f) for f in selected], "combine": "max", "threshold": round(threshold, 4),
            "top_k": int(top_k), "metrics": metrics, "per_feature": per_feature, "validation_decision": decision}
