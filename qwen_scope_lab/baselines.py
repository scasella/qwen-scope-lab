"""Baseline detectors for the monitor **shootout** — the honest answer to the field's
central credibility question: *do SAE features actually beat a cheap raw-residual probe?*
(see arXiv 2502.16681 "Are Sparse Autoencoders Useful?" and 2602.14111 "Do SAEs Beat
Random Baselines?"). If an interpretable SAE-feature monitor cannot beat a linear probe
on the raw residual stream, the interpretability is not buying detection power — and the
bench should say so.

Pure / given-vectors: this module operates on the SAE activation maps and pooled residual
vectors that ``service`` captures, so it is unit-testable with no model (mirrors the
``monitor.py`` discipline: a shared held-out split, a random control, and a verdict that
only credits the SAE monitor when it genuinely wins).

It also adds a deployment-relevant operating point absent from ``monitor.discover``: the
**TPR at a target false-positive rate** (a safety monitor runs at a fixed FPR budget, not
an F1-optimal threshold).
"""
from __future__ import annotations

import statistics as st
from typing import Any, Sequence

import numpy as np

from . import monitor as _mon

Vec = Sequence[float]


# --------------------------- generic score-based metrics ---------------------------
def eval_scores(pos: list[float], neg: list[float], thr: float) -> dict[str, float]:
    """Confusion-matrix metrics for any real-valued detector score at threshold ``thr``."""
    tp = sum(p >= thr for p in pos)
    fn = len(pos) - tp
    fp = sum(n >= thr for n in neg)
    tn = len(neg) - fp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / (len(pos) + len(neg)) if (pos or neg) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"auc": round(_mon.auc(pos, neg), 4), "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "accuracy": round(acc, 4), "fpr": round(fpr, 4)}


def best_threshold_f1(pos: list[float], neg: list[float]) -> float:
    """Threshold on the raw score that maximizes F1 on the given (train) split."""
    vals = sorted(set(pos) | set(neg))
    cands = [vals[0] - 1.0] if vals else [0.0]
    for i, v in enumerate(vals):
        cands.append(v)
        if i + 1 < len(vals):
            cands.append((v + vals[i + 1]) / 2)
    best, best_f1 = cands[0], -1.0
    for t in cands:
        f1 = eval_scores(pos, neg, t)["f1"]
        if f1 > best_f1:
            best_f1, best = f1, t
    return best


def threshold_at_fpr(neg: list[float], target_fpr: float) -> float:
    """Smallest threshold whose false-positive rate on ``neg`` is ≤ ``target_fpr`` — the
    way a deployed safety monitor is actually tuned (a fixed FPR budget)."""
    if not neg:
        return 0.0
    s = sorted(neg, reverse=True)
    k = int(target_fpr * len(neg))  # allowed false positives
    if k <= 0:
        return s[0] + 1e-6
    if k >= len(s):
        return s[-1] - 1e-6
    return (s[k - 1] + s[k]) / 2.0


def _operating_point(pos: list[float], neg: list[float], thr: float, target_fpr: float,
                     extra: dict | None = None) -> dict[str, Any]:
    m = eval_scores(pos, neg, thr)
    fthr = threshold_at_fpr(neg, target_fpr)
    at_fpr = eval_scores(pos, neg, fthr)
    m["tpr_at_fpr"] = at_fpr["recall"]          # detection rate at the fixed FPR budget
    m["fpr_at_op"] = at_fpr["fpr"]              # achieved FPR (≤ target, granularity permitting)
    m["threshold"] = round(float(thr), 4)
    if extra:
        m.update(extra)
    return m


# ------------------------- linear probes on the raw residual -------------------------
def _stack(vecs: list[Vec]) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=float).ravel() for v in vecs])


def diff_means_probe(train_pos: list[Vec], train_neg: list[Vec]) -> tuple[np.ndarray, float]:
    """Difference-of-means probe: the canonical cheap linear baseline. Direction is the
    (normalised) mean-positive minus mean-negative residual; bias centres it at the midpoint."""
    p, n = _stack(train_pos), _stack(train_neg)
    w = p.mean(0) - n.mean(0)
    norm = float(np.linalg.norm(w))
    if norm > 0:
        w = w / norm
    b = -0.5 * float((p.mean(0) + n.mean(0)) @ w)
    return w, b


def logistic_probe(train_pos: list[Vec], train_neg: list[Vec], *, l2: float = 1.0,
                   iters: int = 300, lr: float = 0.5) -> tuple[np.ndarray, float]:
    """Standardised, L2-regularised logistic regression by gradient descent (deterministic,
    no sklearn). A stronger probe than diff-of-means when the classes aren't isotropic;
    folded back to operate on raw residual vectors for scoring."""
    p, n = _stack(train_pos), _stack(train_neg)
    x = np.vstack([p, n])
    y = np.concatenate([np.ones(len(p)), np.zeros(len(n))])
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xs = (x - mu) / sd
    w = np.zeros(xs.shape[1])
    b = 0.0
    m = len(y)
    for _ in range(iters):
        prob = 1.0 / (1.0 + np.exp(-(xs @ w + b)))
        grad = prob - y
        w -= lr * (xs.T @ grad / m + l2 * w / m)
        b -= lr * float(grad.mean())
    w_raw = w / sd
    b_raw = float(b - (w * mu / sd).sum())
    return w_raw, b_raw


def _project(vecs: list[Vec], w: np.ndarray, b: float) -> list[float]:
    x = _stack(vecs)
    return [float(v) for v in (x @ w + b)]


def _select_features(tp_maps: list[dict], tn_maps: list[dict], top_k: int) -> list[int]:
    """Top-k features by mean-activation difference (pos − neg) on the train split — the
    same selection ``monitor.discover`` uses, factored so the shootout splits once."""
    counts: dict[int, int] = {}
    for mp in tp_maps + tn_maps:
        for fid in mp:
            counts[fid] = counts.get(fid, 0) + 1
    feats = list(counts)

    def diff(fid: int) -> float:
        mp = st.mean([m.get(fid, 0.0) for m in tp_maps]) if tp_maps else 0.0
        mn = st.mean([m.get(fid, 0.0) for m in tn_maps]) if tn_maps else 0.0
        return mp - mn

    return sorted(feats, key=lambda f: -diff(f))[: max(1, top_k)]


# ----------------------------------- the shootout -----------------------------------
def shootout(pos_maps: list[dict], neg_maps: list[dict], pos_res: list[Vec], neg_res: list[Vec],
             *, top_k: int = 3, d_sae: int | None = None, target_fpr: float = 0.1,
             seed: int = 0) -> dict[str, Any]:
    """Compare, on one shared held-out split, the SAE-feature monitor against raw-residual
    linear probes (diff-of-means + logistic) and a random-feature control. Returns each
    method's held-out metrics (incl. TPR@FPR) and an **honest verdict**: the SAE monitor is
    only credited as winning if it beats the best cheap probe by a margin."""
    import random as _random

    if not pos_maps or not neg_maps:
        raise ValueError("shootout needs both positive and negative examples")
    if len(pos_res) != len(pos_maps) or len(neg_res) != len(neg_maps):
        raise ValueError("residual vectors must align 1:1 with activation maps")

    # one interleaved train/test split, shared by every method (fall back to train on tiny sets)
    trp_m, tep_m = pos_maps[0::2], pos_maps[1::2] or pos_maps[0::2]
    trn_m, ten_m = neg_maps[0::2], neg_maps[1::2] or neg_maps[0::2]
    trp_r, tep_r = pos_res[0::2], pos_res[1::2] or pos_res[0::2]
    trn_r, ten_r = neg_res[0::2], neg_res[1::2] or neg_res[0::2]

    methods: dict[str, dict] = {}

    # --- SAE feature monitor ---
    feats = _select_features(trp_m, trn_m, top_k)
    sae_tr_p = [_mon._combined(feats, m) for m in trp_m]
    sae_tr_n = [_mon._combined(feats, m) for m in trn_m]
    sae_thr = best_threshold_f1(sae_tr_p, sae_tr_n)
    methods["sae_monitor"] = _operating_point(
        [_mon._combined(feats, m) for m in tep_m], [_mon._combined(feats, m) for m in ten_m],
        sae_thr, target_fpr, extra={"features": [int(f) for f in feats]})

    # --- random-feature control (same k, fit+eval the same way, averaged) ---
    rng = _random.Random(seed)
    universe = list(range(d_sae)) if d_sae else list({fid for m in pos_maps + neg_maps for fid in m})
    ctrl_aucs = []
    for _ in range(5):
        rand = rng.sample(universe, min(top_k, len(universe))) if universe else []
        rtr_p = [_mon._combined(rand, m) for m in trp_m]
        rtr_n = [_mon._combined(rand, m) for m in trn_m]
        rthr = best_threshold_f1(rtr_p, rtr_n)
        ctrl_aucs.append(eval_scores([_mon._combined(rand, m) for m in tep_m],
                                     [_mon._combined(rand, m) for m in ten_m], rthr)["auc"])
    methods["random_control"] = {"auc": round(sum(ctrl_aucs) / len(ctrl_aucs), 4) if ctrl_aucs else float("nan")}

    # --- raw-residual probes ---
    for name, fit in (("residual_diffmeans", diff_means_probe), ("residual_logistic", logistic_probe)):
        try:
            w, b = fit(trp_r, trn_r)
            ptr_p, ptr_n = _project(trp_r, w, b), _project(trn_r, w, b)
            pthr = best_threshold_f1(ptr_p, ptr_n)
            methods[name] = _operating_point(_project(tep_r, w, b), _project(ten_r, w, b), pthr, target_fpr)
        except Exception as exc:  # a degenerate split shouldn't sink the whole shootout
            methods[name] = {"auc": float("nan"), "error": str(exc)}

    # --- honest verdict: does the interpretable SAE detector earn its keep? ---
    sae_auc = methods["sae_monitor"]["auc"]
    probe_aucs = [methods[k]["auc"] for k in ("residual_diffmeans", "residual_logistic")
                  if isinstance(methods[k].get("auc"), float) and methods[k]["auc"] == methods[k]["auc"]]
    best_probe = max(probe_aucs) if probe_aucs else float("nan")
    margin = round(sae_auc - best_probe, 4) if probe_aucs else None
    if margin is None:
        winner, reason = "inconclusive", "no usable residual probe (degenerate split) — cannot compare."
    elif margin >= 0.05:
        winner = "sae_monitor"
        reason = (f"SAE feature monitor (AUC {sae_auc:.2f}) beats the best raw-residual probe "
                  f"(AUC {best_probe:.2f}) by {margin:+.2f} — the interpretable detector earns its keep.")
    elif margin <= -0.05:
        winner = "residual_probe"
        reason = (f"a raw-residual linear probe (AUC {best_probe:.2f}) beats the SAE monitor "
                  f"(AUC {sae_auc:.2f}) by {-margin:.2f} — interpretability is not buying detection "
                  f"power here (cf. arXiv 2502.16681).")
    else:
        winner = "tie"
        reason = (f"SAE monitor (AUC {sae_auc:.2f}) and the raw-residual probe (AUC {best_probe:.2f}) "
                  f"are within 0.05 AUC — the SAE detector does not clearly beat the cheap baseline.")

    return {
        "methods": methods,
        "verdict": {"winner": winner, "margin": margin, "sae_auc": sae_auc, "best_probe_auc": best_probe,
                    "control_auc": methods["random_control"]["auc"], "reason": reason},
        "top_k": int(top_k), "target_fpr": float(target_fpr),
        "n_pos": len(pos_maps), "n_neg": len(neg_maps), "n_test_pos": len(tep_m), "n_test_neg": len(ten_m),
    }
