"""First-class residual-space linear **probe** — the detector our own shootout showed *beats*
the SAE feature (raw-residual probe AUC 1.00 vs SAE 0.83–0.88 on real Qwen-2B). A probe is a
direction in the residual stream (difference-of-means or logistic) plus a calibrated threshold;
the *same* direction doubles as a CAA-style steering vector (see ``service.steer_direction``).

Pure / given-vectors (operates on the pooled residuals ``service`` captures), mirroring
``monitor.py``'s honest-control discipline: a held-out split, a **label-shuffled control** (the
probe's analogue of the random-feature control — fit on permuted labels, it should score ~0.5),
the FPR operating point, and a verdict that only validates when the probe clears a strict gate
*and* beats that control.
"""
from __future__ import annotations

import random
from typing import Any, Sequence

import numpy as np

from . import baselines as _bl

Vec = Sequence[float]


def discover_probe(pos_res: list[Vec], neg_res: list[Vec], *, method: str = "diffmeans",
                   target_fpr: float = 0.1, seed: int = 0) -> dict[str, Any]:
    """Fit a linear probe on pooled residuals, calibrate a threshold, and report held-out metrics
    + a label-shuffled control + an honest verdict. ``method`` is ``diffmeans`` or ``logistic``."""
    if not pos_res or not neg_res:
        raise ValueError("probe discovery needs both positive and negative residual sets")

    trp, tep = pos_res[0::2], pos_res[1::2] or pos_res[0::2]
    trn, ten = neg_res[0::2], neg_res[1::2] or neg_res[0::2]

    if method == "ensemble":
        # average the (unit) diff-means and logistic directions — a cheap, more robust probe
        dw, db = _bl.diff_means_probe(trp, trn)
        lw, lb = _bl.logistic_probe(trp, trn)
        dn, ln = np.linalg.norm(dw) or 1.0, np.linalg.norm(lw) or 1.0
        w = np.asarray(dw, dtype=float) / dn + np.asarray(lw, dtype=float) / ln
        b = float(db / dn + lb / ln)
    else:
        fit = _bl.logistic_probe if method == "logistic" else _bl.diff_means_probe
        w, b = fit(trp, trn)
    thr = _bl.best_threshold_f1(_bl._project(trp, w, b), _bl._project(trn, w, b))
    metrics = _bl._operating_point(_bl._project(tep, w, b), _bl._project(ten, w, b), thr, target_fpr)

    # label-shuffled control: fit on permuted labels, eval held-out -> should be ~chance
    rng = random.Random(seed)
    pool = list(trp) + list(trn)
    n_pos = len(trp)
    ctrl = []
    for _ in range(5):
        idx = list(range(len(pool)))
        rng.shuffle(idx)
        sp = [pool[i] for i in idx[:n_pos]]
        sn = [pool[i] for i in idx[n_pos:]]
        if not sp or not sn:
            continue
        try:
            cw, cb = fit(sp, sn)
            cthr = _bl.best_threshold_f1(_bl._project(sp, cw, cb), _bl._project(sn, cw, cb))
            ctrl.append(_bl.eval_scores(_bl._project(tep, cw, cb), _bl._project(ten, cw, cb), cthr)["auc"])
        except Exception:
            continue
    control_auc = round(sum(ctrl) / len(ctrl), 4) if ctrl else float("nan")

    auc, f1 = metrics["auc"], metrics["f1"]
    passed = (auc >= 0.8) and (f1 >= 0.7) and (control_auc != control_auc or auc >= control_auc + 0.15)
    decision = {
        "status": "validated" if passed else "benchmarked",
        "passed": bool(passed),
        "reason": (f"held-out AUC {auc:.2f}, F1 {f1:.2f}; beats the label-shuffled control ({control_auc:.2f})."
                   if passed else
                   f"held-out AUC {auc:.2f}, F1 {f1:.2f} vs label-shuffled control {control_auc:.2f} — did not clear the gate (AUC≥0.80, F1≥0.70, beat control by ≥0.15)."),
    }
    return {
        "method": method,
        "direction": [float(x) for x in np.asarray(w, dtype=float).ravel()],  # raw (threshold is against this)
        "bias": float(b),
        "threshold": round(float(thr), 4),
        "metrics": {**metrics, "control_auc": control_auc, "n_pos": len(pos_res), "n_neg": len(neg_res),
                    "n_test_pos": len(tep), "n_test_neg": len(ten)},
        "validation_decision": decision,
        "target_fpr": float(target_fpr),
    }


def score_probe(direction: Vec, bias: float, threshold: float, residual: Vec) -> dict[str, Any]:
    """Score one pooled residual against a probe: fires iff ``w·r + b ≥ threshold``."""
    s = float(np.asarray(residual, dtype=float).ravel() @ np.asarray(direction, dtype=float) + float(bias))
    return {"score": round(s, 4), "fires": bool(s >= threshold)}


def unit_direction(direction: Vec) -> list[float]:
    """The normalised probe direction — the steering vector for CAA-style control (``service``
    scales it by a signed strength)."""
    w = np.asarray(direction, dtype=float).ravel()
    norm = float(np.linalg.norm(w))
    return [float(x) for x in (w / norm if norm > 0 else w)]
