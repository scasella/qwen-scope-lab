#!/usr/bin/env python3
"""Harden the layer-12 diff-of-means residual jailbreak probe against the in-the-wild corpus.

Executes the plan in RESULTS.md:382-418 ("direction robust, threshold doesn't transfer"; fix =
per-distribution threshold recalibration + an in-the-wild corpus). Inference-only, single seed for
the probe; the random-direction control is averaged over many seeds. Everything runs on MLX (real
Qwen3.5-2B on-device), no paid APIs.

The DEPLOYABLE probe = diff-of-means fit on the author's clean jailbreak set (JAILBREAK_POS vs
JAILBREAK_NEG) — exactly the direction `service.jailbreak_detection` / `jailbreak_screen` ship,
with its F1-calibrated author threshold. We do NOT retrain it before the wild eval (per the plan);
we retrain only in the held-out-source generalisation section (step 4).

Splits evaluated (each: AUC, TPR@1%FPR, TPR@5%FPR, plus recall/FPR at author-thr vs recalibrated-thr):
  A  itw_jailbreak  vs  benign_ordinary   — wild jailbreaks vs ordinary benign
  B  itw_jailbreak  vs  hard_negative     — wild jailbreaks vs jailbreak-shaped-but-benign
  C  jbb_harmful    vs  hard_negative     — bare harmful intent (no JB framing) vs spicy-benign

Honest controls: a random unit-direction probe on every split (mean AUC over 20 seeds), and the
author threshold transfer is quantified explicitly (how much FPR/recall move when you don't
recalibrate).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from qwen_scope_lab import baselines as _bl
from qwen_scope_lab import behavior_sets as _bs
from qwen_scope_lab.mlx_backend import MlxModel

OUT = Path("data/experiments/jailbreak_wild")
REPORT = Path("reports/jailbreak_wild")
LAYER = 12
MODEL = "mlx-community/Qwen3.5-2B-bf16"
SEED = 0


# ---------- metrics ----------
def roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC = P(score_pos > score_neg) via tie-aware Mann-Whitney U (ties counted as 0.5)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    n_neg, n_pos = len(neg), len(pos)
    combined = np.concatenate([neg, pos])
    order2 = np.argsort(combined, kind="mergesort")
    sv = combined[order2]
    avg_ranks = np.empty(len(sv))
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        avg_ranks[i:j + 1] = (i + j) / 2 + 1  # 1-based average rank
        i = j + 1
    full = np.empty(len(combined))
    full[order2] = avg_ranks
    rank_pos_sum = full[n_neg:].sum()
    auc = (rank_pos_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def threshold_at_fpr(neg: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold s.t. FPR on neg <= target_fpr (fires iff score >= thr)."""
    if len(neg) == 0:
        return float("inf")
    s = np.sort(neg)[::-1]
    k = int(np.floor(target_fpr * len(neg)))
    if k <= 0:
        return float(s[0]) + 1e-9
    return float(s[k - 1]) + 1e-12


def tpr_at_fpr(pos: np.ndarray, neg: np.ndarray, target_fpr: float) -> float:
    thr = threshold_at_fpr(neg, target_fpr)
    return float((pos >= thr).mean())


def rate_at_thr(scores: np.ndarray, thr: float) -> float:
    return float((scores >= thr).mean()) if len(scores) else float("nan")


# ---------- residual capture ----------
def load_split(name: str) -> list[dict]:
    return [json.loads(l) for l in (OUT / f"{name}.jsonl").read_text().splitlines() if l.strip()]


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[load] {MODEL} ...")
    m = MlxModel(MODEL, default_layer=LAYER)

    def residuals(texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), m.d_model), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i] = m.pooled_residual(t, LAYER)
            if (i + 1) % 100 == 0:
                print(f"    .. {i+1}/{len(texts)} residuals ({time.time()-t0:.0f}s)")
        return out

    # ---- the DEPLOYABLE probe: fit on the author clean set (what jailbreak_screen ships) ----
    print("[probe] fitting deployable diff-means probe on author clean set ...")
    pos_clean, neg_clean = _bs.BEHAVIORS["jailbreak"]["clean"]
    pos_res = residuals(list(pos_clean))
    neg_res = residuals(list(neg_clean))
    w, b = _bl.diff_means_probe(list(pos_res), list(neg_res))
    author_thr = _bl.best_threshold_f1(_bl._project(list(pos_res), w, b), _bl._project(list(neg_res), w, b))
    w = np.asarray(w, dtype=np.float32)
    print(f"    direction dim {w.shape}, author F1 threshold {author_thr:.4f}, bias {b:.4f}")

    def score(res: np.ndarray, direction=w, bias=b) -> np.ndarray:
        return res @ direction + bias

    # ---- compute residuals for every eval split (once) ----
    splits = {}
    for name in ["itw_jailbreak", "benign_ordinary", "hard_negative", "jbb_harmful"]:
        rows = load_split(name)
        texts = [r["text"] for r in rows]
        print(f"[res] {name}: {len(texts)} prompts")
        res = residuals(texts)
        splits[name] = {"rows": rows, "res": res, "scores": score(res)}

    # random-direction control: many unit directions, averaged AUC per comparison
    def control_auc(pos_res, neg_res, n=20):
        rng = np.random.default_rng(SEED)
        aucs = []
        for _ in range(n):
            d = rng.standard_normal(m.d_model).astype(np.float32)
            d /= np.linalg.norm(d)
            aucs.append(roc_auc(pos_res @ d, neg_res @ d))
        return float(np.mean(aucs)), float(np.std(aucs))

    def eval_pair(pos_key, neg_key):
        ps, ns = splits[pos_key]["scores"], splits[neg_key]["scores"]
        pr, nr = splits[pos_key]["res"], splits[neg_key]["res"]
        auc = roc_auc(ps, ns)
        c_mean, c_std = control_auc(pr, nr)
        # recalibrated threshold on THIS distribution (target 5% FPR on the negatives here)
        recal_thr = threshold_at_fpr(ns, 0.05)
        return {
            "n_pos": int(len(ps)), "n_neg": int(len(ns)),
            "auc": round(auc, 4),
            "tpr_at_1pct_fpr": round(tpr_at_fpr(ps, ns, 0.01), 4),
            "tpr_at_5pct_fpr": round(tpr_at_fpr(ps, ns, 0.05), 4),
            "control_auc_mean": round(c_mean, 4), "control_auc_std": round(c_std, 4),
            "author_thr": round(float(author_thr), 4),
            "recall_at_author_thr": round(rate_at_thr(ps, author_thr), 4),
            "fpr_at_author_thr": round(rate_at_thr(ns, author_thr), 4),
            "recal_thr_5pct_fpr": round(float(recal_thr), 4),
            "recall_at_recal_thr": round(rate_at_thr(ps, recal_thr), 4),
            "fpr_at_recal_thr": round(rate_at_thr(ns, recal_thr), 4),
        }

    results = {
        "A_wild_jb_vs_ordinary": eval_pair("itw_jailbreak", "benign_ordinary"),
        "B_wild_jb_vs_hard_neg": eval_pair("itw_jailbreak", "hard_negative"),
        "C_harmful_intent_vs_hard_neg": eval_pair("jbb_harmful", "hard_negative"),
    }

    # ---- held-out-SOURCE generalisation (step 4): retrain on some sources, test on held-out ----
    print("[heldout] held-out-source generalisation ...")
    by_src = [json.loads(l) for l in (OUT / "itw_by_source.jsonl").read_text().splitlines() if l.strip()]
    # train sources vs held-out sources (disjoint), benign from benign_ordinary (split in half)
    train_srcs = {"flowgpt", "ChatGPT", "jailbreak_chat"}
    test_srcs = {"ChatGPTJailbreak", "LLM Promptwriting", "BreakGPT", "aiprm",
                 "ChatGPT Prompt Engineering", "Spreadsheet Warriors", "ChatGPTPromptGenius",
                 "AI Prompt Sharing", "awesome_chatgpt_prompts"}
    rng = np.random.default_rng(SEED)
    jb_train_txt = [r["text"] for r in by_src if r["src"] in train_srcs]
    jb_test_txt = [r["text"] for r in by_src if r["src"] in test_srcs]
    rng.shuffle(jb_train_txt); rng.shuffle(jb_test_txt)
    jb_train_txt, jb_test_txt = jb_train_txt[:150], jb_test_txt[:150]
    bn_rows = load_split("benign_ordinary")
    bn_txt = [r["text"] for r in bn_rows]
    rng.shuffle(bn_txt)
    bn_train, bn_test = bn_txt[:150], bn_txt[150:300]

    jb_tr_res, jb_te_res = residuals(jb_train_txt), residuals(jb_test_txt)
    bn_tr_res, bn_te_res = residuals(bn_train), residuals(bn_test)
    w2, b2 = _bl.diff_means_probe(list(jb_tr_res), list(bn_tr_res))
    w2 = np.asarray(w2, dtype=np.float32)
    # in-distribution (train sources) threshold at 5% FPR
    tr_neg_scores = bn_tr_res @ w2 + b2
    thr_id = threshold_at_fpr(tr_neg_scores, 0.05)
    te_pos = jb_te_res @ w2 + b2
    te_neg = bn_te_res @ w2 + b2
    c_mean, c_std = control_auc(jb_te_res, bn_te_res)
    heldout = {
        "train_sources": sorted(train_srcs), "test_sources": sorted(test_srcs),
        "n_train_jb": len(jb_train_txt), "n_test_jb": len(jb_test_txt),
        "in_dist_auc": round(roc_auc(jb_tr_res @ w2 + b2, tr_neg_scores), 4),
        "heldout_source_auc": round(roc_auc(te_pos, te_neg), 4),
        "heldout_tpr_at_5pct_fpr": round(tpr_at_fpr(te_pos, te_neg, 0.05), 4),
        "heldout_recall_at_id_thr": round(rate_at_thr(te_pos, thr_id), 4),
        "heldout_fpr_at_id_thr": round(rate_at_thr(te_neg, thr_id), 4),
        "control_auc_mean": round(c_mean, 4), "control_auc_std": round(c_std, 4),
    }
    results["D_heldout_source_generalisation"] = heldout

    verdict = {
        "model": MODEL, "layer": LAYER, "seed": SEED,
        "probe": "diff_means on author clean jailbreak set (the deployable direction)",
        "author_threshold": round(float(author_thr), 4),
        "elapsed_s": round(time.time() - t0, 1),
        "results": results,
    }
    (REPORT / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\n[done] {time.time()-t0:.0f}s -> {REPORT/'verdict.json'}")


if __name__ == "__main__":
    main()
