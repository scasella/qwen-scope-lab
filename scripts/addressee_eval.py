#!/usr/bin/env python3
"""Addressee-probe eval: is "who is this instruction for?" linearly readable at layer 12 —
and does presence+addressee stacking recover the firewall operating point?

Pre-registered in reports/agent_boundary_addressee/preregistration.md BEFORE this ran.
Requires both corpora on disk (scripts/agent_boundary_build_corpus.py and
scripts/addressee_build_corpus.py). Real Qwen3.5-2B on MLX, layer 12, seed 0.

Arms: addressee_dm / addressee_lg (fit on matched pairs), lexical_matched (char-3-gram, same
fit pairs), presence_lg (the parent experiment's probe, refit identically), combined_min
(min of per-probe z-scores: fires only if instruction-present AND model-directed),
random_control (20 dirs).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from qwen_scope_lab import baselines as _bl
from qwen_scope_lab.mlx_backend import MAX_SEQ, MlxModel

ADDR = Path("data/experiments/agent_boundary_addressee")
PARENT = Path("data/experiments/agent_boundary")
REPORT = Path("reports/agent_boundary_addressee")
LAYER = 12
MODEL = "mlx-community/Qwen3.5-2B-bf16"
SEED = 0
LEX_DIM = 4096


def roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:
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
        avg_ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    full = np.empty(len(combined))
    full[order2] = avg_ranks
    return float((full[n_neg:].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def char3_hash_vec(text: str, dim: int = LEX_DIM) -> np.ndarray:
    t = " ".join(text.lower().split())
    v = np.zeros(dim, dtype=np.float32)
    for i in range(len(t) - 2):
        v[hash(t[i:i + 3]) % dim] += 1.0
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def load(path: Path, name: str) -> list[dict]:
    return [json.loads(l) for l in (path / f"{name}.jsonl").read_text().splitlines() if l.strip()]


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[load] {MODEL} ...")
    m = MlxModel(MODEL, default_layer=LAYER)

    def window_residuals(text: str) -> np.ndarray:
        ids = list(m.tokenizer.encode(text))
        chunks = [ids[i:i + MAX_SEQ] for i in range(0, len(ids), MAX_SEQ)]
        if len(chunks) > 1 and len(chunks[-1]) < 8:
            chunks = chunks[:-2] + [chunks[-2] + chunks[-1]]
        out = []
        for ch in chunks:
            t = m.tokenizer.decode(ch).strip()
            if t:
                out.append(m.pooled_residual(t, LAYER))
        return np.asarray(out, dtype=np.float32)

    data = {
        "model_fit": load(ADDR, "model_fit"), "reader_fit": load(ADDR, "reader_fit"),
        "model_eval": load(ADDR, "model_eval"), "reader_eval": load(ADDR, "reader_eval"),
        "injected_fit": load(PARENT, "injected_fit"), "clean_fit": load(PARENT, "clean_fit"),
        "injected_test": load(PARENT, "injected_test"), "clean_test": load(PARENT, "clean_test"),
        "hard_negative": load(PARENT, "hard_negative"),
    }
    n_docs = sum(len(v) for v in data.values())
    print(f"[res] window residuals for {n_docs} docs ...")
    done = 0
    for s, rows in data.items():
        for r in rows:
            r["wres"] = window_residuals(r["text"])
            done += 1
            if done % 100 == 0:
                print(f"    .. {done}/{n_docs} ({time.time() - t0:.0f}s)")

    # ---- directions ----
    print("[fit] addressee probes (matched pairs) + presence probe (parent fit set) ...")
    a_pos = [r["wres"][0] for r in data["model_fit"]]
    a_neg = [r["wres"][0] for r in data["reader_fit"]]
    w_adm, b_adm = _bl.diff_means_probe(a_pos, a_neg)
    w_alg, b_alg = _bl.logistic_probe(a_pos, a_neg)
    p_pos = [r["wres"][0] for r in data["injected_fit"]]
    p_neg = [r["wres"][0] for r in data["clean_fit"]]
    w_plg, b_plg = _bl.logistic_probe(p_pos, p_neg)
    lex_pos = [char3_hash_vec(r["text"]) for r in data["model_fit"]]
    lex_neg = [char3_hash_vec(r["text"]) for r in data["reader_fit"]]
    w_lex, b_lex = _bl.logistic_probe(lex_pos, lex_neg)

    # per-probe z-normalization stats from each probe's OWN pooled fit-set window-0 scores
    def zstats(pos, neg, w, b):
        s = np.asarray([v @ w + b for v in pos + neg], dtype=np.float64)
        return float(s.mean()), float(s.std() + 1e-9)
    mu_a, sd_a = zstats(a_pos, a_neg, np.asarray(w_alg, dtype=np.float32), b_alg)
    mu_p, sd_p = zstats(p_pos, p_neg, np.asarray(w_plg, dtype=np.float32), b_plg)

    rng = np.random.default_rng(SEED)
    random_dirs = []
    for _ in range(20):
        d = rng.standard_normal(len(w_adm)).astype(np.float32)
        random_dirs.append(d / np.linalg.norm(d))

    # ---- per-doc scores (windowed-max) ----
    for s, rows in data.items():
        for r in rows:
            W = r["wres"]
            r["sc"] = {
                "addressee_dm": float((W @ np.asarray(w_adm, dtype=np.float32) + b_adm).max()),
                "addressee_lg": float((W @ np.asarray(w_alg, dtype=np.float32) + b_alg).max()),
                "presence_lg": float((W @ np.asarray(w_plg, dtype=np.float32) + b_plg).max()),
                "lexical_matched": float(char3_hash_vec(r["text"]) @ w_lex + b_lex),
                "random": [float((W @ d).max()) for d in random_dirs],
            }
            r["sc"]["combined_min"] = min((r["sc"]["presence_lg"] - mu_p) / sd_p,
                                          (r["sc"]["addressee_lg"] - mu_a) / sd_a)

    def sc(split, key, pred=None):
        return np.asarray([r["sc"][key] for r in data[split] if pred is None or pred(r)], dtype=np.float64)

    def rand_auc(ps, ns, pred=None):
        aucs = [roc_auc(np.asarray([r["sc"]["random"][i] for r in data[ps] if pred is None or pred(r)]),
                        np.asarray([r["sc"]["random"][i] for r in data[ns]])) for i in range(20)]
        return round(float(np.mean(aucs)), 4), round(float(np.std(aucs)), 4)

    results: dict = {}

    # ---- A. held-out matched pairs: model-directed vs reader-directed ----
    A = {}
    for key in ["addressee_dm", "addressee_lg", "lexical_matched", "presence_lg"]:
        A[key] = {
            "overall": round(roc_auc(sc("model_eval", key), sc("reader_eval", key)), 4),
            "explicit": round(roc_auc(sc("model_eval", key, lambda r: r["mode"] == "explicit"),
                                      sc("reader_eval", key, lambda r: r["mode"] == "explicit")), 4),
            "implicit": round(roc_auc(sc("model_eval", key, lambda r: r["mode"] == "implicit"),
                                      sc("reader_eval", key, lambda r: r["mode"] == "implicit")), 4),
        }
    A["by_position_addressee_lg"] = {
        pos: round(roc_auc(sc("model_eval", "addressee_lg", lambda r, p=pos: r["position"] == p),
                           sc("reader_eval", "addressee_lg", lambda r, p=pos: r["position"] == p)), 4)
        for pos in ["start", "middle", "end"]}
    A["random"] = rand_auc("model_eval", "reader_eval")
    results["A_matched_heldout"] = A

    # ---- B. transfer to real payloads: parent injected_test vs hard_negative / clean_test ----
    B = {}
    for key in ["addressee_lg", "addressee_dm", "lexical_matched", "presence_lg", "combined_min"]:
        B[key] = {
            "injected_vs_hard_neg": round(roc_auc(sc("injected_test", key), sc("hard_negative", key)), 4),
            "injected_vs_clean": round(roc_auc(sc("injected_test", key), sc("clean_test", key)), 4),
        }
    B["random_vs_hard_neg"] = rand_auc("injected_test", "hard_negative")
    results["B_transfer_real_payloads"] = B

    # ---- C. where does "no instruction" sit on the addressee axis? ----
    results["C_axis_anatomy"] = {
        "reader_vs_clean_addressee_lg": round(roc_auc(sc("hard_negative", "addressee_lg"),
                                                      sc("clean_test", "addressee_lg")), 4),
        "model_vs_clean_addressee_lg": round(roc_auc(sc("injected_test", "addressee_lg"),
                                                     sc("clean_test", "addressee_lg")), 4),
    }

    verdict = {"model": MODEL, "layer": LAYER, "seed": SEED, "n_docs": n_docs,
               "n_windows": int(sum(len(r["wres"]) for rows in data.values() for r in rows)),
               "elapsed_s": round(time.time() - t0, 1), "results": results}
    (REPORT / "verdict.json").write_text(json.dumps(verdict, indent=2))
    with (REPORT / "scores.jsonl").open("w") as f:
        for s, rows in data.items():
            for r in rows:
                f.write(json.dumps({"split": s, **{k: r.get(k) for k in ("pair_id", "mode", "position", "form", "family", "flavor")},
                                    "scores": {k: v for k, v in r["sc"].items() if k != "random"}}) + "\n")
    print(json.dumps(results, indent=2))
    print(f"\n[done] {time.time() - t0:.0f}s -> {REPORT / 'verdict.json'}")


if __name__ == "__main__":
    main()
