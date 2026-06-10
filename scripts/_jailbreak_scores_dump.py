"""Dump per-prompt probe scores (no prompt text) for the in-the-wild jailbreak visual write-up.

Mirrors scripts/jailbreak_wild_eval.py exactly — same model, layer, deployable diff-means probe
fit on the author clean set, same author F1 threshold — and saves one record per evaluated prompt:
{split, src, score}. The scraped prompt texts are NOT included (the corpus itself stays local /
rebuildable); the scores power the draggable-threshold figure in
docs/writeups/jailbreak-probe-in-the-wild.html.

Run:  python scripts/_jailbreak_scores_dump.py
Out:  reports/jailbreak_wild/scores.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from qwen_scope_lab import baselines as _bl
from qwen_scope_lab import behavior_sets as _bs
from qwen_scope_lab.mlx_backend import MlxModel

DATA = Path("data/experiments/jailbreak_wild")
OUT = Path("reports/jailbreak_wild/scores.json")
LAYER = 12
MODEL = "mlx-community/Qwen3.5-2B-bf16"
SPLITS = ["itw_jailbreak", "benign_ordinary", "hard_negative", "jbb_harmful"]


def main() -> None:
    t0 = time.time()
    print(f"[load] {MODEL} ...")
    m = MlxModel(MODEL, default_layer=LAYER)

    def residuals(texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), m.d_model), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i] = m.pooled_residual(t, LAYER)
        return out

    pos_clean, neg_clean = _bs.BEHAVIORS["jailbreak"]["clean"]
    pos_res = residuals(list(pos_clean))
    neg_res = residuals(list(neg_clean))
    w, b = _bl.diff_means_probe(list(pos_res), list(neg_res))
    author_thr = _bl.best_threshold_f1(_bl._project(list(pos_res), w, b),
                                       _bl._project(list(neg_res), w, b))
    w = np.asarray(w, dtype=np.float32)
    print(f"[probe] author threshold {author_thr:.4f}")

    records = []
    for name in SPLITS:
        rows = [json.loads(line) for line in (DATA / f"{name}.jsonl").read_text().splitlines() if line.strip()]
        res = residuals([r["text"] for r in rows])
        scores = res @ w + b
        for r, s in zip(rows, scores):
            records.append({"split": name, "src": r.get("src", ""), "score": round(float(s), 4)})
        print(f"[score] {name}: {len(rows)} prompts ({time.time()-t0:.0f}s)")

    OUT.write_text(json.dumps({"model": MODEL, "layer": LAYER,
                               "author_threshold": round(float(author_thr), 4),
                               "records": records}, indent=0))
    print(f"[done] {OUT} · {len(records)} records · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
