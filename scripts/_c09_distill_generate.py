"""C09 DISTRIBUTION-distillation data generation (MLX, on-device).

The text-SFT formulation of manifold-to-data is structurally refuted (greedy text is invariant
under the replace intervention; manifold/linear arms coincide at the endpoint in text — see
docs/experiments/MANIFOLD_TO_DATA_PROVENANCE.md). The only live formulation is DISTRIBUTION /
soft-label (KL) distillation: teach a hook-free student LoRA the *next-token distribution shift*
the manifold intervention induces, where the within-sweep behavior-energy signal actually lives.

For each concept this script, per carrier template x source value, walks the source->target sweep
along several arms and stores, **per waypoint**, the teacher's next-token distribution under the
position-replace hook (top-k truncated + renormalized; k documented). The supervision prompt is the
carrier filled with the *waypoint's intended value label* — making `(prompt -> teacher_dist)` a
learnable hook-free map (otherwise a constant carrier with a varying value is contradictory
supervision). The arms differ in the residual vector injected at each waypoint:

  gated_manifold    manifold-spline waypoints kept by the geometry gate (energy <= linear chord)
  ungated_manifold  ALL manifold-spline waypoints (isolates whether the GATE matters)
  linear            linear-chord (ambient straight line) waypoints (baseline)
  pullback          activation path optimized to induce the target behavior (gradient pullback)
  prompt_only       NO hook — teacher = base next-token dist on the value-filled carrier (instruction-style)
  shuffled_label    gated_manifold teacher dists with their value labels permuted (negative control)

Writes reports/manifold_c09/distribution_distill/<concept>/<arm>/{train,valid,test}.jsonl, where each
row is {"prompt", "top_ids", "top_probs", "value", "waypoint_index", "template_index", "arm", "kept"}.

    python3 scripts/_c09_distill_generate.py --concept rank      --readout first_token
    python3 scripts/_c09_distill_generate.py --concept education  --readout full_string
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_scope_lab.concept_presets import get_concept
from qwen_scope_lab.mlx_backend import build_mlx_service

DEFAULT_MLX_MODEL = "mlx-community/Qwen3.5-2B-bf16"


def _topk_dist(logits: np.ndarray, k: int):
    """Top-k truncated + renormalized next-token distribution from full-vocab logits."""
    e = np.exp(logits - logits.max())
    probs = e / max(float(e.sum()), 1e-9)
    idx = np.argpartition(-probs, k)[:k]
    idx = idx[np.argsort(-probs[idx])]
    p = probs[idx]
    p = p / max(float(p.sum()), 1e-9)
    return idx.astype(int).tolist(), [float(x) for x in p]


def _waypoint_vectors(service, concept_obj, manifold, layer, src_i, tgt_i, n_waypoints, path):
    """Replicate manifold_steer's per-waypoint replacement vectors (residual-space) for a given path.
    Returns list of (value_label, replacement_vec_np)."""
    import torch  # noqa: F401 (service expects torch tensors for replacement; we hand np to last_logits)
    items, n = manifold["items"], manifold["n_items"]
    spline, pca, cpca = manifold["spline"], manifold["pca"], manifold["centroids_pca"]
    if manifold["kind"] == "cyclic":
        d = (tgt_i - src_i) % n
        if d > n / 2:
            d -= n
        us = [src_i + d * t for t in np.linspace(0, 1, max(2, n_waypoints))]
    else:
        us = list(np.linspace(float(src_i), float(tgt_i), max(2, n_waypoints)))
    steps = len(us)
    out = []
    for wi, u in enumerate(us):
        t = wi / (steps - 1) if steps > 1 else 1.0
        if path == "linear":
            pca_pt = (1 - t) * cpca[src_i] + t * cpca[tgt_i]
            lbl = items[int(round((1 - t) * src_i + t * tgt_i)) % n]
        else:
            uu = (float(u) % n) if manifold["kind"] == "cyclic" else float(u)
            pca_pt = spline(uu)
            lbl = items[int(round(uu)) % n]
        rep = pca.inverse_transform(np.asarray(pca_pt).reshape(1, -1))[0].astype(np.float32)
        out.append((lbl, rep))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="rank")
    ap.add_argument("--readout", choices=["first_token", "full_string"], default="first_token")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--n-waypoints", type=int, default=5)
    ap.add_argument("--topk", type=int, default=256, help="teacher dist truncation (renormalized)")
    ap.add_argument("--arms", nargs="*",
                    default=["gated_manifold", "ungated_manifold", "linear", "prompt_only", "shuffled_label"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/manifold_c09/distribution_distill")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    concept = get_concept(args.concept)
    layer = args.layer if args.layer is not None else (concept.best_layer or 8)
    out_root = Path(args.out) / args.concept
    out_root.mkdir(parents=True, exist_ok=True)
    arms = list(args.arms)

    print(f"[c09-distill] loading MLX {DEFAULT_MLX_MODEL} …", flush=True)
    service = build_mlx_service(DEFAULT_MLX_MODEL, default_layer=12)
    model = service.bundle.model
    manifold, concept_obj, layer = service._build_manifold(args.concept, layer)
    items, n = manifold["items"], manifold["n_items"]
    tgt_i = n - 1
    sources = list(range(n - 1))
    templates = list(concept_obj.templates)

    # carrier-template held-out split (mirror the text-SFT script): train 0..n-3, valid n-2, test n-1
    n_t = len(templates)
    train_t, valid_t = list(range(n_t - 2)), [n_t - 2]
    test_t = [n_t - 1]

    # rows[arm][template_index] -> list of supervision rows
    rows = {a: {ti: [] for ti in range(n_t)} for a in arms}
    energy_gap = []  # linear - manifold mean energy per (template, source)

    def teacher_dist(tmpl, value_label, rep_np):
        """Teacher next-token top-k dist under position-replace at the value token, or base (rep=None).

        The replace position is the value-label token position inside the supervision prompt (the
        carrier ``tmpl`` filled with ``value_label``) — so the residual is overwritten exactly where
        the concept value lives, matching manifold_steer's intervention site."""
        prompt = tmpl.format(item=value_label)
        if rep_np is None:
            logits = model.last_logits(prompt, replace=None)
        else:
            import torch
            position = service._locate_item_position(model.tokenizer, tmpl, value_label, prompt)
            rep = torch.tensor(rep_np, dtype=torch.float32)
            logits = model.last_logits(prompt, replace=(layer, rep, position))
        return _topk_dist(np.asarray(logits, dtype=float), args.topk)

    for ti, tmpl in enumerate(templates):
        for src_i in sources:
            # geometry-gate inputs: per-path mean energy via the real behavior manifold
            try:
                cmp = service.manifold_compare(args.concept, items[tgt_i], layer=layer, source=items[src_i],
                                               prompt=tmpl.format(item=items[src_i]), n_waypoints=args.n_waypoints,
                                               max_new_tokens=2, behavior_readout=args.readout)
            except Exception as exc:  # noqa: BLE001
                print(f"[c09-distill]   skip cmp {tmpl!r}/{items[src_i]!r}: {exc!r}", flush=True)
                continue
            m_e, l_e = cmp["manifold"]["mean_energy"], cmp["linear"]["mean_energy"]
            if m_e is not None and l_e is not None:
                energy_gap.append(l_e - m_e)
            gate_keep = (m_e is not None and l_e is not None and m_e <= l_e)  # energy <= linear chord

            man_vecs = _waypoint_vectors(service, concept_obj, manifold, layer, src_i, tgt_i, args.n_waypoints, "manifold")
            lin_vecs = _waypoint_vectors(service, concept_obj, manifold, layer, src_i, tgt_i, args.n_waypoints, "linear")

            for wi, ((lbl, rep), (_, lrep)) in enumerate(zip(man_vecs, lin_vecs)):
                prompt = tmpl.format(item=lbl)  # supervision prompt encodes the waypoint's value
                man_ids = man_ps = None  # cache manifold teacher dist (reused by gated/ungated)

                def _row(ids, ps, kept, arm):
                    return {"prompt": prompt, "top_ids": ids, "top_probs": ps, "value": lbl,
                            "waypoint_index": wi, "template_index": ti, "arm": arm, "kept": bool(kept)}

                if "linear" in arms:
                    lids, lps = teacher_dist(tmpl, lbl, lrep)
                    rows["linear"][ti].append(_row(lids, lps, True, "linear"))
                if "prompt_only" in arms:
                    pids, pps = teacher_dist(tmpl, lbl, None)  # NO hook — instruction-style base dist
                    rows["prompt_only"][ti].append(_row(pids, pps, True, "prompt_only"))
                if any(a in arms for a in ("gated_manifold", "ungated_manifold")):
                    man_ids, man_ps = teacher_dist(tmpl, lbl, rep)
                if "ungated_manifold" in arms:
                    rows["ungated_manifold"][ti].append(_row(man_ids, man_ps, True, "ungated_manifold"))
                if "gated_manifold" in arms and gate_keep:
                    rows["gated_manifold"][ti].append(_row(man_ids, man_ps, True, "gated_manifold"))
        print(f"[c09-distill] template {ti} '{tmpl}' done", flush=True)

    # shuffled_label control: take the gated_manifold rows and permute which (prompt,value) each
    # teacher dist is attached to, within each template — breaking the value->dist correspondence
    # while keeping the exact same set of teacher dists. A real geometry signal should die here.
    if "shuffled_label" in arms:
        src_arm = "gated_manifold" if "gated_manifold" in arms else "ungated_manifold"
        rng = random.Random(args.seed + 1)
        for ti in range(n_t):
            base = rows.get(src_arm, {}).get(ti, [])
            if len(base) <= 1:
                rows["shuffled_label"][ti] = [dict(r, arm="shuffled_label") for r in base]
                continue
            prompts = [(r["prompt"], r["value"]) for r in base]
            perm = list(range(len(base)))
            rng.shuffle(perm)
            shuffled = []
            for i, r in enumerate(base):
                pr, vl = prompts[perm[i]]
                shuffled.append(dict(r, prompt=pr, value=vl, arm="shuffled_label"))
            rows["shuffled_label"][ti] = shuffled

    # equal-size training cap across arms (smallest arm's train count)
    train_counts = {a: sum(len(rows[a][ti]) for ti in train_t) for a in arms}
    cap = min(train_counts.values()) if train_counts else 0
    summary = {"concept": args.concept, "layer": layer, "readout": args.readout, "topk": args.topk,
               "n_sources": len(sources), "n_templates": n_t, "n_waypoints": args.n_waypoints,
               "train_templates": train_t, "valid_templates": valid_t, "test_templates": test_t,
               "mean_energy_gap_linear_minus_manifold": round(float(np.mean(energy_gap)), 4) if energy_gap else None,
               "n_sweeps_manifold_better": int(sum(1 for g in energy_gap if g > 0)),
               "n_sweeps_total": len(energy_gap), "equal_size_train_cap": cap,
               "train_counts_raw": train_counts, "arms": {}}
    for arm in arms:
        d = out_root / arm
        d.mkdir(parents=True, exist_ok=True)
        tr = [r for ti in train_t for r in rows[arm][ti]]
        random.Random(args.seed).shuffle(tr)
        tr = tr[:cap]
        va = [r for ti in valid_t for r in rows[arm][ti]]
        te = [r for ti in test_t for r in rows[arm][ti]]
        for name, data in (("train", tr), ("valid", va), ("test", te)):
            with (d / f"{name}.jsonl").open("w", encoding="utf-8") as f:
                for r in data:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary["arms"][arm] = {"train": len(tr), "valid": len(va), "test": len(te), "dir": str(d)}

    (out_root / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
