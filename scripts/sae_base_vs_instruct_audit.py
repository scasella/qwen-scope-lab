"""Audit: how much does pairing the BASE-trained Qwen-Scope SAE with the INSTRUCT model cost?

Two faithful, code-path-identical measurements (no fabrication — runs the real models + real SAE):

  Part A — SAE fidelity (base vs instruct), per-token at L12:
    * reconstruction FVU / explained-variance / cosine (the standard SAE health metric),
      computed under several forward conventions; the BASE model's own FVU reveals the SAE's
      true convention, which is then applied IDENTICALLY to the instruct model.
    * L0 (mean active features) — distribution-shift tell.
    * feature-activation AGREEMENT: same raw text → same token ids (shared tokenizer) → do the
      SAME features fire on base-resid vs instruct-resid? (Jaccard of top-K active + activation cosine.)
      This is what the lab actually relies on — it uses the SAE as an encoder, never to reconstruct.
    * raw residual cosine base-vs-instruct (the root-cause shift the SAE faces).

  Part B — the jailbreak SAE-feature-vs-residual-probe shootout, re-run on BOTH models via the
    SAME service.jailbreak_detection() path. The original "probe beats the SAE feature" number was
    produced on the instruct model (SAE off its home distribution); this re-runs it on the base model
    (the faithful pairing) so the comparison is fair. Only the model changes; the SAE is identical.

Usage:
    set -a; . ./.env; set +a
    python scripts/sae_base_vs_instruct_audit.py --out reports/sae_audit
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_scope_lab.mlx_backend import build_mlx_service

INSTRUCT = "mlx-community/Qwen3.5-2B-bf16"
BASE = "Qwen/Qwen3.5-2B-Base"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100"
D_SAE = 32768
LAYER = 12
TOP_K_SERVICE = 64

# Diverse neutral text — general-distribution prompts so FVU/L0 are representative (not just one genre).
SAMPLE_TEXTS = [
    "The capital of France is Paris, a city on the Seine.",
    "Water boils at 100 degrees Celsius at sea level.",
    "She opened the old wooden door and stepped into the dim hallway.",
    "To compute a mean, sum the values and divide by the count.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose.",
    "The quarterly report shows revenue grew twelve percent year over year.",
    "Could you explain how a binary search algorithm works?",
    "In 1969, astronauts first walked on the surface of the Moon.",
    "The recipe calls for two cups of flour and a pinch of salt.",
    "Gravity causes objects to accelerate toward the Earth at 9.8 meters per second squared.",
    "He tuned the guitar carefully before the evening performance.",
    "A prime number has exactly two distinct positive divisors.",
    "The committee will reconvene next Tuesday to finalize the budget.",
    "Rivers carry sediment downstream and deposit it in deltas.",
    "Please summarize the main argument of the second chapter.",
    "The museum's new exhibit features paintings from the Dutch Golden Age.",
    "Electrons carry a negative charge and orbit the atomic nucleus.",
    "The hikers reached the summit just as the sun broke through the clouds.",
    "Compound interest grows the principal faster than simple interest.",
    "The novel explores themes of memory, loss, and reconciliation.",
    "A balanced diet includes proteins, carbohydrates, and healthy fats.",
    "The server returned a 404 error because the page was missing.",
    "Bees communicate the location of flowers through a waggle dance.",
    "The treaty was signed after months of careful negotiation.",
    "Let me know if you would prefer the meeting in the morning or afternoon.",
    "The orchestra rehearsed the symphony's final movement twice.",
    "Tectonic plates shift slowly, reshaping continents over millions of years.",
    "The spreadsheet automatically recalculates totals when a cell changes.",
    "Migratory birds navigate using the Earth's magnetic field.",
    "The lecture covered the causes and consequences of the Industrial Revolution.",
    "A good password mixes letters, numbers, and symbols.",
    "The garden bloomed with tulips, daffodils, and crocuses in spring.",
    "Light travels approximately three hundred thousand kilometers per second.",
    "The startup pivoted to a subscription model after the first year.",
    "Could you translate this paragraph into clear, plain language?",
    "The bridge's suspension cables bear the weight of the roadway.",
    "Antibiotics treat bacterial infections but not viral ones.",
    "The chess player sacrificed a knight to open the king's defenses.",
    "Quarterly earnings beat analyst expectations across most divisions.",
    "The children built a sandcastle near the gentle, rolling waves.",
]


# --------------------------------------------------------------------------- SAE math
def load_sae_np() -> dict[str, np.ndarray]:
    """Load the raw SAE tensors (encoder convention matches sae_math: pre = x @ W_enc.T + b_enc)."""
    import torch
    from huggingface_hub import hf_hub_download
    from huggingface_hub.constants import HF_HUB_CACHE
    path = hf_hub_download(repo_id=SAE_REPO, filename=f"layer{LAYER}.sae.pt", cache_dir=HF_HUB_CACHE)
    state = torch.load(path, map_location="cpu")
    out = {k: state[k].detach().float().numpy() for k in ("W_enc", "b_enc", "W_dec", "b_dec")}
    assert out["W_enc"].shape == (D_SAE, out["W_enc"].shape[1]), out["W_enc"].shape
    assert out["W_dec"].shape == (out["W_enc"].shape[1], D_SAE), out["W_dec"].shape
    return out


def sae_acts(x: np.ndarray, sae: dict, *, presub: bool, mode: str, k: int = 100) -> np.ndarray:
    """x: [N, d_model] -> activations [N, d_sae]. mode in {relu, topk}. presub subtracts b_dec first."""
    z = (x - sae["b_dec"]) if presub else x
    pre = z @ sae["W_enc"].T + sae["b_enc"]              # [N, d_sae]
    if mode == "relu":
        return np.maximum(pre, 0.0)
    # topk: keep the k largest (post-relu) per row, zero the rest
    relu = np.maximum(pre, 0.0)
    if k >= relu.shape[1]:
        return relu
    idx = np.argpartition(relu, -k, axis=1)[:, -k:]
    out = np.zeros_like(relu)
    np.put_along_axis(out, idx, np.take_along_axis(relu, idx, axis=1), axis=1)
    return out


def reconstruct(x: np.ndarray, sae: dict, acts: np.ndarray) -> np.ndarray:
    return acts @ sae["W_dec"].T + sae["b_dec"]          # [N, d_model]


def fvu(x: np.ndarray, x_hat: np.ndarray) -> float:
    num = float(np.mean(np.sum((x - x_hat) ** 2, axis=1)))
    den = float(np.mean(np.sum((x - x.mean(0, keepdims=True)) ** 2, axis=1)))
    return num / den if den > 0 else float("nan")


def row_cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a, axis=1); nb = np.linalg.norm(b, axis=1)
    ok = (na > 0) & (nb > 0)
    if not ok.any():
        return float("nan")
    return float(np.mean(np.sum(a[ok] * b[ok], axis=1) / (na[ok] * nb[ok])))


def recon_metrics(x: np.ndarray, sae: dict, *, presub: bool, mode: str, k: int = 100) -> dict[str, float]:
    acts = sae_acts(x, sae, presub=presub, mode=mode, k=k)
    xh = reconstruct(x, sae, acts)
    return {"fvu": round(fvu(x, xh), 4), "explained_var": round(1 - fvu(x, xh), 4),
            "recon_cosine": round(row_cosine(x, xh), 4), "l0": round(float(np.mean(np.sum(acts > 0, axis=1))), 1)}


# --------------------------------------------------------------- per-model capture
def capture_resids(runtime: Any, texts: list[str], layer: int):
    """Return (stacked [N_tok, d_model] float32, per-text token-id lists, per-text [seq,d] arrays)."""
    per_text, ids_per_text, rows = [], [], []
    for t in texts:
        ids = runtime._encode_ids(t)
        cap = runtime._forward_capture(ids, layer)[0]    # mlx [seq, d_model]
        runtime._mx.eval(cap)
        arr = np.asarray(cap.tolist(), dtype=np.float32)  # [seq, d_model]
        per_text.append(arr); ids_per_text.append(list(ids)); rows.append(arr)
    return np.concatenate(rows, axis=0), ids_per_text, per_text


def free(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


# --------------------------------------------------------------- feature agreement
def feature_agreement(per_text_i: list[np.ndarray], per_text_b: list[np.ndarray], ids_i, ids_b,
                      sae: dict, *, presub: bool, mode: str, k: int, topn: int = 20) -> dict[str, float]:
    """For aligned tokens (shared tokenizer), compare which SAE features fire on instruct- vs base-resid."""
    jacc, acos = [], []
    aligned_tokens = 0
    for ai, ab, ii, ib in zip(per_text_i, per_text_b, ids_i, ids_b):
        m = min(len(ii), len(ib))
        if ii[:m] != ib[:m]:  # tokenizers diverged for this text — align by common prefix only
            common = 0
            for x, y in zip(ii, ib):
                if x != y:
                    break
                common += 1
            m = common
        if m == 0:
            continue
        acts_i = sae_acts(ai[:m], sae, presub=presub, mode=mode, k=k)
        acts_b = sae_acts(ab[:m], sae, presub=presub, mode=mode, k=k)
        aligned_tokens += m
        for r in range(m):
            ti = set(np.argsort(-acts_i[r])[:topn][acts_i[r][np.argsort(-acts_i[r])[:topn]] > 0])
            tb = set(np.argsort(-acts_b[r])[:topn][acts_b[r][np.argsort(-acts_b[r])[:topn]] > 0])
            if ti or tb:
                jacc.append(len(ti & tb) / len(ti | tb))
        acos.append(row_cosine(acts_i[:m], acts_b[:m]))
    return {"aligned_tokens": aligned_tokens, f"top{topn}_feature_jaccard": round(float(np.mean(jacc)), 4) if jacc else float("nan"),
            "activation_cosine": round(float(np.nanmean(acos)), 4) if acos else float("nan")}


# --------------------------------------------------------------------------- main
def run(out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    sae = load_sae_np()
    captured: dict[str, dict] = {}
    partB: dict[str, Any] = {}

    for tag, repo in (("instruct", INSTRUCT), ("base", BASE)):
        print(f"[load] {tag}: {repo} (base converts on first load — may be slow)", flush=True)
        service = build_mlx_service(repo, default_layer=LAYER, d_sae=D_SAE, sae_repo=SAE_REPO, top_k=TOP_K_SERVICE)
        runtime = service.bundle.model
        print(f"[partA] capturing L{LAYER} residuals on {len(SAMPLE_TEXTS)} texts ({tag})", flush=True)
        stacked, ids_pt, per_text = capture_resids(runtime, SAMPLE_TEXTS, LAYER)
        captured[tag] = {"stacked": stacked, "ids": ids_pt, "per_text": per_text}
        np.save(out / f"resid_{tag}_L{LAYER}.npy", stacked)
        print(f"[partB] jailbreak shootout via service.jailbreak_detection ({tag})", flush=True)
        try:
            jd = service.jailbreak_detection(layer=LAYER, use_judge=False)
            partB[tag] = jd
        except Exception as exc:  # noqa: BLE001
            partB[tag] = {"error": repr(exc)}
            print(f"[partB] ERROR ({tag}): {exc!r}", flush=True)
        free(service, runtime)

    # ---- Part A analysis ----
    xi, xb = captured["instruct"]["stacked"], captured["base"]["stacked"]
    conventions = [("relu_nopresub", dict(presub=False, mode="relu")),
                   ("relu_presub", dict(presub=True, mode="relu")),
                   ("topk100_nopresub", dict(presub=False, mode="topk", k=100)),
                   ("topk100_presub", dict(presub=True, mode="topk", k=100))]
    recon = {"base": {}, "instruct": {}}
    for name, kw in conventions:
        recon["base"][name] = recon_metrics(xb, sae, **kw)
        recon["instruct"][name] = recon_metrics(xi, sae, **kw)
    # the SAE's TRUE convention = the one that best reconstructs the BASE model (its training distribution)
    true_conv = min(conventions, key=lambda c: recon["base"][c[0]]["fvu"])[0]
    tk = dict(conventions)[true_conv]

    # residual shift base<->instruct on aligned tokens (root cause)
    n = min(len(xi), len(xb))
    # align per-token using the shared tokenizer alignment used in feature_agreement
    agree = feature_agreement(captured["instruct"]["per_text"], captured["base"]["per_text"],
                              captured["instruct"]["ids"], captured["base"]["ids"], sae, k=tk.get("k", 100),
                              presub=tk["presub"], mode=tk["mode"])
    # raw residual cosine on aligned tokens
    ri, rb = [], []
    for ai, ab, ii, ib in zip(captured["instruct"]["per_text"], captured["base"]["per_text"],
                              captured["instruct"]["ids"], captured["base"]["ids"]):
        m = min(len(ii), len(ib))
        common = 0
        for x, y in zip(ii, ib):
            if x != y:
                break
            common += 1
        m = common
        if m:
            ri.append(ai[:m]); rb.append(ab[:m])
    raw_cos = row_cosine(np.concatenate(ri), np.concatenate(rb)) if ri else float("nan")

    partA = {
        "layer": LAYER, "n_texts": len(SAMPLE_TEXTS), "n_tokens_instruct": int(len(xi)), "n_tokens_base": int(len(xb)),
        "reconstruction_by_convention": recon, "sae_true_convention_from_base_fvu": true_conv,
        "reconstruction_true_convention": {"base": recon["base"][true_conv], "instruct": recon["instruct"][true_conv],
                                           "fvu_delta_instruct_minus_base": round(recon["instruct"][true_conv]["fvu"] - recon["base"][true_conv]["fvu"], 4)},
        "feature_agreement_base_vs_instruct": agree,
        "raw_residual_cosine_base_vs_instruct": round(raw_cos, 4),
    }

    # ---- Part B summary ----
    def shoot(tag):
        jd = partB.get(tag, {})
        sh = jd.get("in_distribution", {}) if isinstance(jd, dict) else {}
        methods, verdict = sh.get("methods", {}), sh.get("verdict", {})
        return {"sae_auc": (methods.get("sae_monitor") or {}).get("auc"),
                "residual_diffmeans_auc": (methods.get("residual_diffmeans") or {}).get("auc"),
                "residual_logistic_auc": (methods.get("residual_logistic") or {}).get("auc"),
                "control_auc": (methods.get("random_control") or {}).get("auc"),
                "winner": verdict.get("winner"), "margin": verdict.get("margin"), "reason": verdict.get("reason"),
                "n_test_pos": sh.get("n_test_pos"), "n_test_neg": sh.get("n_test_neg"),
                "sae_features": (methods.get("sae_monitor") or {}).get("features")}
    partB_summary = {"instruct": shoot("instruct"), "base": shoot("base")}

    result = {"config": {"instruct": INSTRUCT, "base": BASE, "sae": SAE_REPO, "d_sae": D_SAE, "layer": LAYER},
              "partA_sae_fidelity": partA, "partB_jailbreak_shootout": partB_summary}
    (out / "sae_base_vs_instruct.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out / "sae_base_vs_instruct_FULL_partB.json").write_text(json.dumps(partB, indent=2, default=str) + "\n", encoding="utf-8")
    (out / "sae_base_vs_instruct.md").write_text(render_md(result), encoding="utf-8")
    return result


def render_md(r: dict) -> str:
    a = r["partA_sae_fidelity"]; tc = a["sae_true_convention_from_base_fvu"]
    rt = a["reconstruction_true_convention"]; ag = a["feature_agreement_base_vs_instruct"]
    b = r["partB_jailbreak_shootout"]
    def row(tag):
        m = rt[tag]
        return f"| {tag} | {m['fvu']} | {m['explained_var']} | {m['recon_cosine']} | {m['l0']} |"
    def brow(tag):
        s = b[tag]
        return f"| {tag} | {s['sae_auc']} | {s['residual_diffmeans_auc']} | {s['residual_logistic_auc']} | {s['control_auc']} | {s['winner']} | {s['margin']} |"
    return (
        "# Base-trained Qwen-Scope SAE on the instruct model — does the mismatch matter?\n\n"
        f"Models: instruct `{r['config']['instruct']}` vs base `{r['config']['base']}`; SAE `{r['config']['sae']}` (L{r['config']['layer']}, d_sae {r['config']['d_sae']}). "
        "Same SAE in every cell; only the model changes.\n\n"
        f"## Part A — SAE fidelity (per-token, L{a['layer']}, {a['n_texts']} neutral texts)\n\n"
        f"SAE's true forward convention (the one that best reconstructs the **base** model): **`{tc}`**.\n\n"
        "| model | FVU ↓ | explained var ↑ | recon cosine ↑ | L0 |\n|---|---|---|---|---|\n"
        f"{row('base')}\n{row('instruct')}\n\n"
        f"- **FVU delta (instruct − base): {rt['fvu_delta_instruct_minus_base']:+}** — higher FVU on instruct = the SAE reconstructs instruct activations worse (the mismatch cost).\n"
        f"- **Feature-activation agreement** (what the lab actually uses the SAE for — encoding, not reconstruction): "
        f"top-{20} feature Jaccard **{ag.get('top20_feature_jaccard')}**, activation cosine **{ag.get('activation_cosine')}** "
        f"(over {ag.get('aligned_tokens')} aligned tokens). 1.0 = identical features fire; lower = the features you read on instruct differ from the base ones the SAE was trained to.\n"
        f"- **Raw residual cosine base↔instruct: {a['raw_residual_cosine_base_vs_instruct']}** — how far the instruct activations themselves drifted from base (the root cause).\n"
        "- Full per-convention table in the JSON (`reconstruction_by_convention`).\n\n"
        "## Part B — jailbreak shootout (SAE feature vs raw-residual probe), re-run on each model\n\n"
        "Same `service.jailbreak_detection()` path; the original 'probe beats SAE' number was the instruct row.\n\n"
        "| model | SAE AUC | residual diff-means AUC | residual logistic AUC | random control | winner | margin (SAE−probe) |\n|---|---|---|---|---|---|---|\n"
        f"{brow('instruct')}\n{brow('base')}\n\n"
        f"- instruct: {b['instruct'].get('reason')}\n"
        f"- base: {b['base'].get('reason')}\n"
        f"- test split per class: instruct {b['instruct'].get('n_test_pos')}+/{b['instruct'].get('n_test_neg')}−, "
        f"base {b['base'].get('n_test_pos')}+/{b['base'].get('n_test_neg')}− (8/8 banks → coarse AUC, granularity ~0.06; read margins, not third decimals).\n"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="reports/sae_audit")
    args = ap.parse_args(argv)
    res = run(Path(args.out))
    print(json.dumps({"partA": {"true_convention": res["partA_sae_fidelity"]["sae_true_convention_from_base_fvu"],
                                "recon": res["partA_sae_fidelity"]["reconstruction_true_convention"],
                                "feature_agreement": res["partA_sae_fidelity"]["feature_agreement_base_vs_instruct"],
                                "raw_residual_cosine": res["partA_sae_fidelity"]["raw_residual_cosine_base_vs_instruct"]},
                      "partB": res["partB_jailbreak_shootout"]}, indent=2))


if __name__ == "__main__":
    main()
