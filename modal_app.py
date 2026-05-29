from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import modal


app = modal.App("qwen-scope-steering-gui")

hf_cache = modal.Volume.from_name("qwen-scope-hf-cache", create_if_missing=True)
modal_secret = modal.Secret.from_dotenv(__file__) if Path(".env").exists() else modal.Secret.from_dict({})
GUI_SCALEDOWN_WINDOW_SECONDS = 300
GUI_TARGET_ENV_VAR = "QWEN_GUI_TARGET"
DEFAULT_GUI_TARGET = "2b-l4"


@dataclass(frozen=True)
class GuiProfile:
    name: str
    config_path: str
    gpu: str
    cpu: int
    memory: int
    description: str
    aliases: tuple[str, ...] = ()


GUI_PROFILES: dict[str, GuiProfile] = {
    "2b-l4": GuiProfile(
        name="2b-l4",
        config_path="/root/configs/qwen35_2b_dev_l0_100.yaml",
        gpu="L4",
        cpu=4,
        memory=32768,
        description="cheap 2B dev GUI on L4",
        aliases=("2b", "l4", "dev"),
    ),
    "27b-a100": GuiProfile(
        name="27b-a100",
        config_path="/root/configs/qwen35_27b_l0_100.yaml",
        gpu="A100-80GB",
        cpu=8,
        memory=131072,
        description="lower-cost real 27B GUI on A100 80GB",
        aliases=("27b", "a100", "a100-80gb"),
    ),
    "27b-h100": GuiProfile(
        name="27b-h100",
        config_path="/root/configs/qwen35_27b_l0_100.yaml",
        gpu="H100",
        cpu=8,
        memory=131072,
        description="faster/headroom real 27B GUI on H100",
        aliases=("h100",),
    ),
}
GUI_TARGET_ALIASES = {
    alias: name
    for name, profile in GUI_PROFILES.items()
    for alias in (name, *profile.aliases)
}


def _normalize_gui_target(target: str | None) -> str:
    return (target or DEFAULT_GUI_TARGET).strip().lower().replace("_", "-")


def select_gui_profile(target: str | None = None) -> GuiProfile:
    normalized = _normalize_gui_target(target)
    profile_name = GUI_TARGET_ALIASES.get(normalized)
    if profile_name is None:
        choices = ", ".join(sorted(GUI_PROFILES))
        aliases = ", ".join(sorted(alias for alias in GUI_TARGET_ALIASES if alias not in GUI_PROFILES))
        raise ValueError(
            f"Unsupported {GUI_TARGET_ENV_VAR}={target!r}. "
            f"Choose one of: {choices}. Aliases: {aliases}."
        )
    return GUI_PROFILES[profile_name]


SELECTED_GUI_PROFILE = select_gui_profile(os.environ.get(GUI_TARGET_ENV_VAR))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate",
        "fastapi",
        "gradio",
        "httpx",
        "huggingface-hub",
        "numpy",
        "pandas",
        "python-dotenv",
        "pyyaml",
        "safetensors",
        "scikit-learn",
        "torch",
        "transformers",
        "uvicorn",
    )
    .add_local_dir("qwen_scope_steering_gui", remote_path="/root/qwen_scope_steering_gui")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir("scripts", remote_path="/root/scripts")
    .add_local_dir("web", remote_path="/root/web")
    .add_local_file("app.py", remote_path="/root/app.py")
    .add_local_file("serve_web.py", remote_path="/root/serve_web.py")
)


def _smoke(config_path: str, layer: int, max_new_tokens: int) -> dict:
    from qwen_scope_steering_gui.service import SteeringService

    service = SteeringService.from_config_path(config_path)
    prompt = "The capital of France is"
    inspection = service.inspect_prompt(prompt, layer=layer, top_k=3, max_seq_len=32)
    feature_id = int(inspection["top_features_by_token"][-1]["features"][0]["feature_id"])
    steering = service.steer(
        "Write one sentence about Paris.",
        layer=layer,
        feature_id=feature_id,
        strength=5.0,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        mode="all_positions",
    )
    result = {
        "config": config_path,
        "layer": layer,
        "selected_feature_id": feature_id,
        "inspection_metadata": inspection["metadata"],
        "hook_fired": steering["hook_fired"],
        "hidden_delta_norm": steering["hidden_delta_norm"],
        "logits_delta_norm": steering["logits_delta_norm"],
        "unsteered_text": steering["unsteered_text"],
        "steered_text": steering["steered_text"],
    }
    print(json.dumps(result, indent=2))
    try:
        hf_cache.commit()
    except Exception:
        pass
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=3600,
    retries=0,
    volumes={"/cache": hf_cache},
    secrets=[modal_secret],
)
def smoke_2b() -> dict:
    return _smoke("/root/configs/qwen35_2b_dev_l0_100.yaml", layer=12, max_new_tokens=16)


@app.function(
    image=image,
    # H100 is the first choice for the 27B path because one 27B BF16 model plus
    # one W80K SAE layer is an 80GB-class inference job.
    gpu="H100",
    cpu=8,
    memory=131072,
    timeout=5400,
    retries=0,
    volumes={"/cache": hf_cache},
    secrets=[modal_secret],
)
def smoke_27b() -> dict:
    return _smoke("/root/configs/qwen35_27b_l0_100.yaml", layer=32, max_new_tokens=12)


def _web_parity(config_path: str, layer: int, max_new_tokens: int) -> dict:
    """Exercise every web_api endpoint against the REAL model via TestClient.

    Bounded: loads weights once, tiny generation budgets, then returns. Proves the
    Lab Bench web layer (web_gui) reaches parity with the dev backend on real Qwen.
    """
    from fastapi.testclient import TestClient

    from qwen_scope_steering_gui.service import SteeringService
    from qwen_scope_steering_gui.web_api import create_app

    service = SteeringService.from_config_path(config_path)
    client = TestClient(create_app(service, recipes_root="/root/recipes"))
    out: dict = {"config": config_path, "layer": layer, "checks": {}}

    out["model_id"] = client.get("/api/status").json()["configured_model_id"]

    insp = client.post("/api/inspect", json={"prompt": "The capital of France is", "layer": layer, "top_k": 8, "max_seq_len": 32}).json()
    top = insp["top_features_by_token"][-1]["features"][0]
    feat = int(top["feature_id"])
    out["checks"]["inspect"] = {"tokens": insp["tokens"], "top_feature": feat, "top_activation": round(float(top["activation"]), 3)}

    cmp = client.post("/api/compare", json={"positive": "Write a concise factual answer.", "negative": "Write a long rambling story.", "layer": layer, "limit": 5}).json()
    out["checks"]["compare"] = {"positive_stronger": [r["feature_id"] for r in cmp["positive_stronger"][:5]]}

    atl = client.post("/api/atlas", json={"prompts": ["The capital of France is Paris.", "Return a JSON object with name and age."], "layer": layer, "top_k": 8}).json()
    out["checks"]["atlas"] = {"n_prompts": atl["n_prompts"], "features_mapped": len(atl["features"]), "top_feature": atl["features"][0]["feature_id"] if atl["features"] else None}

    steer = client.post("/api/steer", json={"prompt": "Write one sentence about Paris.", "layer": layer, "feature_id": feat, "strength": 6.0, "max_new_tokens": max_new_tokens, "temperature": 0.0}).json()
    out["checks"]["steer"] = {
        "hook_fired": steer["hook_fired"],
        "hidden_delta_norm": steer["hidden_delta_norm"],
        "logits_delta_norm": steer["logits_delta_norm"],
        "unsteered_text": steer["unsteered_text"],
        "steered_text": steer["steered_text"],
    }

    sw = client.post("/api/sweep", json={"prompt": "Write one sentence about Paris.", "layer": layer, "feature_id": feat, "strengths": [0, 6], "max_new_tokens": max_new_tokens}).json()
    out["checks"]["sweep"] = [{"strength": f["strength"], "text": f["text"]} for f in sw["frames"]]

    bench = client.post("/api/benchmark", json={"prompt_set": '{"id":"b1","prompt":"Explain sparse autoencoders in one sentence."}', "feature_id": feat, "strength": 6.0, "layer": layer, "target_behavior": "concise", "max_new_tokens": max_new_tokens}).json()
    out["checks"]["benchmark"] = {"methods": len(bench["methods"]), "verdict": bench["validation_decision"]["status"], "scores": {k: round(float(v), 2) for k, v in bench["method_scores"].items()}}

    ap = client.post("/api/autopilot", json={
        "positive_examples": "Paris is the capital of France.",
        "negative_examples": "Once upon a time there was a long rambling tale.",
        "validation_prompts": '{"id":"v1","prompt":"What is the capital of France?"}',
        "candidate_count": 1, "candidate_layers": [layer], "max_new_tokens": max(4, max_new_tokens // 2),
    }).json()
    out["checks"]["autopilot"] = {"candidates": len(ap.get("candidates", [])), "best_feature": ap.get("best_candidate", {}).get("feature_id"), "recipe_id": ap.get("recipe_id"), "verdict": ap.get("validation_decision", {}).get("status")}

    out["parity_ok"] = bool(
        insp["tokens"]
        and out["checks"]["steer"]["hook_fired"]
        and out["checks"]["steer"]["hidden_delta_norm"] > 0
        and out["checks"]["benchmark"]["methods"] == 7
        and out["checks"]["atlas"]["features_mapped"] > 0
        and out["checks"]["autopilot"]["candidates"] >= 1
    )
    print(json.dumps(out, indent=2))
    try:
        hf_cache.commit()
    except Exception:
        pass
    return out


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=3600,
    retries=0,
    volumes={"/cache": hf_cache},
    secrets=[modal_secret],
)
def web_parity_2b() -> dict:
    return _web_parity("/root/configs/qwen35_2b_dev_l0_100.yaml", layer=12, max_new_tokens=12)


def _spearman(a, b):
    import numpy as np

    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float((ra @ rb) / d) if d > 0 else 0.0


def _manifold_metrics(X, labels, n_items: int, kind: str, k_pca: int = 8, seed: int = 7):
    """(var_top3, ordering_metric) for per-class centroids of X. Ordinal -> best |Spearman|
    of top-3 PCs vs index; cyclic -> ring neighbor-adjacency on top-2 PCs."""
    import numpy as np
    from sklearn.decomposition import PCA

    k = max(2, min(k_pca, X.shape[0] - 1, X.shape[1]))
    pca = PCA(n_components=k, random_state=seed)
    Y = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    C = np.array([Y[labels == ci].mean(0) for ci in range(n_items)])
    idx = np.arange(n_items)
    if kind == "cyclic":
        cc = C[:, :2] - C[:, :2].mean(0)
        dd = np.linalg.norm(cc[:, None, :] - cc[None, :, :], axis=2)
        np.fill_diagonal(dd, np.inf)
        nn = dd.argmin(1)
        metric = float(np.mean([(nn[i] == (i - 1) % n_items) or (nn[i] == (i + 1) % n_items) for i in range(n_items)]))
    else:
        metric = float(max(abs(_spearman(C[:, j], idx)) for j in range(min(3, C.shape[1]))))
    return round(float(var[:3].sum()), 4), round(metric, 4)


# Concepts with MULTIPLE carrier templates per item (fixed-prefix, item at the end so
# the last sub-token carries it). Multiple templates -> per-class centroids over varied
# contexts, matching causalab's centroid construction. Includes a negative control whose
# ordinal index is arbitrary (the ordering metric should come back ~0). Shared: the live
# _residual_manifold_sweep below uses this; the archived _residual_manifold_probe (see
# archive/research_probes.py) also referenced it.
_CONCEPTS_V2 = [
    {"name": "integers_0_20", "kind": "ordinal", "items": [str(i) for i in range(21)],
     "templates": ["The number is {item}", "I counted {item}", "There were {item}",
                   "It costs {item} dollars", "Page {item}", "Chapter {item}"]},
    {"name": "size", "kind": "ordinal",
     "items": ["tiny", "small", "medium", "large", "huge", "enormous"],
     "templates": ["The box was {item}", "It felt {item}", "A {item} thing",
                   "The size was {item}", "Quite {item}", "Remarkably {item}"]},
    {"name": "days_of_week", "kind": "cyclic",
     "items": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
     "templates": ["Today is {item}", "The meeting is on {item}", "See you {item}",
                   "It happened last {item}", "Every {item}", "By {item}"]},
    {"name": "months", "kind": "cyclic",
     "items": ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"],
     "templates": ["The month is {item}", "We met in {item}", "It was a cold {item}",
                   "Born in {item}", "Since last {item}", "By {item}"]},
    {"name": "random_control", "kind": "control",
     "items": ["lamp", "river", "honest", "beneath", "guitar", "whisper", "cement", "orbit"],
     "templates": ["Here is {item}", "Think of {item}", "The next word is {item}",
                   "Consider {item}", "I noticed {item}", "About {item}"]},
]


def _residual_manifold_sweep(config_path: str, layers: list[int], k_pca: int = 8, seed: int = 7) -> dict:
    """Residual-only manifold sweep across layers (no SAE -> no per-layer downloads). Captures
    every swept layer's residual in ONE forward pass per prompt, then scores each concept's
    centroid geometry per layer to find where concept manifolds peak."""
    import json as _json
    import warnings

    import numpy as np
    import torch

    warnings.filterwarnings("ignore")

    from qwen_scope_steering_gui.hooks import register_capture_hook
    from qwen_scope_steering_gui.service import SteeringService

    service = SteeringService.from_config_path(config_path)
    bundle = service.ensure_model()
    tokenizer, model, device = bundle.tokenizer, bundle.model, bundle.device
    n_layers = int(service.config.num_layers)
    layers = [L for L in layers if 0 <= L < n_layers]

    prompts = []  # (concept_idx, item_idx, text)
    for cidx, concept in enumerate(_CONCEPTS_V2):
        for iidx, it in enumerate(concept["items"]):
            for tmpl in concept["templates"]:
                prompts.append((cidx, iidx, tmpl.format(item=it)))

    res = {L: [] for L in layers}
    meta = []
    for cidx, iidx, text in prompts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)
        caps = {L: {} for L in layers}
        handles = [register_capture_hook(model, L, caps[L], to_cpu=False) for L in layers]
        try:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn)
        finally:
            for h in handles:
                h.remove()
        for L in layers:
            res[L].append(caps[L]["residual"][0].float()[-1].detach().cpu().numpy())
        meta.append((cidx, iidx))
    meta = np.asarray(meta)

    per_layer = {}
    for L in layers:
        X_all = np.asarray(res[L], dtype=np.float32)
        per_concept = {}
        for cidx, concept in enumerate(_CONCEPTS_V2):
            mask = meta[:, 0] == cidx
            v3, order = _manifold_metrics(X_all[mask], meta[mask, 1], len(concept["items"]), concept["kind"], k_pca, seed)
            per_concept[concept["name"]] = {"kind": concept["kind"], "var_top3": v3, "order_metric": order}
        real = [(n, c) for n, c in per_concept.items() if c["kind"] in ("ordinal", "cyclic")]
        present = [n for n, c in real if c["order_metric"] >= 0.80 and c["var_top3"] >= 0.50]
        ctrl = per_concept["random_control"]["order_metric"]
        per_layer[str(L)] = {"concepts": per_concept, "n_present": len(present), "present": present,
                             "mean_order": round(float(np.mean([c["order_metric"] for _, c in real])), 4),
                             "control": ctrl}

    best = max(layers, key=lambda L: (per_layer[str(L)]["n_present"], per_layer[str(L)]["mean_order"] - per_layer[str(L)]["control"]))
    out = {"config": config_path, "model_id": service.config.model_id, "num_layers": n_layers,
           "layers_swept": layers, "k_pca": k_pca, "probe": "residual_manifold_sweep",
           "per_layer": per_layer, "best_layer": best, "best_summary": per_layer[str(best)]}
    print(_json.dumps(out, indent=2))
    try:
        hf_cache.commit()
    except Exception:
        pass
    return out


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=3600,
    retries=0,
    volumes={"/cache": hf_cache},
    secrets=[modal_secret],
)
def residual_manifold_sweep_2b() -> dict:
    return _residual_manifold_sweep("/root/configs/qwen35_2b_dev_l0_100.yaml",
                                    layers=[4, 6, 8, 10, 12, 14, 16, 18, 20, 22], k_pca=8)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=131072,
    timeout=5400,
    retries=0,
    volumes={"/cache": hf_cache},
    secrets=[modal_secret],
)
def residual_manifold_sweep_27b() -> dict:
    # 64-layer model: bracket late-mid depth (~0.375-0.75) where the 2B manifolds peaked.
    return _residual_manifold_sweep("/root/configs/qwen35_27b_l0_100.yaml",
                                    layers=[24, 32, 40, 48], k_pca=8)


def _manifold_steer_demo(config_path: str, layer: int, concept: str, source: str, target: str,
                         n_waypoints: int = 7, max_new_tokens: int = 24) -> dict:
    """Real-model manifold steering: fit the concept manifold (residual centroids -> PCA ->
    spline) and steer along it by replacing the concept token's residual with manifold points,
    printing the behavioral trajectory. The honest, geometry-grounded steering feature."""
    import json as _json

    from qwen_scope_steering_gui.service import SteeringService

    service = SteeringService.from_config_path(config_path)
    fit = service.manifold_fit(concept, layer)
    steer = service.manifold_steer(concept, target, layer=layer, source=source,
                                   n_waypoints=n_waypoints, max_new_tokens=max_new_tokens)
    out = {
        "config": config_path, "model_id": service.config.model_id, "layer": layer, "concept": concept,
        "fit_quality": fit["quality"], "synthetic": fit["synthetic"],
        "source": steer["source"], "target": steer["target"], "position": steer["position"],
        "prompt": steer["prompt"], "hook_fired": steer["hook_fired"],
        "unsteered_text": steer["unsteered_text"], "steered_text": steer["steered_text"],
        "trajectory": [{"value": w["value"], "text": w["text"]} for w in steer["waypoints"]],
    }
    print(_json.dumps(out, indent=2))
    try:
        hf_cache.commit()
    except Exception:
        pass
    return out


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=3600,
    retries=0,
    volumes={"/cache": hf_cache},
    secrets=[modal_secret],
)
def manifold_steer_demo_2b() -> dict:
    return _manifold_steer_demo("/root/configs/qwen35_2b_dev_l0_100.yaml", layer=14,
                                concept="days_of_week", source="Monday", target="Thursday")


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=131072,
    timeout=5400,
    retries=0,
    volumes={"/cache": hf_cache},
    secrets=[modal_secret],
)
def manifold_steer_demo_27b() -> dict:
    return _manifold_steer_demo("/root/configs/qwen35_27b_l0_100.yaml", layer=48,
                                concept="days_of_week", source="Monday", target="Thursday")


def _manifold_vs_linear(config_path: str, layer: int, concept: str, source: str, target: str) -> dict:
    """Print manifold-path vs linear-path steering side by side (the paper's headline:
    manifold stays natural / lower perplexity; linear cuts off-manifold)."""
    import json as _json

    from qwen_scope_steering_gui.service import SteeringService

    service = SteeringService.from_config_path(config_path)
    cmp = service.manifold_compare(concept, target, layer=layer, source=source, n_waypoints=7, max_new_tokens=24)
    out = {
        "config": config_path, "model_id": service.config.model_id, "layer": layer, "concept": concept,
        "source": cmp["source"], "target": cmp["target"], "prompt": cmp["prompt"],
        "unsteered_text": cmp["unsteered_text"],
        "manifold": {"steered_text": cmp["manifold"]["steered_text"], "perplexity": cmp["manifold"]["perplexity"],
                     "mean_perplexity": cmp["manifold"]["mean_perplexity"]},
        "linear": {"steered_text": cmp["linear"]["steered_text"], "perplexity": cmp["linear"]["perplexity"],
                   "mean_perplexity": cmp["linear"]["mean_perplexity"]},
        "manifold_trajectory": [{"value": w["value"], "text": w["text"]} for w in cmp["manifold"]["waypoints"]],
        "linear_trajectory": [{"value": w["value"], "text": w["text"]} for w in cmp["linear"]["waypoints"]],
    }
    print(_json.dumps(out, indent=2))
    try:
        hf_cache.commit()
    except Exception:
        pass
    return out


def _manifold_atlas(config_path: str, layers: list[int]) -> dict:
    """Census: for every candidate continuous concept, find the layer where its residual
    manifold is cleanest. Residual-only (no SAE); all layers captured per forward pass."""
    import json as _json
    import warnings

    import numpy as np
    import torch

    warnings.filterwarnings("ignore")

    from qwen_scope_steering_gui.concept_presets import ATLAS_CONCEPTS
    from qwen_scope_steering_gui.hooks import register_capture_hook
    from qwen_scope_steering_gui.service import SteeringService

    service = SteeringService.from_config_path(config_path)
    bundle = service.ensure_model()
    tok, model, device = bundle.tokenizer, bundle.model, bundle.device
    n_layers = int(service.config.num_layers)
    layers = [L for L in layers if 0 <= L < n_layers]

    atlas = []
    for concept in ATLAS_CONCEPTS:
        res = {L: [] for L in layers}
        labels = []
        for ci, item in enumerate(concept.items):
            for tmpl in concept.templates:
                enc = tok(tmpl.format(item=item), return_tensors="pt", truncation=True, max_length=32)
                input_ids = enc["input_ids"].to(device)
                attn = enc.get("attention_mask")
                if attn is not None:
                    attn = attn.to(device)
                caps = {L: {} for L in layers}
                handles = [register_capture_hook(model, L, caps[L], to_cpu=False) for L in layers]
                try:
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attn)
                finally:
                    for h in handles:
                        h.remove()
                for L in layers:
                    res[L].append(caps[L]["residual"][0].float()[-1].detach().cpu().numpy())
                labels.append(ci)
        labels = np.asarray(labels)
        n = len(concept.items)
        best = {"best_layer": None, "best_metric": -1.0, "var_top3": None}
        for L in layers:
            v3, order = _manifold_metrics(np.asarray(res[L], dtype=np.float32), labels, n, concept.kind)
            if order > best["best_metric"]:
                best = {"best_layer": L, "best_metric": order, "var_top3": v3}
        verdict = ("clean" if best["best_metric"] >= 0.8 and (best["var_top3"] or 0) >= 0.5
                   else "partial" if best["best_metric"] >= 0.6 else "diffuse")
        atlas.append({"concept": concept.name, "label": concept.label, "kind": concept.kind,
                      "n_items": n, **best, "verdict": verdict})

    atlas.sort(key=lambda a: -a["best_metric"])
    out = {"config": config_path, "model_id": service.config.model_id, "num_layers": n_layers,
           "layers_swept": layers, "n_concepts": len(atlas), "atlas": atlas}
    print(_json.dumps(out, indent=2))
    try:
        hf_cache.commit()
    except Exception:
        pass
    return out


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=5400,
    retries=0,
    volumes={"/cache": hf_cache},
    secrets=[modal_secret],
)
def manifold_vs_linear_2b() -> dict:
    return _manifold_vs_linear("/root/configs/qwen35_2b_dev_l0_100.yaml", layer=14,
                               concept="days_of_week", source="Monday", target="Thursday")


def _manifold_naturalness_probe(config_path: str, specs: list[tuple]) -> dict:
    """The decisive test: does manifold steering beat linear on the PAPER'S metric —
    distance of the output distribution to the behavior manifold ℳ_y (not raw perplexity)?
    Also reports the activation↔behavior isometry r per concept."""
    import json as _json

    import numpy as np

    from qwen_scope_steering_gui.service import SteeringService

    service = SteeringService.from_config_path(config_path)

    def cumdist(pts, metric):
        n = len(pts)
        seg = [metric(pts[k], pts[k + 1]) for k in range(n - 1)]
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        return np.abs(cum[:, None] - cum[None, :])

    eucl = lambda a, b: float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
    hell = lambda a, b: float(np.linalg.norm(np.sqrt(np.clip(a, 0, None)) - np.sqrt(np.clip(b, 0, None))) / np.sqrt(2))

    rows = []
    for concept, layer, source, target in specs:
        try:
            c = service.manifold_compare(concept, target, layer=layer, source=source, n_waypoints=9, max_new_tokens=8)
            m_e, l_e = c["manifold"]["mean_energy"], c["linear"]["mean_energy"]
            mk = service._manifold_cache.get((concept, layer))
            bk = service._behavior_cache.get((concept, layer))
            iso = None
            if mk is not None and bk is not None:
                d_h = cumdist(np.asarray(mk["centroids_pca"]), eucl)
                d_y = cumdist(np.asarray(bk["centroids"]), hell)
                iu = np.triu_indices(d_h.shape[0], 1)
                a, bb = d_h[iu], d_y[iu]
                if a.std() > 0 and bb.std() > 0:
                    iso = float(np.corrcoef(a, bb)[0, 1])
            rows.append({"concept": concept, "layer": layer,
                         "manifold_energy": m_e, "linear_energy": l_e,
                         "energy_gap": round(l_e - m_e, 4) if (m_e is not None and l_e is not None) else None,
                         "manifold_more_faithful": bool(m_e is not None and l_e is not None and m_e < l_e),
                         "isometry_r": round(iso, 4) if iso is not None else None})
        except Exception as exc:  # noqa: BLE001
            rows.append({"concept": concept, "error": str(exc)})

    out = {"config": config_path, "model_id": service.config.model_id,
           "n_manifold_more_faithful": sum(1 for r in rows if r.get("manifold_more_faithful")),
           "results": rows}
    print(_json.dumps(out, indent=2))
    try:
        hf_cache.commit()
    except Exception:
        pass
    return out


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=5400,
    retries=0,
    volumes={"/cache": hf_cache},
    secrets=[modal_secret],
)
def manifold_naturalness_probe_2b() -> dict:
    return _manifold_naturalness_probe("/root/configs/qwen35_2b_dev_l0_100.yaml", [
        ("integers_0_20", 8, "0", "20"),
        ("rank", 20, "private", "general"),
        ("education", 8, "kindergarten", "doctorate"),
        ("valence", 16, "miserable", "ecstatic"),
        ("size", 16, "tiny", "enormous"),
        ("agreement", 8, "strongly disagree", "strongly agree"),
        ("days_of_week", 14, "Monday", "Thursday"),
    ])


def _manifold_pullback_probe(config_path: str, specs: list[tuple]) -> dict:
    """Pullback test (paper §pullback): optimize the activation path that induces the target
    ℳ_y behavior. Does it (a) induce on-manifold behavior (energy ≤ linear) and (b) RECOVER ℳ_h
    (recovered_r ≫ linear's)? + pullback LBFGS loss start→end (autograd sanity)."""
    import json as _json

    from qwen_scope_steering_gui.service import SteeringService

    service = SteeringService.from_config_path(config_path)
    rows = []
    for concept, layer, source, target in specs:
        try:
            r = service.manifold_pullback(concept, target, layer=layer, source=source,
                                          n_waypoints=5, max_new_tokens=12, lbfgs_iters=25)
            m, lin, pb = r["manifold"], r["linear"], r["pullback"]
            rows.append({
                "concept": concept, "layer": layer,
                "energy": {"manifold": m["mean_energy"], "linear": lin["mean_energy"], "pullback": pb["mean_energy"]},
                "recovered_r": {"manifold": m["recovered_r"], "linear": lin["recovered_r"], "pullback": pb["recovered_r"]},
                "pullback_loss": [pb["loss_start"], pb["loss_end"]],
                "pullback_energy_le_linear": bool(pb["mean_energy"] is not None and lin["mean_energy"] is not None and pb["mean_energy"] <= lin["mean_energy"]),
                "pullback_recovers_manifold": bool(pb["recovered_r"] is not None and lin["recovered_r"] is not None and pb["recovered_r"] > lin["recovered_r"]),
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({"concept": concept, "error": str(exc)})
    out = {"config": config_path, "model_id": service.config.model_id,
           "n_pullback_energy_le_linear": sum(1 for r in rows if r.get("pullback_energy_le_linear")),
           "n_pullback_recovers_manifold": sum(1 for r in rows if r.get("pullback_recovers_manifold")),
           "results": rows}
    print(_json.dumps(out, indent=2))
    try:
        hf_cache.commit()
    except Exception:
        pass
    return out


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=5400,
    retries=0,
    volumes={"/cache": hf_cache},
    secrets=[modal_secret],
)
def manifold_pullback_probe_2b() -> dict:
    return _manifold_pullback_probe("/root/configs/qwen35_2b_dev_l0_100.yaml", [
        ("days_of_week", 14, "Monday", "Thursday"),
        ("rank", 20, "private", "general"),
        ("education", 8, "kindergarten", "doctorate"),
    ])


_MONITOR_SETS = {
    "sentiment": (
        ["I love this, it's wonderful and makes me so happy.", "What a fantastic, delightful experience.",
         "This is amazing; I'm thrilled.", "Absolutely beautiful work, I adore it.",
         "A joyful, uplifting, brilliant result.", "I'm so pleased and excited.",
         "Best day ever, everything is perfect.", "Such a kind and generous gesture, thank you."],
        ["I hate this, it's awful and makes me miserable.", "What a terrible, disappointing experience.",
         "This is horrible; I'm furious.", "Absolutely ugly work, I despise it.",
         "A depressing, miserable, broken result.", "I'm so upset and annoyed.",
         "Worst day ever, everything is ruined.", "Such a cruel and selfish act, shame on you."]),
    "refusal": (
        ["I'm sorry, but I can't help with that request.", "I won't be able to assist with this.",
         "That's not something I'm able to do.", "I cannot provide that information.",
         "Unfortunately I have to decline this request.", "I'm not able to help with that, sorry.",
         "I must refuse to answer this question.", "Apologies, that request is something I can't fulfill."],
        ["Sure, here's how you can do it step by step.", "Absolutely, let me explain the process.",
         "Of course! The steps are as follows.", "Yes, I'd be happy to help with that.",
         "Certainly, here is the information you asked for.", "Great question, here's a detailed answer.",
         "Definitely, I can walk you through it.", "Happy to help, let's get started right away."]),
    "pii": (
        ["My name is John Smith and my email is john.smith@gmail.com.", "Call me at 415-555-0192 or write to 22 Oak Street.",
         "Her SSN is 123-45-6789 and DOB is 04/12/1988.", "Contact alice.jones@company.org, phone (212) 555-7788.",
         "The patient Robert Lee lives at 9 Elm Ave, ZIP 90210.", "My credit card is 4111 1111 1111 1111, exp 09/27.",
         "Reach David at +44 7700 900123 or david@uk.co.", "Account holder Maria Garcia, routing 021000021."],
        ["The weather today is sunny with a light breeze.", "Photosynthesis converts sunlight into energy.",
         "The meeting covered quarterly strategy.", "Mountains form through tectonic collisions.",
         "The recipe needs flour, sugar, and two eggs.", "Our train departs from the station at noon.",
         "The river flows gently past the old mill.", "Gravity causes objects to accelerate downward."]),
}


def _monitor_demo(config_path: str, layer: int = 12) -> dict:
    """Feature-as-monitor reproducibility check on the real model: discover a detector for each
    behavior and report held-out AUC / F1 + the random-feature control + verdict."""
    import json as _json

    from qwen_scope_steering_gui.service import SteeringService

    service = SteeringService.from_config_path(config_path)
    rows = []
    for name, (pos, neg) in _MONITOR_SETS.items():
        try:
            r = service.discover_monitor(pos, neg, layer=layer, top_k=3)
            m = r["metrics"]
            rows.append({"behavior": name, "layer": layer, "features": r["features"],
                         "auc": m["auc"], "f1": m["f1"], "control_auc": m["control_auc"],
                         "verdict": r["validation_decision"]["status"]})
        except Exception as exc:  # noqa: BLE001
            rows.append({"behavior": name, "error": str(exc)})
    out = {"config": config_path, "model_id": service.config.model_id, "layer": layer,
           "n_validated": sum(1 for r in rows if r.get("verdict") == "validated"), "results": rows}
    print(_json.dumps(out, indent=2))
    try:
        hf_cache.commit()
    except Exception:
        pass
    return out


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=5400,
    retries=0,
    volumes={"/cache": hf_cache},
    secrets=[modal_secret],
)
def monitor_demo_2b() -> dict:
    return _monitor_demo("/root/configs/qwen35_2b_dev_l0_100.yaml", layer=12)


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=5400,
    retries=0,
    volumes={"/cache": hf_cache},
    secrets=[modal_secret],
)
def manifold_atlas_2b() -> dict:
    return _manifold_atlas("/root/configs/qwen35_2b_dev_l0_100.yaml", layers=[6, 8, 10, 12, 14, 16, 18, 20])


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=131072,
    timeout=7200,
    retries=0,
    volumes={"/cache": hf_cache},
    secrets=[modal_secret],
)
def manifold_atlas_27b() -> dict:
    return _manifold_atlas("/root/configs/qwen35_27b_l0_100.yaml", layers=[24, 32, 40, 48, 56])


def _bench_smoke(config_path: str, layer: int, max_new_tokens: int, prompt_count: int, recipe_name: str) -> dict:
    from pathlib import Path

    from qwen_scope_steering_gui.benchmark import ServiceGenerationBackend, attach_benchmark_to_recipe, run_benchmark, save_benchmark_result
    from qwen_scope_steering_gui.recipe_schema import FeatureRecipe, Intervention, ModelMetadata, TargetBehavior
    from qwen_scope_steering_gui.recipe_store import RecipeStore
    from qwen_scope_steering_gui.service import SteeringService

    service = SteeringService.from_config_path(config_path)
    inspection = service.inspect_prompt("The capital of France is", layer=layer, top_k=3, max_seq_len=32)
    feature_id = int(inspection["top_features_by_token"][-1]["features"][0]["feature_id"])
    recipe = FeatureRecipe.create(
        target_behavior=TargetBehavior(name=recipe_name, description="Compact Modal benchmark smoke recipe."),
        model=ModelMetadata(
            model_id=service.config.model_id,
            sae_id=service.config.sae_id,
            dtype=service.config.torch_dtype,
            config_name=config_path,
        ),
        interventions=[Intervention(layer=layer, feature_id=feature_id, strength=4.0)],
        created_by="qwen-scope-modal-smoke",
    )
    prompts = [
        {"id": "m001", "prompt": "Explain sparse autoencoders in one sentence."},
        {"id": "m002", "prompt": "Return a JSON object with key city and value Paris."},
    ][:prompt_count]
    result = run_benchmark(
        recipe,
        prompts,
        ServiceGenerationBackend(service),
        prompt_only_instruction="Answer directly. Prompt: {prompt}",
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        seed=0,
        objective="maximize_rule_score",
        prompt_set_id="modal_smoke_inline",
    )
    recipe = attach_benchmark_to_recipe(recipe, result)
    report_dir = Path("/root/reports")
    recipe_dir = Path("/root/recipes")
    benchmark_path = save_benchmark_result(result, report_dir / f"{recipe.recipe_id}_bench.json")
    RecipeStore(recipe_dir).save(recipe, benchmark_results=result, examples=recipe.examples)
    try:
        hf_cache.commit()
    except Exception:
        pass
    summary = {
        "recipe_id": recipe.recipe_id,
        "config": config_path,
        "layer": layer,
        "selected_feature_id": feature_id,
        "benchmark_path": benchmark_path,
        "recipe_json_path": str(recipe_dir / recipe.recipe_id / "recipe.json"),
        "methods": result["methods"],
        "validation_decision": result["validation_decision"],
        "hook_fired": any(
            metrics.get("hook_fired")
            for row in result["per_prompt_results"]
            for method, metrics in row["metrics"].items()
            if method in {"steering_only", "prompt_plus_steering"}
        ),
        "max_hidden_delta_norm": max(
            [
                metrics.get("hidden_delta_norm", 0.0)
                for row in result["per_prompt_results"]
                for method, metrics in row["metrics"].items()
                if method in {"steering_only", "prompt_plus_steering"}
            ]
            or [0.0]
        ),
    }
    print(json.dumps(summary, indent=2))
    return summary


def _autopilot_smoke(config_path: str, layer: int, max_new_tokens: int, candidate_count: int, recipe_dir_name: str) -> dict:
    from pathlib import Path

    from qwen_scope_steering_gui.autopilot import run_autopilot
    from qwen_scope_steering_gui.benchmark import ServiceGenerationBackend
    from qwen_scope_steering_gui.service import SteeringService

    service = SteeringService.from_config_path(config_path)
    result = run_autopilot(
        config=service.config,
        config_path=config_path,
        target_name=recipe_dir_name,
        target_description="Produce direct, compact outputs for Modal smoke testing.",
        positive_examples=["A sparse autoencoder learns sparse features.", '{"city":"Paris"}'],
        negative_examples=["Here is a long rambling answer with many caveats.", "```json\n{\"city\":\"Paris\"}\n```"],
        validation_prompts=[{"id": "a001", "prompt": "Return a JSON object with key city and value Paris."}],
        candidate_layers=[layer],
        candidate_count=candidate_count,
        objective="maximize_rule_score",
        backend=ServiceGenerationBackend(service),
        service=service,
        output_dir=Path("/root/recipes") / recipe_dir_name,
        prompt_only_instruction="Answer directly. Prompt: {prompt}",
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        seed=0,
    )
    try:
        hf_cache.commit()
    except Exception:
        pass
    summary = {
        "config": config_path,
        "best_candidate": result["best_candidate"],
        "output_paths": result["output_paths"],
        "validation_decision": result["benchmark"]["validation_decision"],
        "hook_fired": any(
            metrics.get("hook_fired")
            for row in result["benchmark"]["per_prompt_results"]
            for method, metrics in row["metrics"].items()
            if method in {"steering_only", "prompt_plus_steering"}
        ),
        "max_hidden_delta_norm": max(
            [
                metrics.get("hidden_delta_norm", 0.0)
                for row in result["benchmark"]["per_prompt_results"]
                for method, metrics in row["metrics"].items()
                if method in {"steering_only", "prompt_plus_steering"}
            ]
            or [0.0]
        ),
        "warning": result["warning"],
    }
    print(json.dumps(summary, indent=2))
    return summary


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=3600,
    retries=0,
    volumes={"/cache": hf_cache},
    secrets=[modal_secret],
)
def bench_smoke_2b() -> dict:
    return _bench_smoke("/root/configs/qwen35_2b_dev_l0_100.yaml", layer=12, max_new_tokens=6, prompt_count=2, recipe_name="modal_2b_bench_smoke")


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=3600,
    retries=0,
    volumes={"/cache": hf_cache},
    secrets=[modal_secret],
)
def autopilot_smoke_2b() -> dict:
    return _autopilot_smoke("/root/configs/qwen35_2b_dev_l0_100.yaml", layer=12, max_new_tokens=4, candidate_count=2, recipe_dir_name="modal_2b_autopilot_smoke")


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=131072,
    timeout=5400,
    retries=0,
    volumes={"/cache": hf_cache},
    secrets=[modal_secret],
)
def bench_smoke_27b() -> dict:
    return _bench_smoke("/root/configs/qwen35_27b_l0_100.yaml", layer=32, max_new_tokens=4, prompt_count=2, recipe_name="modal_27b_bench_smoke")


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=131072,
    timeout=5400,
    retries=0,
    volumes={"/cache": hf_cache},
    secrets=[modal_secret],
)
def autopilot_smoke_27b() -> dict:
    return _autopilot_smoke("/root/configs/qwen35_27b_l0_100.yaml", layer=32, max_new_tokens=3, candidate_count=1, recipe_dir_name="modal_27b_autopilot_smoke")


def _mount_gradio(config_path: str):
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    from app import build_demo

    fastapi_app = FastAPI()
    demo = build_demo(config_path)
    return mount_gradio_app(app=fastapi_app, blocks=demo, path="/")


def _mount_web(config_path: str):
    from qwen_scope_steering_gui.service import SteeringService
    from qwen_scope_steering_gui.web_api import create_app

    service = SteeringService.from_config_path(config_path)
    return create_app(service, recipes_root="/root/recipes", experiments_root="/root/experiments",
                      monitors_root="/root/monitors")


@app.function(
    image=image,
    gpu=SELECTED_GUI_PROFILE.gpu,
    cpu=SELECTED_GUI_PROFILE.cpu,
    memory=SELECTED_GUI_PROFILE.memory,
    timeout=7200,
    scaledown_window=GUI_SCALEDOWN_WINDOW_SECONDS,
    max_containers=1,
    volumes={"/cache": hf_cache},
    secrets=[modal_secret],
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def gradio_gui():
    print(
        json.dumps(
            {
                "gui": "gradio",
                "gui_target": SELECTED_GUI_PROFILE.name,
                "description": SELECTED_GUI_PROFILE.description,
                "gpu": SELECTED_GUI_PROFILE.gpu,
                "config_path": SELECTED_GUI_PROFILE.config_path,
            },
            indent=2,
        )
    )
    return _mount_gradio(SELECTED_GUI_PROFILE.config_path)


@app.function(
    image=image,
    gpu=SELECTED_GUI_PROFILE.gpu,
    cpu=SELECTED_GUI_PROFILE.cpu,
    memory=SELECTED_GUI_PROFILE.memory,
    timeout=7200,
    scaledown_window=GUI_SCALEDOWN_WINDOW_SECONDS,
    max_containers=1,
    volumes={"/cache": hf_cache},
    secrets=[modal_secret],
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web_gui():
    # The Lab Bench: same SteeringService backend as gradio_gui, served through the
    # FastAPI web UI in web/. Selected by the same QWEN_GUI_TARGET env var.
    print(
        json.dumps(
            {
                "gui": "lab-bench",
                "gui_target": SELECTED_GUI_PROFILE.name,
                "description": SELECTED_GUI_PROFILE.description,
                "gpu": SELECTED_GUI_PROFILE.gpu,
                "config_path": SELECTED_GUI_PROFILE.config_path,
            },
            indent=2,
        )
    )
    return _mount_web(SELECTED_GUI_PROFILE.config_path)
