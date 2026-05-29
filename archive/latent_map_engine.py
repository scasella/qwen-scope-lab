"""ARCHIVED — the dead "latent space map" engine (decoder-cosine 2D projection).

These were methods on `SteeringService` (qwen_scope_steering_gui/service.py) that backed
the removed `POST /api/layout` route and the Explore→Map view. They project the SAE's
decoder directions (`W_dec` columns) to 2D (t-SNE / PCA), KMeans-cluster them, and overlay
a prompt's per-feature activation.

WHY THIS IS ARCHIVED, NOT SHIPPED
---------------------------------
The premise — that the SAE's decoder geometry encodes a navigable "map of meaning" — was
tested on the real model and failed decisively:
  * real 2B L12: decoder-direction silhouette ≈ 0.003 (near-isotropic, no clustering);
    co-activating features land no closer than random (concentration ratio ≈ 0.97).
  * real 27B L32: silhouette ≈ 0.0016 (even more isotropic).
The co-activation and Ising-coupling follow-ups (see research_probes.py) confirmed it: any
apparent structure is organized by SYNTAX/POSITION, not semantics. A global SAE-feature map
is dead. The project pivoted to steering along concept manifolds in the RESIDUAL stream,
which ARE real and replicate across 2B/27B. See docs/MANIFOLD.md and the project memory.

Preserved verbatim (as standalone functions; they were `SteeringService` methods) for
provenance. `research_probes.py::_latent_map` called `latent_layout`; `_coactivation_map`
called `_project_2d`. NOT executable as-is — depends on a live model + SAE loader.
"""
from __future__ import annotations

from typing import Any


def latent_layout(
    self,
    prompt: str | None = None,
    layer: int | None = None,
    method: str = "tsne",
    max_features: int = 400,
    n_clusters: int = 8,
) -> dict[str, Any]:
    """Project SAE decoder directions to 2D + cluster them, then overlay the
    prompt's per-feature activation. The prompt's active features are always
    included in the laid-out set (so they are visible at real width); the rest of
    the budget is filled with the highest-norm features. Geometry (coords +
    clusters) is cached by (layer, method, selected-set signature)."""
    import hashlib

    import numpy as np
    import torch

    layer = self.config.default_layer if layer is None else int(layer)
    method = "pca" if str(method).lower() == "pca" else "tsne"
    self.ensure_model()
    sae = self.sae_loader.load_layer(layer)

    active: dict[int, float] = {}
    if prompt and prompt.strip():
        insp = self.inspect_prompt(prompt, layer=layer, top_k=self.config.top_k)
        for row in insp["top_features_by_token"]:
            for f in row["features"]:
                fid = int(f["feature_id"])
                active[fid] = max(active.get(fid, 0.0), float(f["activation"]))

    norms = _feature_norms(self, layer, sae)  # (d_sae,)
    d_sae = int(norms.shape[0])
    if d_sae <= max_features:
        sel = list(range(d_sae))
    else:
        top = np.argsort(-norms)[:max_features].tolist()
        sel = sorted(set(top) | {fid for fid in active if 0 <= fid < d_sae})

    sig = (layer, method, hashlib.sha1(",".join(map(str, sel)).encode()).hexdigest()[:16])
    geom = self._layout_cache.get(sig)
    if geom is None:
        directions = sae.W_dec.detach().to(dtype=torch.float32, device="cpu").t().numpy()  # (d_sae, d_model)
        X = directions[sel]
        coords = _project_2d(X, method)
        k = max(1, min(int(n_clusters), len(sel)))
        if k > 1 and len(sel) > k:
            from sklearn.cluster import KMeans

            cluster_ids = KMeans(n_clusters=k, n_init=10, random_state=7).fit_predict(X).tolist()
        else:
            cluster_ids = [0] * len(sel)
        geom = {"coords": coords.tolist(), "clusters": cluster_ids}
        self._layout_cache[sig] = geom

    features = []
    cluster_acc: dict[int, list[float]] = {}
    for (fid, (x, y), c) in zip(sel, geom["coords"], geom["clusters"]):
        a = float(active.get(int(fid), 0.0))
        features.append({"feature_id": int(fid), "x": round(float(x), 4), "y": round(float(y), 4),
                         "cluster": int(c), "activation": round(a, 4), "norm": round(float(norms[fid]), 4)})
        cluster_acc.setdefault(int(c), [0.0, 0.0, 0])
        cluster_acc[int(c)][0] += float(x)
        cluster_acc[int(c)][1] += float(y)
        cluster_acc[int(c)][2] += 1
    clusters = [{"cluster": c, "x": round(v[0] / v[2], 4), "y": round(v[1] / v[2], 4), "size": int(v[2])}
                for c, v in sorted(cluster_acc.items())]

    return {
        "layer": layer,
        "method": method,
        "d_sae": d_sae,
        "n_features": len(features),
        "prompt": prompt or "",
        "active_count": len(active),
        "features": features,
        "clusters": clusters,
    }


def _feature_norms(self, layer: int, sae) -> "Any":
    import numpy as np
    import torch

    cached = self._norm_cache.get(layer)
    if cached is None:
        w = sae.W_dec.detach().to(dtype=torch.float32, device="cpu").numpy()  # (d_model, d_sae)
        cached = np.linalg.norm(w, axis=0)  # per-feature decoder norm
        self._norm_cache[layer] = cached
    return cached


def _project_2d(x, method: str):
    import numpy as np

    n = int(x.shape[0])
    if method == "pca" or n < 5:
        from sklearn.decomposition import PCA

        y = PCA(n_components=2, random_state=7).fit_transform(x)
    else:
        from sklearn.manifold import TSNE

        perplexity = max(5.0, min(30.0, (n - 1) / 3.0))
        y = TSNE(n_components=2, random_state=7, init="pca", perplexity=perplexity).fit_transform(x)
    y = np.asarray(y, dtype=float)
    y = y - y.mean(axis=0, keepdims=True)
    scale = np.abs(y).max(axis=0)
    scale[scale < 1e-9] = 1.0
    return y / scale  # each axis normalised to roughly [-1, 1]
