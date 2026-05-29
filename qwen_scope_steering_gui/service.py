from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .activations import extract_prompt_features
from .concept_presets import get_concept, preset_summaries
from .config import SteeringConfig, config_to_dict, load_config
from .env import load_environment
from .feature_compare import contrast_features
from .feature_labels import label_feature
from .feature_selection import select_active_feature
from .generation import manifold_generate, sequence_perplexity, steer_generation
from .hooks import register_capture_hook
from .model_loader import ModelBundle, gpu_memory_summary, load_model
from .notebook import load_notebook, save_notebook_entry
from .sae_loader import LazySAELoader


class SteeringService:
    def __init__(self, config: SteeringConfig, config_path: str | Path):
        self.config = config
        self.config_path = str(config_path)
        self.bundle: ModelBundle | None = None
        self.sae_loader = LazySAELoader(config)
        self._manifold_cache: dict[tuple[str, int], Any] = {}
        self._behavior_cache: dict[tuple, Any] = {}

    @classmethod
    def from_config_path(cls, config_path: str | Path, env_path: str | Path | None = ".env") -> "SteeringService":
        load_environment(env_path)
        return cls(load_config(config_path), config_path)

    def ensure_model(self) -> ModelBundle:
        if self.bundle is None:
            self.bundle = load_model(self.config)
        return self.bundle

    def inspect_prompt(self, prompt: str, layer: int | None = None, top_k: int | None = None, max_seq_len: int | None = None) -> dict[str, Any]:
        layer = self.config.default_layer if layer is None else int(layer)
        bundle = self.ensure_model()
        sae = self.sae_loader.load_layer(layer)
        return extract_prompt_features(bundle, sae, self.config, prompt, layer, top_k, max_seq_len)

    def compare_prompts(self, positive_prompt: str, negative_prompt: str, layer: int | None = None, limit: int = 20) -> dict[str, Any]:
        layer = self.config.default_layer if layer is None else int(layer)
        positive = self.inspect_prompt(positive_prompt, layer=layer, top_k=self.config.top_k)
        negative = self.inspect_prompt(negative_prompt, layer=layer, top_k=self.config.top_k)
        return {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "layer": layer,
            **contrast_features(positive, negative, limit),
        }

    def steer(
        self,
        prompt: str,
        layer: int | None,
        feature_id: int,
        strength: float,
        max_new_tokens: int | None = None,
        temperature: float = 0.7,
        mode: str = "all_positions",
        compute_logits_delta: bool = True,
    ) -> dict[str, Any]:
        layer = self.config.default_layer if layer is None else int(layer)
        bundle = self.ensure_model()
        sae = self.sae_loader.load_layer(layer)
        return steer_generation(
            bundle=bundle,
            sae=sae,
            config=self.config,
            prompt=prompt,
            layer=layer,
            feature_id=int(feature_id),
            strength=float(strength),
            max_new_tokens=max_new_tokens or self.config.default_max_new_tokens,
            temperature=float(temperature),
            mode=mode,
            compute_logits_delta=compute_logits_delta,
        )

    def auto_steer(self, prompt: str, layer: int | None, strength: float, max_new_tokens: int, temperature: float) -> dict[str, Any]:
        inspection = self.inspect_prompt(prompt, layer=layer, top_k=1)
        selected = select_active_feature(inspection)
        result = self.steer(prompt, layer, selected["feature_id"], strength, max_new_tokens, temperature)
        result["auto_feature_source"] = selected
        return result

    # ------------------------------- behavior monitors -------------------------------
    def discover_monitor(self, positive: list[str], negative: list[str], layer: int | None = None, top_k: int = 3) -> dict[str, Any]:
        from . import monitor as _mon
        pos = [p for p in (positive or []) if p and p.strip()]
        neg = [n for n in (negative or []) if n and n.strip()]
        if not pos or not neg:
            raise ValueError("provide at least one positive and one negative example")
        layer = self.config.default_layer if layer is None else int(layer)
        pos_maps = [_mon.activation_map(self.inspect_prompt(t, layer=layer, top_k=40)) for t in pos]
        neg_maps = [_mon.activation_map(self.inspect_prompt(t, layer=layer, top_k=40)) for t in neg]
        result = _mon.discover(pos_maps, neg_maps, top_k=int(top_k), d_sae=getattr(self.config, "d_sae", None))
        result["layer"] = layer
        return result

    def score_monitor(self, text: str, features: list[int], layer: int | None, threshold: float) -> dict[str, Any]:
        from . import monitor as _mon
        layer = self.config.default_layer if layer is None else int(layer)
        amap = _mon.activation_map(self.inspect_prompt(text, layer=layer, top_k=40))
        return _mon.score([int(f) for f in features], float(threshold), amap)

    # ------------------------------- manifold steering -------------------------------
    def manifold_presets(self) -> dict[str, Any]:
        return {"presets": preset_summaries(), "default_layer": self.config.default_layer}

    def _capture_last_residual(self, prompt: str, layer: int):
        import torch

        bundle = self.ensure_model()
        enc = bundle.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = enc["input_ids"].to(bundle.device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(bundle.device)
        cap: dict = {}
        handle = register_capture_hook(bundle.model, layer, cap, to_cpu=False)
        try:
            with torch.no_grad():
                bundle.model(input_ids=input_ids, attention_mask=attn)
        finally:
            handle.remove()
        return cap["residual"][0].float()[-1].detach().cpu().numpy()  # [d_model]

    def _build_manifold(self, concept_name: str, layer: int | None):
        import numpy as np
        from scipy.interpolate import CubicSpline
        from sklearn.decomposition import PCA

        concept = get_concept(concept_name)
        if layer is None:
            bl = concept.best_layer  # atlas-derived; use only when valid for the loaded model (dev model has few layers)
            layer = bl if (bl is not None and 0 <= bl < self.config.num_layers) else self.config.default_layer
        else:
            layer = int(layer)
        key = (concept.name, layer)
        cached = self._manifold_cache.get(key)
        if cached is not None:
            return cached, concept, layer

        rows, labels = [], []
        for ci, item in enumerate(concept.items):
            for tmpl in concept.templates:
                rows.append(self._capture_last_residual(tmpl.format(item=item), layer))
                labels.append(ci)
        rows = np.asarray(rows, dtype=np.float64)
        labels = np.asarray(labels)
        n = len(concept.items)
        centroids_dmodel = np.asarray([rows[labels == ci].mean(0) for ci in range(n)])
        k = max(2, min(64, rows.shape[0] - 1, rows.shape[1]))
        pca = PCA(n_components=k, random_state=7).fit(rows)
        centroids_pca = pca.transform(centroids_dmodel)

        if concept.kind == "cyclic":
            u_nodes = np.arange(n + 1, dtype=float)
            spline = CubicSpline(u_nodes, np.vstack([centroids_pca, centroids_pca[:1]]), bc_type="periodic", axis=0)
            u_min, u_max = 0.0, float(n)
        else:
            spline = CubicSpline(np.arange(n, dtype=float), centroids_pca, bc_type="natural", axis=0)
            u_min, u_max = 0.0, float(n - 1)

        u_dense = np.linspace(u_min, u_max, 160)
        pts3 = np.vstack([centroids_pca[:, :3], spline(u_dense)[:, :3]])
        center = pts3.mean(0)
        scale = float(np.abs(pts3 - center).max()) or 1.0
        manifold = {
            "concept": concept.name, "kind": concept.kind, "layer": layer, "items": list(concept.items),
            "n_items": n, "pca": pca, "spline": spline, "centroids_pca": centroids_pca,
            "centroids_dmodel": centroids_dmodel,
            "u_values": list(range(n)), "u_min": u_min, "u_max": u_max, "u_dense": u_dense,
            "center3": center, "scale3": scale, "synthetic": self.config.model_id.startswith("dev/"),
        }
        self._manifold_cache[key] = manifold
        return manifold, concept, layer

    def _u_to_3d(self, manifold: dict, u: float) -> list[float]:
        import numpy as np

        n = manifold["n_items"]
        if manifold["synthetic"]:  # clean ring/line so the 3D UI is developable on the toy model
            if manifold["kind"] == "cyclic":
                a = 2 * np.pi * (float(u) % n) / n
                return [float(np.cos(a)), float(np.sin(a)), 0.0]
            t = (float(u) / (n - 1)) if n > 1 else 0.0
            return [float(2 * t - 1), 0.0, 0.0]
        p3 = np.asarray(manifold["spline"](float(u)))[:3]
        return [float(v) for v in (p3 - manifold["center3"]) / manifold["scale3"]]

    def _manifold_quality(self, manifold: dict) -> dict[str, Any]:
        import numpy as np

        c = manifold["centroids_pca"]
        n = manifold["n_items"]
        idx = np.arange(n)
        if manifold["kind"] == "cyclic":
            xy = c[:, :2] - c[:, :2].mean(0)
            dd = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
            np.fill_diagonal(dd, np.inf)
            nn = dd.argmin(1)
            metric = float(np.mean([(nn[i] == (i - 1) % n) or (nn[i] == (i + 1) % n) for i in range(n)]))
            name = "ring_adjacency"
        else:
            def _rank(a):
                return np.argsort(np.argsort(a)).astype(float)
            best = 0.0
            for j in range(min(3, c.shape[1])):
                ra, rb = _rank(c[:, j]) - (n - 1) / 2, idx - (n - 1) / 2
                d = float(np.linalg.norm(ra) * np.linalg.norm(rb))
                best = max(best, abs(float((ra @ rb) / d)) if d else 0.0)
            metric, name = best, "abs_spearman"
        return {"metric_name": name, "metric": round(metric, 4)}

    def manifold_fit(self, concept: str, layer: int | None = None) -> dict[str, Any]:
        manifold, concept_obj, layer = self._build_manifold(concept, layer)
        n = manifold["n_items"]
        points_3d = [{"value": manifold["items"][i], "index": i, "xyz": self._u_to_3d(manifold, i)} for i in range(n)]
        return {
            "concept": concept_obj.name, "label": concept_obj.label, "kind": manifold["kind"],
            "layer": layer, "n_items": n, "items": manifold["items"], "synthetic": manifold["synthetic"],
            "points_3d": points_3d, "curve_3d": [self._u_to_3d(manifold, u) for u in manifold["u_dense"]],
            "u_min": manifold["u_min"], "u_max": manifold["u_max"], "steer_prompt": concept_obj.steer_prompt,
            "quality": self._manifold_quality(manifold),
        }

    @staticmethod
    def _locate_item_position(tokenizer, steer_prompt: str, item: str, prompt: str) -> int:
        prefix = steer_prompt.split("{item}")[0]
        plen2 = len(tokenizer(prefix + item)["input_ids"])
        total = len(tokenizer(prompt)["input_ids"])
        return max(0, min(plen2 - 1, total - 1))

    def manifold_steer(self, concept: str, target: str, layer: int | None = None, source: str | None = None,
                       prompt: str | None = None, n_waypoints: int = 7, max_new_tokens: int = 24,
                       temperature: float = 0.0, path: str = "manifold", compute_unsteered: bool = True,
                       compute_energy: bool = False) -> dict[str, Any]:
        import numpy as np
        import torch

        manifold, concept_obj, layer = self._build_manifold(concept, layer)
        bundle = self.ensure_model()
        items, n = manifold["items"], manifold["n_items"]
        path = "linear" if str(path).lower() == "linear" else "manifold"
        beh = self._build_behavior_manifold(concept_obj, layer) if compute_energy else None

        def to_index(v, default):
            if v is None:
                return default
            if v in items:
                return items.index(v)
            try:
                iv = int(v)
            except (TypeError, ValueError):
                raise ValueError(f"unknown value {v!r} for concept {concept_obj.name}")
            if not 0 <= iv < n:
                raise ValueError(f"index {iv} out of range for concept {concept_obj.name}")
            return iv

        src_i, tgt_i = to_index(source, 0), to_index(target, n - 1)
        prompt = prompt or concept_obj.steer_prompt.format(item=items[src_i])
        position = self._locate_item_position(bundle.tokenizer, concept_obj.steer_prompt, items[src_i], prompt)

        if manifold["kind"] == "cyclic":  # traverse the short way around the ring
            d = (tgt_i - src_i) % n
            if d > n / 2:
                d -= n
            us = [src_i + d * t for t in np.linspace(0, 1, max(2, n_waypoints))]
        else:
            us = list(np.linspace(float(src_i), float(tgt_i), max(2, n_waypoints)))

        spline, pca, cpca = manifold["spline"], manifold["pca"], manifold["centroids_pca"]
        src3, tgt3 = self._u_to_3d(manifold, src_i), self._u_to_3d(manifold, tgt_i)
        steps = len(us)
        waypoints, unsteered, path_3d = [], None, []
        for wi, u in enumerate(us):
            t = wi / (steps - 1) if steps > 1 else 1.0
            if path == "linear":  # straight chord through ambient space (Euclidean) — cuts off-manifold
                pca_pt = (1 - t) * cpca[src_i] + t * cpca[tgt_i]
                p3 = [(1 - t) * a + t * b for a, b in zip(src3, tgt3)]
                lbl = items[int(round((1 - t) * src_i + t * tgt_i)) % n]
            else:                 # follow the fitted manifold (spline)
                uu = (float(u) % n) if manifold["kind"] == "cyclic" else float(u)
                pca_pt = spline(uu)
                p3 = self._u_to_3d(manifold, uu)
                lbl = items[int(round(uu)) % n]
            replacement = torch.tensor(pca.inverse_transform(np.asarray(pca_pt).reshape(1, -1))[0], dtype=torch.float32)
            gen = manifold_generate(bundle, prompt, layer, replacement, position, max_new_tokens, temperature,
                                    compute_unsteered=(compute_unsteered and wi == 0))
            if compute_unsteered and wi == 0:
                unsteered = gen["unsteered_text"]
            ppl = sequence_perplexity(bundle, prompt, gen["steered_text"])
            wp = {"value": lbl, "text": gen["steered_text"],
                  "perplexity": round(ppl, 3) if ppl is not None else None, "hook_fired": gen["hook_fired"]}
            if beh is not None:
                q = self._output_distribution(prompt, layer, replacement, position, beh["token_ids"])
                wp["energy"] = round(self._behavior_energy(beh, q), 4)
            waypoints.append(wp)
            path_3d.append([round(float(c), 4) for c in p3])

        steered_text = waypoints[-1]["text"]
        ppls = [w["perplexity"] for w in waypoints if w["perplexity"] is not None]
        energies = [w["energy"] for w in waypoints if w.get("energy") is not None]
        return {
            "concept": concept_obj.name, "kind": manifold["kind"], "layer": layer, "prompt": prompt, "path": path,
            "position": position, "source": items[src_i], "target": items[tgt_i],
            "unsteered_text": unsteered, "steered_text": steered_text,
            "perplexity": waypoints[-1]["perplexity"],
            "mean_perplexity": round(sum(ppls) / len(ppls), 3) if ppls else None,  # raw fluency of the whole path
            "mean_energy": round(sum(energies) / len(energies), 4) if energies else None,  # distance to behavior manifold (lower=more faithful)
            "unsteered_perplexity": sequence_perplexity(bundle, prompt, unsteered) if unsteered else None,
            "waypoints": waypoints, "path_3d": path_3d,
            "hook_fired": all(w["hook_fired"] for w in waypoints),
        }

    def manifold_compare(self, concept: str, target: str, layer: int | None = None, source: str | None = None,
                         prompt: str | None = None, n_waypoints: int = 7, max_new_tokens: int = 24,
                         temperature: float = 0.0) -> dict[str, Any]:
        """Run manifold-path AND linear-path steering from the same source to target and
        return both (with perplexity) — the paper's manifold-vs-linear comparison."""
        m = self.manifold_steer(concept, target, layer, source, prompt, n_waypoints, max_new_tokens,
                                temperature, path="manifold", compute_unsteered=True, compute_energy=True)
        lin = self.manifold_steer(concept, target, layer, source, m["prompt"], n_waypoints, max_new_tokens,
                                  temperature, path="linear", compute_unsteered=False, compute_energy=True)
        lin["unsteered_text"] = m["unsteered_text"]
        lin["unsteered_perplexity"] = m["unsteered_perplexity"]
        return {
            "concept": m["concept"], "kind": m["kind"], "layer": m["layer"], "prompt": m["prompt"],
            "source": m["source"], "target": m["target"],
            "unsteered_text": m["unsteered_text"], "unsteered_perplexity": m["unsteered_perplexity"],
            "manifold": m, "linear": lin,
        }

    def manifold_sae_coverage(self, concept: str, layer: int | None = None, top_k: int = 5) -> dict[str, Any]:
        """Which SAE atoms tile each point of the concept manifold (the paper's 'features
        tile the manifold'). For each value's centroid, the top-k SAE features by activation."""
        from collections import Counter

        import numpy as np
        import torch

        from .sae_math import topk_features

        manifold, concept_obj, layer = self._build_manifold(concept, layer)
        sae = self.sae_loader.load_layer(layer)
        we = sae.W_enc.to(dtype=torch.float32)
        be = sae.b_enc.to(dtype=torch.float32)

        labels: dict[int, str] = {}
        try:
            for en in (self.notebook().get("features") or []):
                if en.get("feature_id") is not None and en.get("human_label"):
                    labels[int(en["feature_id"])] = en["human_label"]
        except Exception:
            pass

        cents = manifold["centroids_dmodel"]
        per_value, cover = [], {}
        for i, item in enumerate(manifold["items"]):
            vec = torch.tensor(np.asarray(cents[i]), dtype=torch.float32, device=we.device)
            vals, idx = topk_features(vec, we, be, top_k)
            feats = [{"feature_id": int(fid), "activation": round(float(a), 4), "label": labels.get(int(fid))}
                     for a, fid in zip(vals.detach().cpu().tolist(), idx.detach().cpu().tolist())]
            per_value.append({"value": item, "index": i, "xyz": self._u_to_3d(manifold, i),
                              "dominant_feature": (feats[0]["feature_id"] if feats else None), "features": feats})
            for f in feats:
                cover.setdefault(f["feature_id"], []).append(item)

        tiling = sorted(
            ({"feature_id": fid, "label": labels.get(fid), "covers": vlist, "n_values": len(vlist)}
             for fid, vlist in cover.items()),
            key=lambda x: -x["n_values"])
        return {
            "concept": concept_obj.name, "kind": manifold["kind"], "layer": layer,
            "synthetic": manifold["synthetic"], "n_items": manifold["n_items"], "top_k": top_k,
            "per_value": per_value, "tiling": tiling, "n_distinct_features": len(cover),
        }

    # --- behavior manifold ℳ_y (paper-faithful naturalness: distance of the output
    #     distribution to the fitted behavior manifold, not raw perplexity) ---
    def _concept_token_ids(self, concept) -> list[int]:
        cached = self._behavior_cache.get(("ids", concept.name))
        if cached is not None:
            return cached
        tok = self.ensure_model().tokenizer
        ids = [int(tok(" " + v, add_special_tokens=False)["input_ids"][0]) for v in concept.items]
        self._behavior_cache[("ids", concept.name)] = ids
        return ids

    def _output_distribution(self, prompt, layer, replacement, position, token_ids):
        import numpy as np
        import torch

        from .hooks import HookTrace, register_replace_hook

        bundle = self.ensure_model()
        enc = bundle.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = enc["input_ids"].to(bundle.device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(bundle.device)
        handle = register_replace_hook(bundle.model, layer, replacement, position, HookTrace()) if replacement is not None else None
        try:
            with torch.no_grad():
                logits = bundle.model(input_ids=input_ids, attention_mask=attn).logits[0, -1].float()
        finally:
            if handle is not None:
                handle.remove()
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        vocab = probs.shape[0]
        sub = np.array([probs[t] if 0 <= t < vocab else 0.0 for t in token_ids], dtype=float)  # guard toy-model OOV ids
        total = sub.sum()
        return sub / total if total > 0 else np.full(len(token_ids), 1.0 / len(token_ids))

    def _build_behavior_manifold(self, concept, layer: int):
        import numpy as np
        from scipy.interpolate import CubicSpline

        key = (concept.name, layer)
        cached = self._behavior_cache.get(key)
        if cached is not None:
            return cached
        import torch

        bundle = self.ensure_model()
        tok = bundle.tokenizer
        token_ids = self._concept_token_ids(concept)
        n = len(concept.items)
        probe = tok("the", return_tensors="pt")["input_ids"].to(bundle.device)
        with torch.no_grad():
            vocab = int(bundle.model(input_ids=probe).logits.shape[-1])
        valid_pos = [i for i, t in enumerate(token_ids) if 0 <= t < vocab]   # toy-model guard
        valid_ids = [token_ids[i] for i in valid_pos]
        P = []
        for v in concept.items:
            prompt = concept.steer_prompt.format(item=v)
            pos = self._locate_item_position(tok, concept.steer_prompt, v, prompt)
            P.append(self._output_distribution(prompt, layer, None, pos, token_ids))
        P = np.asarray(P)                       # (n, n) behavior centroids (unintervened)
        sq = np.sqrt(np.clip(P, 0.0, None))     # Hellinger coordinates
        if concept.kind == "cyclic":
            spline = CubicSpline(np.arange(n + 1, dtype=float), np.vstack([sq, sq[:1]]), bc_type="periodic", axis=0)
            u_min, u_max = 0.0, float(n)
        else:
            spline = CubicSpline(np.arange(n, dtype=float), sq, bc_type="natural", axis=0)
            u_min, u_max = 0.0, float(n - 1)
        u_dense = np.linspace(u_min, u_max, 120)
        dense = np.clip(spline(u_dense), 0.0, None) ** 2
        dense = dense / dense.sum(axis=1, keepdims=True).clip(1e-9)
        beh = {"token_ids": token_ids, "centroids": P, "dense_p": dense, "n": n,
               "spline": spline, "u_min": u_min, "u_max": u_max,
               "valid_pos": valid_pos, "valid_ids": valid_ids, "vocab": vocab}
        self._behavior_cache[key] = beh
        return beh

    @staticmethod
    def _behavior_energy(beh, q) -> float:
        import numpy as np

        coef = (np.sqrt(beh["dense_p"]) * np.sqrt(np.asarray(q)[None, :])).sum(axis=1)
        return float(-np.log(np.clip(coef.max(), 1e-12, None)))  # min Bhattacharyya distance to ℳ_y

    # --- pullback steering: optimize the activation path that INDUCES a target ℳ_y behavior ---
    def _pca_to_3d(self, manifold, pca_pt) -> list[float]:
        import numpy as np

        p3 = np.asarray(pca_pt)[:3]
        return [round(float(v), 4) for v in (p3 - manifold["center3"]) / manifold["scale3"]]

    def _recover_intrinsic_r(self, manifold, pca_points, src_i, tgt_i):
        """Project each path point onto ℳ_h (nearest intrinsic u), correlate recovered u with
        the ideal src→tgt sweep. High = the path traces the manifold (paper's R²_pullback)."""
        import numpy as np

        n = manifold["n_items"]
        du = np.linspace(0.0, float(n if manifold["kind"] == "cyclic" else n - 1), 240)
        dpts = manifold["spline"](du)
        rec = np.array([du[np.linalg.norm(dpts - np.asarray(p)[None, :], axis=1).argmin()] for p in pca_points])
        t = np.linspace(0.0, 1.0, len(pca_points))
        if manifold["kind"] == "cyclic":
            d = (tgt_i - src_i) % n
            if d > n / 2:
                d -= n
            ideal = src_i + d * t
        else:
            ideal = src_i + t * (tgt_i - src_i)
        if rec.std() > 1e-9 and ideal.std() > 1e-9:
            return round(float(np.corrcoef(rec, ideal)[0, 1]), 4)
        return None

    def _pullback_path(self, manifold, beh, layer, prompt, position, src_i, tgt_i, n_waypoints, lbfgs_iters):
        import numpy as np
        import torch

        from .hooks import layer_module

        bundle = self.ensure_model()
        dev = bundle.device
        pca = manifold["pca"]
        comps = torch.tensor(np.asarray(pca.components_), dtype=torch.float32, device=dev)
        mean = torch.tensor(np.asarray(pca.mean_), dtype=torch.float32, device=dev)
        cpca = manifold["centroids_pca"]
        n = manifold["n_items"]
        enc = bundle.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = enc["input_ids"].to(dev)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(dev)
        valid_pos, valid_ids, vocab = beh["valid_pos"], beh["valid_ids"], beh["vocab"]
        sq_spline = beh["spline"]
        d = (tgt_i - src_i) % n
        if manifold["kind"] == "cyclic" and d > n / 2:
            d -= n
        u_at = (lambda tt: src_i + d * tt) if manifold["kind"] == "cyclic" else (lambda tt: src_i + tt * (tgt_i - src_i))

        steps = max(2, n_waypoints)
        if len(valid_ids) < 2:  # toy/degenerate backend: no concept tokens in vocab -> skip optimization
            pts = [((1 - wi / (steps - 1)) * cpca[src_i] + (wi / (steps - 1)) * cpca[tgt_i]).astype(np.float32) for wi in range(steps)]
            uni = np.full(len(beh["token_ids"]), 1.0 / len(beh["token_ids"]))
            return pts, [uni for _ in pts], None, None

        holder = {"rep": None}

        def hook(_m, _i, output):
            hidden = output[0] if isinstance(output, tuple) else output
            rep = holder["rep"]
            if rep is None or not (0 <= position < hidden.shape[1]):
                return output
            mask = torch.zeros(hidden.shape[1], device=hidden.device, dtype=hidden.dtype)
            mask[position] = 1.0
            new = hidden * (1 - mask.view(1, -1, 1)) + rep.to(hidden.dtype).view(1, 1, -1) * mask.view(1, -1, 1)
            return (new, *output[1:]) if isinstance(output, tuple) else new

        handle = layer_module(bundle.model, layer).register_forward_hook(hook)
        steps = max(2, n_waypoints)
        pca_points, induced = [], []
        loss_start = loss_end = None
        try:
            for wi in range(steps):
                t = wi / (steps - 1)
                uu = (u_at(t) % n) if manifold["kind"] == "cyclic" else u_at(t)
                tgt_full = np.clip(sq_spline(uu), 0.0, None) ** 2
                tv = np.array([tgt_full[i] for i in valid_pos], dtype=float)
                tv = tv / max(tv.sum(), 1e-9)
                target = torch.tensor(tv, dtype=torch.float32, device=dev)
                z0 = ((1 - t) * cpca[src_i] + t * cpca[tgt_i]).astype(np.float32)
                z = torch.tensor(z0, requires_grad=True, device=dev)
                z_init = torch.tensor(z0, device=dev)
                opt = torch.optim.LBFGS([z], max_iter=int(lbfgs_iters), line_search_fn="strong_wolfe")
                tracker = {"first": None, "last": None}

                def closure():
                    opt.zero_grad()
                    rep = z @ comps + mean
                    holder["rep"] = rep
                    logits = bundle.model(input_ids=input_ids, attention_mask=attn).logits[0, -1].float()
                    q = torch.softmax(logits, dim=-1)[valid_ids]
                    q = q / q.sum().clamp_min(1e-9)
                    loss = 0.5 * ((q.clamp_min(1e-12).sqrt() - target.sqrt()) ** 2).sum() + 1e-3 * ((z - z_init) ** 2).sum()
                    loss.backward()
                    lv = float(loss.detach())
                    if tracker["first"] is None:
                        tracker["first"] = lv
                    tracker["last"] = lv
                    return loss

                opt.step(closure)
                if wi == 0:
                    loss_start = tracker["first"]
                loss_end = tracker["last"]
                zf = z.detach()
                pca_points.append(zf.cpu().numpy())
                with torch.no_grad():
                    holder["rep"] = zf @ comps + mean
                    pr = torch.softmax(bundle.model(input_ids=input_ids, attention_mask=attn).logits[0, -1].float(), dim=-1).cpu().numpy()
                full = np.array([pr[tk] if tk < vocab else 0.0 for tk in beh["token_ids"]], dtype=float)
                induced.append(full / full.sum() if full.sum() > 0 else full)
                holder["rep"] = None
        finally:
            handle.remove()
        return pca_points, induced, loss_start, loss_end

    def manifold_pullback(self, concept: str, target: str, layer: int | None = None, source: str | None = None,
                          n_waypoints: int = 5, max_new_tokens: int = 20, lbfgs_iters: int = 25) -> dict[str, Any]:
        """Pullback: optimize the activation path that INDUCES the smooth ℳ_y behavior sweep from
        source to target, vs manifold and linear paths. Tests (a) does pullback induce on-manifold
        behavior (low energy) and (b) does the optimized path recover ℳ_h (recovered_r)."""
        import numpy as np
        import torch

        manifold, concept_obj, layer = self._build_manifold(concept, layer)
        beh = self._build_behavior_manifold(concept_obj, layer)
        bundle = self.ensure_model()
        items, n = manifold["items"], manifold["n_items"]

        def to_index(v, default):
            if v is None:
                return default
            if v in items:
                return items.index(v)
            iv = int(v)
            if not 0 <= iv < n:
                raise ValueError(f"index {iv} out of range for concept {concept_obj.name}")
            return iv

        src_i, tgt_i = to_index(source, 0), to_index(target, n - 1)
        prompt = concept_obj.steer_prompt.format(item=items[src_i])
        position = self._locate_item_position(bundle.tokenizer, concept_obj.steer_prompt, items[src_i], prompt)

        m = self.manifold_steer(concept, items[tgt_i], layer, items[src_i], prompt, n_waypoints,
                                max_new_tokens, 0.0, path="manifold", compute_unsteered=True, compute_energy=True)
        lin = self.manifold_steer(concept, items[tgt_i], layer, items[src_i], prompt, n_waypoints,
                                  max_new_tokens, 0.0, path="linear", compute_unsteered=False, compute_energy=True)

        pca_points, induced, l0, l1 = self._pullback_path(manifold, beh, layer, prompt, position, src_i, tgt_i, n_waypoints, lbfgs_iters)
        cpca, spline = manifold["centroids_pca"], manifold["spline"]
        du = np.linspace(0.0, float(n if manifold["kind"] == "cyclic" else n - 1), 240)
        dpts = spline(du)
        pb_wps, pb_path3d, pb_e = [], [], []
        for z, q in zip(pca_points, induced):
            e = self._behavior_energy(beh, q); pb_e.append(e)
            rep = torch.tensor(np.asarray(manifold["pca"].inverse_transform(np.asarray(z).reshape(1, -1))[0]), dtype=torch.float32)
            gen = manifold_generate(bundle, prompt, layer, rep, position, max_new_tokens, 0.0, compute_unsteered=False)
            uu = du[np.linalg.norm(dpts - np.asarray(z)[None, :], axis=1).argmin()]
            pb_wps.append({"value": items[int(round(uu)) % n], "text": gen["steered_text"], "energy": round(e, 4)})
            pb_path3d.append(self._pca_to_3d(manifold, z))

        steps = max(2, n_waypoints)
        ts = np.linspace(0.0, 1.0, steps)
        d = (tgt_i - src_i) % n
        if manifold["kind"] == "cyclic" and d > n / 2:
            d -= n
        man_pca = [spline((src_i + d * t) % n if manifold["kind"] == "cyclic" else src_i + t * (tgt_i - src_i)) for t in ts]
        lin_pca = [(1 - t) * cpca[src_i] + t * cpca[tgt_i] for t in ts]
        pullback = {"path": "pullback", "steered_text": pb_wps[-1]["text"],
                    "mean_energy": round(sum(pb_e) / len(pb_e), 4) if pb_e else None,
                    "recovered_r": self._recover_intrinsic_r(manifold, pca_points, src_i, tgt_i),
                    "waypoints": pb_wps, "path_3d": pb_path3d,
                    "loss_start": round(l0, 4) if l0 is not None else None,
                    "loss_end": round(l1, 4) if l1 is not None else None}
        m["recovered_r"] = self._recover_intrinsic_r(manifold, man_pca, src_i, tgt_i)
        lin["recovered_r"] = self._recover_intrinsic_r(manifold, lin_pca, src_i, tgt_i)
        return {
            "concept": concept_obj.name, "kind": manifold["kind"], "layer": layer, "prompt": prompt,
            "source": items[src_i], "target": items[tgt_i], "unsteered_text": m["unsteered_text"],
            "manifold": m, "linear": lin, "pullback": pullback,
        }

    def notebook(self) -> dict[str, Any]:
        return load_notebook(self.config.notebook_path)

    def save_notebook_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        return save_notebook_entry(self.config.notebook_path, entry)

    def label_feature(self, payload: dict[str, Any]) -> dict[str, Any]:
        return label_feature(payload)

    def status(self) -> dict[str, Any]:
        return {
            "config_file": self.config_path,
            "config": config_to_dict(self.config),
            "loaded_model_id": self.config.model_id if self.bundle else None,
            "loaded_sae_id": self.config.sae_id if self.sae_loader.cached_layers else None,
            "configured_model_id": self.config.model_id,
            "configured_sae_id": self.config.sae_id,
            "model_loaded": self.bundle is not None,
            "loaded_device": str(self.bundle.device) if self.bundle else None,
            "loaded_dtype": str(self.bundle.dtype) if self.bundle else None,
            "active_layer_cache": self.sae_loader.cached_layers,
            "gpu": gpu_memory_summary(),
            "mode": "modal" if os.environ.get("MODAL_TASK_ID") else "local",
            "warning": "Experimental feature steering. Interpret effects cautiously.",
        }
