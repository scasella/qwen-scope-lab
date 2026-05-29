"""Archived dead-end geometry-investigation Modal probes (PROVENANCE ONLY).

These functions were extracted VERBATIM from ``modal_app.py`` when the live ops
file was trimmed down to the shipping product/ops entrypoints. They are the
"geometry investigation" research probes that searched for a global SAE-feature
"map of meaning" and came back negative:

  * ``_latent_map`` / ``latent_map_2b`` / ``latent_map_27b`` --
    decoder-cosine latent map (t-SNE/KMeans over W_dec). Decoder space is
    near-isotropic: silhouette ~= 0.003.
  * ``_coactivation_map`` / ``coact_map_2b`` (+ ``_COACT_CORPUS``) --
    co-activation map (features that fire together across a diverse corpus).
    Co-activation silhouette ~= 0.024, and the little structure that exists is
    organized by syntax, not by semantic domain.
  * ``_concept_manifold_probe`` / ``manifold_probe_2b``
    (+ ``_CONCEPT_*`` / ``_CONCEPTS``) -- conditional-coupling (Ising)
    pseudolikelihood probe. Coupling silhouette ~= 0.005, BELOW the marginal
    same-feature control -- no conditional-coupling communities either.
  * ``_residual_manifold_probe`` / ``residual_manifold_2b``
    (+ ``_CONCEPTS_V2``) -- early residual-stream manifold probe (the first,
    superseded version of the residual test).
  * ``_manifold_vs_linear_probe`` / ``manifold_vs_linear_probe_2b`` -- raw
    mean-perplexity sweep of manifold-path vs linear-chord steering across
    several concepts (superseded by the naturalness/energy and pullback probes
    that use the paper's actual on-manifold metric).

Taken together these three negative results -- decoder silhouette ~= 0.003,
co-activation ~= 0.024 (and syntactic), Ising coupling ~= 0.005 < the marginal
control -- proved that a global SAE-feature "map of meaning" does not exist at
this layer/SAE, which is exactly what justified PIVOTING to residual-stream
manifold steering (concept manifolds live in the residual stream, not in the
SAE-feature map). See ``docs/MANIFOLD.md`` and the project memory for the full
write-up.

PRESERVED, NOT EXECUTABLE AS-IS. This file is a historical record, not a live
Modal app. The code below references ``app``, ``image``, ``modal_secret``,
``hf_cache``, ``modal`` and the shared helpers ``_spearman`` /
``_manifold_metrics`` (all of which remain in ``modal_app.py`` at the revision
when this split happened), plus ``qwen_scope_steering_gui`` service internals as
they existed then. Nothing here is wired into a ``modal.App`` in this module, so
the ``@app.function(...)`` decorators below are intentionally INERT references
left exactly as they were in the original source. Do not import or deploy this
file; read it for provenance.

NOTE: ``_concept_manifold_probe`` and ``_residual_manifold_probe`` below call the
shared helper ``_spearman`` (and the sweep/atlas family in modal_app.py calls
``_manifold_metrics``). Those helpers were deliberately KEPT in ``modal_app.py``
because live functions there still use them, so the calls below are dangling
references in this archived copy -- expected and harmless for a preserved record.
"""

# ---------------------------------------------------------------------------
# Everything below is verbatim source extracted from modal_app.py.
# The leading ``@app.function(...)`` decorators are preserved as-is but are
# inert here (there is no ``app`` defined in this module). See module docstring.
# ---------------------------------------------------------------------------


def _latent_map(config_path: str, layer: int, method: str, max_features: int, n_clusters: int, prompt: str) -> dict:
    """Run /api/layout's engine against the REAL model and quantify whether the learned
    decoder geometry actually clusters -- the question Phase A cannot answer on the
    synthetic dev backend. Geometry is CPU work (t-SNE/KMeans over W_dec); the model is
    loaded only to overlay the prompt's activation.

    Reports two honest coherence metrics:
      * silhouette_highdim -- silhouette of the KMeans clusters in the *original*
        decoder space (>~0.1 = real clusters; ~0 = isotropic / no structure).
      * active_spatial_concentration_ratio -- mean pairwise 2D distance among the
        prompt's co-activating features / the same for a random null (<1 = they group).
    """
    import json as _json
    from collections import Counter

    import numpy as np
    import torch
    from sklearn.metrics import silhouette_score

    from qwen_scope_steering_gui.service import SteeringService

    service = SteeringService.from_config_path(config_path)
    lay = service.latent_layout(prompt=prompt, layer=layer, method=method, max_features=max_features, n_clusters=n_clusters)
    feats = lay["features"]

    sae = service.sae_loader.load_layer(layer)
    directions = sae.W_dec.detach().to(dtype=torch.float32, device="cpu").t().numpy()  # (d_sae, d_model)
    sel = [f["feature_id"] for f in feats]
    X = directions[sel]
    labels = [f["cluster"] for f in feats]
    n_lab = len(set(labels))
    sil_high = float(silhouette_score(X, labels)) if 2 <= n_lab < len(sel) else None

    P = np.array([[f["x"], f["y"]] for f in feats], dtype=float)
    act_idx = [i for i, f in enumerate(feats) if f["activation"] > 0]

    def mean_pair_dist(idx: list[int]):
        if len(idx) < 2:
            return None
        pts = P[idx]
        tot, cnt = 0.0, 0
        for i in range(len(pts)):
            dd = np.linalg.norm(pts[i + 1:] - pts[i], axis=1)
            tot += float(dd.sum()); cnt += len(dd)
        return tot / cnt if cnt else None

    active_spread = mean_pair_dist(act_idx)
    rng = np.random.default_rng(0)
    nulls = [mean_pair_dist(rng.choice(len(feats), size=len(act_idx), replace=False).tolist())
             for _ in range(20) if 2 <= len(act_idx) <= len(feats)]
    null_spread = float(np.mean([x for x in nulls if x])) if nulls else None
    conc_ratio = (active_spread / null_spread) if (active_spread and null_spread) else None

    act_clusters = Counter(f["cluster"] for f in feats if f["activation"] > 0)
    top_frac = (max(act_clusters.values()) / sum(act_clusters.values())) if act_clusters else None

    geom_clusters = sil_high is not None and sil_high > 0.10
    act_concentrated = conc_ratio is not None and conc_ratio < 0.80
    if geom_clusters and act_concentrated:
        verdict = "coherent: decoder space clusters AND co-activating features are spatially concentrated"
    elif geom_clusters or act_concentrated:
        verdict = "partial: some structure, weaker than a clean fixture"
    else:
        verdict = "diffuse: little cluster structure (near-isotropic) -- a 3D topography would be mostly decorative"

    out = {
        "config": config_path,
        "model_id": service.config.model_id,
        "layer": layer,
        "method": method,
        "d_sae": lay["d_sae"],
        "n_features_laid_out": lay["n_features"],
        "active_count": lay["active_count"],
        "n_clusters": n_lab,
        "cluster_sizes": sorted(Counter(labels).values(), reverse=True),
        "silhouette_highdim": round(sil_high, 4) if sil_high is not None else None,
        "active_spatial_concentration_ratio": round(conc_ratio, 3) if conc_ratio is not None else None,
        "active_top_cluster_fraction": round(top_frac, 3) if top_frac is not None else None,
        "coherence_verdict": verdict,
        "note": "laid-out set = highest-norm features UNION prompt-active features (not a uniform sample)",
        "top_active": [
            {"feature_id": f["feature_id"], "activation": f["activation"], "cluster": f["cluster"], "x": f["x"], "y": f["y"]}
            for f in sorted((f for f in feats if f["activation"] > 0), key=lambda f: -f["activation"])[:8]
        ],
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
def latent_map_2b() -> dict:
    return _latent_map("/root/configs/qwen35_2b_dev_l0_100.yaml", layer=12, method="tsne",
                       max_features=1500, n_clusters=12, prompt="The capital of France is Paris")


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
def latent_map_27b() -> dict:
    return _latent_map("/root/configs/qwen35_27b_l0_100.yaml", layer=32, method="tsne",
                       max_features=1500, n_clusters=12, prompt="The capital of France is Paris")


# A deliberately diverse corpus spanning distinct semantic domains, so that if a
# *co-activation* geometry exists (features that fire together across inputs), it has
# room to show up as separable communities.
_COACT_CORPUS = [
    # geography / places
    "The capital of France is Paris.", "Tokyo is the largest metropolitan area in Japan.",
    "The Nile flows north through Egypt to the Mediterranean.", "Mount Everest sits on the border of Nepal and Tibet.",
    "Brazil is the largest country in South America.", "The Sahara is the world's largest hot desert.",
    "Venice is famous for its canals and arched bridges.", "The Amazon rainforest spans nine countries.",
    "Iceland has many active volcanoes and glaciers.", "Kenya lies along the equator in East Africa.",
    # science
    "Photosynthesis converts sunlight into chemical energy.", "Water boils at one hundred degrees Celsius at sea level.",
    "DNA carries the genetic instructions of living cells.", "Gravity causes objects to accelerate toward the Earth.",
    "The mitochondria is the powerhouse of the cell.", "Light travels faster than sound through air.",
    "Atoms consist of protons, neutrons, and electrons.", "Evolution proceeds through natural selection over generations.",
    "The human heart pumps blood through the circulatory system.", "Electrons carry a negative electric charge.",
    "Vaccines train the immune system to recognize pathogens.", "The greenhouse effect traps heat in the atmosphere.",
    # code / programming
    "Write a Python function to sort a list of integers.", "def add(a, b):\n    return a + b",
    "The loop iterates over every element in the array.", "Use a dictionary to map keys to values quickly.",
    "The function returns None when the input is empty.", "Import the numpy library to work with arrays.",
    "A recursive function calls itself with smaller inputs.", "Catch the exception and log the error message.",
    "The variable is assigned inside the for loop.", "Compile the program before running the tests.",
    "Refactor the class to remove duplicated code.", "The pointer references a location in memory.",
    # data / config
    "Return a JSON object with name and age fields.", '{"city": "Paris", "country": "France"}',
    "The API responded with a 404 not found error.", "Parse the CSV file and load it into a table.",
    "Set the configuration value in the YAML file.", "The database query returned three matching rows.",
    "Serialize the object to a JSON string.", "The schema defines required and optional fields.",
    # math
    "Two plus two equals four.", "The derivative of x squared is two x.",
    "A triangle has three sides and three angles.", "The square root of sixteen is four.",
    "Multiply the matrix by its inverse to get the identity.", "The probability of heads is one half.",
    "Solve the equation for the unknown variable x.", "The sum of the angles in a triangle is 180 degrees.",
    "A prime number has exactly two divisors.", "Integrate the function over the given interval.",
    # food
    "The recipe calls for flour, butter, sugar, and eggs.", "Fresh bread smells wonderful in the early morning.",
    "Simmer the sauce gently for twenty minutes.", "Chop the onions finely before adding the garlic.",
    "The cake needs to bake for forty-five minutes.", "Season the soup with salt and black pepper.",
    "Whisk the eggs until they become light and fluffy.", "Serve the pasta with a sprinkle of parmesan.",
    # emotion
    "I feel happy and deeply grateful today.", "He was overcome with a profound and quiet sadness.",
    "She trembled with excitement before the show.", "They felt anxious waiting for the results.",
    "A wave of relief washed over the tired travelers.", "The child laughed with pure unguarded joy.",
    "He clenched his fists in sudden anger.", "Loneliness settled over the empty house.",
    # narrative / fiction
    "Once upon a time in a distant kingdom there lived a queen.", "The old sailor told a long and winding tale of the sea.",
    "She wandered slowly through the storied marble halls.", "A dragon slept beneath the crumbling stone tower.",
    "The detective examined the dimly lit room for clues.", "Rain hammered the windows as the storm rolled in.",
    "He drew his sword and faced the silent forest.", "The letter arrived on a grey and ordinary morning.",
    "Their ship vanished into the thickening fog.", "In the year 3000, the colony awoke from its long sleep.",
    # business / finance
    "The stock market fell sharply after the earnings report.", "Quarterly revenue grew by twelve percent year over year.",
    "The startup raised a new round of venture funding.", "Investors worried about rising interest rates.",
    "The company announced layoffs across three divisions.", "Profit margins narrowed as costs increased.",
    "The board approved the merger on Friday.", "Inflation eroded consumers' purchasing power.",
    # sports
    "The striker scored a goal in the final minute.", "She set a new world record in the marathon.",
    "The team won the championship after extra time.", "He served three aces in the opening set.",
    "The pitcher threw a fastball down the middle.", "Fans cheered as the runner crossed the finish line.",
    # music / art
    "The orchestra played a slow and beautiful symphony.", "The museum displayed a hall of impressionist paintings.",
    "She practiced the piano sonata for hours.", "The sculptor carved the figure from white marble.",
    "A gentle melody drifted from the open window.", "The choir sang in perfect four-part harmony.",
    "He sketched the skyline in charcoal and ink.", "The film's score swelled during the final scene.",
    # history
    "The empire collapsed after centuries of slow decline.", "The treaty was signed at the end of the war.",
    "Ancient Rome built stone roads across the continent.", "The revolution overthrew the old monarchy.",
    "Explorers mapped the coastline in the sixteenth century.", "The dynasty ruled for over two hundred years.",
    # medicine / health
    "The doctor prescribed antibiotics for the infection.", "Regular exercise lowers the risk of heart disease.",
    "The patient's fever broke after two days.", "Surgeons performed the operation early in the morning.",
    "A balanced diet supports a healthy immune system.", "The nurse checked the patient's blood pressure.",
    "Drink plenty of water to stay hydrated.", "The vaccine reduced infections across the population.",
    # weather / nature
    "Dark clouds gathered before the afternoon thunderstorm.", "Snow fell silently over the quiet mountain village.",
    "The river overflowed its banks after heavy rain.", "A warm breeze drifted across the open meadow.",
    "Autumn leaves turned red and gold in October.", "The drought left the fields cracked and dry.",
    # technology
    "The server crashed because of a memory leak.", "Machine learning models require large training datasets.",
    "The new phone has a faster processor and brighter screen.", "Engineers deployed the update to production overnight.",
    "The network latency spiked during peak hours.", "Quantum computers exploit superposition and entanglement.",
    "The algorithm compresses images without visible loss.", "Cloud storage scales automatically with demand.",
    # law / government
    "The lawyer presented the closing argument to the jury.", "Congress passed the bill after a long debate.",
    "The judge ruled in favor of the defendant.", "Citizens lined up early to cast their votes.",
    "The contract includes a strict confidentiality clause.", "The court overturned the lower ruling on appeal.",
    # everyday / instructions
    "Answer concisely in no more than one sentence.", "List three concrete benefits of regular exercise.",
    "Plant the seeds in spring and water them daily.", "Turn left at the second traffic light.",
    "Please summarize the article in a short paragraph.", "Remember to back up your files before the upgrade.",
    "Set an alarm for six o'clock tomorrow morning.", "Fold the laundry and put it away neatly.",
    # travel
    "The train departs from platform nine at noon.", "They booked a small hotel near the old town.",
    "The flight was delayed by two hours due to weather.", "Pack light and bring a comfortable pair of shoes.",
    "The ferry crosses the strait twice each day.", "Tourists gathered to watch the sunset over the bay.",
    # philosophy / abstract
    "Justice requires treating similar cases alike.", "Free will and determinism have long been debated.",
    "Knowledge begins with curiosity and honest doubt.", "Happiness may depend more on meaning than pleasure.",
    "The mind and body relationship remains deeply mysterious.", "Ethics asks what we owe to one another.",
]


def _coactivation_map(config_path: str, layer: int, max_features: int, n_clusters: int, focus_prompt: str, method: str = "tsne") -> dict:
    """Pivot test: lay features out by *co-activation* (their activation fingerprint
    across a diverse corpus) instead of by decoder direction, and re-run the same
    coherence metrics. Answers: does a co-activation geometry cluster where the
    decoder-cosine geometry did not?"""
    import json as _json
    from collections import Counter

    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    from qwen_scope_steering_gui.service import SteeringService

    service = SteeringService.from_config_path(config_path)
    top_k = max(20, int(service.config.top_k))
    n = len(_COACT_CORPUS)
    peak: dict[int, "np.ndarray"] = {}
    tokens: dict[int, Counter] = {}
    for pi, p in enumerate(_COACT_CORPUS):
        insp = service.inspect_prompt(p, layer=layer, top_k=top_k, max_seq_len=64)
        for row in insp["top_features_by_token"]:
            tok = (row.get("token_text") or "").strip()
            for f in row["features"]:
                fid = int(f["feature_id"]); a = float(f["activation"])
                v = peak.get(fid)
                if v is None:
                    v = peak[fid] = np.zeros(n, dtype=float)
                if a > v[pi]:
                    v[pi] = a
                if tok:
                    tk = tokens.setdefault(fid, Counter())
                    if a > tk.get(tok, -1.0):
                        tk[tok] = a

    # Selectivity filter: keep features that fire in a few prompts but are NOT
    # near-ubiquitous -- drops both noise (breadth 1) and the generic always-on core
    # (the ~876-feature blob that dominated the 30-prompt run), leaving the
    # domain-selective features where any real co-activation community would live.
    lo, hi = 3, max(4, int(round(0.40 * n)))
    items = [(fid, v) for fid, v in peak.items() if lo <= int((v > 0).sum()) <= hi]
    items.sort(key=lambda kv: -float(kv[1].sum()))
    items = items[:max_features]
    fids = [fid for fid, _ in items]
    M = np.array([v for _, v in items], dtype=float)  # (n_features, n_corpus_prompts)

    coords = SteeringService._project_2d(M, method)
    k = max(2, min(int(n_clusters), len(fids) - 1)) if len(fids) > 2 else 1
    labels = (KMeans(n_clusters=k, n_init=10, random_state=7).fit_predict(M).tolist()
              if (k > 1 and len(fids) > k) else [0] * len(fids))
    n_lab = len(set(labels))
    sil = float(silhouette_score(M, labels)) if 2 <= n_lab < len(fids) else None

    # label each cluster by the tokens its member features fire on most
    cluster_tokens: dict[int, Counter] = {}
    for fid, lab in zip(fids, labels):
        acc = cluster_tokens.setdefault(lab, Counter())
        for t, _a in tokens.get(fid, Counter()).most_common(3):
            acc[t] += 1
    cluster_summary = sorted(
        ({"cluster": lab, "size": labels.count(lab),
          "top_tokens": [t for t, _ in cluster_tokens.get(lab, Counter()).most_common(8)]}
         for lab in set(labels)),
        key=lambda c: -c["size"],
    )

    focus = service.inspect_prompt(focus_prompt, layer=layer, top_k=top_k, max_seq_len=64)
    focus_active = {int(f["feature_id"]) for row in focus["top_features_by_token"] for f in row["features"]}
    P = coords
    act_idx = [i for i, fid in enumerate(fids) if fid in focus_active]

    def mean_pair_dist(idx: list[int]):
        if len(idx) < 2:
            return None
        pts = P[idx]; tot, cnt = 0.0, 0
        for i in range(len(pts)):
            dd = np.linalg.norm(pts[i + 1:] - pts[i], axis=1)
            tot += float(dd.sum()); cnt += len(dd)
        return tot / cnt if cnt else None

    active_spread = mean_pair_dist(act_idx)
    rng = np.random.default_rng(0)
    nulls = [mean_pair_dist(rng.choice(len(fids), size=len(act_idx), replace=False).tolist())
             for _ in range(20) if 2 <= len(act_idx) <= len(fids)]
    null_spread = float(np.mean([x for x in nulls if x])) if nulls else None
    conc = (active_spread / null_spread) if (active_spread and null_spread) else None

    # honest, graded read against the decoder baseline (~0.003)
    if sil is None:
        verdict = "inconclusive"
    elif sil >= 0.20:
        verdict = "strong: clear co-activation communities -- build the co-activation map"
    elif sil >= 0.12:
        verdict = "moderate: real co-activation structure well above decoder -- a 2D co-activation map is justified"
    elif sil >= 0.07:
        verdict = "weak-but-real: well above the decoder baseline yet soft -- defensible as a soft overview, not a crisp map"
    else:
        verdict = "diffuse: co-activation no better than decoder -- the spatial map is not the right tool"

    out = {
        "config": config_path,
        "model_id": service.config.model_id,
        "layer": layer,
        "basis": "coactivation",
        "n_corpus_prompts": n,
        "breadth_filter": [lo, hi],
        "features_after_filter": len(fids),
        "n_clusters": n_lab,
        "cluster_sizes": sorted(Counter(labels).values(), reverse=True),
        "silhouette_coactivation": round(sil, 4) if sil is not None else None,
        "decoder_baseline_silhouette": 0.003,
        "focus_prompt": focus_prompt,
        "focus_active_in_set": len(act_idx),
        "focus_spatial_concentration_ratio": round(conc, 3) if conc is not None else None,
        "coherence_verdict": verdict,
        "clusters": cluster_summary,
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
def coact_map_2b() -> dict:
    return _coactivation_map("/root/configs/qwen35_2b_dev_l0_100.yaml", layer=12,
                             max_features=1500, n_clusters=12, focus_prompt="The capital of France is Paris")


# Known CONTINUOUS concepts for the manifold probe. The paper shows manifolds for
# ordinal/cyclic concepts; we hand-feed a few and check whether their features recover
# the expected line/ring under PCA. Fixed-prefix templates so only the item token varies.
_CONCEPT_INTEGERS = {"name": "integers_0_20", "kind": "ordinal",
                     "items": [str(i) for i in range(21)], "template": "The number is {item}"}
_CONCEPT_DAYS = {"name": "days_of_week", "kind": "cyclic",
                 "items": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                 "template": "Today is {item}"}
_CONCEPT_MONTHS = {"name": "months", "kind": "cyclic",
                   "items": ["January", "February", "March", "April", "May", "June",
                             "July", "August", "September", "October", "November", "December"],
                   "template": "The month is {item}"}
_CONCEPTS = [_CONCEPT_INTEGERS, _CONCEPT_DAYS, _CONCEPT_MONTHS]


# NOTE: _spearman lives in modal_app.py (kept there because live functions use it).
# The probe below calls it; the reference is dangling in this archived copy.
def _concept_manifold_probe(config_path: str, layer: int, coupling_feature_cap: int = 800,
                            l1_C: float = 0.05, n_communities: int = 12, seed: int = 7) -> dict:
    """Go/no-go probe (arXiv 2604.28119): does a CONDITIONAL-coupling (Ising) basis cluster
    SAE features where decoder-cosine (~0.003) and marginal co-activation (~0.024) did not,
    and do known continuous concepts recover a line/ring? Geometry is CPU work; the model is
    loaded only to capture dense codes."""
    import json as _json
    import warnings
    from collections import Counter

    import numpy as np
    import torch
    from sklearn.cluster import SpectralClustering
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import silhouette_score

    warnings.filterwarnings("ignore")  # one-shot container: keep the JSON report clean

    from qwen_scope_steering_gui.hooks import register_capture_hook
    from qwen_scope_steering_gui.sae_math import compute_pre_activations
    from qwen_scope_steering_gui.service import SteeringService

    service = SteeringService.from_config_path(config_path)
    bundle = service.ensure_model()
    sae = service.sae_loader.load_layer(layer)
    tokenizer, model, device = bundle.tokenizer, bundle.model, bundle.device
    W_enc = sae.W_enc.to(device=device, dtype=torch.float32)
    b_enc = sae.b_enc.to(device=device, dtype=torch.float32)

    def code_rows(prompt: str, max_seq_len: int = 64):
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)
        capture: dict = {}
        handle = register_capture_hook(model, layer, capture, to_cpu=False)
        try:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn)
        finally:
            handle.remove()
        residual = capture["residual"][0].float()
        pre = torch.relu(compute_pre_activations(residual, W_enc, b_enc))  # [seq, d_sae]
        toks = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())
        return toks, pre.detach().cpu().numpy()

    def cluster_and_score(affinity):
        aff = np.abs(affinity).astype(float).copy()
        np.fill_diagonal(aff, 0.0)
        kk = max(2, min(int(n_communities), aff.shape[0] - 1))
        lab = SpectralClustering(n_clusters=kk, affinity="precomputed",
                                 assign_labels="discretize", random_state=seed).fit_predict(aff)
        mx = aff.max() or 1.0
        dist = 1.0 - aff / mx
        np.fill_diagonal(dist, 0.0)
        sil = float(silhouette_score(dist, lab, metric="precomputed")) if len(set(lab)) >= 2 else None
        return lab, sil

    def modularity(affinity, lab):
        a = np.abs(affinity).astype(float).copy()
        np.fill_diagonal(a, 0.0)
        m = a.sum() / 2.0
        if m <= 0:
            return None
        deg = a.sum(1)
        uniq = {c: i for i, c in enumerate(sorted(set(lab)))}
        s = np.zeros((len(lab), len(uniq)))
        for i, l in enumerate(lab):
            s[i, uniq[l]] = 1.0
        bmod = a - np.outer(deg, deg) / (2 * m)
        return float(np.trace(s.T @ bmod @ s) / (2 * m))

    # ---------- Part 1: conditional-coupling basis ----------
    rows, row_tokens = [], []
    for p in _COACT_CORPUS:
        toks, pre = code_rows(p)
        for r in range(pre.shape[0]):
            rows.append(pre[r]); row_tokens.append(toks[r])
    A_full = np.asarray(rows, dtype=np.float32)  # [N, d_sae]
    N = A_full.shape[0]

    freq = (A_full > 0).mean(0)
    lo_rate, hi_rate = 8.0 / N, 0.40
    cand = np.where((freq >= lo_rate) & (freq <= hi_rate))[0]
    mass = A_full[:, cand].sum(0)
    sel = np.sort(cand[np.argsort(-mass)][:coupling_feature_cap])
    A = A_full[:, sel]
    B = (A > 0).astype(np.int8)
    colsum = B.sum(0)
    good = np.where((colsum > 0) & (colsum < N))[0]
    sel, A, B = sel[good], A[:, good], B[:, good]
    c = B.shape[1]

    Jraw = np.zeros((c, c), dtype=float)
    for i in range(c):
        y = B[:, i]
        if y.min() == y.max():
            continue
        X = np.delete(B, i, axis=1)
        clf = LogisticRegression(penalty="l1", C=l1_C, solver="liblinear",
                                 max_iter=200, class_weight="balanced")
        clf.fit(X, y)
        coef = clf.coef_[0]
        Jraw[i, :i] = coef[:i]
        Jraw[i, i + 1:] = coef[i:]
    J = (Jraw + Jraw.T) / 2.0
    coupling_density = float((np.abs(J) > 1e-8).mean())

    # same-feature marginal control: feature-feature correlation of binarized codes
    Bf = B.astype(float)
    Bc = Bf - Bf.mean(0)
    norms = np.linalg.norm(Bc, axis=0) + 1e-9
    corr = (Bc.T @ Bc) / np.outer(norms, norms)

    labels, sil_coupling = cluster_and_score(J)
    _, sil_marginal = cluster_and_score(corr)
    modularity_Q = modularity(J, labels)
    absJ = np.abs(J).copy(); np.fill_diagonal(absJ, np.nan)
    same = labels[:, None] == labels[None, :]
    within = float(np.nanmean(np.where(same, absJ, np.nan)))
    between = float(np.nanmean(np.where(~same, absJ, np.nan)))
    block_concentration = float(within / between) if between > 0 else None

    feat_tok: dict[int, Counter] = {}
    for ri, tok in enumerate(row_tokens):
        active = np.where(B[ri] > 0)[0]
        for fi in active:
            feat_tok.setdefault(int(fi), Counter())[tok] += float(A[ri, fi])
    comm_tok: dict[int, Counter] = {}
    for fi, lab in enumerate(labels):
        acc = comm_tok.setdefault(int(lab), Counter())
        for t, _w in feat_tok.get(int(fi), Counter()).most_common(3):
            acc[t] += 1
    communities = sorted(
        ({"community": int(l), "size": int((labels == l).sum()),
          "top_tokens": [t for t, _ in comm_tok.get(int(l), Counter()).most_common(8)]}
         for l in set(labels)), key=lambda c: -c["size"])

    if sil_coupling is not None and sil_coupling >= 0.10 and (sil_marginal is None or sil_coupling >= 2 * sil_marginal) and (modularity_Q or 0) >= 0.30:
        q1 = "strong"
    elif sil_coupling is not None and sil_coupling >= 0.06 and (sil_marginal is None or sil_coupling > sil_marginal) and (modularity_Q or 0) >= 0.20:
        q1 = "moderate"
    else:
        q1 = "dead"

    bg_mean = A_full.mean(0)  # [d_sae] background per-feature activation

    # ---------- Part 2: known continuous concepts ----------
    concept_out = {}
    for concept in _CONCEPTS:
        items = concept["items"]
        item_vecs, chosen = [], []
        for it in items:
            toks, pre = code_rows(concept["template"].format(item=it))
            item_vecs.append(pre[-1]); chosen.append(toks[-1])  # last sub-token of the item
        M = np.asarray(item_vecs, dtype=np.float32)  # [n_items, d_sae]
        on_items = M.mean(0)
        fire_items = (M > 0).sum(0)
        sel_ratio = on_items / (bg_mean + 1e-6)
        cc = np.where((fire_items >= 3) & (on_items > 0))[0]
        cc = cc[np.argsort(-sel_ratio[cc])][:60]
        verdict = "diffuse"; var2d = None; order_metric = None; coords = None; metric_name = None
        if len(cc) >= 2 and M.shape[0] >= 3:
            Xc = M[:, cc]
            Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-6)
            pca = PCA(n_components=2, random_state=seed)
            coords = pca.fit_transform(Xc)
            var2d = float(pca.explained_variance_ratio_[:2].sum())
            idx = np.arange(len(items))
            if concept["kind"] == "ordinal":
                s1, s2 = abs(_spearman(coords[:, 0], idx)), abs(_spearman(coords[:, 1], idx))
                order_metric = float(max(s1, s2)); metric_name = "abs_spearman_pc_vs_index"
            else:
                ctr = coords - coords.mean(0)
                dd = np.linalg.norm(ctr[:, None, :] - ctr[None, :, :], axis=2)
                np.fill_diagonal(dd, np.inf)
                nn = dd.argmin(1)
                n = len(items)
                order_metric = float(np.mean([(nn[i] == (i - 1) % n) or (nn[i] == (i + 1) % n) for i in range(n)]))
                metric_name = "neighbor_adjacency"
            if var2d >= 0.85 and order_metric >= 0.80:
                verdict = "clean"
            elif var2d >= 0.70 and order_metric >= 0.60:
                verdict = "partial"
        concept_out[concept["name"]] = {
            "kind": concept["kind"], "n_items": len(items), "features_selected": int(len(cc)),
            "variance_explained_2d": round(var2d, 4) if var2d is not None else None,
            "order_metric_name": metric_name,
            "order_metric": round(order_metric, 4) if order_metric is not None else None,
            "verdict": verdict,
            "coords": ([{"item": items[i], "index": i, "pc1": round(float(coords[i, 0]), 3),
                         "pc2": round(float(coords[i, 1]), 3)} for i in range(len(items))]
                       if coords is not None else None),
        }

    q2_any_clean = any(v["verdict"] == "clean" for v in concept_out.values())
    overall = ("GREEN: build the concept-manifold workbench" if (q1 in ("strong", "moderate") or q2_any_clean)
               else "DEAD: 2D map is the wrong tool at this layer/SAE")

    out = {
        "config": config_path, "model_id": service.config.model_id, "layer": layer, "probe": "concept_manifold",
        "part1_coupling": {
            "sample_unit": "token", "n_samples": int(N), "features_scoped": int(c),
            "estimator": "logistic_pseudolikelihood_l1", "l1_C": l1_C, "coupling_density": round(coupling_density, 4),
            "n_communities": len(set(labels)), "community_sizes": sorted(Counter(labels.tolist()).values(), reverse=True),
            "silhouette_coupling": round(sil_coupling, 4) if sil_coupling is not None else None,
            "silhouette_marginal_same_features": round(sil_marginal, 4) if sil_marginal is not None else None,
            "decoder_baseline_silhouette": 0.003, "marginal_corpus_baseline_silhouette": 0.024,
            "modularity_Q": round(modularity_Q, 4) if modularity_Q is not None else None,
            "block_concentration": round(block_concentration, 3) if block_concentration is not None else None,
            "verdict_q1": q1, "communities": communities,
        },
        "part2_concepts": concept_out,
        "go_no_go": {"q1": q1, "q2_any_clean": q2_any_clean, "overall": overall},
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
def manifold_probe_2b() -> dict:
    return _concept_manifold_probe("/root/configs/qwen35_2b_dev_l0_100.yaml", layer=12,
                                   coupling_feature_cap=800, l1_C=0.05, n_communities=12)


# Concepts with MULTIPLE carrier templates per item (fixed-prefix, item at the end so
# the last sub-token carries it). Multiple templates -> per-class centroids over varied
# contexts, matching causalab's centroid construction. Includes a negative control whose
# ordinal index is arbitrary (the ordering metric should come back ~0).
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


# NOTE: _spearman lives in modal_app.py (kept there because live functions use it).
# The probe below calls it; the reference is dangling in this archived copy.
def _residual_manifold_probe(config_path: str, layer: int, k_pca: int = 8, seed: int = 7) -> dict:
    """FAITHFUL test of the paper's manifold concept (mirrors causalab activation_manifold
    + subspace.pca): does a concept's geometry live in the RESIDUAL STREAM, and do the SAE
    codes dilute it? For each concept we collect per-class centroids over several carrier
    templates, PCA the residual activations (centered), and measure whether the centroids
    recover the expected ordinal line / cyclic ring -- then repeat in SAE-code space to
    show the dilution. A random-word control should score ~0 on the ordering metric."""
    import json as _json
    import warnings

    import numpy as np
    import torch
    from sklearn.decomposition import PCA

    warnings.filterwarnings("ignore")

    from qwen_scope_steering_gui.hooks import register_capture_hook
    from qwen_scope_steering_gui.service import SteeringService

    service = SteeringService.from_config_path(config_path)
    bundle = service.ensure_model()
    sae = service.sae_loader.load_layer(layer)
    tokenizer, model, device = bundle.tokenizer, bundle.model, bundle.device
    W_enc = sae.W_enc.to(device=device, dtype=torch.float32)
    b_enc = sae.b_enc.to(device=device, dtype=torch.float32)

    def capture(prompt: str):
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)
        cap: dict = {}
        h = register_capture_hook(model, layer, cap, to_cpu=False)
        try:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn)
        finally:
            h.remove()
        last = cap["residual"][0].float()[-1]  # [d_model] residual of the last token
        code = torch.relu(last @ W_enc.transpose(-1, -2) + b_enc)  # [d_sae] SAE code
        return last.detach().cpu().numpy(), code.detach().cpu().numpy()

    def analyze(X, labels, n_items, kind):
        k = max(2, min(k_pca, X.shape[0] - 1, X.shape[1]))
        pca = PCA(n_components=k, random_state=seed)
        Y = pca.fit_transform(X)  # PCA centers by default
        var = pca.explained_variance_ratio_
        C = np.array([Y[labels == ci].mean(0) for ci in range(n_items)])  # centroids in PCA space
        idx = np.arange(n_items)
        if kind == "cyclic":
            cc = C[:, :2] - C[:, :2].mean(0)
            dd = np.linalg.norm(cc[:, None, :] - cc[None, :, :], axis=2)
            np.fill_diagonal(dd, np.inf)
            nn = dd.argmin(1)
            metric = float(np.mean([(nn[i] == (i - 1) % n_items) or (nn[i] == (i + 1) % n_items) for i in range(n_items)]))
            mname = "neighbor_adjacency"
        else:  # ordinal or control
            metric = float(max(abs(_spearman(C[:, j], idx)) for j in range(min(3, C.shape[1]))))
            mname = "abs_spearman_pc_vs_index"
        return {"var_top2": round(float(var[:2].sum()), 4), "var_top3": round(float(var[:3].sum()), 4),
                "order_metric_name": mname, "order_metric": round(metric, 4),
                "centroids_2d": [[round(float(C[i, 0]), 3), round(float(C[i, 1]), 3)] for i in range(n_items)]}

    concept_out = {}
    for concept in _CONCEPTS_V2:
        items, kind = concept["items"], concept["kind"]
        R, S, labels = [], [], []
        for ci, it in enumerate(items):
            for tmpl in concept["templates"]:
                r, s = capture(tmpl.format(item=it))
                R.append(r); S.append(s); labels.append(ci)
        R = np.asarray(R, dtype=np.float32); S = np.asarray(S, dtype=np.float32); labels = np.asarray(labels)
        res = analyze(R, labels, len(items), kind)
        cod = analyze(S, labels, len(items), kind)
        present = (res["order_metric"] >= 0.80 and res["var_top3"] >= 0.50)
        concept_out[concept["name"]] = {
            "kind": kind, "n_items": len(items), "n_samples": int(len(R)),
            "residual": res, "sae_code": cod,
            "residual_manifold_present": bool(present),
            "sae_dilutes": bool(present and (cod["order_metric"] < res["order_metric"] - 0.15 or cod["var_top3"] < res["var_top3"] - 0.15)),
        }

    real = [v for k, v in concept_out.items() if v["kind"] in ("ordinal", "cyclic")]
    n_present = sum(1 for v in real if v["residual_manifold_present"])
    ctrl = concept_out["random_control"]["residual"]["order_metric"]
    control_ok = ctrl < 0.6
    any_dilution = any(v["sae_dilutes"] for v in real)
    if n_present >= 2 and control_ok:
        overall = ("GREEN: concept manifolds ARE present in residual-stream activations" +
                   (" and the SAE dilutes them (use residual/subspace directions, not an SAE-feature map)" if any_dilution
                    else " (and survive in SAE codes)"))
    elif n_present >= 1 and control_ok:
        overall = "PARTIAL: some residual manifolds present; mixed"
    else:
        overall = "DEAD: no clean concept manifold even in residual-stream activations at this layer"

    out = {
        "config": config_path, "model_id": service.config.model_id, "layer": layer,
        "probe": "residual_manifold", "method": "PCA of residual activations, per-class centroids (causalab-faithful)",
        "k_pca": k_pca, "concepts": concept_out,
        "go_no_go": {"residual_manifolds_present": int(n_present), "of_real_concepts": len(real),
                     "control_order_metric": round(float(ctrl), 4), "control_ok": bool(control_ok),
                     "sae_dilution_observed": bool(any_dilution), "overall": overall},
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
def residual_manifold_2b() -> dict:
    return _residual_manifold_probe("/root/configs/qwen35_2b_dev_l0_100.yaml", layer=12, k_pca=8)


def _manifold_vs_linear_probe(config_path: str, specs: list[tuple]) -> dict:
    """Sweep several clean/curved concepts: does manifold-path mean perplexity beat the
    linear chord anywhere (where intermediate chord points go genuinely off-manifold)?"""
    import json as _json

    from qwen_scope_steering_gui.service import SteeringService

    service = SteeringService.from_config_path(config_path)
    rows = []
    for concept, layer, source, target in specs:
        try:
            c = service.manifold_compare(concept, target, layer=layer, source=source, n_waypoints=9, max_new_tokens=20)
            mp, lp = c["manifold"]["mean_perplexity"], c["linear"]["mean_perplexity"]
            rows.append({"concept": concept, "layer": layer, "source": source, "target": target,
                         "manifold_mean_ppl": mp, "linear_mean_ppl": lp,
                         "gap": (round(lp - mp, 3) if (mp is not None and lp is not None) else None),
                         "manifold_better": bool(mp is not None and lp is not None and mp < lp),
                         "manifold_end": c["manifold"]["steered_text"][:70],
                         "linear_end": c["linear"]["steered_text"][:70]})
        except Exception as exc:  # noqa: BLE001
            rows.append({"concept": concept, "error": str(exc)})
    out = {"config": config_path, "model_id": service.config.model_id,
           "n_manifold_better": sum(1 for r in rows if r.get("manifold_better")), "results": rows}
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
def manifold_vs_linear_probe_2b() -> dict:
    return _manifold_vs_linear_probe("/root/configs/qwen35_2b_dev_l0_100.yaml", [
        ("integers_0_20", 8, "0", "20"),
        ("rank", 20, "private", "general"),
        ("education", 8, "kindergarten", "doctorate"),
        ("valence", 16, "miserable", "ecstatic"),
        ("size", 16, "tiny", "enormous"),
        ("agreement", 8, "strongly disagree", "strongly agree"),
    ])
