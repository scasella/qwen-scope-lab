/* The Lab Bench — live frontend over the FastAPI backend. */

/* ----------------------------- state ----------------------------- */
const S = {
  mode: "explore",
  exploreView: "inspect",
  status: null,
  labels: {},            // feature_id -> human label (from notebook)
  prompt: "The capital of France is Paris",
  layer: 0,
  lastInspection: null,
  selToken: 0,
  lastCompare: null,
  posPrompt: "Write a concise factual answer.",
  negPrompt: "Write a long rambling story.",
  atlasData: null,
  atlasPrompts: "The capital of France is Paris.\nWrite a concise factual answer.\nReturn a JSON object with name and age.\nOnce upon a time, a long rambling story unfolded.\nThe museum was full of art and golden evening light.",
  atlasSort: "peak",
  atlasLabeledOnly: false,
  manifoldConcept: "days_of_week",
  manifoldLayer: null,
  manifoldPresets: null,
  manifoldFit: null,
  manifoldResult: null,
  manifoldSource: null,
  manifoldTarget: null,
  manifoldWaypoints: 7,
  manifoldCompare: null,
  manifoldPullback: null,
  manifoldCoverage: null,
  manifoldShowSae: false,
  manifoldCoverageColors: null,
  monitorBehavior: "refusal",
  monitorPos: "I'm sorry, but I can't help with that.\nI won't be able to assist with this.\nThat's not something I'm able to do.\nUnfortunately I have to decline this request.",
  monitorNeg: "Sure, here's how you can do it step by step.\nAbsolutely, let me explain the process.\nOf course! The steps are as follows.\nYes, I'd be happy to help with that.",
  monitorLayer: null,
  monitorTopK: 3,
  monitorResult: null,
  monitorTestText: "I'm afraid I cannot do that.",
  monitorScore: null,
  monitors: [],
  pinned: null,          // {id, label, topTokens:[], fingerprint:[]}
  steerPrompt: "Write one sentence about Paris.",
  strength: 8,
  steerResult: null,
  sweepResult: null,
  recipes: [],
  libFilter: "all",
  recipeDetail: null,
  benchResult: null,
  benchPrompts: '{"id":"p001","prompt":"Explain sparse autoencoders in one paragraph."}\n{"id":"p002","prompt":"Describe why feature steering needs controls."}',
  benchObjective: "maximize_rule_score",
  apPositive: "Paris is the capital of France.\nThe capital of France is Paris.",
  apNegative: "Once upon a time there was a long and rambling tale.\nA winding story with many gentle digressions.",
  apCount: 3,
  autopilotResult: null,
  busy: false,
};

/* ----------------------------- helpers ----------------------------- */
const $ = s => document.querySelector(s);
const esc = s => (s == null ? "" : String(s)).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const heatClass = f => "h" + Math.min(4, Math.max(0, Math.floor(f * 4.999)));
const fLabel = id => S.labels[id] || null;
const short = (s, n) => (s && s.length > n ? s.slice(0, n) + "…" : (s || ""));

async function api(path, body) {
  const opts = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = "HTTP " + r.status;
    try { detail = (await r.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return r.json();
}

function toast(msg, isErr) {
  const t = $("#toast");
  $("#toastMsg").textContent = msg;
  t.classList.toggle("err", !!isErr);
  t.classList.add("show");
  clearTimeout(t._t);
  t._t = setTimeout(() => t.classList.remove("show"), isErr ? 3200 : 1900);
}

async function withBusy(btn, fn) {
  if (S.busy) return;
  S.busy = true;
  const html = btn ? btn.innerHTML : null;
  if (btn) { btn.disabled = true; btn.innerHTML = `<span class="spinner"></span> running…`; }
  try {
    return await fn();
  } catch (e) {
    toast(e.message || "request failed", true);
    throw e;
  } finally {
    S.busy = false;
    if (btn) { btn.disabled = false; btn.innerHTML = html; }
  }
}

function sparkSVG(pts, color) {
  if (!pts || !pts.length) return `<svg width="100%" height="26"></svg>`;
  const mx = Math.max(...pts, 1e-6), n = pts.length;
  const xs = n === 1 ? [0] : pts.map((_, i) => (i / (n - 1)) * 118);
  const p = pts.map((v, i) => `${xs[i].toFixed(1)},${(24 - (Math.max(v, 0) / mx) * 22).toFixed(1)}`).join(" ");
  return `<svg width="100%" height="26" viewBox="0 0 118 26" preserveAspectRatio="none"><polyline points="${p}" fill="none" stroke="${color}" stroke-width="1.6"/></svg>`;
}

/* ----- motion helpers (refined / instrument-grade) ----- */
const REDUCED = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
let animateNext = false;
function reveal(root, sel, step = 34, max = 18) {
  if (!root || REDUCED) return;
  root.querySelectorAll(sel).forEach((el, i) => {
    if (i >= max) return;
    el.style.animation = "rise .42s var(--e-out) both";
    el.style.animationDelay = (i * step) + "ms";
    el.addEventListener("animationend", () => { el.style.animation = ""; el.style.animationDelay = ""; }, { once: true });
  });
}
function animateGauges(root) {
  (root || document).querySelectorAll(".gv[data-to]").forEach(el => {
    const to = parseFloat(el.dataset.to);
    if (isNaN(to)) return;
    if (REDUCED) { el.textContent = to.toFixed(2); return; }
    const dur = 600, t0 = performance.now();
    const tick = t => { const k = Math.min(1, (t - t0) / dur); el.textContent = (to * (1 - Math.pow(1 - k, 3))).toFixed(2); if (k < 1) requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  });
}

/* feature aggregation from an inspection response */
function aggregateFeatures(inspection) {
  const agg = {};
  const tokens = inspection.tokens;
  inspection.top_features_by_token.forEach((row, ti) => {
    row.features.forEach(f => {
      const id = f.feature_id, a = f.activation;
      if (!agg[id]) agg[id] = { id, max: -1e9, perToken: new Array(tokens.length).fill(0), tokens: [] };
      agg[id].perToken[ti] = a;
      if (a > agg[id].max) agg[id].max = a;
      agg[id].tokens.push({ tok: row.token_text, act: a });
    });
  });
  Object.values(agg).forEach(e => { e.tokens.sort((x, y) => y.act - x.act); e.topTokens = e.tokens.slice(0, 5).map(t => t.tok); });
  return agg;
}

function pinFromInspection(id) {
  const agg = S.lastInspection ? aggregateFeatures(S.lastInspection) : {};
  const e = agg[id];
  S.pinned = {
    id,
    label: fLabel(id),
    topTokens: e ? e.topTokens : [],
    fingerprint: e ? e.perToken : [],
  };
}

/* ----------------------------- boot ----------------------------- */
async function boot() {
  try {
    S.status = await api("/api/status");
    S.layer = S.status.config.default_layer;
  } catch (e) {
    document.getElementById("stage").innerHTML = `<div class="errbar">Could not reach the backend: ${esc(e.message)}. Is serve_web.py running?</div>`;
    return;
  }
  try {
    const nb = await api("/api/notebook");
    const entries = (nb && nb.features) || (Array.isArray(nb) ? nb : []);
    entries.forEach(en => { if (en && en.feature_id != null && en.human_label) S.labels[en.feature_id] = en.human_label; });
  } catch (_) {}
  try { S.recipes = await api("/api/recipes"); } catch (_) { S.recipes = []; }
  try { S.monitors = await api("/api/monitors"); } catch (_) { S.monitors = []; }
  try { S.manifoldPresets = (await api("/api/manifold/presets")).presets; } catch (_) {}
  renderRail();
  render();
  // auto-run a first inspection so Explore opens populated
  await runInspect(null, true);
}

function renderRail() {
  document.querySelectorAll(".navitem").forEach(n => n.classList.toggle("on", n.dataset.arg === S.mode));
  const names = { explore: "Explore", steer: "Steer", measure: "Measure", manifold: "Manifold", monitor: "Monitor", library: "Library" };
  $("#crumbs").innerHTML = "Lab Bench · <b>" + names[S.mode] + "</b>";
  const st = S.status; if (!st) return;
  const cfg = st.config;
  $("#ctxModel").textContent = (cfg.model_id || "").split("/").pop();
  $("#ctxModel").title = cfg.model_id;
  $("#ctxSae").textContent = (cfg.sae_id || "").split("/").pop();
  $("#ctxSae").title = cfg.sae_id;
  $("#ctxLayer").textContent = S.layer;
  $("#ctxDtype").textContent = cfg.torch_dtype;
  $("#ctxDevice").textContent = st.loaded_device || cfg.device;
  $("#ctxStatus").textContent = st.model_loaded ? "● loaded" : "○ idle";
  $("#ctxStatus").style.color = st.model_loaded ? "var(--ok)" : "var(--faint)";
  const gpu = st.gpu || {};
  if (gpu.cuda_available) {
    $("#ctxGpu").textContent = `${gpu.allocated_gb} / ${gpu.total_gb} GB`;
    $("#ctxBar").style.width = Math.min(100, (gpu.allocated_gb / Math.max(gpu.total_gb, 1)) * 100).toFixed(0) + "%";
  } else {
    $("#ctxGpu").textContent = "cpu";
    $("#ctxBar").style.width = "0%";
  }
  const isDev = (cfg.model_id || "").startsWith("dev/");
  $("#backendChip").textContent = isDev ? "LIVE · DEV BACKEND" : "LIVE · " + (cfg.model_id || "").split("/").pop();
  $("#warmText").textContent = st.model_loaded ? `warm · ${st.loaded_device || cfg.device}` : "idle";
  $("#warmChip").classList.toggle("cold", !st.model_loaded);
}

/* ----------------------------- render ----------------------------- */
function render() { renderRail(); renderStage(); renderInspector(); }
function renderStage() {
  const st = $("#stage");
  st.classList.remove("fade"); void st.offsetWidth; st.classList.add("fade");
  if (S.mode === "explore") st.innerHTML = viewExplore();
  else if (S.mode === "steer") { st.innerHTML = viewSteer(); if (S.pinned) syncDial(); }
  else if (S.mode === "measure") st.innerHTML = viewMeasure();
  else if (S.mode === "manifold") st.innerHTML = viewManifold();
  else if (S.mode === "monitor") st.innerHTML = viewMonitor();
  else if (S.mode === "library") st.innerHTML = viewLibrary();
  if (animateNext) { reveal(st, ".panel", 55); animateNext = false; }
  if ((S.mode === "library" && !S.recipeDetail) || S.mode === "monitor") reveal(st, ".rcard", 30);
  if (S.mode === "manifold") mountManifold();
  animateGauges(st);
}

/* ----------------------------- EXPLORE ----------------------------- */
function viewExplore() {
  const seg = `<div class="seg">
    <button data-act="exploreView" data-arg="inspect" class="${S.exploreView === 'inspect' ? 'on' : ''}">Inspect</button>
    <button data-act="exploreView" data-arg="atlas" class="${S.exploreView === 'atlas' ? 'on' : ''}">Atlas</button>
    <button data-act="exploreView" data-arg="contrast" class="${S.exploreView === 'contrast' ? 'on' : ''}">Contrast</button>
  </div>`;
  let body = "";
  if (S.exploreView === "inspect") body = viewInspect();
  else if (S.exploreView === "atlas") body = viewAtlas();
  else body = viewContrast();
  return `<div class="stage-head"><div class="k">01 · Explore</div>
    <h1>Find a feature worth steering</h1>
    <p>Inspect where SAE features fire, browse the features active in your prompt, or contrast two prompts. Click any feature to pin it — it rides with you into Steer and Measure.</p></div>${seg}${body}`;
}

function viewInspect() {
  const ins = S.lastInspection;
  let micro = `<div class="placeholder"><div class="big">◎</div>Press <b>Inspect</b> to capture SAE activations for this prompt.</div>`;
  if (ins) {
    const toks = ins.tokens.map((t, i) => {
      const acts = ins.top_features_by_token[i].features.map(f => f.activation);
      return { text: t, max: acts.length ? Math.max(...acts) : 0 };
    });
    const mx = Math.max(...toks.map(t => t.max), 1e-6);
    if (S.selToken >= toks.length) S.selToken = toks.length - 1;
    const heat = toks.map((t, i) => {
      const frac = t.max / mx;
      const badge = frac > 0.6 ? `<small>${t.max.toFixed(1)}</small>` : "";
      return `<span class="tok ${heatClass(frac)} ${i === S.selToken ? 'sel' : ''}" data-act="token" data-arg="${i}">${esc(t.text)}${badge}</span>`;
    }).join("");
    micro = `<div class="tokrow">${heat}</div><div class="heat-legend">low <span class="scale"></span> high</div>`;
  }
  return `<div class="panel">
    <div class="panel-h"><h3>Prompt</h3><span class="tag">inspect_prompt · layer ${S.layer}</span></div>
    <textarea class="field" id="promptInput">${esc(S.prompt)}</textarea>
    <div class="row" style="margin-top:12px"><button class="btn" id="inspectBtn" data-act="inspect">Inspect activations</button>
      <span class="muted">tokens tinted by max SAE activation · click one to expand its features</span></div>
  </div>
  <div class="panel">
    <div class="panel-h"><h3>Activation microscope</h3>${ins ? `<span class="tag">${ins.tokens.length} tokens</span>` : ''}</div>
    ${micro}
  </div>
  ${ins ? `<div class="panel" id="featPanel">${featPanelInner()}</div>` : ''}`;
}
function featPanelInner() {
  const ins = S.lastInspection; if (!ins) return "";
  const sel = ins.top_features_by_token[S.selToken];
  const feats = sel ? sel.features : [];
  const fmx = Math.max(...feats.map(f => f.activation), 1e-6);
  const bars = feats.map(f => {
    const lab = fLabel(f.feature_id);
    return `<div class="fbar" data-act="pin" data-arg="${f.feature_id}">
      <span class="fid">#${f.feature_id}</span>
      <span class="flabel ${lab ? '' : 'un'}">${lab ? esc(lab) : 'unlabeled'}</span>
      <span class="track"><span class="fill" style="width:${(f.activation / fmx * 100).toFixed(0)}%"></span></span>
      <span class="val">${f.activation.toFixed(2)}</span><span class="pin">⊕</span>
    </div>`;
  }).join("");
  const selTok = esc(ins.tokens[S.selToken] || "");
  return `<div class="panel-h"><h3>Top features · token <span class="mono" style="color:var(--brand)">${selTok}</span></h3><span class="tag">click ⊕ to pin</span></div>
    ${bars || '<p class="muted">No active features for this token.</p>'}`;
}

function viewAtlas() {
  const d = S.atlasData;
  const tag = d ? `${d.features.length} features · ${d.n_prompts} prompts · layer ${d.layer}` : "scan prompts to build a map";
  const labeledCount = Object.keys(S.labels).length;
  const controls = d ? `
    <div class="row" style="margin-bottom:14px;gap:10px;flex-wrap:wrap">
      <input class="field" id="atlasSearch" placeholder="⌕  search by id, token, or label…" style="flex:1;min-width:180px">
      <select class="field" id="atlasSort" style="width:auto;padding:9px 12px">
        <option value="peak" ${S.atlasSort === 'peak' ? 'selected' : ''}>sort: peak activation</option>
        <option value="breadth" ${S.atlasSort === 'breadth' ? 'selected' : ''}>sort: breadth (# prompts)</option>
        <option value="id" ${S.atlasSort === 'id' ? 'selected' : ''}>sort: feature id</option>
      </select>
      <button class="btn ghost sm" id="atlasLabeledBtn" data-act="atlasLabeled"
        style="${S.atlasLabeledOnly ? 'border-color:var(--brand-dim);color:var(--brand)' : ''}">labeled only${S.atlasLabeledOnly ? ' ✓' : ''}${labeledCount ? ` (${labeledCount})` : ''}</button>
    </div>
    <div class="atlas-grid" id="atlasGrid">${atlasCards("")}</div>`
    : `<div class="placeholder"><div class="big">▦</div>Scan a set of prompts to build a cross-prompt feature map. Each card shows where a feature fires, how broadly across the corpus, and any saved label.</div>`;
  return `<div class="panel">
    <div class="panel-h"><h3>Feature atlas</h3><span class="tag">${tag}</span></div>
    <textarea class="field" id="atlasPrompts" style="min-height:78px;font-family:var(--mono);font-size:12.5px">${esc(S.atlasPrompts)}</textarea>
    <div class="row" style="margin:12px 0">
      <button class="btn" id="atlasScanBtn" data-act="atlasScan">Scan corpus</button>
      <span class="muted">inspects each prompt (one per line) and maps every feature that fires — across prompts</span>
    </div>
    ${controls}
  </div>`;
}
function atlasCards(q) {
  q = (q || "").toLowerCase().trim();
  const d = S.atlasData; if (!d) return "";
  let list = d.features.slice();
  if (S.atlasLabeledOnly) list = list.filter(e => fLabel(e.feature_id));
  if (q) list = list.filter(e => ("" + e.feature_id).includes(q) || (e.top_tokens || []).some(t => t.toLowerCase().includes(q)) || (fLabel(e.feature_id) || "").toLowerCase().includes(q));
  const sort = S.atlasSort;
  list.sort((a, b) => sort === "breadth" ? (b.n_prompts - a.n_prompts || b.peak - a.peak) : sort === "id" ? a.feature_id - b.feature_id : b.peak - a.peak);
  list = list.slice(0, 90);
  if (!list.length) return `<p class="muted" style="grid-column:1/-1">No features match${q ? ` "${esc(q)}"` : (S.atlasLabeledOnly ? " — none labeled yet" : "")}.</p>`;
  return list.map(e => {
    const lab = fLabel(e.feature_id), pinned = S.pinned && S.pinned.id === e.feature_id;
    return `<div class="fcard ${pinned ? 'pinned' : ''} ${lab ? 'labeled' : ''}" data-act="pinAtlas" data-arg="${e.feature_id}">
      <div class="row" style="justify-content:space-between;align-items:center"><div class="fid">#${e.feature_id}</div>${lab ? '<span class="lbldot" title="labeled">●</span>' : ''}</div>
      <h4 class="${lab ? '' : 'un'}">${lab ? esc(lab) : 'unlabeled'}</h4>
      <div class="spark" style="padding:6px 8px">${sparkSVG(e.fingerprint, lab ? 'var(--brand)' : 'var(--brand-dim)')}</div>
      <div class="fires"><b>peak</b> ${e.peak.toFixed(2)} · <b>in</b> ${e.n_prompts}/${d.n_prompts} · ${esc((e.top_tokens || []).slice(0, 3).join(", "))}</div>
    </div>`;
  }).join("");
}
async function runAtlasScan(btn) {
  S.atlasPrompts = ($("#atlasPrompts") && $("#atlasPrompts").value) || S.atlasPrompts;
  const prompts = S.atlasPrompts.split("\n").map(s => s.trim()).filter(Boolean);
  await withBusy(btn, async () => {
    S.atlasData = await api("/api/atlas", { prompts, layer: S.layer, top_k: 12, max_features: 90 });
    renderStage();
    reveal($("#atlasGrid"), ".fcard", 24, 24);
    toast("Mapped " + S.atlasData.features.length + " features across " + S.atlasData.n_prompts + " prompts");
  });
}

/* ----------------------------- manifold steering ----------------------------- */
const MANIFOLD_PRESETS_FALLBACK = [
  { name: "days_of_week", label: "Days of the week", kind: "cyclic", best_layer: 14 },
  { name: "months", label: "Months", kind: "cyclic", best_layer: null },
  { name: "integers_0_20", label: "Integers 0–20", kind: "ordinal", best_layer: 8 },
  { name: "size", label: "Size", kind: "ordinal", best_layer: 16 },
  { name: "letters", label: "Letters A–Z", kind: "ordinal", best_layer: 16 },
];

// The atlas-derived best layer for a concept, used as the default — but only when it's
// valid for the loaded model (the dev model has few layers); otherwise the global layer.
function manifoldNumLayers() { return (S.status && S.status.config && S.status.config.num_layers) || 0; }
function recommendedLayer(conceptName) {
  const p = (S.manifoldPresets || MANIFOLD_PRESETS_FALLBACK).find(x => x.name === conceptName);
  const bl = p && p.best_layer;
  const nl = manifoldNumLayers();
  return (bl != null && (nl === 0 || bl < nl)) ? bl : S.layer;
}

function viewManifold() {
  const presets = S.manifoldPresets || MANIFOLD_PRESETS_FALLBACK;
  const fit = S.manifoldFit;
  const opts = presets.map(p => `<option value="${p.name}" ${S.manifoldConcept === p.name ? "selected" : ""}>${esc(p.label)} · ${p.kind}</option>`).join("");
  const quality = fit ? `<span class="tag">${fit.kind} · ${fit.n_items} pts · ${fit.quality.metric_name} ${fit.quality.metric}${fit.synthetic ? " · synthetic (dev)" : ""}</span>` : "";
  let steer = "";
  if (fit) {
    const itemOpts = sel => fit.items.map(v => `<option value="${esc(v)}" ${sel === v ? "selected" : ""}>${esc(v)}</option>`).join("");
    const src = S.manifoldSource || fit.items[0];
    const tgt = S.manifoldTarget || fit.items[fit.items.length - 1];
    steer = `<div class="panel">
      <div class="panel-h"><h3>Steer along the manifold</h3><span class="tag">replace · ${fit.kind === "cyclic" ? "shortest arc" : "interpolate"}</span></div>
      <div class="row" style="gap:10px;flex-wrap:wrap;align-items:center">
        <span class="muted">from</span><select class="field" id="mSource" style="width:auto">${itemOpts(src)}</select>
        <span class="muted">to</span><select class="field" id="mTarget" style="width:auto">${itemOpts(tgt)}</select>
        <span class="muted">waypoints</span><input class="field" id="mWaypoints" type="number" min="2" max="16" value="${S.manifoldWaypoints}" style="width:70px">
        <button class="btn" id="mSteerBtn" data-act="manifoldSteer">Steer</button>
        <button class="btn ghost" id="mCompareBtn" data-act="manifoldCompare">Compare vs linear</button>
        <button class="btn ghost" id="mPullbackBtn" data-act="manifoldPullback">Pullback</button>
        <span class="muted">— or click a point in 3D</span>
      </div>
      <div id="manifoldOut">${manifoldOutHTML()}</div>
    </div>`;
  }
  return `<div class="stage-head"><div class="k">04 · Manifold</div>
    <h1>Steer along the concept manifold</h1>
    <p>Fit a concept's activation manifold (per-value residual centroids → spline), see its 3D geometry, then traverse it to steer — the intervention replaces the concept's residual with the point you move to (paper-faithful), staying on-manifold.</p></div>
  <div class="panel">
    <div class="panel-h"><h3>Concept manifold</h3>${quality}</div>
    <div class="row" style="gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:12px">
      <select class="field" id="manifoldConcept" style="width:auto">${opts}</select>
      <span class="muted">layer</span><input class="field" id="mLayer" type="number" min="0" value="${S.manifoldLayer == null ? recommendedLayer(S.manifoldConcept) : S.manifoldLayer}" style="width:64px">
      <button class="btn" id="manifoldFitBtn" data-act="manifoldFit">Fit manifold</button>
      ${manifoldLayerHint()}
    </div>
    <div class="mwrap"><div id="manifoldCanvas" class="manifold-canvas"></div></div>
    ${fit
      ? `<div class="row" style="gap:12px;align-items:center;margin-top:8px;flex-wrap:wrap">
          <button class="btn ghost sm" id="manifoldSaeBtn" data-act="manifoldSae" style="${S.manifoldShowSae ? 'border-color:var(--brand-dim);color:var(--brand)' : ''}">SAE coverage${S.manifoldShowSae ? ' ✓' : ''}</button>
          <span class="maplegend" style="margin:0"><span class="ml"><span class="dot" style="background:#5eead4"></span> value</span><span class="ml"><span class="dot" style="background:#f59e0b"></span> handle</span><span class="ml muted">drag to orbit · click a point to steer</span></span>
        </div>
        <div id="manifoldSae">${manifoldSaeHTML()}</div>`
      : `<div class="placeholder" style="margin-top:8px"><div class="big">∿</div>Pick a concept and press <b>Fit manifold</b> to build its 3D geometry.</div>`}
  </div>
  ${steer}`;
}

function perpBadge(p) {
  if (p == null) return "";
  const cls = p < 30 ? "good" : p < 120 ? "mid" : "bad";
  return `<span class="perp ${cls}">ppl ${p.toFixed(1)}</span>`;
}
function energyBadge(e) {
  if (e == null) return "";
  const cls = e < 0.3 ? "good" : e < 0.8 ? "mid" : "bad";
  return `<span class="perp ${cls}" title="distance to behavior manifold (lower=more faithful)">E ${e.toFixed(2)}</span>`;
}
const trajHTML = wps => wps.map(w => `<div class="wrow"><span class="wv">${esc(w.value)}</span><span class="wt">${esc(short(w.text, 90))}</span></div>`).join("");

function manifoldOutHTML() {
  if (S.manifoldPullback) {
    const c = S.manifoldPullback;
    const rBadge = r => `<span class="perp" title="recovers ℳ_h (higher=path traces the manifold)">R ${r == null ? "—" : r.toFixed(2)}</span>`;
    const pane = (leg, name, col) => `<div class="bapane ${name === "manifold" ? "after" : ""}"><div class="h" style="color:${col}">${name} → ${esc(c.target)} ${energyBadge(leg.mean_energy)} ${rBadge(leg.recovered_r)}</div><div class="txt">${esc(short(leg.steered_text || "", 150)) || '<i>(empty)</i>'}</div></div>`;
    return `<div class="ba" style="margin-top:12px;grid-template-columns:1fr 1fr 1fr">
        ${pane(c.manifold, "manifold", "#5eead4")}${pane(c.linear, "linear", "#fca5a5")}${pane(c.pullback, "pullback", "#a78bfa")}
      </div>
      <div class="muted" style="margin-top:8px"><b>E</b> = behavior-manifold distance (lower = more faithful) · <b>R</b> = recovers ℳ_h (higher = traces the manifold) · pullback opt loss ${c.pullback.loss_start == null ? "—" : c.pullback.loss_start} → ${c.pullback.loss_end == null ? "—" : c.pullback.loss_end}</div>
      <div class="row" style="margin-top:12px;align-items:center;gap:10px"><button class="btn" id="mSaveBtn" data-act="saveManifold">Save to Library</button><span class="muted">saves this concept ${esc(S.manifoldSource || "")}→${esc(c.target)} steer as a manifold recipe (verdict from the energy comparison)</span></div>`;
  }
  if (S.manifoldCompare) {
    const c = S.manifoldCompare;
    return `<div class="ba" style="margin-top:12px">
        <div class="bapane after"><div class="h">manifold → ${esc(c.target)} ${energyBadge(c.manifold.mean_energy)} ${perpBadge(c.manifold.mean_perplexity)}</div><div class="txt">${esc(short(c.manifold.steered_text || "", 170)) || '<i>(empty)</i>'}</div></div>
        <div class="bapane"><div class="h" style="color:#fca5a5">linear → ${esc(c.target)} ${energyBadge(c.linear.mean_energy)} ${perpBadge(c.linear.mean_perplexity)}</div><div class="txt">${esc(short(c.linear.steered_text || "", 170)) || '<i>(empty)</i>'}</div></div>
      </div>
      <div class="muted" style="margin-top:8px"><b>E</b> = mean distance to the behavior manifold (lower = more faithful) · ppl = raw fluency · <span class="mono">unsteered:</span> ${esc(short(c.unsteered_text || "", 90))}</div>
      <div class="ba" style="margin-top:12px">
        <div><div class="panel-h"><h3>Manifold path</h3></div><div class="traj">${trajHTML(c.manifold.waypoints)}</div></div>
        <div><div class="panel-h"><h3>Linear path</h3></div><div class="traj">${trajHTML(c.linear.waypoints)}</div></div>
      </div>`;
  }
  const r = S.manifoldResult;
  if (!r) return "";
  return `<div class="ba" style="margin-top:12px">
      <div class="bapane before"><div class="h">unsteered</div><div class="txt">${esc(short(r.unsteered_text || "", 170)) || '<i>(empty)</i>'}</div></div>
      <div class="bapane after"><div class="h">steered → ${esc(r.target)} ${perpBadge(r.perplexity)}</div><div class="txt">${esc(short(r.steered_text || "", 170)) || '<i>(empty)</i>'}</div></div>
    </div>
    <div class="panel-h" style="margin-top:14px"><h3>Trajectory</h3><span class="tag">${r.waypoints.length} waypoints · hook ${r.hook_fired ? "✓" : "—"}</span></div>
    <div class="traj">${trajHTML(r.waypoints)}</div>`;
}

function coverageColors(cov) {
  const doms = [...new Set(cov.per_value.map(p => p.dominant_feature))];
  const fcol = {};
  doms.forEach((f, i) => { fcol[f] = `hsl(${(i * 67) % 360} 70% 62%)`; });
  const valColor = {};
  cov.per_value.forEach(p => { valColor[p.value] = fcol[p.dominant_feature]; });
  return { fcol, valColor };
}

function manifoldSaeHTML() {
  const cov = S.manifoldCoverage;
  if (!cov || !S.manifoldShowSae) return "";
  const fcol = (S.manifoldCoverageColors || { fcol: {} }).fcol;
  const rows = cov.tiling.slice(0, 18).map(t => {
    const col = fcol[t.feature_id] || "var(--brand)";
    return `<div class="wrow" data-act="pinCoverage" data-arg="${t.feature_id}"><span class="wv" style="color:${col}">#${t.feature_id}</span><span class="wt">${t.label ? esc(t.label) : '<i style="color:var(--faint)">unlabeled</i>'} · tiles ${t.n_values} (${esc(t.covers.slice(0, 4).join(", "))})</span><span data-act="coverageToSteer" data-arg="${t.feature_id}" title="pin this feature and open Steer at L${manifoldLayer()}" style="margin-left:auto;flex:none;font-size:11px;color:var(--brand);cursor:pointer">steer →</span></div>`;
  }).join("");
  return `<div class="panel-h" style="margin-top:12px"><h3>SAE features that tile this manifold</h3><span class="tag">${cov.n_distinct_features} distinct · top-${cov.top_k}/value${cov.synthetic ? " · synthetic (dev)" : ""}</span></div>
    <div class="traj">${rows}</div>`;
}

function mountManifold(retries = 25) {
  if (!window.LabManifold) { if (retries > 0) setTimeout(() => mountManifold(retries - 1), 100); return; }
  const ok = window.LabManifold.mount("#manifoldCanvas", {
    onPick: v => { S.manifoldTarget = v; const t = $("#mTarget"); if (t) t.value = v; runManifoldSteer(); },
  });
  if (ok && S.manifoldFit) { window.LabManifold.render(S.manifoldFit); if (S.manifoldTarget) window.LabManifold.setActive(S.manifoldTarget); }
}

function manifoldLayer() { return S.manifoldLayer == null ? recommendedLayer(S.manifoldConcept) : S.manifoldLayer; }
function manifoldLayerHint() {
  const p = (S.manifoldPresets || MANIFOLD_PRESETS_FALLBACK).find(x => x.name === S.manifoldConcept);
  const bl = p && p.best_layer, nl = manifoldNumLayers();
  if (bl != null && (nl === 0 || bl < nl)) return `<span class="muted">recommended <b>L${bl}</b> (atlas) · residual-stream PCA → spline</span>`;
  if (bl == null) return `<span class="muted">diffuse geometry — no clean manifold layer · residual-stream PCA → spline</span>`;
  return `<span class="muted">residual-stream PCA → spline</span>`;
}

async function runManifoldFit(btn) {
  S.manifoldConcept = ($("#manifoldConcept") && $("#manifoldConcept").value) || S.manifoldConcept;
  if ($("#mLayer") && $("#mLayer").value !== "") S.manifoldLayer = +$("#mLayer").value;
  await withBusy(btn, async () => {
    S.manifoldFit = await api("/api/manifold/fit", { concept: S.manifoldConcept, layer: manifoldLayer() });
    S.manifoldResult = null; S.manifoldCompare = null; S.manifoldPullback = null; S.manifoldCoverage = null; S.manifoldShowSae = false;
    S.manifoldSource = S.manifoldFit.items[0];
    S.manifoldTarget = S.manifoldFit.items[S.manifoldFit.items.length - 1];
    renderStage();
    toast(`Fit ${S.manifoldFit.label} · ${S.manifoldFit.quality.metric_name} ${S.manifoldFit.quality.metric}`);
  });
}

async function runManifoldSteer(btn) {
  if (!S.manifoldFit) return;
  S.manifoldSource = ($("#mSource") && $("#mSource").value) || S.manifoldSource;
  S.manifoldTarget = ($("#mTarget") && $("#mTarget").value) || S.manifoldTarget;
  S.manifoldWaypoints = ($("#mWaypoints") && +$("#mWaypoints").value) || S.manifoldWaypoints;
  await withBusy(btn || $("#mSteerBtn"), async () => {
    S.manifoldCompare = null; S.manifoldPullback = null;
    S.manifoldResult = await api("/api/manifold/steer", {
      concept: S.manifoldFit.concept, source: S.manifoldSource, target: S.manifoldTarget,
      layer: manifoldLayer(), n_waypoints: S.manifoldWaypoints,
    });
    const out = $("#manifoldOut");
    if (out) { out.innerHTML = manifoldOutHTML(); reveal(out, ".wrow", 24, 16); }
    if (window.LabManifold) { window.LabManifold.setActive(S.manifoldTarget); window.LabManifold.animatePath(S.manifoldResult.path_3d); }
    toast(`Steered ${S.manifoldSource} → ${S.manifoldTarget}`);
  });
}

async function runManifoldCompare(btn) {
  if (!S.manifoldFit) return;
  S.manifoldSource = ($("#mSource") && $("#mSource").value) || S.manifoldSource;
  S.manifoldTarget = ($("#mTarget") && $("#mTarget").value) || S.manifoldTarget;
  S.manifoldWaypoints = ($("#mWaypoints") && +$("#mWaypoints").value) || S.manifoldWaypoints;
  await withBusy(btn || $("#mCompareBtn"), async () => {
    S.manifoldResult = null; S.manifoldPullback = null;
    S.manifoldCompare = await api("/api/manifold/compare", {
      concept: S.manifoldFit.concept, source: S.manifoldSource, target: S.manifoldTarget,
      layer: manifoldLayer(), n_waypoints: S.manifoldWaypoints,
    });
    const out = $("#manifoldOut");
    if (out) { out.innerHTML = manifoldOutHTML(); reveal(out, ".wrow", 14, 24); }
    if (window.LabManifold) {
      window.LabManifold.setActive(S.manifoldTarget);
      window.LabManifold.renderComparePaths(S.manifoldCompare.manifold.path_3d, S.manifoldCompare.linear.path_3d);
    }
    const c = S.manifoldCompare;
    toast(`manifold ppl ${c.manifold.perplexity ? c.manifold.perplexity.toFixed(1) : "—"} vs linear ${c.linear.perplexity ? c.linear.perplexity.toFixed(1) : "—"}`);
  });
}

async function runManifoldPullback(btn) {
  if (!S.manifoldFit) return;
  S.manifoldSource = ($("#mSource") && $("#mSource").value) || S.manifoldSource;
  S.manifoldTarget = ($("#mTarget") && $("#mTarget").value) || S.manifoldTarget;
  S.manifoldWaypoints = ($("#mWaypoints") && +$("#mWaypoints").value) || S.manifoldWaypoints;
  await withBusy(btn || $("#mPullbackBtn"), async () => {
    S.manifoldResult = null; S.manifoldCompare = null;
    S.manifoldPullback = await api("/api/manifold/pullback", {
      concept: S.manifoldFit.concept, source: S.manifoldSource, target: S.manifoldTarget,
      layer: manifoldLayer(), n_waypoints: Math.min(S.manifoldWaypoints, 6),
    });
    const out = $("#manifoldOut");
    if (out) { out.innerHTML = manifoldOutHTML(); }
    if (window.LabManifold) {
      window.LabManifold.setActive(S.manifoldTarget);
      const c = S.manifoldPullback;
      window.LabManifold.renderPaths([
        { points: c.manifold.path_3d, color: 0x5eead4 },
        { points: c.linear.path_3d, color: 0xef4444 },
        { points: c.pullback.path_3d, color: 0xa78bfa },
      ]);
    }
    const pb = S.manifoldPullback.pullback;
    toast(`pullback E ${pb.mean_energy ?? "—"} · recovers ℳ_h R ${pb.recovered_r ?? "—"}`);
  });
}

async function runManifoldSaeCoverage(btn) {
  if (!S.manifoldFit) return;
  if (S.manifoldShowSae) {  // toggle off
    S.manifoldShowSae = false; S.manifoldCoverage = null; S.manifoldCoverageColors = null;
    const el = $("#manifoldSae"); if (el) el.innerHTML = "";
    const b = $("#manifoldSaeBtn"); if (b) { b.style.color = ""; b.style.borderColor = ""; b.textContent = "SAE coverage"; }
    if (window.LabManifold) window.LabManifold.colorByGroup(null);
    return;
  }
  await withBusy(btn || $("#manifoldSaeBtn"), async () => {
    S.manifoldCoverage = await api("/api/manifold/sae_coverage", { concept: S.manifoldFit.concept, layer: manifoldLayer(), top_k: 5 });
    S.manifoldShowSae = true;
    S.manifoldCoverageColors = coverageColors(S.manifoldCoverage);
    const el = $("#manifoldSae"); if (el) { el.innerHTML = manifoldSaeHTML(); reveal(el, ".wrow", 16, 18); }
    const b = $("#manifoldSaeBtn"); if (b) { b.style.color = "var(--brand)"; b.style.borderColor = "var(--brand-dim)"; b.textContent = "SAE coverage ✓"; }
    if (window.LabManifold) window.LabManifold.colorByGroup(S.manifoldCoverageColors.valColor);
    toast(`${S.manifoldCoverage.n_distinct_features} SAE features tile ${S.manifoldFit.label}`);
  });
}

function viewContrast() {
  let bars = `<div class="placeholder"><div class="big">⇄</div>Enter two prompts and press <b>Compare</b> to see which features separate them.</div>`;
  if (S.lastCompare) {
    const rows = [...S.lastCompare.positive_stronger, ...S.lastCompare.negative_stronger];
    const mx = Math.max(...rows.map(r => Math.abs(r.difference)), 1e-6);
    const sorted = rows.slice().sort((a, b) => b.difference - a.difference);
    bars = `<div class="divlabels"><span style="color:var(--neg)">◀ pulls negative</span><span style="color:var(--pos)">pulls positive ▶</span></div>` +
      sorted.map(r => {
        const w = (Math.abs(r.difference) / mx * 100).toFixed(0);
        const bar = r.difference >= 0 ? `<span class="br" style="width:${w / 2}%"></span>` : `<span class="bl" style="width:${w / 2}%"></span>`;
        const lab = fLabel(r.feature_id);
        return `<div class="divrow" data-act="pinContrast" data-arg="${r.feature_id}" title="${lab ? esc(lab) : 'unlabeled'}">
          <div class="divtrack"><span class="mid"></span>${bar}</div>
          <span class="dfid">#${r.feature_id} <span style="color:var(--faint)">${r.difference > 0 ? '+' : ''}${r.difference.toFixed(2)}</span></span>
        </div>`;
      }).join("");
  }
  return `<div class="panel">
    <div class="panel-h"><h3>Contrast lens</h3><span class="tag">compare_prompts · layer ${S.layer}</span></div>
    <div class="row" style="align-items:stretch;gap:12px">
      <textarea class="field" id="posPrompt" style="flex:1">${esc(S.posPrompt)}</textarea>
      <textarea class="field" id="negPrompt" style="flex:1">${esc(S.negPrompt)}</textarea>
    </div>
    <div class="row" style="margin-top:12px"><button class="btn" id="compareBtn" data-act="compare">Compare</button>
      <span class="muted">difference = positive_max − negative_max · click a bar to pin</span></div>
  </div>
  <div class="panel"><div class="panel-h"><h3>Contrastive features</h3></div>${bars}</div>`;
}

async function runInspect(btn, silent) {
  S.prompt = ($("#promptInput") && $("#promptInput").value) || S.prompt;
  await withBusy(btn, async () => {
    S.lastInspection = await api("/api/inspect", { prompt: S.prompt, layer: S.layer, top_k: 12, max_seq_len: 128 });
    S.selToken = S.lastInspection.tokens.length - 1;
    if (!silent) toast("Inspected " + S.lastInspection.tokens.length + " tokens");
    animateNext = true; renderRail(); renderStage();
    reveal($("#stage .tokrow"), ".tok", 22, 40);
  }).catch(() => { renderStage(); });
}
async function runCompare(btn) {
  S.posPrompt = $("#posPrompt").value; S.negPrompt = $("#negPrompt").value;
  await withBusy(btn, async () => {
    S.lastCompare = await api("/api/compare", { positive: S.posPrompt, negative: S.negPrompt, layer: S.layer, limit: 12 });
    toast("Compared prompts"); animateNext = true; renderStage();
    reveal($("#stage"), ".divrow", 26);
  });
}

/* ----------------------------- STEER ----------------------------- */
function viewSteer() {
  if (!S.pinned) return `<div class="stage-head"><div class="k">02 · Steer</div><h1>No feature armed</h1>
    <p>Pin a feature in Explore first, then come back to steer with it.</p></div>
    <button class="btn" data-act="mode" data-arg="explore">← Go to Explore</button>`;
  const lab = S.pinned.label;
  return `<div class="stage-head"><div class="k">02 · Steer</div>
    <h1>Steering studio</h1>
    <p>Add <span class="mono">strength × W_dec[:, ${S.pinned.id}]</span> to the layer-${S.layer} residual stream, then generate. Watch the output move.</p></div>
  <div class="armhdr">
    <span class="fid">#${S.pinned.id}</span><span class="lab">${lab ? esc(lab) : '<i style="color:var(--faint)">unlabeled feature</i>'}</span>
    <button class="btn ghost sm swap" data-act="mode" data-arg="explore">swap feature</button>
  </div>
  <div class="panel">
    <div class="panel-h"><h3>Prompt</h3><span class="tag">steer · all positions</span></div>
    <textarea class="field" id="steerPrompt">${esc(S.steerPrompt)}</textarea>
  </div>
  <div class="panel">
    <div class="studio">
      <div class="dialbox">
        <svg width="168" height="168" viewBox="0 0 170 170">
          <circle cx="85" cy="85" r="60" fill="none" stroke="var(--border)" stroke-width="10"/>
          <circle id="dialArc" cx="85" cy="85" r="60" fill="none" stroke="var(--pos)" stroke-width="10" stroke-linecap="round"
            stroke-dasharray="377" stroke-dashoffset="377" transform="rotate(-90 85 85)"/>
          <line id="dialNeedle" x1="85" y1="85" x2="85" y2="33" stroke="#fff" stroke-width="3" stroke-linecap="round"/>
          <circle cx="85" cy="85" r="5" fill="#fff"/>
        </svg>
        <div class="dialval" id="dialVal">+0.0</div>
        <div class="dialcap">strength</div>
        <input type="range" id="strengthSlider" min="-15" max="15" step="0.5" value="${S.strength}">
        <button class="btn" id="genBtn" data-act="generate" style="width:100%;margin-top:14px;justify-content:center">Generate</button>
      </div>
      <div id="steerOut">${steerOutHTML()}</div>
    </div>
  </div>
  <div class="panel">
    <div class="panel-h"><h3>Strength sweep</h3><span class="tag">includes 0 · click a frame to set strength</span></div>
    <div class="row" style="margin-bottom:12px"><button class="btn ghost sm" id="sweepBtn" data-act="sweep">Run sweep (−10 … +10)</button>
      <span class="muted">generates at five strengths</span></div>
    <div class="strip" id="strip">${sweepHTML()}</div>
  </div>
  <div class="row"><button class="btn" data-act="sendMeasure">Send to Measure →</button>
    <span class="muted">a fired hook ≠ a real behavior change. Confirm it in Measure.</span></div>`;
}
function steerOutHTML() {
  const r = S.steerResult;
  if (!r) return `<div class="placeholder" style="height:100%"><div class="big">↗</div>Set a strength and press <b>Generate</b> to see the steered output and evidence.</div>`;
  const hooked = r.hook_fired;
  return `<div class="ba">
    <div class="bapane before"><div class="h">unsteered · baseline</div><div class="txt">${esc(r.unsteered_text) || '<i>(empty)</i>'}</div></div>
    <div class="bapane after"><div class="h">steered · ${r.strength > 0 ? '+' : ''}${r.strength}</div><div class="txt">${esc(r.steered_text) || '<i>(empty)</i>'}</div></div>
  </div>
  <div class="gauges">
    <div class="gauge"><div class="gk">hook</div><div class="gv ${hooked ? 'green' : 'off'}">${hooked ? '✓ fired' : '— off'}</div></div>
    <div class="gauge"><div class="gk">hidden Δ</div><div class="gv teal" data-to="${Number(r.hidden_delta_norm).toFixed(2)}">0.00</div></div>
    <div class="gauge"><div class="gk">logits Δ</div><div class="gv teal" ${r.logits_delta_norm == null ? '' : `data-to="${Number(r.logits_delta_norm).toFixed(2)}"`}>${r.logits_delta_norm == null ? '—' : '0.00'}</div></div>
  </div>`;
}
function sweepHTML() {
  const sw = S.sweepResult;
  if (!sw) return `<span class="muted">No sweep yet.</span>`;
  return sw.frames.map(f => {
    const s = f.strength, on = Math.abs(s - S.strength) < 2.5;
    const cls = s < 0 ? "neg" : s === 0 ? "zero" : "";
    return `<div class="frame ${cls} ${on ? 'on' : ''}" data-act="setStrength" data-arg="${s}">
      <div class="s">str <b>${s > 0 ? '+' : ''}${s}</b></div>
      <div class="t">${esc(short(f.text, 110)) || '<i>(empty)</i>'}</div></div>`;
  }).join("");
}
function syncDial() {
  const str = S.strength, dv = $("#dialVal"); if (!dv) return;
  dv.textContent = (str > 0 ? "+" : "") + str.toFixed(1);
  dv.className = "dialval " + (str > 0 ? "pos" : str < 0 ? "neg" : "zero");
  const C = 377, fracMag = Math.min(1, Math.abs(str) / 15);
  const arc = $("#dialArc");
  arc.setAttribute("stroke-dashoffset", (C * (1 - fracMag)).toFixed(1));
  arc.setAttribute("stroke", str >= 0 ? "var(--pos)" : "var(--neg)");
  $("#dialNeedle").setAttribute("transform", `rotate(${(str / 15 * 135).toFixed(1)} 85 85)`);
}
async function runSteer(btn) {
  S.steerPrompt = ($("#steerPrompt") && $("#steerPrompt").value) || S.steerPrompt;
  await withBusy(btn, async () => {
    S.steerResult = await api("/api/steer", { prompt: S.steerPrompt, layer: S.layer, feature_id: S.pinned.id, strength: S.strength });
    const out = $("#steerOut"); if (out) { out.innerHTML = steerOutHTML(); reveal(out, ".bapane", 80); animateGauges(out); }
    renderRail();
  });
}
async function runSweep(btn) {
  S.steerPrompt = ($("#steerPrompt") && $("#steerPrompt").value) || S.steerPrompt;
  await withBusy(btn, async () => {
    S.sweepResult = await api("/api/sweep", { prompt: S.steerPrompt, layer: S.layer, feature_id: S.pinned.id, strengths: [-10, -5, 0, 5, 10], max_new_tokens: 18 });
    const strip = $("#strip"); if (strip) { strip.innerHTML = sweepHTML(); reveal(strip, ".frame", 50); }
    toast("Swept 5 strengths");
  });
}

/* ----------------------------- MEASURE ----------------------------- */
const OBJECTIVES = ["maximize_rule_score", "maximize_json_validity", "minimize_length_without_empty_output"];
function viewMeasure() {
  const armed = S.pinned
    ? `<p class="muted" style="margin-bottom:10px">Armed · feature <span class="mono" style="color:var(--brand)">#${S.pinned.id}</span> @ strength <span class="mono" style="color:var(--pos)">${S.strength > 0 ? '+' : ''}${S.strength.toFixed(1)}</span> · layer ${S.layer}</p>`
    : `<p class="muted" style="margin-bottom:10px">Pin a feature in Explore to measure it.</p>`;
  const live = S.steerResult ? steerOutHTML() : `<div class="placeholder"><div class="big">▤</div>Run a live check to generate one paired unsteered / steered output with real hook evidence.</div>`;
  return `<div class="stage-head"><div class="k">03 · Measure</div>
    <h1>Did it actually work?</h1>
    <p>A fired hook alone isn't proof. The seven-control benchmark asks the real question: does steering beat a plain prompt instruction and every control — or do they tie it?</p></div>
  <div class="panel">
    <div class="panel-h"><h3>Live causal check</h3><span class="tag">unsteered vs steered</span></div>
    ${armed}
    <div class="row" style="margin-bottom:14px">
      <button class="btn" id="measureBtn" data-act="generate" ${S.pinned ? '' : 'disabled'}>Run live check</button>
      <span class="muted">single generation pair at the current strength</span>
    </div>
    <div id="steerOut">${live}</div>
  </div>
  <div class="panel">
    <div class="panel-h"><h3>Seven-control benchmark</h3><span class="tag">the validation gate</span></div>
    <textarea class="field" id="benchPrompts" style="min-height:84px;font-family:var(--mono);font-size:12.5px">${esc(S.benchPrompts)}</textarea>
    <div class="row" style="margin:12px 0">
      <select class="field" id="benchObjective" style="width:auto;padding:9px 12px">
        ${OBJECTIVES.map(o => `<option ${S.benchObjective === o ? 'selected' : ''}>${o}</option>`).join("")}
      </select>
      <button class="btn" id="benchBtn" data-act="runBenchmark" ${S.pinned ? '' : 'disabled'}>Run benchmark</button>
      <span class="muted">${S.pinned ? 'runs all 7 methods over each prompt' : 'pin a feature first'}</span>
    </div>
    <div id="benchOut">${benchHTML(S.benchResult)}</div>
  </div>
  <div class="panel">
    <div class="panel-h"><h3>Autopilot · discovery run</h3><span class="tag">examples → recipe</span></div>
    <p class="muted" style="margin-bottom:12px">Skip the manual hunt: give behavior examples and autopilot searches candidate features, benchmarks each against all seven controls, sweeps strength, and saves the best as a recipe. It validates on the prompt set above.</p>
    <div class="row" style="align-items:stretch;gap:12px">
      <textarea class="field" id="apPos" style="flex:1" placeholder="positive examples (one per line)">${esc(S.apPositive)}</textarea>
      <textarea class="field" id="apNeg" style="flex:1" placeholder="negative examples (one per line)">${esc(S.apNegative)}</textarea>
    </div>
    <div class="row" style="margin:12px 0">
      <span class="muted">candidates</span>
      <input class="field" id="apCount" type="number" min="1" max="6" value="${S.apCount}" style="width:74px">
      <button class="btn" id="apBtn" data-act="runAutopilot">Run autopilot</button>
      <span class="muted">searches layer ${S.layer} · uses the objective above</span>
    </div>
    <div id="apOut">${S.autopilotResult ? autopilotHTML(S.autopilotResult) : '<div class="placeholder"><div class="big">◎</div>Run autopilot to auto-discover, benchmark, and save a steering feature.</div>'}</div>
  </div>`;
}
function apStepper(active) {
  const steps = ["examples", "candidates", "bench", "sweep", "recipe"];
  return `<div class="stepper">` + steps.map((s, i) =>
    `<span class="st ${i < active ? 'done' : i === active ? 'active' : ''}"><span class="d"></span>${s}</span>`).join('<span class="dash">──</span>') + `</div>`;
}
function autopilotHTML(r) {
  const cands = r.candidates || [];
  const cmx = Math.max(...cands.map(c => Math.abs(c.combined_score)), 1e-6);
  const board = cands.map(c => {
    const best = r.best_candidate && r.best_candidate.feature_id === c.feature_id;
    return `<div class="fbar">
      <span class="fid">#${c.feature_id}</span>
      <span class="flabel">L${c.layer} · contrast ${Number(c.contrast || 0).toFixed(2)}</span>
      <span class="track"><span class="fill" style="width:${(Math.abs(c.combined_score) / cmx * 100).toFixed(0)}%;${best ? 'background:var(--brand)' : ''}"></span></span>
      <span class="val">${Number(c.combined_score || 0).toFixed(2)}</span>${best ? '<span class="pin" style="color:var(--brand)">★</span>' : '<span class="pin"></span>'}
    </div>`;
  }).join("");
  const vals = r.methods.map(m => r.method_scores[m]);
  const mx = Math.max(...vals), lo = Math.min(0, ...vals), range = (mx - lo) || 1;
  const vb = r.methods.slice().sort((a, b) => r.method_scores[b] - r.method_scores[a]).map(m => {
    const s = r.method_scores[m];
    const ctrl = !/^(steering_only|prompt_plus_steering|prompt_only)$/.test(m);
    const tint = /^(steering_only|prompt_plus_steering)$/.test(m) ? "var(--brand)" : (m === "prompt_only" ? "var(--warn)" : "var(--faint)");
    return `<div class="score"><div class="top"><span class="m">${esc(m)}${ctrl ? ' <span class="ctrl">⊘ control</span>' : ''}</span><span class="v">${s.toFixed(2)}</span></div>
      <div class="tr"><span class="fl" style="width:${((s - lo) / range * 100).toFixed(0)}%;background:${tint}"></span></div></div>`;
  }).join("");
  const vd = r.validation_decision || {};
  const status = (r.best_recipe || {}).status || vd.status || "candidate";
  const cls = status === "validated" ? "val" : status === "candidate" ? "cand" : "benchmarked";
  return `${apStepper(5)}
    <p class="muted" style="margin:6px 0 14px">Searched ${cands.length} candidate${cands.length === 1 ? '' : 's'} · best <span class="mono" style="color:var(--brand)">#${(r.best_candidate || {}).feature_id}</span> → recipe <span class="mono" style="color:var(--brand)">${esc(r.recipe_id || '')}</span> saved to the Library.</p>
    <div class="ins-block"><div class="t">CANDIDATE LEADERBOARD</div>${board}</div>
    <div class="ins-block"><div class="t">BEST CANDIDATE · VALIDATION</div>${vb}
      <div style="margin-top:12px"><span class="verdict ${cls}">${status.toUpperCase()}</span></div>
      ${vd.reason ? `<p class="reason">${esc(vd.reason)}</p>` : ''}
      ${r.warning ? `<p class="reason" style="color:var(--warn)">⚠ ${esc(r.warning)}</p>` : ''}
    </div>`;
}
async function runAutopilot(btn) {
  S.apPositive = ($("#apPos") && $("#apPos").value) || S.apPositive;
  S.apNegative = ($("#apNeg") && $("#apNeg").value) || S.apNegative;
  S.apCount = +($("#apCount") && $("#apCount").value || S.apCount);
  S.benchPrompts = ($("#benchPrompts") && $("#benchPrompts").value) || S.benchPrompts;
  S.benchObjective = ($("#benchObjective") && $("#benchObjective").value) || S.benchObjective;
  const out = $("#apOut");
  let step = 0, running = true;
  const msgs = ["reading examples…", "ranking candidates by contrast…", "benchmarking vs 7 controls…", "sweeping strength…", "writing recipe…"];
  const tick = () => { if (!running || !out) return; out.innerHTML = apStepper(Math.min(step, 4)) + `<p class="muted" style="margin-top:8px">${msgs[Math.min(step, 4)]}</p>`; step++; };
  tick(); const iv = setInterval(tick, 750);
  try {
    await withBusy(btn, async () => {
      S.autopilotResult = await api("/api/autopilot", {
        target_name: "discovered_behavior", positive_examples: S.apPositive, negative_examples: S.apNegative,
        validation_prompts: S.benchPrompts, candidate_count: S.apCount, candidate_layers: [S.layer],
        objective: S.benchObjective, max_new_tokens: 14,
      });
    });
    running = false; clearInterval(iv);
    if (out) { out.innerHTML = autopilotHTML(S.autopilotResult); reveal(out, ".ins-block", 90); }
    S.recipes = await api("/api/recipes").catch(() => S.recipes);
    toast("Autopilot → " + (S.autopilotResult.recipe_id || "done"));
  } catch (e) {
    running = false; clearInterval(iv);
    if (out) out.innerHTML = `<div class="errbar">${esc(e.message)}</div>`;
  }
}
function benchHTML(d) {
  if (!d) return `<div class="placeholder"><div class="big">▤</div>Run the benchmark to score unsteered, prompt-only, steering, and four controls — then read the verdict.</div>`;
  const vals = d.methods.map(m => d.method_scores[m]);
  const mx = Math.max(...vals), lo = Math.min(0, ...vals), range = (mx - lo) || 1;
  const tint = m => /^(steering_only|prompt_plus_steering)$/.test(m) ? "var(--brand)" : (m === "prompt_only" ? "var(--warn)" : "var(--faint)");
  const isCtrl = m => !/^(steering_only|prompt_plus_steering|prompt_only)$/.test(m);
  const board = d.methods.slice().sort((a, b) => d.method_scores[b] - d.method_scores[a]).map(m => {
    const s = d.method_scores[m], w = ((s - lo) / range * 100).toFixed(0);
    return `<div class="score"><div class="top"><span class="m">${esc(m)}${isCtrl(m) ? ' <span class="ctrl">⊘ control</span>' : ''}</span><span class="v">${s.toFixed(2)}</span></div>
      <div class="tr"><span class="fl" style="width:${w}%;background:${tint(m)}"></span></div></div>`;
  }).join("");
  const vd = d.validation_decision || {};
  const status = vd.status || "benchmarked";
  const cls = status === "validated" ? "val" : status === "candidate" ? "cand" : "benchmarked";
  const ex = d.examples && d.examples[0];
  const exHTML = ex ? `<div class="sec-t" style="font-family:var(--mono);font-size:11px;letter-spacing:.14em;color:var(--faint);text-transform:uppercase;margin:20px 0 10px">first prompt · before / after</div>
    <div class="ba"><div class="bapane before"><div class="h">unsteered</div><div class="txt">${esc(short(ex.outputs.unsteered_baseline, 220)) || '<i>(empty)</i>'}</div></div>
    <div class="bapane after"><div class="h">steering only</div><div class="txt">${esc(short(ex.outputs.steering_only, 220)) || '<i>(empty)</i>'}</div></div></div>` : '';
  return `${board}
    <div style="margin-top:16px;display:flex;align-items:center;gap:14px;flex-wrap:wrap">
      <span class="verdict ${cls}">${status.toUpperCase()}</span>
      <button class="btn ghost sm" data-act="saveRecipe" id="saveRecipeBtn">Save as recipe →</button>
    </div>
    ${vd.reason ? `<p class="reason">${esc(vd.reason)}</p>` : ''}
    ${exHTML}`;
}
async function runBenchmark(btn) {
  if (!S.pinned) return;
  S.benchPrompts = ($("#benchPrompts") && $("#benchPrompts").value) || S.benchPrompts;
  S.benchObjective = ($("#benchObjective") && $("#benchObjective").value) || S.benchObjective;
  const slug = (S.pinned.label || "steered_behavior").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "") || "steered_behavior";
  await withBusy(btn, async () => {
    S.benchResult = await api("/api/benchmark", {
      prompt_set: S.benchPrompts, feature_id: S.pinned.id, strength: S.strength, layer: S.layer,
      target_behavior: slug, target_description: S.pinned.label || "", objective: S.benchObjective, max_new_tokens: 20,
    });
    const out = $("#benchOut"); if (out) { out.innerHTML = benchHTML(S.benchResult); reveal(out, ".bapane", 80); }
    toast("Benchmark complete — " + ((S.benchResult.validation_decision || {}).status || "benchmarked"));
  });
}
async function saveRecipe(btn) {
  await withBusy(btn, async () => {
    const r = await api("/api/recipes", {});
    S.recipes = await api("/api/recipes").catch(() => S.recipes);
    toast("Saved recipe " + r.recipe_id + " (" + r.status + ")");
  });
}
async function saveManifoldRecipe(btn) {
  await withBusy(btn, async () => {
    const r = await api("/api/recipes", { kind: "manifold" });
    S.recipes = await api("/api/recipes").catch(() => S.recipes);
    toast("Saved manifold recipe " + r.recipe_id + " (" + r.status + ")");
  });
}
async function loadManifoldRecipe(d) {
  const m = (d && d.manifold) || {};
  if (!m.concept) return;
  S.recipeDetail = null; S.mode = "manifold";
  S.manifoldConcept = m.concept; S.manifoldLayer = m.layer;
  S.manifoldResult = null; S.manifoldCompare = null; S.manifoldPullback = null; S.manifoldCoverage = null; S.manifoldShowSae = false;
  S.manifoldFit = await api("/api/manifold/fit", { concept: m.concept, layer: m.layer }).catch(() => null);
  if (S.manifoldFit) {
    S.manifoldSource = S.manifoldFit.items.includes(m.source) ? m.source : S.manifoldFit.items[0];
    S.manifoldTarget = S.manifoldFit.items.includes(m.target) ? m.target : S.manifoldFit.items[S.manifoldFit.items.length - 1];
  }
  render();
  toast("Loaded " + m.concept + " " + (m.source || "") + "→" + (m.target || "") + " into Manifold");
}

/* ----------------------------- MONITOR ----------------------------- */
function viewMonitor() {
  const r = S.monitorResult;
  const layerVal = S.monitorLayer == null ? S.layer : S.monitorLayer;
  const gallery = (S.monitors || []).map(m => {
    const cls = m.status === "validated" ? "val" : m.status === "benchmarked" ? "benchmarked" : "draft";
    return `<div class="rcard ${cls}" data-act="loadMonitor" data-arg="${esc(m.monitor_id)}">
      <span class="st-chip ${cls}">${esc(m.status)}</span>
      <h4>${esc(m.behavior)}</h4>
      <div class="meta"><b>L${m.layer} · ${m.n_features} feat${m.n_features === 1 ? '' : 's'}</b><br>AUC ${m.auc == null ? '—' : (+m.auc).toFixed(2)} · F1 ${m.f1 == null ? '—' : (+m.f1).toFixed(2)}<br>${esc(m.monitor_id)}</div>
    </div>`;
  }).join("");
  return `<div class="stage-head"><div class="k">05 · Monitor</div>
    <h1>Find a feature-based detector</h1>
    <p>Give labeled examples of a behavior; the bench finds the SAE feature(s) whose activation best separates them and reports a <b>held-out</b> AUC / precision / recall plus a <b>random-feature control</b> — a cheap, interpretable runtime monitor, proven not to be chance.</p></div>
  <div class="panel">
    <div class="panel-h"><h3>Discover a monitor</h3><span class="tag">labeled examples → detector</span></div>
    <div class="row" style="gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:10px">
      <span class="muted">behavior</span><input class="field" id="monBehavior" value="${esc(S.monitorBehavior)}" style="width:auto">
      <span class="muted">layer</span><input class="field" id="monLayer" type="number" min="0" value="${layerVal}" style="width:64px">
      <span class="muted">top-k</span><input class="field" id="monTopK" type="number" min="1" max="8" value="${S.monitorTopK}" style="width:56px">
    </div>
    <div class="row" style="align-items:stretch;gap:12px">
      <textarea class="field" id="monPos" style="flex:1;min-height:96px;font-family:var(--mono);font-size:12px" placeholder="positive — behavior present (one per line)">${esc(S.monitorPos)}</textarea>
      <textarea class="field" id="monNeg" style="flex:1;min-height:96px;font-family:var(--mono);font-size:12px" placeholder="negative — behavior absent (one per line)">${esc(S.monitorNeg)}</textarea>
    </div>
    <div class="row" style="margin:12px 0"><button class="btn" id="monDiscoverBtn" data-act="monitorDiscover">Discover monitor</button>
      <span class="muted">inspects each example, ranks features by separation, evaluates held-out + a random-feature control</span></div>
    <div id="monitorOut">${monitorOutHTML()}</div>
  </div>
  ${r ? `<div class="panel">
    <div class="panel-h"><h3>Test on new text</h3><span class="tag">runtime flag</span></div>
    <textarea class="field" id="monTest" style="min-height:54px">${esc(S.monitorTestText)}</textarea>
    <div class="row" style="margin:12px 0"><button class="btn ghost" id="monScoreBtn" data-act="monitorScore">Flag text</button>
      ${S.monitorScore ? `<span class="perp ${S.monitorScore.fires ? 'bad' : 'good'}">${S.monitorScore.fires ? '⚑ FLAGGED' : '— clear'} · score ${(+S.monitorScore.score).toFixed(2)} / thr ${(+r.threshold).toFixed(2)}</span>` : '<span class="muted">score a new input against the discovered features</span>'}</div>
  </div>` : ''}
  <div class="panel">
    <div class="panel-h"><h3>Saved monitors</h3><span class="tag">${(S.monitors || []).length} saved</span></div>
    <div class="rgrid">${gallery || '<p class="muted">No saved monitors yet. Discover one above, then Save.</p>'}</div>
  </div>`;
}

function monitorOutHTML() {
  const r = S.monitorResult;
  if (!r) return `<div class="placeholder"><div class="big">◉</div>Paste positive and negative examples, then press <b>Discover monitor</b>.</div>`;
  const m = r.metrics || {}, vd = r.validation_decision || {};
  const cls = vd.status === "validated" ? "val" : "benchmarked";
  const feats = (r.per_feature || []).map(f => `<div class="wrow"><span class="wv">#${f.feature_id}</span><span class="wt">AUC ${(+f.auc).toFixed(2)} · fires pos ${f.fires_pos}/${f.n_pos}, neg ${f.fires_neg}/${f.n_neg}</span></div>`).join("");
  const bar = v => `width:${v == null ? 0 : Math.round(Math.max(0, Math.min(1, +v)) * 100)}%`;
  const metric = (k, label, tint) => `<div class="score"><div class="top"><span class="m">${label}</span><span class="v">${m[k] == null ? '—' : (+m[k]).toFixed(2)}</span></div><div class="tr"><span class="fl" style="${bar(m[k])};background:${tint || 'var(--brand)'}"></span></div></div>`;
  return `<div class="sec-t" style="margin-top:14px">selected features · combine ${r.combine} · threshold ${(+r.threshold).toFixed(2)}</div>
    <div class="traj">${feats}</div>
    <div class="sec-t" style="margin-top:16px">held-out evaluation</div>
    ${metric('auc', 'AUC (held-out)')}${metric('precision', 'precision')}${metric('recall', 'recall')}${metric('f1', 'F1')}
    ${metric('control_auc', 'random-feature control AUC ⊘', 'var(--faint)')}
    <div style="margin-top:14px;display:flex;align-items:center;gap:14px;flex-wrap:wrap">
      <span class="verdict ${cls}">${(vd.status || 'benchmarked').toUpperCase()}</span>
      <span class="muted">n_pos ${m.n_pos} · n_neg ${m.n_neg} (held-out ${m.n_test_pos}/${m.n_test_neg})</span>
      <button class="btn ghost sm" id="monSaveBtn" data-act="monitorSave">Save monitor →</button>
    </div>
    ${vd.reason ? `<p class="reason" style="margin-top:8px">${esc(vd.reason)}</p>` : ''}`;
}

async function runMonitorDiscover(btn) {
  S.monitorBehavior = ($("#monBehavior") && $("#monBehavior").value) || S.monitorBehavior;
  S.monitorPos = ($("#monPos") && $("#monPos").value) || S.monitorPos;
  S.monitorNeg = ($("#monNeg") && $("#monNeg").value) || S.monitorNeg;
  if ($("#monLayer") && $("#monLayer").value !== "") S.monitorLayer = +$("#monLayer").value;
  S.monitorTopK = +(($("#monTopK") && $("#monTopK").value) || S.monitorTopK);
  await withBusy(btn, async () => {
    S.monitorResult = await api("/api/monitor/discover", { behavior: S.monitorBehavior, positive_examples: S.monitorPos,
      negative_examples: S.monitorNeg, layer: S.monitorLayer == null ? S.layer : S.monitorLayer, top_k: S.monitorTopK });
    S.monitorScore = null; renderStage();
    toast("Monitor: " + (S.monitorResult.validation_decision || {}).status);
  });
}
async function saveMonitor(btn) {
  await withBusy(btn, async () => {
    const r = await api("/api/monitors", {});
    S.monitors = await api("/api/monitors").catch(() => S.monitors);
    renderStage();
    toast("Saved monitor " + r.monitor_id + " (" + r.status + ")");
  });
}
async function runMonitorScore(btn) {
  if (!S.monitorResult) return;
  S.monitorTestText = ($("#monTest") && $("#monTest").value) || S.monitorTestText;
  await withBusy(btn, async () => {
    S.monitorScore = await api("/api/monitor/score", { text: S.monitorTestText, features: S.monitorResult.features,
      layer: S.monitorResult.layer, threshold: S.monitorResult.threshold });
    renderStage();
  });
}
async function loadMonitor(id) {
  const d = await api("/api/monitors/" + encodeURIComponent(id)).catch(() => null);
  if (!d) return;
  const ev = d.evaluation || {};
  S.monitorBehavior = (d.behavior || {}).name || S.monitorBehavior;
  S.monitorLayer = d.layer;
  S.monitorResult = { features: d.features, combine: d.combine, threshold: d.threshold, top_k: d.top_k, layer: d.layer,
    metrics: ev, per_feature: ev.per_feature || [], validation_decision: ev.validation_decision || { status: d.status } };
  S.monitorScore = null; renderStage();
  toast("Loaded monitor " + id);
}

/* ----------------------------- LIBRARY ----------------------------- */
function viewLibrary() {
  if (S.recipeDetail) return recipeDetailHTML(S.recipeDetail);
  const filters = ["all", "validated", "candidate", "benchmarked", "draft"];
  const fbtns = filters.map(f => `<button class="${S.libFilter === f ? 'on' : ''}" data-act="libFilter" data-arg="${f}">${f}</button>`).join("");
  const list = S.recipes.filter(r => S.libFilter === "all" || r.status === S.libFilter);
  const cards = list.map(r => {
    const cls = statusClass(r.status);
    const isM = r.kind === "manifold";
    const meta = isM
      ? `<b>${esc(r.concept || '')} · ${esc(r.source || '')}→${esc(r.target || '')}</b><br>L${r.layer} · ${esc(r.manifold_path || 'manifold')} path<br>${esc(r.recipe_id)}`
      : `<b>L${r.layer} · #${r.feature_id}</b><br>${esc((r.model_id || '').split('/').pop())}<br>${esc(r.recipe_id)}`;
    return `<div class="rcard ${cls}" data-act="recipe" data-arg="${esc(r.recipe_id)}">
      <span class="st-chip ${cls}">${esc(r.status)}</span>${isM ? '<span class="tag" style="margin-left:6px">∿ manifold</span>' : ''}
      <h4>${esc(r.target_behavior)}</h4>
      <div class="meta">${meta}</div>
    </div>`;
  }).join("");
  return `<div class="stage-head"><div class="k">04 · Library</div>
    <h1>Recipe gallery</h1>
    <p>Every benchmarked experiment becomes a reproducible recipe card. The status edge shows how far it climbed the validation ladder.</p></div>
  <div class="libfilters">${fbtns}</div>
  <div class="rgrid">${cards || '<p class="muted">No recipes with this status. Run a benchmark via the CLI / Modal to create one.</p>'}</div>`;
}
function statusClass(s) {
  return { validated: "val", candidate: "cand", benchmarked: "benchmarked", draft: "draft", failed: "bad", blocked: "bad" }[s] || "draft";
}
function recipeDetailHTML(d) {
  const iv = (d.interventions && d.interventions[0]) || {};
  const tb = d.target_behavior || {};
  const model = d.model || {};
  const bench = d.benchmark || {};
  const vd = bench.validation_decision || {};
  const cls = statusClass(d.status);
  const isM = d.kind === "manifold" && d.manifold;
  const m = d.manifold || {};
  const prov = isM
    ? `<div class="pv"><div class="k">model</div><div class="v">${esc((model.model_id || '').split('/').pop())}</div></div>
       <div class="pv"><div class="k">concept</div><div class="v">${esc(m.concept || '')}</div></div>
       <div class="pv"><div class="k">steer</div><div class="v">${esc(m.source || '')}→${esc(m.target || '')}</div></div>
       <div class="pv"><div class="k">layer</div><div class="v">${esc(m.layer ?? '—')}</div></div>
       <div class="pv"><div class="k">path</div><div class="v">${esc(m.path || 'manifold')}</div></div>
       <div class="pv"><div class="k">status</div><div class="v">${esc(d.status || '')}</div></div>`
    : `<div class="pv"><div class="k">model</div><div class="v">${esc((model.model_id || '').split('/').pop())}</div></div>
       <div class="pv"><div class="k">layer</div><div class="v">${esc(iv.layer ?? '—')}</div></div>
       <div class="pv"><div class="k">feature</div><div class="v">#${esc(iv.feature_id ?? '—')}</div></div>
       <div class="pv"><div class="k">strength</div><div class="v">${esc(iv.strength ?? '—')}</div></div>
       <div class="pv"><div class="k">mode</div><div class="v">${esc(iv.mode || 'all_pos')}</div></div>
       <div class="pv"><div class="k">status</div><div class="v">${esc(d.status || '')}</div></div>`;
  const legs = bench.legs || {};
  const legHTML = (isM && Object.keys(legs).length) ? `<div class="sec-t">paths compared</div>
    <div class="traj">${["manifold", "linear", "pullback"].filter(k => legs[k]).map(k => {
      const L = legs[k];
      return `<div class="wrow"><span class="wv">${k}</span><span class="wt">${energyBadge(L.mean_energy)} ${L.recovered_r != null ? `<span class="perp">R ${(+L.recovered_r).toFixed(2)}</span>` : ''} ${esc(short(L.steered_text || '', 90))}</span></div>`;
    }).join("")}</div>` : "";
  const ex = (d.examples && d.examples[0]) || null;
  const exHTML = ex
    ? (isM
      ? `<div class="ba">
          <div class="bapane before"><div class="h">unsteered</div><div class="txt">${esc(short(ex.unsteered || '', 240)) || '<i>(empty)</i>'}</div></div>
          <div class="bapane after"><div class="h">steered → ${esc(m.target || '')}</div><div class="txt">${esc(short(ex.steered || '', 240)) || '<i>(empty)</i>'}</div></div>
        </div>`
      : `<div class="ba">
          <div class="bapane before"><div class="h">${esc(Object.keys(ex)[0] || 'input')}</div><div class="txt">${esc(short(JSON.stringify(ex[Object.keys(ex)[0]] ?? ex), 240))}</div></div>
          <div class="bapane after"><div class="h">recorded example</div><div class="txt">${esc(short(JSON.stringify(ex), 240))}</div></div>
        </div>`)
    : `<p class="muted">No before/after examples recorded.</p>`;
  const lims = (d.limitations && d.limitations.length) ? d.limitations.map(l => `<li>${esc(l)}</li>`).join("") : "<li class='muted'>None recorded.</li>";
  return `<div class="rdetail fade">
    <button class="back" data-act="libBack">← back to gallery</button>
    <span class="st-chip ${cls}">${esc(d.status || '')}</span>${isM ? '<span class="tag" style="margin-left:6px">∿ manifold</span>' : ''}
    <h2>${esc(tb.name || d.recipe_id || 'recipe')}</h2>
    <p style="color:var(--dim)">${esc(tb.description || '')}</p>
    <div class="prov">${prov}</div>
    ${legHTML}
    <div class="sec-t">recorded example</div>
    ${exHTML}
    <div class="sec-t">verdict</div>
    <p class="reason" style="margin-top:0">${esc(vd.reason || bench.summary || 'No benchmark summary recorded.')}</p>
    <div class="sec-t">limitations</div>
    <ul style="color:var(--dim);font-size:13px;padding-left:18px">${lims}</ul>
    <div class="row" style="margin-top:22px">
      ${isM
        ? `<button class="btn" data-act="loadManifold" data-arg="${esc(d.recipe_id)}">Load into Manifold</button>`
        : `${iv.feature_id != null ? `<button class="btn" data-act="loadSteer" data-arg="${esc(d.recipe_id)}">Load into Steer</button>` : ''}
           <button class="btn ghost" data-act="openMeasure">Open in Measure</button>`}
    </div>
  </div>`;
}
async function openRecipe(id) {
  await withBusy(null, async () => {
    S.recipeDetail = await api("/api/recipes/" + encodeURIComponent(id));
    renderStage(); renderInspector();
  });
}

/* ----------------------------- INSPECTOR ----------------------------- */
function renderInspector() {
  const ins = $("#inspector");
  if (S.mode === "library") {
    ins.innerHTML = `<div class="ins-h">TIP</div><div class="ins-empty"><div class="big">▦</div>Recipe cards capture everything needed to reproduce a steer: model, layer, feature, strength, controls, and before/after evidence.</div>`;
    return;
  }
  if (!S.pinned) {
    ins.innerHTML = `<div class="ins-h">INSPECTOR</div><div class="ins-empty"><div class="big">⊕</div>Pin a feature to carry it across Explore, Steer, and Measure.</div>`;
    return;
  }
  const p = S.pinned;
  const inSteer = S.mode === "steer";
  ins.innerHTML = `
    <div class="ins-h">${inSteer ? 'ARMED FEATURE' : 'PINNED FEATURE'}</div>
    <div class="ins-feat"><div class="fid">#${p.id}</div>
      <div class="lab ${p.label ? '' : 'un'}">${p.label ? esc(p.label) : 'unlabeled'}</div></div>
    ${p.topTokens && p.topTokens.length ? `<div class="chips">${p.topTokens.map(t => `<span class="chip">${esc(t)}</span>`).join("")}</div>` : ''}
    ${p.fingerprint && p.fingerprint.length ? `<div class="ins-block"><div class="t">ACTIVATION ACROSS PROMPT</div><div class="spark">${sparkSVG(p.fingerprint, 'var(--brand)')}</div></div>` : ''}
    <div class="ins-block"><div class="t">NOTEBOOK</div>
      <textarea class="note" id="featNote" placeholder="what does this feature seem to do?">${esc(p.label || "")}</textarea>
      <button class="btn ghost sm" id="saveNoteBtn" data-act="saveNote" style="margin-top:8px;width:100%;justify-content:center">Save label</button></div>
    <button class="ins-cta" data-act="${inSteer ? 'sendMeasure' : 'steerWith'}">${inSteer ? 'Send to Measure →' : 'Steer with this →'}</button>
    ${S.mode !== 'measure' ? `<button class="ins-sub" data-act="mode" data-arg="measure">Open Measure</button>` : ''}
  `;
}

/* ----------------------------- events ----------------------------- */
document.body.addEventListener("click", async e => {
  const el = e.target.closest("[data-act]"); if (!el) return;
  const act = el.dataset.act, arg = el.dataset.arg;
  switch (act) {
    case "mode": S.mode = arg; if (arg !== "library") S.recipeDetail = null; animateNext = true; render(); break;
    case "exploreView": S.exploreView = arg; animateNext = true; renderStage(); break;
    case "inspect": runInspect($("#inspectBtn")); break;
    case "token": {
      S.selToken = +arg;
      document.querySelectorAll("#stage .tok").forEach((t, i) => t.classList.toggle("sel", i === S.selToken));
      const fp = $("#featPanel"); if (fp) fp.innerHTML = featPanelInner();
      break;
    }
    case "pin": pinFromInspection(+arg); toast("Pinned #" + arg); renderInspector();
      document.querySelectorAll(".fcard").forEach(c => c.classList.toggle("pinned", c.dataset.arg === arg)); break;
    case "pinContrast": {
      const id = +arg; const row = [...(S.lastCompare.positive_stronger || []), ...(S.lastCompare.negative_stronger || [])].find(r => r.feature_id === id);
      const toks = row ? [...(row.positive_tokens || []), ...(row.negative_tokens || [])] : [];
      S.pinned = { id, label: fLabel(id), topTokens: toks.slice(0, 5), fingerprint: [] };
      toast("Pinned #" + id); renderInspector(); break;
    }
    case "compare": runCompare($("#compareBtn")); break;
    case "steerWith": S.mode = "steer"; render(); toast("Armed #" + S.pinned.id); break;
    case "generate": runSteer(el); break;
    case "sweep": runSweep($("#sweepBtn")); break;
    case "setStrength": S.strength = +arg; { const sl = $("#strengthSlider"); if (sl) sl.value = arg; } syncDial();
      document.querySelectorAll("#strip .frame").forEach(f => f.classList.toggle("on", Math.abs(+f.querySelector(".s b").textContent.replace('+','') - S.strength) < 2.5)); break;
    case "sendMeasure": S.steerPrompt = ($("#steerPrompt") && $("#steerPrompt").value) || S.steerPrompt; S.mode = "measure"; render(); break;
    case "saveNote": saveNote(el); break;
    case "libFilter": S.libFilter = arg; renderStage(); break;
    case "recipe": openRecipe(arg); break;
    case "libBack": S.recipeDetail = null; renderStage(); renderInspector(); break;
    case "runBenchmark": runBenchmark(el); break;
    case "saveRecipe": saveRecipe(el); break;
    case "runAutopilot": runAutopilot(el); break;
    case "atlasScan": runAtlasScan($("#atlasScanBtn")); break;
    case "manifoldFit": runManifoldFit($("#manifoldFitBtn")); break;
    case "manifoldSteer": runManifoldSteer($("#mSteerBtn")); break;
    case "manifoldCompare": runManifoldCompare($("#mCompareBtn")); break;
    case "manifoldPullback": runManifoldPullback($("#mPullbackBtn")); break;
    case "manifoldSae": runManifoldSaeCoverage($("#manifoldSaeBtn")); break;
    case "saveManifold": saveManifoldRecipe($("#mSaveBtn")); break;
    case "loadManifold": loadManifoldRecipe(S.recipeDetail); break;
    case "monitorDiscover": runMonitorDiscover($("#monDiscoverBtn")); break;
    case "monitorSave": saveMonitor($("#monSaveBtn")); break;
    case "monitorScore": runMonitorScore($("#monScoreBtn")); break;
    case "loadMonitor": loadMonitor(arg); break;
    case "pinCoverage": {
      const id = +arg;
      S.pinned = { id, label: fLabel(id), topTokens: [], fingerprint: [] };
      toast("Pinned #" + id); renderInspector();
      break;
    }
    case "coverageToSteer": {
      const id = +arg;
      S.pinned = { id, label: fLabel(id), topTokens: [], fingerprint: [] };
      S.layer = manifoldLayer();  // steer at the layer the manifold was fit on
      S.mode = "steer"; render(); toast("Pinned #" + id + " → Steer (L" + S.layer + ")");
      break;
    }
    case "atlasLabeled": {
      S.atlasLabeledOnly = !S.atlasLabeledOnly;
      const g = $("#atlasGrid"); if (g) g.innerHTML = atlasCards($("#atlasSearch") ? $("#atlasSearch").value : "");
      const b = $("#atlasLabeledBtn");
      if (b) { b.style.color = S.atlasLabeledOnly ? "var(--brand)" : ""; b.style.borderColor = S.atlasLabeledOnly ? "var(--brand-dim)" : ""; }
      break;
    }
    case "pinAtlas": {
      const e2 = (S.atlasData && S.atlasData.features || []).find(x => x.feature_id === +arg) || {};
      S.pinned = { id: +arg, label: fLabel(+arg), topTokens: e2.top_tokens || [], fingerprint: e2.fingerprint || [] };
      toast("Pinned #" + arg); renderInspector();
      document.querySelectorAll(".fcard").forEach(c => c.classList.toggle("pinned", c.dataset.arg === arg));
      break;
    }
    case "loadSteer": {
      const d = S.recipeDetail; const iv = (d.interventions && d.interventions[0]) || {};
      S.pinned = { id: iv.feature_id, label: (d.target_behavior || {}).name || null, topTokens: [], fingerprint: [] };
      if (iv.strength != null) { S.strength = Math.max(-15, Math.min(15, iv.strength)); }
      if (iv.layer != null) S.layer = iv.layer;
      S.recipeDetail = null; S.mode = "steer"; render(); toast("Loaded #" + iv.feature_id + " into Steer"); break;
    }
    case "openMeasure": {
      const d = S.recipeDetail; const iv = (d.interventions && d.interventions[0]) || {};
      if (iv.feature_id != null) {
        S.pinned = { id: iv.feature_id, label: (d.target_behavior || {}).name || null, topTokens: [], fingerprint: [] };
        if (iv.strength != null) S.strength = Math.max(-15, Math.min(15, iv.strength));
        if (iv.layer != null) S.layer = iv.layer;
      }
      S.benchResult = null; S.recipeDetail = null; S.mode = "measure"; render(); break;
    }
  }
});
document.body.addEventListener("input", e => {
  if (e.target.id === "strengthSlider") { S.strength = parseFloat(e.target.value); syncDial(); }
  else if (e.target.id === "atlasSearch") { const g = $("#atlasGrid"); if (g) g.innerHTML = atlasCards(e.target.value); }
  else if (e.target.id === "atlasSort") { S.atlasSort = e.target.value; const g = $("#atlasGrid"); if (g) g.innerHTML = atlasCards($("#atlasSearch") ? $("#atlasSearch").value : ""); }
  else if (e.target.id === "atlasPrompts") S.atlasPrompts = e.target.value;
  else if (e.target.id === "manifoldConcept") { S.manifoldConcept = e.target.value; S.manifoldLayer = null; renderStage(); }
  else if (e.target.id === "mLayer") S.manifoldLayer = e.target.value === "" ? null : +e.target.value;
  else if (e.target.id === "mSource") S.manifoldSource = e.target.value;
  else if (e.target.id === "mTarget") S.manifoldTarget = e.target.value;
  else if (e.target.id === "promptInput") S.prompt = e.target.value;
  else if (e.target.id === "steerPrompt") S.steerPrompt = e.target.value;
  else if (e.target.id === "posPrompt") S.posPrompt = e.target.value;
  else if (e.target.id === "negPrompt") S.negPrompt = e.target.value;
  else if (e.target.id === "benchPrompts") S.benchPrompts = e.target.value;
  else if (e.target.id === "benchObjective") S.benchObjective = e.target.value;
  else if (e.target.id === "apPos") S.apPositive = e.target.value;
  else if (e.target.id === "apNeg") S.apNegative = e.target.value;
  else if (e.target.id === "apCount") S.apCount = +e.target.value;
});
async function saveNote(btn) {
  if (!S.pinned) return;
  const label = ($("#featNote") && $("#featNote").value) || "";
  await withBusy(btn, async () => {
    await api("/api/notebook", { feature_id: S.pinned.id, layer: S.layer, human_label: label, notes: "" });
    if (label) S.labels[S.pinned.id] = label; else delete S.labels[S.pinned.id];
    S.pinned.label = label || null;
    toast("Saved label for #" + S.pinned.id); renderInspector();
    if (S.mode === "explore" && S.exploreView === "atlas") { const g = $("#atlasGrid"); if (g) g.innerHTML = atlasCards($("#atlasSearch") ? $("#atlasSearch").value : ""); }
  });
}

boot();
