/* The Lab Bench — 3D concept-manifold renderer (Three.js).
   Exposes window.LabManifold for the classic app.js to drive:
   mount(selector,{onPick}) · render(fit) · setActive(value) · animatePath(path3d,onDone) · dispose(). */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const SCALE = 4;
const BRAND = 0x5eead4;
const ACCENT = 0xf59e0b;
let S = null;

function makeLabel(text) {
  const c = document.createElement("canvas");
  c.width = 256; c.height = 128;
  const ctx = c.getContext("2d");
  ctx.font = "600 46px Inter, system-ui, sans-serif";
  ctx.fillStyle = "#cfe9e4"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(String(text), 128, 64);
  const tex = new THREE.CanvasTexture(c);
  const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
  spr.scale.set(0.95, 0.48, 1);
  return spr;
}

const v3 = xyz => new THREE.Vector3(xyz[0] * SCALE, xyz[1] * SCALE, xyz[2] * SCALE);

function teardown() {
  if (!S) return;
  cancelAnimationFrame(S.raf);
  window.removeEventListener("resize", S.onResize);
  try { S.controls.dispose(); } catch (_) {}
  try { S.renderer.dispose(); } catch (_) {}
  if (S.renderer.domElement && S.renderer.domElement.parentNode) S.renderer.domElement.parentNode.removeChild(S.renderer.domElement);
  S = null;
}

function mount(selector, opts = {}) {
  const container = document.querySelector(selector);
  if (!container) return false;
  teardown();
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  } catch (e) {
    container.innerHTML = '<div class="m3d-fallback">3D unavailable — no WebGL in this browser.</div>';
    return false;
  }
  const w = container.clientWidth || 760, h = container.clientHeight || 460;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(w, h);
  renderer.setClearColor(0x000000, 0);
  container.innerHTML = "";
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 100);
  camera.position.set(3.6, 2.4, 6.6);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.08;
  const reduced = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  controls.autoRotate = !reduced; controls.autoRotateSpeed = 0.6;
  scene.add(new THREE.AmbientLight(0xffffff, 0.85));
  const dl = new THREE.DirectionalLight(0xffffff, 0.55); dl.position.set(5, 8, 6); scene.add(dl);
  const group = new THREE.Group(); scene.add(group);

  S = { renderer, scene, camera, controls, group, container, onPick: opts.onPick, points: [], handle: null, pathObjs: [], reduced, raf: 0 };

  const ray = new THREE.Raycaster(); const m = new THREE.Vector2();
  renderer.domElement.addEventListener("click", ev => {
    if (!S || !S.onPick) return;
    const r = renderer.domElement.getBoundingClientRect();
    m.x = ((ev.clientX - r.left) / r.width) * 2 - 1;
    m.y = -((ev.clientY - r.top) / r.height) * 2 + 1;
    ray.setFromCamera(m, camera);
    const hits = ray.intersectObjects(S.points.map(p => p.mesh));
    if (hits.length) S.onPick(hits[0].object.userData.value);
  });
  S.onResize = () => {
    const W = container.clientWidth, H = container.clientHeight;
    if (!W || !H) return;
    camera.aspect = W / H; camera.updateProjectionMatrix(); renderer.setSize(W, H);
  };
  window.addEventListener("resize", S.onResize);

  const loop = () => { S && (S.raf = requestAnimationFrame(loop), S.controls.update(), S.renderer.render(scene, camera)); };
  loop();
  return true;
}

function render(fit) {
  if (!S || !fit) return;
  while (S.group.children.length) S.group.remove(S.group.children[0]);
  S.points = [];
  S.pathObjs = [];

  const curve = (fit.curve_3d || []).map(v3);
  if (curve.length) {
    const geo = new THREE.BufferGeometry().setFromPoints(curve);
    S.group.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color: BRAND, transparent: true, opacity: 0.5 })));
  }
  (fit.points_3d || []).forEach((p, i) => {
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.12, 24, 24),
      new THREE.MeshStandardMaterial({ color: BRAND, emissive: 0x0b3b35, roughness: 0.4 }));
    mesh.position.copy(v3(p.xyz)); mesh.userData = { value: p.value, index: i };
    S.group.add(mesh);
    const label = makeLabel(p.value);
    label.position.copy(v3(p.xyz)).add(new THREE.Vector3(0, 0.34, 0));
    S.group.add(label);
    S.points.push({ mesh, label, value: p.value });
  });
  S.handle = new THREE.Mesh(
    new THREE.SphereGeometry(0.19, 24, 24),
    new THREE.MeshStandardMaterial({ color: ACCENT, emissive: 0x5a3a00, roughness: 0.3 }));
  S.handle.visible = false; S.group.add(S.handle);
  S.controls.target.set(0, 0, 0);
}

function setActive(value) {
  if (!S) return;
  S.points.forEach(p => {
    const on = p.value === value;
    p.mesh.scale.setScalar(on ? 1.7 : 1);
    p.mesh.material.color.setHex(on ? 0xffffff : BRAND);
  });
}

function clearPaths() {
  if (!S) return;
  S.pathObjs.forEach(o => S.group.remove(o));
  S.pathObjs = [];
}

function addPathLine(pts, color, opacity) {
  const ln = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
  S.group.add(ln); S.pathObjs.push(ln);
}

function animatePath(path3d, onDone) {
  if (!S || !S.handle || !path3d || !path3d.length) { onDone && onDone(); return; }
  clearPaths();
  const pts = path3d.map(v3);
  addPathLine(pts, ACCENT, 0.9);
  S.handle.visible = true;
  if (S.reduced || pts.length < 2) { S.handle.position.copy(pts[pts.length - 1]); onDone && onDone(); return; }
  const seg = pts.length - 1, dur = 1100, t0 = performance.now();
  const tick = t => {
    const k = Math.min(1, (t - t0) / dur);
    const f = k * seg, i = Math.min(seg - 1, Math.floor(f));
    S.handle.position.lerpVectors(pts[i], pts[i + 1], f - i);
    if (k < 1) requestAnimationFrame(tick); else onDone && onDone();
  };
  requestAnimationFrame(tick);
}

// manifold (amber, follows curve) vs linear (red, straight chord) — the comparison
function renderComparePaths(manifoldPath, linearPath) {
  if (!S) return;
  clearPaths();
  const m = (manifoldPath || []).map(v3), l = (linearPath || []).map(v3);
  if (m.length) addPathLine(m, ACCENT, 0.95);
  if (l.length) addPathLine(l, 0xef4444, 0.95);
  if (S.handle && m.length) S.handle.visible = true;
  const lh = new THREE.Mesh(new THREE.SphereGeometry(0.17, 20, 20),
    new THREE.MeshStandardMaterial({ color: 0xef4444, emissive: 0x5a0000, roughness: 0.3 }));
  S.group.add(lh); S.pathObjs.push(lh);
  if (S.reduced || m.length < 2) {
    if (m.length) S.handle.position.copy(m[m.length - 1]);
    if (l.length) lh.position.copy(l[l.length - 1]);
    return;
  }
  const dur = 1200, t0 = performance.now();
  const tick = t => {
    const k = Math.min(1, (t - t0) / dur);
    if (m.length > 1) { const f = k * (m.length - 1), i = Math.min(m.length - 2, Math.floor(f)); S.handle.position.lerpVectors(m[i], m[i + 1], f - i); }
    if (l.length > 1) { const f = k * (l.length - 1), i = Math.min(l.length - 2, Math.floor(f)); lh.position.lerpVectors(l[i], l[i + 1], f - i); }
    if (k < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function colorByGroup(map) {
  if (!S) return;
  S.points.forEach(p => p.mesh.material.color.set((map && map[p.value]) || BRAND));
}

// N labeled paths (manifold / linear / pullback) each with its own animated handle
function renderPaths(specs) {
  if (!S) return;
  clearPaths();
  const handles = [];
  (specs || []).forEach(sp => {
    const pts = (sp.points || []).map(v3);
    if (!pts.length) return;
    addPathLine(pts, sp.color, 0.92);
    const h = new THREE.Mesh(new THREE.SphereGeometry(0.16, 18, 18),
      new THREE.MeshStandardMaterial({ color: sp.color, emissive: 0x111111, roughness: 0.3 }));
    S.group.add(h); S.pathObjs.push(h); handles.push({ h, pts });
  });
  if (S.reduced) { handles.forEach(({ h, pts }) => h.position.copy(pts[pts.length - 1])); return; }
  const dur = 1200, t0 = performance.now();
  const tick = t => {
    const k = Math.min(1, (t - t0) / dur);
    handles.forEach(({ h, pts }) => {
      if (pts.length > 1) { const f = k * (pts.length - 1), i = Math.min(pts.length - 2, Math.floor(f)); h.position.lerpVectors(pts[i], pts[i + 1], f - i); }
    });
    if (k < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

window.LabManifold = { mount, render, setActive, animatePath, renderComparePaths, renderPaths, colorByGroup, dispose: teardown, ready: true };
