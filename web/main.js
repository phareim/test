import init, { App } from "./pkg/wgpu_test.js";

// name, x, z, size (height, or length when floating), yaw, floats on water
const MODELS = [
  ["stabbur", -2.5, 0.5, 3.0, 0.4, false],
  ["lighthouse", 6.5, -4.5, 6.5, 0.0, false],
  ["tree-oak", 1.8, 3.2, 4.5, 0.0, false],
  ["badger", 0.2, -1.2, 0.7, 2.2, false],
  ["stump", -4.5, -2.8, 0.7, 0.0, false],
  ["treasure-chest", -0.8, 2.6, 0.6, -0.5, false],
  ["rowboat", 10.5, 7.0, 2.8, 0.9, true],
  ["fox-pixal3d", 0.6, -3.4, 2.0, 0.6, false],
  ["fox-trellis", 3.4, -2.2, 2.0, 0.6, false],
];

// Floating labels above the two fox test models (same GLB pipeline, 20k triangles each).
const LABELS = [
  { text: "Pixal3D · 7 min", x: 0.6, y: 2.5, z: -3.4, cls: "a" },
  { text: "TRELLIS.2 · 2 t 55 min", x: 3.4, y: 2.5, z: -2.2, cls: "b" },
];

const $ = (id) => document.getElementById(id);
const canvas = $("c");

function fail(msg) {
  $("err").style.display = "block";
  $("err").textContent = `Kunne ikke starte: ${msg}`;
}

function fitCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(canvas.clientWidth * dpr));
  canvas.height = Math.max(1, Math.round(canvas.clientHeight * dpr));
}

async function main() {
  await init();
  fitCanvas();
  const app = await App.create(canvas);
  $("backend").textContent = app.info();
  window.app = app;

  // Input: drag to orbit, wheel or pinch to zoom.
  const pointers = new Map();
  let pinch = 0;
  canvas.addEventListener("pointerdown", (e) => { canvas.setPointerCapture(e.pointerId); pointers.set(e.pointerId, e); });
  canvas.addEventListener("pointerup", (e) => { pointers.delete(e.pointerId); pinch = 0; });
  canvas.addEventListener("pointercancel", (e) => { pointers.delete(e.pointerId); pinch = 0; });
  canvas.addEventListener("pointermove", (e) => {
    const prev = pointers.get(e.pointerId);
    if (!prev) return;
    pointers.set(e.pointerId, e);
    if (pointers.size === 1) {
      app.orbit(e.clientX - prev.clientX, e.clientY - prev.clientY);
    } else if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
      if (pinch) app.zoom(pinch / d);
      pinch = d;
    }
  });
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); app.zoom(Math.exp(e.deltaY * 0.001)); }, { passive: false });

  const slider = $("t");
  slider.addEventListener("input", () => app.set_sun(parseFloat(slider.value)));
  app.set_sun(parseFloat(slider.value));

  new ResizeObserver(() => { fitCanvas(); app.resize(canvas.width, canvas.height); }).observe(canvas);

  let last = performance.now();
  let frames = 0, fpsAt = last, fps = 0;
  function tick(now) {
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    app.frame(now / 1000, dt);
    frames++;
    if (now - fpsAt > 500) {
      fps = (frames * 1000) / (now - fpsAt);
      frames = 0; fpsAt = now;
      $("stats").textContent = `${fps.toFixed(0)} fps · ${canvas.width}×${canvas.height} · ${app.triangles().toLocaleString("no")} trekanter`;
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  const labels = LABELS.map((l) => {
    const el = document.createElement("div");
    el.className = `tag ${l.cls}`;
    el.textContent = l.text;
    document.body.appendChild(el);
    return { ...l, el };
  });
  function placeLabels() {
    for (const l of labels) {
      const [sx, sy, ok] = app.project(l.x, l.y, l.z);
      l.el.style.display = ok && sx > -0.1 && sx < 1.1 ? "block" : "none";
      l.el.style.transform = `translate(${(sx * innerWidth).toFixed(1)}px, ${(sy * innerHeight).toFixed(1)}px) translate(-50%, -100%)`;
    }
    requestAnimationFrame(placeLabels);
  }
  requestAnimationFrame(placeLabels);

  // Models stream in one by one while the island is already rendering.
  let done = 0;
  $("load").textContent = `Laster modeller 0/${MODELS.length}`;
  for (const [name, x, z, size, yaw, float] of MODELS) {
    try {
      const bytes = new Uint8Array(await (await fetch(`models/${name}.glb`)).arrayBuffer());
      app.add_model(bytes, x, z, size, yaw, float);
    } catch (e) {
      console.error(name, e);
    }
    done++;
    $("load").textContent = done < MODELS.length ? `Laster modeller ${done}/${MODELS.length}` : "Dra for å rotere, rull eller knip for å zoome";
  }
}

main().catch((e) => { console.error(e); fail(e?.message || String(e)); });
