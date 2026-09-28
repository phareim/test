# test.phareim.no — wgpu in the browser

A technical test of [wgpu](https://github.com/gfx-rs/wgpu) compiled to WebAssembly.
Rust renders a small island scene: procedural terrain, animated water with shoreline
foam, a sky with a sun you move with a slider, a 2048² shadow map with 3×3 PCF, 4× MSAA,
and seven Eventyrland models loaded from .glb at runtime.

wgpu picks **WebGPU** when the browser has it and falls back to **WebGL2**
(`new_instance_with_webgpu_detection`). The panel shows which backend is running.

## Layout

| Path | What |
|---|---|
| `src/lib.rs` | `App`, exported to JS: device setup, pipelines, render passes, input |
| `src/scene.rs` | vertex/uniform layouts, island height function, sky model, glTF loader |
| `src/scene.wgsl` | all shaders (sky, terrain, models, water, shadow) |
| `web/` | the static site: `index.html`, `main.js`, `models/`, and `pkg/` (build output) |

JS owns the page (canvas size, pointer input, `requestAnimationFrame`) and calls
`app.frame(t, dt)`, `app.orbit`, `app.zoom`, `app.set_sun`, `app.add_model`.

## Models

`web/models/*.glb` are the uncompressed Tripo/Hunyuan exports (PNG textures) from
`~/archive/3d-raw/eventyrland/assets/local3d/s512*/raw/`. The `gltf` crate cannot
read `EXT_meshopt_compression`, KTX2 or WebP, so Eventyrland's optimized files do not
load here. The terrain height function exists twice (`scene.rs` and `scene.wgsl`)
and must stay in sync.

## Build and deploy

```bash
heavy -- wasm-pack build --target web --release --out-dir web/pkg --no-typescript
cd web && python3 -m http.server 8791   # local test
```

Push to `main` → GitHub Actions builds the wasm and deploys `web/` as the static-assets-only
Worker `phareim-test` (`wrangler.toml`), served at https://test.phareim.no. Rust on Sleeper was
installed with rustup in `~/.cargo` for this repo (2026-09-28).

This is a throwaway test. It is redundant once a real project adopts wgpu or the test
has answered its question; then `npx wrangler delete phareim-test` (removes the custom domain too).
