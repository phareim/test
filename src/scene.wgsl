struct Globals {
    view_proj: mat4x4f,
    light_vp: mat4x4f,
    inv_view_proj: mat4x4f,
    camera_pos: vec4f,
    sun_dir: vec4f,
    sun_color: vec4f,
    sky_top: vec4f,
    sky_horizon: vec4f,
    params: vec4f, // time, apply gamma, shadow texel, unused
};
@group(0) @binding(0) var<uniform> g: Globals;

struct Obj {
    model: mat4x4f,
    tint: vec4f,
};
@group(1) @binding(0) var<uniform> o: Obj;
@group(1) @binding(1) var tex: texture_2d<f32>;
@group(1) @binding(2) var samp: sampler;

@group(2) @binding(0) var shadow_map: texture_depth_2d;
@group(2) @binding(1) var shadow_samp: sampler_comparison;

struct VIn {
    @location(0) pos: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
};

struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
};

// ---------- shared helpers ----------

fn hash2(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

fn noise2(p: vec2f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash2(i), hash2(i + vec2f(1.0, 0.0)), u.x),
               mix(hash2(i + vec2f(0.0, 1.0)), hash2(i + vec2f(1.0, 1.0)), u.x), u.y);
}

// Mirror of scene::terrain_height in Rust.
fn terrain_height(x: f32, z: f32) -> f32 {
    let n = sin(x * 0.35) * cos(z * 0.3) * 0.35 + sin(x * 0.9 + z * 0.7) * 0.12 + cos((x - z) * 0.23) * 0.2;
    let r = sqrt(x * x + z * z) + n * 2.0;
    let island = 1.0 - smoothstep(6.0, 12.5, r);
    let knoll = exp(-((x - 6.5) * (x - 6.5) + (z + 4.5) * (z + 4.5)) / 7.0) * 1.3;
    return island * (1.7 + n * 0.5) + knoll * island - 0.9 - smoothstep(12.0, 30.0, r) * 3.0;
}

fn sky_color(dir: vec3f) -> vec3f {
    let up = max(dir.y, 0.0);
    var c = mix(g.sky_horizon.rgb, g.sky_top.rgb, pow(up, 0.55));
    let s = max(dot(dir, normalize(g.sun_dir.xyz)), 0.0);
    c += g.sun_color.rgb * (pow(s, 900.0) * 6.0 + pow(s, 12.0) * 0.18);
    if (dir.y < 0.0) {
        c = mix(g.sky_horizon.rgb, g.sky_horizon.rgb * 0.55, min(-dir.y * 3.0, 1.0));
    }
    return c;
}

fn shadow(world: vec3f, n: vec3f) -> f32 {
    let lp = g.light_vp * vec4f(world + n * 0.03, 1.0);
    let ndc = lp.xyz / lp.w;
    let uv = vec2f(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    let t = g.params.z;
    var sum = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            sum += textureSampleCompareLevel(shadow_map, shadow_samp, uv + vec2f(f32(x), f32(y)) * t * 1.5, ndc.z - 0.0015);
        }
    }
    let inside = all(uv > vec2f(0.0)) && all(uv < vec2f(1.0)) && ndc.z < 1.0;
    return select(1.0, sum / 9.0, inside);
}

fn lit(albedo: vec3f, n: vec3f, world: vec3f, shine: f32) -> vec3f {
    let l = normalize(g.sun_dir.xyz);
    let v = normalize(g.camera_pos.xyz - world);
    let ndl = max(dot(n, l), 0.0);
    let sh = shadow(world, n);
    let ground = vec3f(0.18, 0.16, 0.12);
    let ambient = mix(ground, g.sky_top.rgb, n.y * 0.5 + 0.5) * 0.55 + g.sky_horizon.rgb * 0.08;
    let h = normalize(l + v);
    let spec = pow(max(dot(n, h), 0.0), 48.0) * shine * ndl;
    return albedo * (ambient + g.sun_color.rgb * ndl * sh) + g.sun_color.rgb * spec * sh;
}

fn fog(c: vec3f, world: vec3f) -> vec3f {
    let d = distance(g.camera_pos.xyz, world);
    let f = 1.0 - exp(-max(d - 20.0, 0.0) * 0.012);
    return mix(c, g.sky_horizon.rgb, f);
}

fn finish(c: vec3f, a: f32) -> vec4f {
    // Filmic-ish tone map, then gamma only when the surface is not sRGB.
    let x = max(c, vec3f(0.0));
    var m = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    if (g.params.y > 0.5) {
        m = pow(clamp(m, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2));
    }
    return vec4f(m, a);
}

// ---------- sky ----------

struct SkyOut {
    @builtin(position) clip: vec4f,
    @location(0) ndc: vec2f,
};

@vertex fn vs_sky(@builtin(vertex_index) i: u32) -> SkyOut {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
    var out: SkyOut;
    out.clip = vec4f(p, 1.0, 1.0);
    out.ndc = p;
    return out;
}

@fragment fn fs_sky(in: SkyOut) -> @location(0) vec4f {
    let far = g.inv_view_proj * vec4f(in.ndc, 1.0, 1.0);
    let near = g.inv_view_proj * vec4f(in.ndc, 0.0, 1.0);
    let dir = normalize(far.xyz / far.w - near.xyz / near.w);
    return finish(sky_color(dir), 1.0);
}

// ---------- terrain and models ----------

@vertex fn vs_main(v: VIn) -> VOut {
    let w = o.model * vec4f(v.pos, 1.0);
    var out: VOut;
    out.clip = g.view_proj * w;
    out.world = w.xyz;
    out.normal = normalize((o.model * vec4f(v.normal, 0.0)).xyz);
    out.uv = v.uv;
    return out;
}

@vertex fn vs_shadow(v: VIn) -> @builtin(position) vec4f {
    return g.light_vp * o.model * vec4f(v.pos, 1.0);
}

@fragment fn fs_object(in: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
    let c = textureSample(tex, samp, in.uv) * o.tint;
    if (c.a < 0.5) {
        discard;
    }
    let n = select(-normalize(in.normal), normalize(in.normal), front);
    return finish(fog(lit(c.rgb, n, in.world, 0.25), in.world), 1.0);
}

@fragment fn fs_terrain(in: VOut) -> @location(0) vec4f {
    let n = normalize(in.normal);
    let y = in.world.y;
    let k = noise2(in.uv * 1.7) * 0.6 + noise2(in.uv * 6.0) * 0.4;
    let sand = vec3f(0.78, 0.66, 0.44) * (0.9 + k * 0.2);
    let grass = mix(vec3f(0.16, 0.34, 0.08), vec3f(0.34, 0.46, 0.14), k);
    let rock = vec3f(0.42, 0.40, 0.37) * (0.8 + k * 0.4);
    var c = mix(sand, grass, smoothstep(0.12, 0.35, y + (k - 0.5) * 0.15));
    c = mix(c, rock, smoothstep(0.8, 0.62, n.y));
    c = mix(c, sand * 0.55, smoothstep(-0.05, -0.6, y));
    return finish(fog(lit(c, n, in.world, 0.05), in.world), 1.0);
}

// ---------- water ----------

fn wave(p: vec2f, t: f32) -> vec3f {
    // Height plus d/dx, d/dz of four summed sine waves.
    var h = 0.0;
    var d = vec2f(0.0);
    let dirs = array<vec2f, 4>(vec2f(1.0, 0.3), vec2f(-0.4, 1.0), vec2f(0.7, -0.8), vec2f(-1.0, -0.2));
    let freqs = array<f32, 4>(0.55, 0.8, 1.35, 2.1);
    let amps = array<f32, 4>(0.07, 0.05, 0.025, 0.015);
    for (var i = 0; i < 4; i++) {
        let dir = normalize(dirs[i]);
        let ph = dot(dir, p) * freqs[i] + t * (1.2 + f32(i) * 0.35);
        h += sin(ph) * amps[i];
        d += dir * cos(ph) * amps[i] * freqs[i];
    }
    return vec3f(h, d);
}

@vertex fn vs_water(v: VIn) -> VOut {
    let wv = wave(v.pos.xz, g.params.x);
    let w = vec3f(v.pos.x, wv.x, v.pos.z);
    var out: VOut;
    out.clip = g.view_proj * vec4f(w, 1.0);
    out.world = w;
    out.normal = normalize(vec3f(-wv.y, 1.0, -wv.z));
    out.uv = v.pos.xz;
    return out;
}

@fragment fn fs_water(in: VOut) -> @location(0) vec4f {
    let t = g.params.x;
    let ripple = vec2f(noise2(in.uv * 3.0 + t * 0.6), noise2(in.uv * 3.0 - t * 0.5 + 7.0)) - 0.5;
    let n = normalize(normalize(in.normal) + vec3f(ripple.x, 0.0, ripple.y) * 0.18);
    let v = normalize(g.camera_pos.xyz - in.world);
    let l = normalize(g.sun_dir.xyz);

    let depth = in.world.y - terrain_height(in.world.x, in.world.z);
    let fresnel = 0.03 + 0.97 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
    let deep = vec3f(0.02, 0.10, 0.16);
    let shallow = vec3f(0.10, 0.42, 0.44);
    let body = mix(shallow, deep, smoothstep(0.0, 2.5, depth)) * (0.35 + 0.65 * max(l.y, 0.0));
    let refl = sky_color(reflect(-v, n));
    let sh = shadow(in.world, vec3f(0.0, 1.0, 0.0));
    let spec = pow(max(dot(reflect(-l, n), v), 0.0), 220.0) * 4.0 * sh;
    var c = mix(body * (0.6 + 0.4 * sh), refl, fresnel) + g.sun_color.rgb * spec;

    let foam_line = 1.0 - smoothstep(0.0, 0.28, depth);
    let foam = foam_line * smoothstep(0.35, 0.7, noise2(in.uv * 5.0 + vec2f(t * 0.4, -t * 0.3)) + foam_line * 0.4);
    c = mix(c, vec3f(0.9, 0.95, 0.95) * (0.5 + 0.5 * sh), foam * 0.8);

    let alpha = clamp(smoothstep(0.0, 1.2, depth) * 0.75 + fresnel + foam, 0.25, 1.0);
    return finish(fog(c, in.world), alpha);
}
