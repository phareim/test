//! CPU side of the scene: vertex/uniform layouts, the procedural island,
//! the sky model and glTF loading.

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};
use wasm_bindgen::prelude::*;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

impl Vertex {
    const ATTRS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Globals {
    pub view_proj: [[f32; 4]; 4],
    pub light_vp: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub sun_dir: [f32; 4],
    pub sun_color: [f32; 4],
    pub sky_top: [f32; 4],
    pub sky_horizon: [f32; 4],
    /// time, apply-gamma flag, shadow texel size, unused
    pub params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ObjUniform {
    pub model: [[f32; 4]; 4],
    pub tint: [f32; 4],
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Island height at (x, z). Water is at y = 0. Mirrored in scene.wgsl for shoreline foam.
#[wasm_bindgen]
pub fn terrain_height(x: f32, z: f32) -> f32 {
    let n = (x * 0.35).sin() * (z * 0.3).cos() * 0.35
        + (x * 0.9 + z * 0.7).sin() * 0.12
        + ((x - z) * 0.23).cos() * 0.2;
    let r = (x * x + z * z).sqrt() + n * 2.0;
    let island = 1.0 - smoothstep(6.0, 12.5, r);
    let knoll = (-((x - 6.5).powi(2) + (z + 4.5).powi(2)) / 7.0).exp() * 1.3;
    island * (1.7 + n * 0.5) + knoll * island - 0.9 - smoothstep(12.0, 30.0, r) * 3.0
}

/// Lowest ground in a small ring, so a model never hangs over a slope.
pub fn ground_under(x: f32, z: f32, radius: f32) -> f32 {
    let mut h = terrain_height(x, z);
    for i in 0..8 {
        let a = i as f32 * std::f32::consts::TAU / 8.0;
        h = h.min(terrain_height(x + a.cos() * radius, z + a.sin() * radius));
    }
    h - 0.03
}

fn grid(n: u32, half: f32, height: impl Fn(f32, f32) -> f32) -> (Vec<Vertex>, Vec<u32>) {
    let step = 2.0 * half / n as f32;
    let mut vertices = Vec::with_capacity(((n + 1) * (n + 1)) as usize);
    for j in 0..=n {
        for i in 0..=n {
            let x = -half + i as f32 * step;
            let z = -half + j as f32 * step;
            let e = 0.05;
            let normal = Vec3::new(
                height(x - e, z) - height(x + e, z),
                2.0 * e,
                height(x, z - e) - height(x, z + e),
            )
            .normalize();
            vertices.push(Vertex { pos: [x, height(x, z), z], normal: normal.into(), uv: [x, z] });
        }
    }
    let mut indices = Vec::with_capacity((n * n * 6) as usize);
    for j in 0..n {
        for i in 0..n {
            let a = j * (n + 1) + i;
            let b = a + n + 1;
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    (vertices, indices)
}

pub fn terrain_mesh() -> (Vec<Vertex>, Vec<u32>) {
    grid(220, 30.0, terrain_height)
}

pub fn water_mesh() -> (Vec<Vertex>, Vec<u32>) {
    grid(200, 120.0, |_, _| 0.0)
}

/// Sun direction and sky colours for a time of day in 0..1 (linear RGB).
pub struct Sky {
    pub dir: Vec3,
    pub sun: Vec3,
    pub top: Vec3,
    pub horizon: Vec3,
}

impl Sky {
    pub fn at(t: f32) -> Sky {
        let a = std::f32::consts::PI * (0.04 + 0.92 * t);
        let dir = Vec3::new(a.cos() * 0.9, a.sin() * 0.85 + 0.03, -0.45).normalize();
        let warm = 1.0 - smoothstep(0.02, 0.5, dir.y);
        let sun = Vec3::new(3.0, 2.85, 2.55).lerp(Vec3::new(2.4, 1.1, 0.5), warm);
        let top = Vec3::new(0.16, 0.36, 0.78).lerp(Vec3::new(0.14, 0.16, 0.38), warm);
        let horizon = Vec3::new(0.62, 0.76, 0.92).lerp(Vec3::new(0.98, 0.56, 0.34), warm);
        Sky { dir, sun, top, horizon }
    }
}

pub struct Prim {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub tint: [f32; 4],
    pub image: Option<usize>,
}

pub struct Loaded {
    pub prims: Vec<Prim>,
    /// width, height, RGBA8 pixels
    pub images: Vec<(u32, u32, Vec<u8>)>,
    pub bounds: (Vec3, Vec3),
}

/// Parses a .glb and bakes node transforms into the vertices.
pub fn load_glb(bytes: &[u8]) -> Result<Loaded, String> {
    let (doc, buffers, images) = gltf::import_slice(bytes).map_err(|e| format!("glTF: {e}"))?;
    let mut prims = Vec::new();
    let scene = doc.default_scene().or_else(|| doc.scenes().next()).ok_or("glTF has no scene")?;
    for node in scene.nodes() {
        visit(&node, Mat4::IDENTITY, &buffers, &mut prims);
    }
    if prims.is_empty() {
        return Err("glTF has no triangle meshes".into());
    }

    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for p in &prims {
        for v in &p.vertices {
            min = min.min(v.pos.into());
            max = max.max(v.pos.into());
        }
    }

    let images = images.into_iter().map(to_rgba).collect();
    Ok(Loaded { prims, images, bounds: (min, max) })
}

fn visit(node: &gltf::Node, parent: Mat4, buffers: &[gltf::buffer::Data], out: &mut Vec<Prim>) {
    let m = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
    let nm = Mat3::from_mat4(m).inverse().transpose();
    if let Some(mesh) = node.mesh() {
        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = prim.reader(|b| Some(&*buffers[b.index()]));
            let Some(positions) = reader.read_positions() else { continue };
            let positions: Vec<Vec3> = positions.map(|p| m.transform_point3(p.into())).collect();
            let indices: Vec<u32> = match reader.read_indices() {
                Some(i) => i.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };
            let normals: Vec<Vec3> = match reader.read_normals() {
                Some(n) => n.map(|n| (nm * Vec3::from(n)).normalize_or_zero()).collect(),
                None => smooth_normals(&positions, &indices),
            };
            let uvs: Vec<[f32; 2]> = match reader.read_tex_coords(0) {
                Some(t) => t.into_f32().collect(),
                None => vec![[0.0; 2]; positions.len()],
            };
            let vertices = positions
                .iter()
                .zip(&normals)
                .zip(&uvs)
                .map(|((p, n), uv)| Vertex { pos: (*p).into(), normal: (*n).into(), uv: *uv })
                .collect();

            let pbr = prim.material().pbr_metallic_roughness();
            let image = pbr.base_color_texture().map(|t| t.texture().source().index());
            out.push(Prim { vertices, indices, tint: pbr.base_color_factor(), image });
        }
    }
    for child in node.children() {
        visit(&child, m, buffers, out);
    }
}

fn smooth_normals(positions: &[Vec3], indices: &[u32]) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for t in indices.chunks_exact(3) {
        let [a, b, c] = [t[0] as usize, t[1] as usize, t[2] as usize];
        let n = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
        normals[a] += n;
        normals[b] += n;
        normals[c] += n;
    }
    normals.into_iter().map(|n| n.normalize_or(Vec3::Y)).collect()
}

/// Converts a decoded glTF image to RGBA8, capped at 2048 px on the long side.
fn to_rgba(img: gltf::image::Data) -> (u32, u32, Vec<u8>) {
    use gltf::image::Format;
    let (w, h) = (img.width, img.height);
    let px = &img.pixels;
    let rgba: Vec<u8> = match img.format {
        Format::R8G8B8A8 => px.clone(),
        Format::R8G8B8 => px.chunks_exact(3).flat_map(|c| [c[0], c[1], c[2], 255]).collect(),
        Format::R8G8 => px.chunks_exact(2).flat_map(|c| [c[0], c[0], c[0], c[1]]).collect(),
        Format::R8 => px.iter().flat_map(|&c| [c, c, c, 255]).collect(),
        _ => vec![255; (w * h * 4) as usize],
    };
    let longest = w.max(h);
    if longest <= 2048 {
        return (w, h, rgba);
    }
    let img = image::RgbaImage::from_raw(w, h, rgba).expect("pixel count matches size");
    let (nw, nh) = ((w * 2048 / longest).max(1), (h * 2048 / longest).max(1));
    let small = image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Triangle);
    (nw, nh, small.into_raw())
}
