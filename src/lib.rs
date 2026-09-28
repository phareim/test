//! A small wgpu scene for the browser: a procedural island with water, sky,
//! a moving sun with shadow mapping, and glTF models dropped onto it.
//! JS owns the page (input, resize, requestAnimationFrame); Rust owns the GPU.

mod scene;

use glam::camera::rh::{proj::directx as proj, view::look_at_mat4};
use glam::{Mat4, Vec3};
use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;

pub use scene::terrain_height;
use scene::{Globals, ObjUniform, Vertex};

const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const SHADOW_SIZE: u32 = 2048;

struct Draw {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    count: u32,
    ubuf: wgpu::Buffer,
    bind: wgpu::BindGroup,
    tint: [f32; 4],
}

struct Model {
    draws: Vec<Draw>,
    base: Mat4,
    float: bool,
    phase: f32,
}

#[wasm_bindgen]
pub struct App {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    samples: u32,
    info: String,

    globals_buf: wgpu::Buffer,
    globals_bind: wgpu::BindGroup,
    shadow_bind: wgpu::BindGroup,
    shadow_view: wgpu::TextureView,
    obj_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    white: wgpu::TextureView,

    sky_pipe: wgpu::RenderPipeline,
    terrain_pipe: wgpu::RenderPipeline,
    object_pipe: wgpu::RenderPipeline,
    water_pipe: wgpu::RenderPipeline,
    shadow_pipe: wgpu::RenderPipeline,

    depth_view: wgpu::TextureView,
    msaa_view: Option<wgpu::TextureView>,

    terrain: Draw,
    water: Draw,
    models: Vec<Model>,
    triangles: u32,

    yaw: f32,
    pitch: f32,
    dist: f32,
    sun: f32,
    idle: f32,
}

#[wasm_bindgen]
impl App {
    /// Sets up the GPU on `canvas`: WebGPU when the browser has it, WebGL2 otherwise.
    pub async fn create(canvas: web_sys::HtmlCanvasElement) -> Result<App, JsValue> {
        console_error_panic_hook::set_once();
        let _ = console_log::init_with_level(log::Level::Warn);

        let (width, height) = (canvas.width().max(1), canvas.height().max(1));
        let instance = wgpu::util::new_instance_with_webgpu_detection(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        })
        .await;
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| JsValue::from_str(&format!("surface: {e}")))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .map_err(|e| JsValue::from_str(&format!("adapter: {e}")))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await
            .map_err(|e| JsValue::from_str(&format!("device: {e}")))?;

        let mut config = surface
            .get_default_config(&adapter, width, height)
            .ok_or_else(|| JsValue::from_str("surface not supported by adapter"))?;
        config.present_mode = wgpu::PresentMode::Fifo;
        surface.configure(&device, &config);

        let supports4 = |f: wgpu::TextureFormat| {
            adapter.get_texture_format_features(f).flags.sample_count_supported(4)
        };
        let samples = if supports4(config.format) && supports4(DEPTH) { 4 } else { 1 };

        let ai = adapter.get_info();
        let backend = match ai.backend {
            wgpu::Backend::BrowserWebGpu => "WebGPU".to_string(),
            wgpu::Backend::Gl => "WebGL2".to_string(),
            other => format!("{other:?}"),
        };
        let info = if ai.name.is_empty() {
            format!("{backend} · MSAA {samples}x")
        } else {
            format!("{backend} · {} · MSAA {samples}x", ai.name)
        };

        // Bind group layouts: 0 = per-frame globals, 1 = per-draw object, 2 = shadow map.
        let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[uniform_entry(0)],
        });
        let obj_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("object"),
            entries: &[
                uniform_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let shadow_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });

        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let globals_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout: &globals_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: globals_buf.as_entire_binding() }],
        });

        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow map"),
            size: wgpu::Extent3d { width: SHADOW_SIZE, height: SHADOW_SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_tex.create_view(&Default::default());
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let shadow_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow"),
            layout: &shadow_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&shadow_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&shadow_sampler) },
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("material"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let white = upload_texture(&device, &queue, 1, 1, vec![255; 4]);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene"),
            source: wgpu::ShaderSource::Wgsl(include_str!("scene.wgsl").into()),
        });
        let main_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("main"),
            bind_group_layouts: &[Some(&globals_layout), Some(&obj_layout), Some(&shadow_layout)],
            immediate_size: 0,
        });
        let sky_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sky"),
            bind_group_layouts: &[Some(&globals_layout)],
            immediate_size: 0,
        });
        let shadow_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow"),
            bind_group_layouts: &[Some(&globals_layout), Some(&obj_layout)],
            immediate_size: 0,
        });

        let p = PipeCtx { device: &device, shader: &shader, format: config.format, samples };
        let sky_pipe = p.build(&sky_layout, "vs_sky", "fs_sky", false, false, wgpu::CompareFunction::Always, None);
        let terrain_pipe = p.build(&main_layout, "vs_main", "fs_terrain", true, true, wgpu::CompareFunction::Less, None);
        let object_pipe = p.build(&main_layout, "vs_main", "fs_object", true, true, wgpu::CompareFunction::Less, None);
        let water_pipe = p.build(
            &main_layout,
            "vs_water",
            "fs_water",
            true,
            false,
            wgpu::CompareFunction::Less,
            Some(wgpu::BlendState::ALPHA_BLENDING),
        );
        let shadow_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow"),
            layout: Some(&shadow_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_shadow"),
                compilation_options: Default::default(),
                buffers: &[Some(Vertex::layout())],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
            }),
            multisample: Default::default(),
            fragment: None,
            multiview_mask: None,
            cache: None,
        });

        let (tv, ti) = scene::terrain_mesh();
        let (wv, wi) = scene::water_mesh();
        let white_ref = &white;
        let mk = |v: &[Vertex], i: &[u32], tint: [f32; 4]| {
            make_draw(&device, &obj_layout, &sampler, white_ref, v, i, tint, Mat4::IDENTITY)
        };
        let terrain = mk(&tv, &ti, [1.0; 4]);
        let water = mk(&wv, &wi, [1.0; 4]);
        let triangles = (ti.len() / 3) as u32;

        let (depth_view, msaa_view) = targets(&device, &config, samples);

        Ok(App {
            device,
            queue,
            surface,
            config,
            samples,
            info,
            globals_buf,
            globals_bind,
            shadow_bind,
            shadow_view,
            obj_layout,
            sampler,
            white,
            sky_pipe,
            terrain_pipe,
            object_pipe,
            water_pipe,
            shadow_pipe,
            depth_view,
            msaa_view,
            terrain,
            water,
            models: Vec::new(),
            triangles,
            yaw: 0.7,
            pitch: 0.38,
            dist: 26.0,
            sun: 0.42,
            idle: 10.0,
        })
    }

    /// Loads a .glb, scales it to `size` (height, or length when `float` is set)
    /// and stands it on the terrain at (x, z). Floating models bob on the water.
    pub fn add_model(&mut self, bytes: &[u8], x: f32, z: f32, size: f32, yaw: f32, float: bool) -> Result<(), JsValue> {
        let loaded = scene::load_glb(bytes).map_err(|e| JsValue::from_str(&e))?;
        let (min, max) = loaded.bounds;
        let extent = max - min;
        let scale = if float { size / extent.x.max(extent.z) } else { size / extent.y };
        let ground = if float { 0.0 } else { scene::ground_under(x, z, extent.x.max(extent.z) * scale * 0.4) };
        let pivot = Vec3::new((min.x + max.x) * 0.5, min.y, (min.z + max.z) * 0.5);
        let base = Mat4::from_translation(Vec3::new(x, ground, z))
            * Mat4::from_rotation_y(yaw)
            * Mat4::from_scale(Vec3::splat(scale))
            * Mat4::from_translation(-pivot);

        let mut textures: Vec<Option<wgpu::TextureView>> = Vec::new();
        let mut draws = Vec::new();
        for prim in &loaded.prims {
            let view = match prim.image {
                Some(i) => {
                    if textures.len() <= i {
                        textures.resize_with(i + 1, || None);
                    }
                    if textures[i].is_none() {
                        let img = &loaded.images[i];
                        textures[i] = Some(upload_texture(&self.device, &self.queue, img.0, img.1, img.2.clone()));
                    }
                    textures[i].as_ref().unwrap()
                }
                None => &self.white,
            };
            self.triangles += (prim.indices.len() / 3) as u32;
            draws.push(make_draw(
                &self.device,
                &self.obj_layout,
                &self.sampler,
                view,
                &prim.vertices,
                &prim.indices,
                prim.tint,
                base,
            ));
        }
        let phase = x * 0.37 + z * 0.21;
        self.models.push(Model { draws, base, float, phase });
        Ok(())
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 || (width == self.config.width && height == self.config.height) {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        let (d, m) = targets(&self.device, &self.config, self.samples);
        self.depth_view = d;
        self.msaa_view = m;
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.006;
        self.pitch = (self.pitch + dy * 0.004).clamp(0.06, 1.35);
        self.idle = 0.0;
    }

    pub fn zoom(&mut self, factor: f32) {
        self.dist = (self.dist * factor).clamp(6.0, 60.0);
        self.idle = 0.0;
    }

    /// Time of day, 0 = sunrise, 1 = sunset.
    pub fn set_sun(&mut self, t: f32) {
        self.sun = t.clamp(0.0, 1.0);
    }

    pub fn info(&self) -> String {
        self.info.clone()
    }

    pub fn triangles(&self) -> u32 {
        self.triangles
    }

    /// Renders one frame. `time` is seconds, `dt` the seconds since the last frame.
    pub fn frame(&mut self, time: f32, dt: f32) {
        self.idle += dt;
        if self.idle > 4.0 {
            self.yaw += dt * 0.05;
        }

        let aspect = self.config.width as f32 / self.config.height as f32;
        let target = Vec3::new(0.0, 1.2, 0.0);
        let eye = target
            + self.dist * Vec3::new(self.pitch.cos() * self.yaw.sin(), self.pitch.sin(), self.pitch.cos() * self.yaw.cos());
        let view = look_at_mat4(eye, target, Vec3::Y);
        let proj = proj::perspective(0.8, aspect, 0.1, 400.0);
        let view_proj = proj * view;

        let sky = scene::Sky::at(self.sun);
        let light_view = look_at_mat4(sky.dir * 40.0, Vec3::ZERO, Vec3::Y);
        let light_proj = proj::orthographic(-24.0, 24.0, -24.0, 24.0, 1.0, 90.0);

        let globals = Globals {
            view_proj: view_proj.to_cols_array_2d(),
            light_vp: (light_proj * light_view).to_cols_array_2d(),
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            camera_pos: eye.extend(1.0).to_array(),
            sun_dir: sky.dir.extend(0.0).to_array(),
            sun_color: sky.sun.extend(1.0).to_array(),
            sky_top: sky.top.extend(1.0).to_array(),
            sky_horizon: sky.horizon.extend(1.0).to_array(),
            params: [time, if self.config.format.is_srgb() { 0.0 } else { 1.0 }, 1.0 / SHADOW_SIZE as f32, 0.0],
        };
        self.queue.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));

        for m in &self.models {
            if !m.float {
                continue;
            }
            let bob = Mat4::from_translation(Vec3::new(0.0, (time * 1.3 + m.phase).sin() * 0.05 + 0.1, 0.0))
                * Mat4::from_rotation_z((time * 0.9 + m.phase).sin() * 0.05)
                * Mat4::from_rotation_x((time * 1.1 + m.phase).cos() * 0.04);
            let model = bob * m.base;
            for d in &m.draws {
                write_obj(&self.queue, d, model);
            }
        }

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            _ => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
        };
        let frame_view = frame.texture.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(&Default::default());

        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.shadow_pipe);
            pass.set_bind_group(0, &self.globals_bind, &[]);
            draw(&mut pass, &self.terrain);
            for m in &self.models {
                for d in &m.draws {
                    draw(&mut pass, d);
                }
            }
        }

        {
            let (view, resolve) = match &self.msaa_view {
                Some(ms) => (ms, Some(&frame_view)),
                None => (&frame_view, None),
            };
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: resolve,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_bind_group(0, &self.globals_bind, &[]);
            pass.set_pipeline(&self.sky_pipe);
            pass.draw(0..3, 0..1);

            pass.set_bind_group(2, &self.shadow_bind, &[]);
            pass.set_pipeline(&self.terrain_pipe);
            draw(&mut pass, &self.terrain);
            pass.set_pipeline(&self.object_pipe);
            for m in &self.models {
                for d in &m.draws {
                    draw(&mut pass, d);
                }
            }
            pass.set_pipeline(&self.water_pipe);
            draw(&mut pass, &self.water);
        }

        self.queue.submit([enc.finish()]);
        self.queue.present(frame);
    }
}

fn draw(pass: &mut wgpu::RenderPass<'_>, d: &Draw) {
    pass.set_bind_group(1, &d.bind, &[]);
    pass.set_vertex_buffer(0, d.vbuf.slice(..));
    pass.set_index_buffer(d.ibuf.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..d.count, 0, 0..1);
}

fn write_obj(queue: &wgpu::Queue, d: &Draw, model: Mat4) {
    let u = ObjUniform { model: model.to_cols_array_2d(), tint: d.tint };
    queue.write_buffer(&d.ubuf, 0, bytemuck::bytes_of(&u));
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn make_draw(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    view: &wgpu::TextureView,
    vertices: &[Vertex],
    indices: &[u32],
    tint: [f32; 4],
    model: Mat4,
) -> Draw {
    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    let u = ObjUniform { model: model.to_cols_array_2d(), tint };
    let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&u),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    });
    Draw { vbuf, ibuf, count: indices.len() as u32, ubuf, bind, tint }
}

/// Uploads RGBA8 pixels as an sRGB texture with a CPU-built mip chain.
fn upload_texture(device: &wgpu::Device, queue: &wgpu::Queue, w: u32, h: u32, rgba: Vec<u8>) -> wgpu::TextureView {
    let levels = 32 - w.max(h).leading_zeros();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let mut img = image::RgbaImage::from_raw(w, h, rgba).expect("pixel count matches size");
    for level in 0..levels {
        let (lw, lh) = img.dimensions();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            img.as_raw(),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * lw), rows_per_image: Some(lh) },
            wgpu::Extent3d { width: lw, height: lh, depth_or_array_layers: 1 },
        );
        if level + 1 < levels {
            img = image::imageops::resize(&img, (lw / 2).max(1), (lh / 2).max(1), image::imageops::FilterType::Triangle);
        }
    }
    texture.create_view(&Default::default())
}

fn targets(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    samples: u32,
) -> (wgpu::TextureView, Option<wgpu::TextureView>) {
    let size = wgpu::Extent3d { width: config.width, height: config.height, depth_or_array_layers: 1 };
    let make = |format, label| {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&Default::default())
    };
    let depth = make(DEPTH, "depth");
    let msaa = (samples > 1).then(|| make(config.format, "msaa"));
    (depth, msaa)
}

struct PipeCtx<'a> {
    device: &'a wgpu::Device,
    shader: &'a wgpu::ShaderModule,
    format: wgpu::TextureFormat,
    samples: u32,
}

impl PipeCtx<'_> {
    #[allow(clippy::too_many_arguments)]
    fn build(
        &self,
        layout: &wgpu::PipelineLayout,
        vs: &str,
        fs: &str,
        with_vertices: bool,
        depth_write: bool,
        compare: wgpu::CompareFunction,
        blend: Option<wgpu::BlendState>,
    ) -> wgpu::RenderPipeline {
        let buffers = [Some(Vertex::layout())];
        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(fs),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: self.shader,
                entry_point: Some(vs),
                compilation_options: Default::default(),
                buffers: if with_vertices { &buffers } else { &[] },
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH,
                depth_write_enabled: Some(depth_write),
                depth_compare: Some(compare),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: self.samples, ..Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: self.shader,
                entry_point: Some(fs),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    }
}
