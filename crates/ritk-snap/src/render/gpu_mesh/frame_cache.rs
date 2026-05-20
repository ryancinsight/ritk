//! Per-viewport GPU resource cache for the mesh renderer.
//!
//! `GpuMeshFrameCache` holds all textures and buffers that are tied to the
//! output viewport dimensions.  All resources are reallocated only when
//! `rows` or `cols` changes — otherwise they are reused across frames.
//!
//! # Resource inventory
//!
//! | Resource           | Format / Usage                          | Size              |
//! |--------------------|----------------------------------------|-------------------|
//! | `color_array_tex`  | Rgba8Unorm, RENDER_ATTACHMENT+TEXTURE  | rows×cols×4×N_LAYERS |
//! | `depth_texes[i]`   | Depth32Float, RENDER_ATTACHMENT+TEXTURE| rows×cols×4 each  |
//! | `normal_depth_tex` | Rgba16Float, RENDER_ATTACHMENT+TEXTURE | rows×cols×8       |
//! | `ao_buf`           | STORAGE, f32 per pixel                 | rows×cols×4       |
//! | `output_buf`       | STORAGE\|COPY_SRC, u32 per pixel       | rows×cols×4       |
//! | `staging_buf`      | MAP_READ\|COPY_DST                     | rows×cols×4       |
//! | `scene_buf`        | UNIFORM\|COPY_DST, 144 bytes           | fixed             |
//! | `lights_buf`       | UNIFORM\|COPY_DST, 96 bytes            | fixed             |
//! | `material_buf`     | UNIFORM\|COPY_DST, 48 bytes            | fixed             |
//! | `ssao_uniforms_buf`| UNIFORM\|COPY_DST, 48 bytes            | fixed             |
//! | `ssao_kernel_buf`  | STORAGE (read-only), 256 bytes         | fixed             |
//! | `comp_params_buf`  | UNIFORM\|COPY_DST, 16 bytes            | fixed             |

use wgpu::util::DeviceExt as _;
use super::params::build_ssao_kernel;

/// Number of depth-peel layers. Must match `N_LAYERS` constant in composite.wgsl.
pub(super) const N_PEEL_LAYERS: usize = 4;

/// GPU resource cache scoped to one viewport size.
///
/// Owned by `GpuMeshRenderer`; recreated via `new` whenever dimensions change.
/// The `staging_buf` must NOT be reused while `map_async` is pending.
pub(super) struct GpuMeshFrameCache {
    pub rows: usize,
    pub cols: usize,

    // ── Textures ──────────────────────────────────────────────────────────────
    /// Rgba8Unorm 2D-array texture with N_PEEL_LAYERS array layers.
    /// Layer i holds the Phong-shaded color for peel pass i.
    pub color_array_tex: wgpu::Texture,
    /// Per-layer Depth32Float depth textures (one per peel pass).
    pub depth_texes: Vec<wgpu::Texture>,
    /// Rgba16Float G-buffer: xyz = view-space normal encoded to [0,1], w = linear depth.
    /// Produced only by peel pass 0; read by the SSAO compute pass.
    pub normal_depth_tex: wgpu::Texture,

    // ── Compute / readback buffers ────────────────────────────────────────────
    /// SSAO output: one f32 per pixel. Written by SSAO compute, read by composite.
    pub ao_buf:      wgpu::Buffer,
    /// Composite output: packed u32 RGBA, one per pixel. Copied to staging after composite.
    pub output_buf:  wgpu::Buffer,
    /// Staging buffer: mapped for CPU readback after `map_async`.
    pub staging_buf: wgpu::Buffer,

    // ── Uniform / kernel buffers ──────────────────────────────────────────────
    /// SceneUniforms (144 bytes). Updated via `queue.write_buffer` each frame.
    pub scene_buf:         wgpu::Buffer,
    /// LightBlock (96 bytes). Updated via `queue.write_buffer` each frame.
    pub lights_buf:        wgpu::Buffer,
    /// MaterialUniforms (48 bytes). Updated via `queue.write_buffer` each frame.
    pub material_buf:      wgpu::Buffer,
    /// SsaoUniforms (48 bytes). Updated via `queue.write_buffer` each frame.
    pub ssao_uniforms_buf: wgpu::Buffer,
    /// SSAO hemisphere kernel (16 × vec4 = 256 bytes). Written once at creation.
    pub ssao_kernel_buf:   wgpu::Buffer,
    /// CompositeUniforms (16 bytes). Updated via `queue.write_buffer` each frame.
    pub comp_params_buf:   wgpu::Buffer,
}

impl GpuMeshFrameCache {
    /// Allocate all textures and buffers for a `rows × cols` viewport.
    pub(super) fn new(device: &wgpu::Device, rows: usize, cols: usize) -> Self {
        let w = cols as u32;
        let h = rows as u32;
        let px = (rows * cols) as u64;

        // ── Color array texture (N_PEEL_LAYERS layers, Rgba8Unorm) ─────────────
        let color_array_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mesh_color_array"),
            size: wgpu::Extent3d {
                width: w, height: h,
                depth_or_array_layers: N_PEEL_LAYERS as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // ── Per-layer depth textures (Depth32Float) ────────────────────────────
        let depth_texes: Vec<wgpu::Texture> = (0..N_PEEL_LAYERS)
            .map(|i| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("mesh_depth_{i}")),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
            })
            .collect();

        // ── G-buffer: normal + linear depth (Rgba16Float) ─────────────────────
        let normal_depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mesh_normal_depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // ── Compute / readback buffers ─────────────────────────────────────────
        let ao_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_ao_buf"),
            size: px * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_output_buf"),
            size: px * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_staging_buf"),
            size: px * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Uniform / kernel buffers ───────────────────────────────────────────
        let make_uniform = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let scene_buf         = make_uniform("mesh_scene_buf", 144);
        let lights_buf        = make_uniform("mesh_lights_buf", 96);
        let material_buf      = make_uniform("mesh_material_buf", 48);
        let ssao_uniforms_buf = make_uniform("mesh_ssao_uniforms_buf", 48);
        let comp_params_buf   = make_uniform("mesh_comp_params_buf", 16);

        let kernel_data: [[f32; 4]; 16] = build_ssao_kernel();
        let ssao_kernel_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_ssao_kernel_buf"),
            contents: bytemuck::cast_slice(&kernel_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        Self {
            rows, cols,
            color_array_tex, depth_texes, normal_depth_tex,
            ao_buf, output_buf, staging_buf,
            scene_buf, lights_buf, material_buf,
            ssao_uniforms_buf, ssao_kernel_buf, comp_params_buf,
        }
    }
}
