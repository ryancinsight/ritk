//! wgpu device init and pipeline compilation for the GPU mesh renderer.
//!
//! `GpuMeshContext` owns the `wgpu::Device` and `wgpu::Queue` plus the four
//! compiled pipelines and their bind group layouts:
//!
//! | Pipeline           | Shader            | Stages            | BGL entries |
//! |--------------------|-------------------|-------------------|-------------|
//! | `geom_base`        | geometry.wgsl     | VS + FS (2 color) | 0=scene, 1=lights, 2=material |
//! | `geom_peel`        | peel.wgsl         | VS + FS (1 color) | 0=scene, 1=lights, 2=material, 3=prev_depth |
//! | `ssao`             | ssao.wgsl         | Compute           | 0=ssao_params, 1=normal_depth, 2=ao_buf, 3=kernel |
//! | `composite`        | composite.wgsl    | Compute           | 0=color_layers, 1=ao_buf, 2=output, 3=comp_params |

use super::params::MeshVertex;

pub(super) struct GpuMeshContext {
    pub device: wgpu::Device,
    pub queue:  wgpu::Queue,

    pub geom_base_pipeline: wgpu::RenderPipeline,
    pub geom_peel_pipeline: wgpu::RenderPipeline,
    pub ssao_pipeline:      wgpu::ComputePipeline,
    pub composite_pipeline: wgpu::ComputePipeline,

    pub geom_base_bgl: wgpu::BindGroupLayout,
    pub geom_peel_bgl: wgpu::BindGroupLayout,
    pub ssao_bgl:      wgpu::BindGroupLayout,
    pub composite_bgl: wgpu::BindGroupLayout,
}

impl GpuMeshContext {
    /// Initialize a headless wgpu context and compile all four pipelines.
    ///
    /// Returns `None` when no compatible GPU adapter is available.
    pub fn try_new() -> Option<Self> {
        let (device, queue) = pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });
            let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }).await?;
            adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .ok()
        })?;

        // ── Shader modules ────────────────────────────────────────────────────
        let geom_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("geom_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("geometry.wgsl").into()),
        });
        let peel_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("peel_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("peel.wgsl").into()),
        });
        let ssao_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssao_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ssao.wgsl").into()),
        });
        let comp_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("composite_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("composite.wgsl").into()),
        });

        // ── Bind group layout helpers ─────────────────────────────────────────
        let uniform_entry = |binding: u32, vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_ro_entry = |binding: u32, vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_rw_entry = |binding: u32, vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let vs_fs = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let fs    = wgpu::ShaderStages::FRAGMENT;
        let cs    = wgpu::ShaderStages::COMPUTE;

        // ── geom_base BGL: bindings 0=scene(VS+FS), 1=lights(FS), 2=material(FS) ─
        let geom_base_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("geom_base_bgl"),
            entries: &[
                uniform_entry(0, vs_fs),
                uniform_entry(1, fs),
                uniform_entry(2, fs),
            ],
        });

        // ── geom_peel BGL: same + 3=prev_depth(FS, texture_depth_2d) ────────
        let geom_peel_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("geom_peel_bgl"),
            entries: &[
                uniform_entry(0, vs_fs),
                uniform_entry(1, fs),
                uniform_entry(2, fs),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: fs,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        // ── ssao BGL: 0=ssao_params(U,CS), 1=normal_depth(tex,CS),
        //              2=ao_buf(rw,CS), 3=kernel(ro,CS) ─────────────────────
        let ssao_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao_bgl"),
            entries: &[
                uniform_entry(0, cs),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: cs,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_rw_entry(2, cs),
                storage_ro_entry(3, cs),
            ],
        });

        // ── composite BGL: 0=color_layers(D2Array,CS), 1=ao_buf(ro,CS),
        //                   2=output(rw,CS), 3=comp_params(U,CS) ─────────────
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: cs,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_ro_entry(1, cs),
                storage_rw_entry(2, cs),
                uniform_entry(3, cs),
            ],
        });

        // ── Render pipelines ──────────────────────────────────────────────────
        let depth_stencil = wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };

        let primitive = wgpu::PrimitiveState {
            topology:            wgpu::PrimitiveTopology::TriangleList,
            strip_index_format:  None,
            front_face:          wgpu::FrontFace::Ccw,
            cull_mode:           Some(wgpu::Face::Back),
            polygon_mode:        wgpu::PolygonMode::Fill,
            unclipped_depth:     false,
            conservative:        false,
        };

        // Base pipeline: 2 color targets (Rgba8Unorm + Rgba16Float).
        let base_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("geom_base_pl_layout"),
            bind_group_layouts: &[&geom_base_bgl],
            push_constant_ranges: &[],
        });
        let geom_base_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("geom_base_pipeline"),
            layout: Some(&base_pl_layout),
            vertex: wgpu::VertexState {
                module: &geom_module,
                entry_point: "vs_main",
                buffers: &[MeshVertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive,
            depth_stencil: Some(depth_stencil.clone()),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &geom_module,
                entry_point: "fs_base_main",
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview: None,
        });

        // Peel pipeline: 1 color target (Rgba8Unorm), + prev_depth binding.
        let peel_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("geom_peel_pl_layout"),
            bind_group_layouts: &[&geom_peel_bgl],
            push_constant_ranges: &[],
        });
        let geom_peel_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("geom_peel_pipeline"),
            layout: Some(&peel_pl_layout),
            vertex: wgpu::VertexState {
                module: &peel_module,
                entry_point: "vs_main",
                buffers: &[MeshVertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive,
            depth_stencil: Some(depth_stencil),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &peel_module,
                entry_point: "fs_peel_main",
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview: None,
        });

        // ── Compute pipelines ─────────────────────────────────────────────────
        let ssao_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssao_pl_layout"),
            bind_group_layouts: &[&ssao_bgl],
            push_constant_ranges: &[],
        });
        let ssao_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ssao_pipeline"),
            layout: Some(&ssao_pl_layout),
            module: &ssao_module,
            entry_point: "ssao_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let comp_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite_pl_layout"),
            bind_group_layouts: &[&composite_bgl],
            push_constant_ranges: &[],
        });
        let composite_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("composite_pipeline"),
            layout: Some(&comp_pl_layout),
            module: &comp_module,
            entry_point: "composite_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Some(Self {
            device, queue,
            geom_base_pipeline, geom_peel_pipeline,
            ssao_pipeline, composite_pipeline,
            geom_base_bgl, geom_peel_bgl,
            ssao_bgl, composite_bgl,
        })
    }
}
