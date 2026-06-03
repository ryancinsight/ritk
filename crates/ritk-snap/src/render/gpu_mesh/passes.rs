//! GPU render pass execution: depth peeling, SSAO, composite, async readback.
//!
//! All GPU work is encoded into a single `CommandBuffer` per frame:
//!  1. Pass 0 (base): geometry → color_array[0] + depth_texes[0] + normal_depth_tex
//!  2. Passes 1..N-1 (peel): geometry with depth discard → color_array[i] + depth_texes[i]
//!  3. SSAO compute: normal_depth_tex → ao_buf
//!  4. Composite compute: color_array + ao_buf → output_buf
//!  5. Copy: output_buf → staging_buf (COPY_SRC → COPY_DST)
//!
//! `submit_mesh_async` submits this command buffer and registers `map_async` on
//! the staging buffer.  `collect_mesh_result` reads the mapped data and returns
//! a `ColorImage`.  Both functions mirror the protocol from `mip_pass.rs`.

use bytemuck::cast_slice;
use egui::ColorImage;
use std::sync::mpsc;

use super::{
    context::GpuMeshContext,
    frame_cache::{GpuMeshFrameCache, N_PEEL_LAYERS},
    mesh_buf::GpuMeshBufs,
    params::{CompositeUniforms, LightBlock, MaterialUniforms, SceneUniforms, SsaoUniforms},
};

// ── Public interface ──────────────────────────────────────────────────────────

/// Submit all GPU passes for one mesh frame and register non-blocking readback.
///
/// Returns an `mpsc::Receiver` that fires `Ok(Ok(()))` when the staging buffer
/// is mapped and ready to read via `collect_mesh_result`.
///
/// # Precondition
///
/// `cache.staging_buf` must NOT be currently mapped (i.e., no pending readback).
pub(super) fn submit_mesh_async(
    ctx: &GpuMeshContext,
    cache: &GpuMeshFrameCache,
    mesh: &GpuMeshBufs,
    scene: SceneUniforms,
    lights: LightBlock,
    material: MaterialUniforms,
    ssao: SsaoUniforms,
    peel_layers: usize,
) -> mpsc::Receiver<Result<(), wgpu::BufferAsyncError>> {
    // Write uniform data into cached buffers.
    ctx.queue
        .write_buffer(&cache.scene_buf, 0, cast_slice(&[scene]));
    ctx.queue
        .write_buffer(&cache.lights_buf, 0, cast_slice(&[lights]));
    ctx.queue
        .write_buffer(&cache.material_buf, 0, cast_slice(&[material]));
    ctx.queue
        .write_buffer(&cache.ssao_uniforms_buf, 0, cast_slice(&[ssao]));
    ctx.queue.write_buffer(
        &cache.comp_params_buf,
        0,
        cast_slice(&[CompositeUniforms {
            rows: cache.rows as u32,
            cols: cache.cols as u32,
            _pad: [0, 0],
        }]),
    );

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mesh_encoder"),
        });

    encode_geometry_passes(ctx, cache, mesh, &mut encoder, peel_layers);
    encode_ssao_pass(ctx, cache, &mut encoder);
    encode_composite_pass(ctx, cache, &mut encoder);

    // Copy output buffer to staging buffer for CPU readback.
    let byte_len = (cache.rows * cache.cols * 4) as u64;
    encoder.copy_buffer_to_buffer(&cache.output_buf, 0, &cache.staging_buf, 0, byte_len);

    ctx.queue.submit([encoder.finish()]);

    // Register non-blocking map_async; fire callback when GPU finishes.
    let (tx, rx) = mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
    cache
        .staging_buf
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
    rx
}

/// Read the mapped staging buffer and return a `ColorImage`.
///
/// # Precondition
///
/// The receiver returned by `submit_mesh_async` must have fired `Ok(Ok(()))`.
pub(super) fn collect_mesh_result(cache: &GpuMeshFrameCache) -> ColorImage {
    let data = cache.staging_buf.slice(..).get_mapped_range();
    let pixels: Vec<egui::Color32> = data
        .chunks_exact(4)
        .map(|px| egui::Color32::from_rgba_unmultiplied(px[0], px[1], px[2], px[3]))
        .collect();
    drop(data);
    cache.staging_buf.unmap();
    ColorImage {
        size: [cache.cols, cache.rows],
        pixels,
    }
}

// ── Geometry passes ───────────────────────────────────────────────────────────

fn encode_geometry_passes(
    ctx: &GpuMeshContext,
    cache: &GpuMeshFrameCache,
    mesh: &GpuMeshBufs,
    encoder: &mut wgpu::CommandEncoder,
    peel_layers: usize,
) {
    let layers = peel_layers.clamp(1, N_PEEL_LAYERS);
    let w = cache.cols as u32;
    let h = cache.rows as u32;

    // Pass 0: base pass — writes color_array[0] + depth_texes[0] + normal_depth_tex.
    {
        let color0_view = cache
            .color_array_tex
            .create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: 0,
                array_layer_count: Some(1),
                ..Default::default()
            });
        let nd_view = cache
            .normal_depth_tex
            .create_view(&wgpu::TextureViewDescriptor::default());
        let depth0_view = cache.depth_texes[0].create_view(&wgpu::TextureViewDescriptor::default());

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("geom_base_bg"),
            layout: &ctx.geom_base_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cache.scene_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cache.lights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cache.material_buf.as_entire_binding(),
                },
            ],
        });

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("geom_base_pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &color0_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &nd_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth0_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });

        rpass.set_viewport(0.0, 0.0, w as f32, h as f32, 0.0, 1.0);
        rpass.set_pipeline(&ctx.geom_base_pipeline);
        rpass.set_bind_group(0, &bg, &[]);
        rpass.set_vertex_buffer(0, mesh.vertex_buf.slice(..));
        rpass.set_index_buffer(mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..mesh.n_indices, 0, 0..1);
    }

    // Passes 1..layers-1: peel passes, each discarding geometry ≤ prev depth.
    for i in 1..layers {
        let color_view = cache
            .color_array_tex
            .create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: i as u32,
                array_layer_count: Some(1),
                ..Default::default()
            });
        let depth_view = cache.depth_texes[i].create_view(&wgpu::TextureViewDescriptor::default());
        let prev_depth_view = cache.depth_texes[i - 1].create_view(&wgpu::TextureViewDescriptor {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("geom_peel_bg_{i}")),
            layout: &ctx.geom_peel_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cache.scene_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cache.lights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cache.material_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&prev_depth_view),
                },
            ],
        });

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(&format!("geom_peel_pass_{i}")),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });

        rpass.set_viewport(0.0, 0.0, w as f32, h as f32, 0.0, 1.0);
        rpass.set_pipeline(&ctx.geom_peel_pipeline);
        rpass.set_bind_group(0, &bg, &[]);
        rpass.set_vertex_buffer(0, mesh.vertex_buf.slice(..));
        rpass.set_index_buffer(mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..mesh.n_indices, 0, 0..1);
    }
}

// ── SSAO pass ─────────────────────────────────────────────────────────────────

fn encode_ssao_pass(
    ctx: &GpuMeshContext,
    cache: &GpuMeshFrameCache,
    encoder: &mut wgpu::CommandEncoder,
) {
    let nd_view = cache
        .normal_depth_tex
        .create_view(&wgpu::TextureViewDescriptor::default());

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ssao_bg"),
        layout: &ctx.ssao_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: cache.ssao_uniforms_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&nd_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cache.ao_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cache.ssao_kernel_buf.as_entire_binding(),
            },
        ],
    });

    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("ssao_pass"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(&ctx.ssao_pipeline);
    cpass.set_bind_group(0, &bg, &[]);
    let wx = (cache.cols as u32).div_ceil(8);
    let wy = (cache.rows as u32).div_ceil(8);
    cpass.dispatch_workgroups(wx, wy, 1);
}

// ── Composite pass ────────────────────────────────────────────────────────────

fn encode_composite_pass(
    ctx: &GpuMeshContext,
    cache: &GpuMeshFrameCache,
    encoder: &mut wgpu::CommandEncoder,
) {
    let color_array_view = cache
        .color_array_tex
        .create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("composite_bg"),
        layout: &ctx.composite_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&color_array_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: cache.ao_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cache.output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cache.comp_params_buf.as_entire_binding(),
            },
        ],
    });

    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("composite_pass"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(&ctx.composite_pipeline);
    cpass.set_bind_group(0, &bg, &[]);
    let wx = (cache.cols as u32).div_ceil(8);
    let wy = (cache.rows as u32).div_ceil(8);
    cpass.dispatch_workgroups(wx, wy, 1);
}
