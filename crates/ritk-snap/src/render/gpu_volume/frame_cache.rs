//! Cached per-frame GPU buffers for one render pass.
//!
//! Holds four buffers that are reused across frames to eliminate per-frame
//! GPU allocation overhead:
//!
//! - `output_buf` — compute shader writes packed u32 RGBA pixels here.
//! - `staging_buf` — receives a `copy_buffer_to_buffer` and is mapped for
//!   async CPU readback.
//! - `params_buf` — 32-byte uniform updated each frame via `queue.write_buffer`
//!   (no re-allocation).
//! - `lut_buf` — 256-entry × 4-channel f32 RGBA LUT (4 096 bytes); updated
//!   each frame via `queue.write_buffer`.
//!
//! All four buffers are reallocated only when viewport dimensions change
//! (typical case: stable viewport → zero GPU allocation per frame after
//! the first call).
//!
//! # Non-blocking readback contract
//!
//! `staging_buf` must NOT be reused while an in-flight `map_async` is pending.
//! Callers track in-flight state via [`super::PendingReadback`] and only
//! resubmit once the receiver fires `Ok(())`.

use wgpu::BufferUsages;

/// Cached GPU buffers for one render pass at a fixed output resolution.
pub(super) struct GpuFrameCache {
    /// Output pixel count (rows dimension).
    pub rows: usize,
    /// Output pixel count (cols dimension).
    pub cols: usize,
    /// Compute shader output — STORAGE | COPY_SRC.
    pub output_buf: wgpu::Buffer,
    /// CPU-mappable staging target — MAP_READ | COPY_DST.
    pub staging_buf: wgpu::Buffer,
    /// 32-byte render parameter uniform — UNIFORM | COPY_DST.
    ///
    /// Both [`super::params::RenderParams`] and [`super::params::VrParams`]
    /// are 32 bytes; this single buffer serves both pass types.
    pub params_buf: wgpu::Buffer,
    /// 256-entry RGBA f32 colormap LUT — STORAGE | COPY_DST.
    ///
    /// Layout: `lut[i*4 + c]` = channel `c` ∈ [0,1] for LUT entry `i`.
    /// Size: 256 × 4 × 4 = 4 096 bytes.
    pub lut_buf: wgpu::Buffer,
}

impl GpuFrameCache {
    /// Allocate all four GPU buffers for a `rows × cols` output viewport.
    ///
    /// `bytes_per_pixel` must be 4 (packed u32 RGBA for both MIP and VR passes).
    ///
    /// # Allocation sizes
    ///
    /// | Buffer      | Size                          |
    /// |-------------|-------------------------------|
    /// | output_buf  | rows × cols × bytes_per_pixel |
    /// | staging_buf | rows × cols × bytes_per_pixel |
    /// | params_buf  | 32 bytes (fixed)              |
    /// | lut_buf     | 4 096 bytes (fixed)           |
    pub fn new(
        device: &wgpu::Device,
        rows: usize,
        cols: usize,
        bytes_per_pixel: u64,
    ) -> Self {
        let pixel_bytes = rows as u64 * cols as u64 * bytes_per_pixel;

        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_frame_output"),
            size: pixel_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_frame_staging"),
            size: pixel_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // COPY_DST is required for queue.write_buffer.
        // Both RenderParams and VrParams are exactly 32 bytes (see params.rs).
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_frame_params"),
            size: 32,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 256 entries × 4 channels × sizeof(f32) = 4 096 bytes.
        let lut_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_frame_lut"),
            size: 256 * 4 * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self { rows, cols, output_buf, staging_buf, params_buf, lut_buf }
    }
}
