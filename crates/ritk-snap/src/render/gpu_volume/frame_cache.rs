//! Cached per-frame GPU output and staging buffers.
//!
//! Allocating a new `wgpu::Buffer` on every render call is expensive when the
//! output dimensions are stable (typical case: fixed viewport size).
//! [`GpuFrameCache`] holds one output buffer and one staging buffer sized for
//! a specific `(rows, cols)` output. Callers call [`GpuFrameCache::ensure`]
//! once per frame: if the dimensions match the cached buffers they are reused;
//! if not (e.g. viewport resize) new buffers are allocated and the old ones
//! are dropped.
//!
//! # Buffer lifecycle
//!
//! - **Output buffer**: written by the compute shader each frame.
//! - **Staging buffer**: receives a `copy_buffer_to_buffer` from output, then
//!   is mapped for CPU read.  After [`wgpu::BufferSlice::unmap`] the staging
//!   buffer is returned to the GPU and can be reused on the next frame.
//!
//! # Safety
//!
//! Callers must ensure that `ctx.device.poll(wgpu::Maintain::Wait)` has
//! completed before re-submitting work to the same buffers.  All call sites
//! in this crate satisfy this invariant by synchronously polling to completion
//! before unmapping, matching the single-threaded render model.

use wgpu::BufferUsages;

/// Cached output and staging GPU buffers for one render pass.
///
/// The buffers are valid for exactly `rows × cols` output pixels at
/// `bytes_per_pixel` bytes each.
pub(super) struct GpuFrameCache {
    /// Cached output dimension (rows).
    pub rows: usize,
    /// Cached output dimension (cols).
    pub cols: usize,
    /// GPU storage buffer written by the compute shader.
    pub output_buf: wgpu::Buffer,
    /// CPU-mappable staging buffer for readback.
    pub staging_buf: wgpu::Buffer,
}

impl GpuFrameCache {
    /// Allocate new output and staging buffers for `rows × cols` pixels at
    /// `bytes_per_pixel` bytes each.
    pub fn new(
        device: &wgpu::Device,
        rows: usize,
        cols: usize,
        bytes_per_pixel: u64,
    ) -> Self {
        let size = rows as u64 * cols as u64 * bytes_per_pixel;
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_frame_output"),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_frame_staging"),
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { rows, cols, output_buf, staging_buf }
    }
}
