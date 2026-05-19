//! GPU uniform buffer structs for the MIP/VR compute shaders.
//!
//! `RenderParams` must match the `struct RenderParams` layout in `mip.wgsl`.
//!
//! # Layout invariant (std140)
//!
//! All fields are `u32` (4 bytes each). Total size = 16 bytes, satisfying
//! the 16-byte minimum alignment required by wgpu uniform buffers.

use bytemuck::{Pod, Zeroable};

/// Uniform parameters uploaded to the MIP compute shader.
///
/// # WGSL struct binding
///
/// ```wgsl
/// struct RenderParams {
///     depth: u32,
///     rows:  u32,
///     cols:  u32,
///     _pad:  u32,
/// }
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(super) struct RenderParams {
    /// Number of depth slices (z-axis, first dimension of the volume).
    pub depth: u32,
    /// Number of rows (y-axis, second dimension of the output image).
    pub rows: u32,
    /// Number of columns (x-axis, third dimension of the output image).
    pub cols: u32,
    /// Padding to 16-byte std140 boundary.
    pub _pad: u32,
}
