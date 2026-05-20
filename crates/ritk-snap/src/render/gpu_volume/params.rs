//! GPU uniform buffer structs for the MIP/VR compute shaders.
//!
//! `RenderParams` must match `struct RenderParams` in `mip.wgsl`.
//! `VrParams` must match `struct VrParams` in `vr.wgsl`.
//!
//! # Layout invariant (std140)
//!
//! Both structs use only `u32`/`f32` fields (4 bytes each) and are padded to
//! a multiple of 16 bytes, satisfying the wgpu uniform-buffer minimum
//! alignment requirement.

use bytemuck::{Pod, Zeroable};

/// Uniform parameters for the MIP compute shader.
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
///
/// Total size: 16 bytes (4 × u32).
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

/// Uniform parameters for the VR compute shader.
///
/// # WGSL struct binding
///
/// ```wgsl
/// struct VrParams {
///     depth:       u32,
///     rows:        u32,
///     cols:        u32,
///     _pad0:       u32,
///     wl_lo:       f32,
///     wl_range:    f32,
///     alpha_scale: f32,
///     _pad1:       f32,
/// }
/// ```
///
/// Total size: 32 bytes (8 × 4-byte fields), satisfying 16-byte std140 alignment.
///
/// # Field derivation
///
/// Given a [`WindowLevel`] with `center` and `width`:
/// - `wl_lo    = center − 0.5 × width`
/// - `wl_range = width`  (floored to 1.0 to prevent division by zero)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(super) struct VrParams {
    pub depth: u32,
    pub rows: u32,
    pub cols: u32,
    pub _pad0: u32,
    /// Lower bound of the window/level range: `center − 0.5 × width`.
    pub wl_lo: f32,
    /// Width of the window/level range (≥ 1.0).
    pub wl_range: f32,
    /// Per-voxel opacity scale factor. Canonical app value: `0.06`.
    pub alpha_scale: f32,
    pub _pad1: f32,
}
