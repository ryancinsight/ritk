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
///     depth    : u32,
///     rows     : u32,
///     cols     : u32,
///     _pad0    : u32,
///     center       : f32,
///     width        : f32,
///     presentation : u32,
///     _pad2        : u32,
/// }
/// ```
///
/// Total size: 32 bytes (8 × 4-byte fields), satisfying 16-byte std140 alignment.
///
/// The presentation bitfield stores the DICOM VOI function in bits 0–1 and
/// the MONOCHROME1 inversion flag in bit 2.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(super) struct RenderParams {
    /// Number of depth slices (z-axis, first dimension of the volume).
    pub depth: u32,
    /// Number of rows (y-axis, second dimension of the output image).
    pub rows: u32,
    /// Number of columns (x-axis, third dimension of the output image).
    pub cols: u32,
    /// Padding to first 16-byte boundary.
    pub _pad0: u32,
    /// Window centre.
    pub center: f32,
    /// Window width, floored to one at the host boundary.
    pub width: f32,
    /// DICOM VOI function and MONOCHROME1 inversion bitfield.
    pub presentation: u32,
    /// Padding to the 32-byte uniform size.
    pub _pad2: u32,
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
///     center:       f32,
///     width:        f32,
///     alpha_scale:  f32,
///     presentation: u32,
/// }
/// ```
///
/// Total size: 32 bytes (8 × 4-byte fields), satisfying 16-byte std140 alignment.
///
/// The presentation bitfield stores the DICOM VOI function in bits 0–1 and
/// the MONOCHROME1 inversion flag in bit 2.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(super) struct VrParams {
    pub depth: u32,
    pub rows: u32,
    pub cols: u32,
    pub _pad0: u32,
    /// Window centre.
    pub center: f32,
    /// Window width, floored to one at the host boundary.
    pub width: f32,
    /// Per-voxel opacity scale factor. Canonical app value: `0.06`.
    pub alpha_scale: f32,
    /// DICOM VOI function and MONOCHROME1 inversion bitfield.
    pub presentation: u32,
}
