//! Rendering pipeline for medical image display.
//!
//! This module owns the sub-systems needed to convert raw voxel data into
//! pixels ready for GPU upload:
//!
//! - [`NamedColorMap`] — Iris-owned normalized intensity-to-RGBA mappings.
//! - [`fusion`] — primary/secondary fused compare rendering.
//! - [`grayscale`] — DICOM scalar presentation semantics.
//! - [`slice_render`] — 2-D slice extraction and pixel-buffer assembly.
//! - [`histogram`] — voxel intensity histogram computation SSOT.
//! - [`mip_vr`] — multi-resolution image rendering.

pub mod buffer_pool;
pub mod fusion;
#[cfg(not(target_arch = "wasm32"))]
pub mod gpu_mesh;
#[cfg(not(target_arch = "wasm32"))]
pub mod gpu_volume;
pub mod grayscale;
pub mod histogram;
pub mod mesh_render;
pub mod mip_vr;
pub mod slice_render;

pub(crate) use buffer_pool::RenderBufferPool;
/// Maximum value of a u8 pixel component as f32, used for normalizing to [0, 1].
pub(crate) const U8_MAX_F32: f32 = 255.0;
pub use fusion::{render_fused_slice, secondary_slice_for_primary, FusedSliceParams, FusionError};
#[cfg(not(target_arch = "wasm32"))]
pub use gpu_mesh::{GpuMeshRenderer, MeshRenderConfig, SsaoConfig};
pub use grayscale::{
    GrayscalePresentation, GrayscalePresentationError, VoiLutFunction, WindowLevel,
};
pub use histogram::{compute_histogram, histogram_bin_center, histogram_peak_count, Histogram};
pub use iris::color::NamedColorMap;
pub use mesh_render::{DirectionalLight, MeshCamera, MeshRenderer, PhongMaterial};
pub use mip_vr::{render_mip_axial, render_vr_axial};
pub use slice_render::SliceRenderer;

#[cfg(test)]
#[path = "tests_colormap.rs"]
mod tests_colormap;
