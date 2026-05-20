//! Rendering pipeline for medical image display.
//!
//! This module owns the sub-systems needed to convert raw voxel data into
//! pixels ready for GPU upload:
//!
//! - [`colormap`]  — named intensity-to-RGB mappings.
//! - [`fusion`] — primary/secondary fused compare rendering.
//! - [`slice_render`] — DICOM window/level LUT and 2-D slice extraction.
//! - [`histogram`] — voxel intensity histogram computation SSOT.
//! - [`mip_vr`] — multi-resolution image rendering.

pub mod buffer_pool;
pub mod colormap;
pub mod fusion;
pub mod histogram;
pub mod mesh_render;
pub mod mip_vr;
pub mod slice_render;
#[cfg(not(target_arch = "wasm32"))]
pub mod gpu_volume;
#[cfg(not(target_arch = "wasm32"))]
pub mod gpu_mesh;

pub(crate) use buffer_pool::RenderBufferPool;
pub use colormap::Colormap;
pub use fusion::{render_fused_slice, FusedSliceParams};
pub use histogram::{compute_histogram, histogram_bin_center, histogram_peak_count, Histogram};
pub use mesh_render::{DirectionalLight, MeshCamera, MeshRenderer, PhongMaterial};
pub use mip_vr::{render_mip_axial, render_vr_axial};
pub use slice_render::{SliceRenderer, WindowLevel};
#[cfg(not(target_arch = "wasm32"))]
pub use gpu_mesh::{GpuMeshRenderer, MeshRenderConfig, SsaoConfig};
