//! Rendering pipeline for medical image display.
//!
//! This module owns the sub-systems needed to convert raw voxel data into
//! pixels ready for GPU upload:
//!
//! - [`colormap`]  — named intensity-to-RGB mappings.
//! - [`slice_render`] — DICOM window/level LUT and 2-D slice extraction.
//! - [`histogram`] — voxel intensity histogram computation SSOT.

pub mod colormap;
pub mod histogram;
pub mod slice_render;

pub use colormap::Colormap;
pub use histogram::{compute_histogram, histogram_bin_center, histogram_peak_count, Histogram};
pub use slice_render::{SliceRenderer, WindowLevel};
