//! Rendering pipeline for medical image display.
//!
//! This module owns the two sub-systems needed to convert raw voxel data into
//! pixels ready for GPU upload:
//!
//! - [`colormap`] — named intensity-to-RGB mappings.
//! - [`slice_render`] — DICOM window/level LUT and 2-D slice extraction.

pub mod colormap;
pub mod slice_render;

pub use colormap::Colormap;
pub use slice_render::{SliceRenderer, WindowLevel};
