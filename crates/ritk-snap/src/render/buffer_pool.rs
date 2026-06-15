//! Pre-allocated scratch buffers for per-frame texture rebuild operations.
//!
//! # Allocation invariant
//!
//! Once a scratch buffer has grown to the maximum observed dimension, all
//! subsequent render calls for equal or smaller dimensions incur zero heap
//! allocations. Capacity is monotone non-decreasing; no shrink occurs.
//!
//! # Usage
//!
//! `RenderBufferPool` is stored as a field on `SnapApp` and threaded through
//! the slice render and MIP render helpers as `&mut RenderBufferPool`.
//!
//! ## Eliminated allocations per dirty-texture rebuild
//!
//! | Call site | Eliminated scratch alloc |
//! |----------------------------------|---------------------------------|
//! | `SliceRenderer::render_with_scratch` | `Vec<f32>` from `extract_slice` |
//! | `SliceRenderer::render_with_scratch` | `Vec<u8>` RGBA intermediate |
//! | `render_mip_axial_with_scratch` | `Vec<u8>` RGBA intermediate |
//! | `render_vr_axial_with_scratch` | `Vec<u8>` RGBA intermediate |
//! | `apply_to_image_into` | `Vec<Color32>` for transform output |
//!
//! The `color32` scratch buffer allows viewport orientation transforms
//! (flip/rotate) to write their output into pre-allocated memory instead
//! of allocating a new `Vec<Color32>` per transform step.

/// Pre-allocated scratch buffers eliminating per-frame heap allocation on the
/// slice-render hot path.
///
/// # Invariants
///
/// - `pixel_f32.len()` equals the most-recently-requested `f32` length.
/// - `rgba_u8.len()` equals the most-recently-requested `u8` length.
/// - `color32.len()` equals the most-recently-requested `Color32` length.
/// - `Vec::capacity` is monotone non-decreasing; `Vec::resize` extends when
///   needed and reuses without shrinking otherwise.
#[derive(Debug, Default)]
pub(crate) struct RenderBufferPool {
    /// f32 scratch for `extract_slice_into` output.
    pub(crate) pixel_f32: Vec<f32>,
    /// u8 scratch for RGBA intermediate encoding.
    pub(crate) rgba_u8: Vec<u8>,
    /// Color32 scratch for viewport orientation transform output.
    pub(crate) color32: Vec<egui::Color32>,
}

impl RenderBufferPool {
    /// Resize `rgba_u8` to exactly `len` elements, reusing existing capacity.
    ///
    /// Elements beyond the previous length are set to `0`.
    #[inline]
    pub(crate) fn resize_pixel_bytes(&mut self, len: usize) {
        self.rgba_u8.resize(len, 0_u8);
    }

    /// Resize `color32` to exactly `len` elements, reusing existing capacity.
    ///
    /// Elements beyond the previous length are set to `Color32::BLACK`.
    #[inline]
    pub(crate) fn resize_color32(&mut self, len: usize) {
        self.color32.resize(len, egui::Color32::BLACK);
    }
}

#[cfg(test)]
#[path = "tests_buffer_pool.rs"]
mod tests;
