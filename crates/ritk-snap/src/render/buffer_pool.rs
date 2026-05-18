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
//! | Call site                        | Eliminated scratch alloc        |
//! |----------------------------------|---------------------------------|
//! | `SliceRenderer::render_with_scratch` | `Vec<f32>` from `extract_slice` |
//! | `SliceRenderer::render_with_scratch` | `Vec<u8>` RGBA intermediate     |
//! | `render_mip_axial_with_scratch`  | `Vec<u8>` RGBA intermediate     |
//! | `render_vr_axial_with_scratch`   | `Vec<u8>` RGBA intermediate     |

/// Pre-allocated scratch buffers eliminating per-frame heap allocation on the
/// slice-render hot path.
///
/// # Invariants
///
/// - `pixel_f32.len()` equals the most-recently-requested `f32` length.
/// - `rgba_u8.len()` equals the most-recently-requested `u8` length.
/// - `Vec::capacity` is monotone non-decreasing; `Vec::resize` extends when
///   needed and reuses without shrinking otherwise.
#[derive(Debug, Default)]
pub(crate) struct RenderBufferPool {
    /// f32 scratch for `extract_slice_into` output.
    pub(crate) pixel_f32: Vec<f32>,
    /// u8 scratch for RGBA intermediate encoding.
    pub(crate) rgba_u8: Vec<u8>,
}

impl RenderBufferPool {
    /// Resize `rgba_u8` to exactly `len` elements, reusing existing capacity.
    ///
    /// Elements beyond the previous length are set to `0`.
    #[inline]
    pub(crate) fn resize_u8(&mut self, len: usize) {
        self.rgba_u8.resize(len, 0_u8);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::RenderBufferPool;
    use crate::render::colormap::Colormap;
    use crate::render::mip_vr::{render_mip_axial, render_mip_axial_with_scratch};
    use crate::render::slice_render::{SliceRenderer, WindowLevel};
    use crate::LoadedVolume;
    use std::sync::Arc;

    /// Construct a minimal [`LoadedVolume`] for differential tests.
    ///
    /// Pixel value at voxel `(d, r, c)` is `(d × R × C + r × C + c) as f32`.
    fn make_volume(depth: usize, rows: usize, cols: usize) -> LoadedVolume {
        let n = depth * rows * cols;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        LoadedVolume {
            data: Arc::new(data),
            shape: [depth, rows, cols],
            spacing: [1.0, 1.0, 1.0],
            origin: [0.0, 0.0, 0.0],
            direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            metadata: None,
            source: None,
            modality: None,
            patient_name: None,
            patient_id: None,
            study_date: None,
            series_description: None,
            series_time: None,
            patient_weight_kg: None,
            injected_dose_bq: None,
            radionuclide_half_life_s: None,
            radiopharmaceutical_start_time: None,
            decay_correction: None,
        }
    }

    /// Resizing `pixel_f32` to the same length twice must not shrink capacity.
    ///
    /// Analytical: initial capacity = 0 → after resize(100), capacity ≥ 100 →
    /// second resize(100) must preserve capacity (monotone non-decreasing).
    #[test]
    fn test_resize_f32_capacity_monotone_same_size() {
        let mut pool = RenderBufferPool::default();
        pool.pixel_f32.resize(100, 0.0_f32);
        assert_eq!(pool.pixel_f32.len(), 100, "len must equal requested size");
        let cap_after_first = pool.pixel_f32.capacity();
        assert!(
            cap_after_first >= 100,
            "capacity must be ≥ 100 after resize"
        );
        pool.pixel_f32.resize(100, 0.0_f32);
        assert_eq!(
            pool.pixel_f32.capacity(),
            cap_after_first,
            "capacity must not shrink on same-size resize"
        );
    }

    /// Growing from 200 then requesting 50 must preserve capacity at ≥ 200.
    ///
    /// Analytical: Vec::resize(n) with n < current capacity does not
    /// deallocate — capacity is monotone non-decreasing in observed peak.
    #[test]
    fn test_resize_u8_capacity_monotone_grow_then_smaller_request() {
        let mut pool = RenderBufferPool::default();
        pool.resize_u8(50);
        pool.resize_u8(200);
        let cap_at_200 = pool.rgba_u8.capacity();
        assert!(
            cap_at_200 >= 200,
            "capacity must be ≥ 200 after resize(200)"
        );
        pool.resize_u8(50);
        assert!(
            pool.rgba_u8.capacity() >= cap_at_200,
            "capacity must not shrink when resize requests a smaller length"
        );
        assert_eq!(
            pool.rgba_u8.len(),
            50,
            "len must equal the new requested size even if capacity is larger"
        );
    }

    /// New elements added by direct `Vec::resize` on `pixel_f32` must be initialized to `0.0`.
    #[test]
    fn test_resize_f32_new_elements_zero() {
        let mut pool = RenderBufferPool::default();
        pool.pixel_f32.resize(4, 0.0_f32);
        assert!(
            pool.pixel_f32.iter().all(|&v| v == 0.0_f32),
            "all newly allocated f32 elements must be 0.0"
        );
    }

    /// New elements added by `resize_u8` must be initialized to `0`.
    #[test]
    fn test_resize_u8_new_elements_zero() {
        let mut pool = RenderBufferPool::default();
        pool.resize_u8(4);
        assert!(
            pool.rgba_u8.iter().all(|&v| v == 0u8),
            "all newly allocated u8 elements must be 0"
        );
    }

    /// `render_with_scratch` must produce pixel-identical output to `render`.
    ///
    /// Differential equivalence contract: the scratch-based path and the
    /// allocating path must agree on every pixel for the same input.
    ///
    /// Volume: [D=4, R=5, C=6]. Axis 0 (axial), slice d=2.
    /// WL(centre=12.0, width=24.0): L=0, U=24. Grayscale colormap.
    #[test]
    fn test_render_with_scratch_axial_pixel_identical() {
        let vol = make_volume(4, 5, 6);
        let wl = WindowLevel::new(12.0, 24.0);
        let expected = SliceRenderer::render(&vol, 0, 2, wl, Colormap::Grayscale);
        let mut pool = RenderBufferPool::default();
        let actual =
            SliceRenderer::render_with_scratch(&mut pool, &vol, 0, 2, wl, Colormap::Grayscale);
        assert_eq!(
            actual.size, expected.size,
            "scratch render must produce identical dimensions"
        );
        assert_eq!(
            actual.pixels, expected.pixels,
            "scratch render must produce pixel-identical output for axis=0"
        );
    }

    /// `render_with_scratch` must produce pixel-identical output for coronal (axis=1).
    ///
    /// Volume: [D=4, R=5, C=6]. Axis 1 (coronal), slice r=2.
    #[test]
    fn test_render_with_scratch_coronal_pixel_identical() {
        let vol = make_volume(4, 5, 6);
        let wl = WindowLevel::new(12.0, 24.0);
        let expected = SliceRenderer::render(&vol, 1, 2, wl, Colormap::Grayscale);
        let mut pool = RenderBufferPool::default();
        let actual =
            SliceRenderer::render_with_scratch(&mut pool, &vol, 1, 2, wl, Colormap::Grayscale);
        assert_eq!(actual.size, expected.size, "coronal dimensions must match");
        assert_eq!(
            actual.pixels, expected.pixels,
            "scratch render must produce pixel-identical output for axis=1"
        );
    }

    /// `render_with_scratch` must produce pixel-identical output for sagittal (axis=2).
    ///
    /// Volume: [D=4, R=5, C=6]. Axis 2 (sagittal), slice c=1.
    #[test]
    fn test_render_with_scratch_sagittal_pixel_identical() {
        let vol = make_volume(4, 5, 6);
        let wl = WindowLevel::new(12.0, 24.0);
        let expected = SliceRenderer::render(&vol, 2, 1, wl, Colormap::Grayscale);
        let mut pool = RenderBufferPool::default();
        let actual =
            SliceRenderer::render_with_scratch(&mut pool, &vol, 2, 1, wl, Colormap::Grayscale);
        assert_eq!(actual.size, expected.size, "sagittal dimensions must match");
        assert_eq!(
            actual.pixels, expected.pixels,
            "scratch render must produce pixel-identical output for axis=2"
        );
    }

    /// Pool reuse across two consecutive calls must preserve pixel identity.
    ///
    /// Verifies that the pool correctly resets state between calls so that
    /// a second render produces the same result as a first render.
    #[test]
    fn test_render_with_scratch_pool_reuse_produces_consistent_results() {
        let vol = make_volume(4, 5, 6);
        let wl = WindowLevel::new(12.0, 24.0);
        let mut pool = RenderBufferPool::default();
        let first =
            SliceRenderer::render_with_scratch(&mut pool, &vol, 0, 2, wl, Colormap::Grayscale);
        let second =
            SliceRenderer::render_with_scratch(&mut pool, &vol, 0, 2, wl, Colormap::Grayscale);
        assert_eq!(
            first.pixels, second.pixels,
            "two renders with the same pool and inputs must be identical"
        );
    }

    /// `render_mip_axial_with_scratch` must produce pixel-identical output to
    /// `render_mip_axial`.
    ///
    /// Volume: [D=4, R=5, C=6]. WL(centre=12.0, width=24.0). Grayscale.
    #[test]
    fn test_render_mip_with_scratch_pixel_identical() {
        let vol = make_volume(4, 5, 6);
        let wl = WindowLevel::new(12.0, 24.0);
        let expected = render_mip_axial(&vol, wl, Colormap::Grayscale);
        let mut scratch = Vec::new();
        let actual = render_mip_axial_with_scratch(&mut scratch, &vol, wl, Colormap::Grayscale);
        assert_eq!(
            actual.size, expected.size,
            "MIP scratch render must produce identical dimensions"
        );
        assert_eq!(
            actual.pixels, expected.pixels,
            "MIP scratch render must produce pixel-identical output"
        );
    }
}
