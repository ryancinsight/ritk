//! 2-D slice rendering pipeline for medical volume display.
//!
//! # Mathematical specification
//!
//! ## Window/Level (DICOM PS 3.3 §C.7.6.3.1.5)
//!
//! Given centre `c` and width `w`, define:
//! ```text
//! L = c − w/2        (lower bound)
//! U = c + w/2        (upper bound)
//! ```
//! For pixel value `v`:
//! ```text
//! output = 0                              if v ≤ L
//! output = 255                            if v ≥ U
//! output = round((v − L) / (U − L) × 255) otherwise
//! ```
//!
//! ## Slice extraction (row-major [D, R, C] volume)
//!
//! | `axis` | Fixed index | Pixel at (row, col) in output      | Output dimensions |
//! |--------|-------------|-------------------------------------|-------------------|
//! | 0      | d (axial)   | data[d×R×C + row×C + col]           | (rows=R, cols=C)  |
//! | 1      | r (coronal) | data[depth×R×C + r×C + col]         | (rows=D, cols=C)  |
//! | 2      | c (sagittal)| data[depth×R×C + row×C + c]         | (rows=D, cols=R)  |
//!
//! The `SliceRenderer` converts extracted pixels through the WL LUT and then
//! through a [`Colormap`], producing an [`egui::ColorImage`] of size
//! `[width, height]` = `[cols, rows]` in egui convention.

use super::buffer_pool::RenderBufferPool;
use super::colormap::Colormap;
use crate::LoadedVolume;

// ── WindowLevel ───────────────────────────────────────────────────────────────

/// Window/Level lookup table for linear DICOM intensity windowing.
///
/// # Invariants
/// - The mapping is monotone non-decreasing in `v`.
/// - Output is always in `[0, 255]` regardless of input magnitude.
///
/// # Mathematical specification (DICOM PS 3.3 §C.7.6.3.1.5)
///
/// Let `L = center − width/2`,  `U = center + width/2`.
///
/// For pixel value `v`:
/// ```text
/// output = 0                               if v ≤ L
/// output = 255                             if v ≥ U
/// output = round((v − L) / (U − L) × 255) otherwise
/// ```
#[derive(Debug, Clone, Copy)]
pub struct WindowLevel {
    /// Display window centre — the midpoint of the visible intensity range.
    pub center: f64,
    /// Display window width — the span of the visible intensity range.
    ///
    /// A width of zero or a negative value causes all inputs to saturate:
    /// values equal to or below `center` clamp to 0, values above clamp to 255.
    pub width: f64,
}

impl WindowLevel {
    /// Construct a `WindowLevel` with the given centre and width.
    pub fn new(center: f64, width: f64) -> Self {
        Self { center, width }
    }

    /// Apply the WL LUT to a single pixel value `v`, returning the u8 display value.
    ///
    /// # Safety comment
    /// Division by `(u − l)` is guarded by the branch structure: the division
    /// executes only when `l < v < u`, which implies `u − l = width > 0`.
    #[inline]
    pub fn apply(&self, v: f64) -> u8 {
        let l = self.center - self.width * 0.5;
        let u = self.center + self.width * 0.5;
        if v <= l {
            0
        } else if v >= u {
            255
        } else {
            // SAFETY: l < v < u ⟹ u − l = width > 0, no division by zero.
            ((v - l) / (u - l) * super::U8_MAX_F32 as f64).round() as u8
        }
    }

    /// Apply the WL LUT to every element of `pixels`, returning `Vec<u8>`.
    ///
    /// Each `f32` is widened to `f64` before applying the formula to preserve
    /// arithmetic precision across the full HU range (≈ −32 768 … +32 767).
    pub fn apply_slice(&self, pixels: &[f32]) -> Vec<u8> {
        pixels.iter().map(|&p| self.apply(p as f64)).collect()
    }
}

// ── SliceRenderer ─────────────────────────────────────────────────────────────

/// Renders a single 2-D slice from a [`LoadedVolume`] to an [`egui::ColorImage`].
///
/// # Coordinate conventions
///
/// | `axis` | Name     | Fixed index | Output: `[width, height]` in egui |
/// |--------|----------|-------------|-----------------------------------|
/// | 0      | Axial    | depth `d`   | `[cols, rows]`                    |
/// | 1      | Coronal  | row `r`     | `[cols, depth]`                   |
/// | 2      | Sagittal | column `c`  | `[rows, depth]`                   |
///
/// An out-of-range `index` is silently clamped. An unknown `axis` yields a
/// 1×1 black image rather than a panic.
pub struct SliceRenderer;

impl SliceRenderer {
    /// Extract and render a single slice from `volume`.
    ///
    /// # Parameters
    /// - `volume`   — source volume with row-major `[depth, rows, cols]` layout.
    /// - `axis`     — 0 = axial (fixed depth), 1 = coronal (fixed row),
    ///   2 = sagittal (fixed column).
    /// - `index`    — position along `axis`; clamped to the valid range silently.
    /// - `wl`       — DICOM window/level parameters for intensity mapping.
    /// - `colormap` — colormap applied after WL normalisation.
    ///
    /// # Returns
    /// An [`egui::ColorImage`] of size `[width, height]` (see table above)
    /// containing RGB pixels ready for GPU upload.
    pub fn render(
        volume: &LoadedVolume,
        axis: usize,
        index: usize,
        wl: WindowLevel,
        colormap: Colormap,
    ) -> egui::ColorImage {
        let (pixels, width, height) = volume.extract_slice(axis, index);
        if width == 0 || height == 0 {
            // Return a minimal valid image rather than panic; callers can detect
            // the degenerate case by checking image.size.
            return egui::ColorImage::from_rgb([1, 1], &[0u8, 0, 0]);
        }

        // Fused WL+colormap single pass: apply window/level per pixel, then
        // map the normalised result through the colormap directly, eliminating
        // the intermediate `wl_bytes` allocation.
        let mut rgba = Vec::with_capacity(width * height * 4);
        for &p in &pixels {
            let byte = wl.apply(p as f64);
            let [r, g, b] = colormap.map(byte as f32 / super::U8_MAX_F32);
            rgba.push(r);
            rgba.push(g);
            rgba.push(b);
            rgba.push(255);
        }

        egui::ColorImage::from_rgba_unmultiplied([width, height], &rgba)
    }

    /// Extract and render a single slice using pre-allocated scratch buffers.
    ///
    /// Produces output pixel-identical to [`Self::render`] for the same inputs
    /// while eliminating two per-call heap allocations:
    ///
    /// 1. The `Vec<f32>` created by `extract_slice` (replaced by
    ///    `pool.pixel_f32` reuse via `extract_slice_into`).
    /// 2. The `Vec<u8>` RGBA intermediate (replaced by `pool.rgba_u8` reuse).
    ///
    /// # Differential equivalence invariant
    ///
    /// For all valid (`volume`, `axis`, `index`, `wl`, `colormap`) inputs:
    /// ```text
    /// render_with_scratch(pool, volume, axis, index, wl, colormap).pixels
    ///   == render(volume, axis, index, wl, colormap).pixels
    /// ```
    pub(crate) fn render_with_scratch(
        pool: &mut RenderBufferPool,
        volume: &LoadedVolume,
        axis: usize,
        index: usize,
        wl: WindowLevel,
        colormap: Colormap,
    ) -> egui::ColorImage {
        let (width, height) = volume.extract_slice_into(&mut pool.pixel_f32, axis, index);
        if width == 0 || height == 0 {
            return egui::ColorImage::from_rgb([1, 1], &[0u8, 0, 0]);
        }
        pool.resize_pixel_bytes(width * height * 4);
        // Split-borrow: pool.pixel_f32 (read) and pool.rgba_u8 (write) are
        // distinct fields; Rust NLL permits simultaneous borrows.
        let pixels = pool.pixel_f32.as_slice();
        let rgba = pool.rgba_u8.as_mut_slice();
        for (i, &p) in pixels.iter().enumerate() {
            let byte = wl.apply(p as f64);
            let [r, g, b] = colormap.map(byte as f32 / super::U8_MAX_F32);
            let base = i * 4;
            rgba[base] = r;
            rgba[base + 1] = g;
            rgba[base + 2] = b;
            rgba[base + 3] = 255;
        }
        egui::ColorImage::from_rgba_unmultiplied([width, height], &pool.rgba_u8)
    }
}

#[cfg(test)]
#[path = "tests_slice_render.rs"]
mod tests;
