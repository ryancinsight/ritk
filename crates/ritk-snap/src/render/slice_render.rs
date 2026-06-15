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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Construct a minimal [`LoadedVolume`] for shape and value tests.
    ///
    /// Pixel value at voxel `(d, r, c)` is `(d×R×C + r×C + c) as f32`, giving
    /// each voxel a unique, analytically derivable value.
    fn make_volume(depth: usize, rows: usize, cols: usize) -> LoadedVolume {
        let n = depth * rows * cols;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        LoadedVolume {
            data: std::sync::Arc::new(data),
            shape: [depth, rows, cols],
            channels: 1,
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

    // ── WindowLevel ───────────────────────────────────────────────────────────

    /// v ≤ L must map to 0 (lower saturation).
    ///
    /// Analytical: WL(centre=500, width=1000) → L = 0.
    /// v = −100 < 0 → output = 0.  v = 0 = L → output = 0.
    #[test]
    fn test_window_level_clamp_lower() {
        let wl = WindowLevel::new(500.0, 1000.0);
        // L = 500 − 500 = 0.
        assert_eq!(
            wl.apply(-100.0),
            0u8,
            "v = −100 (below lower bound L=0) must clamp to 0"
        );
        assert_eq!(
            wl.apply(0.0),
            0u8,
            "v = 0 (exactly at lower bound L=0) must clamp to 0"
        );
    }

    /// v ≥ U must map to 255 (upper saturation).
    ///
    /// Analytical: WL(centre=500, width=1000) → U = 1000.
    /// v = 1100 > 1000 → output = 255.  v = 1000 = U → output = 255.
    #[test]
    fn test_window_level_clamp_upper() {
        let wl = WindowLevel::new(500.0, 1000.0);
        // U = 500 + 500 = 1000.
        assert_eq!(
            wl.apply(1100.0),
            255u8,
            "v = 1100 (above upper bound U=1000) must clamp to 255"
        );
        assert_eq!(
            wl.apply(1000.0),
            255u8,
            "v = 1000 (exactly at upper bound U=1000) must clamp to 255"
        );
    }

    /// v = centre must map to ≈ 128 (within ±1 LSB of 127.5).
    ///
    /// Analytical: output = round((centre − L) / (U − L) × 255)
    ///   = round(width/2 / width × 255) = round(0.5 × 255) = round(127.5) = 128.
    ///
    /// Rust's `f64::round` uses round-half-away-from-zero, so 127.5 → 128.
    #[test]
    fn test_window_level_midpoint() {
        let wl = WindowLevel::new(500.0, 1000.0);
        let out = wl.apply(500.0);
        // Accept 127 or 128: the midpoint falls on a rounding boundary.
        assert!(
            out == 127 || out == 128,
            "v = centre = 500 must map to 127 or 128 (round(127.5)), got {out}"
        );
    }

    /// Verify the full 9-voxel axial slice d=0 of a [2,3,3] volume against
    /// the analytically derived WL formula.
    ///
    /// Pixel values 0..=8 with WL(centre=4, width=8): L=0, U=8.
    ///   i=0 → 0 (v ≤ L),  i=1..7 → round(i/8×255),  i=8 → 255 (v ≥ U).
    #[test]
    fn test_window_level_apply_slice_analytic() {
        let wl = WindowLevel::new(4.0, 8.0);
        // L = 0, U = 8.
        let pixels: Vec<f32> = (0..9u32).map(|i| i as f32).collect();
        let out = wl.apply_slice(&pixels);
        // Analytically derived expected values.
        let expected: Vec<u8> = vec![
            0u8,                                   // i=0, v=L → 0
            (1.0_f64 / 8.0 * 255.0).round() as u8, // i=1 → 32
            (2.0_f64 / 8.0 * 255.0).round() as u8, // i=2 → 64
            (3.0_f64 / 8.0 * 255.0).round() as u8, // i=3 → 96
            (4.0_f64 / 8.0 * 255.0).round() as u8, // i=4 → 128
            (5.0_f64 / 8.0 * 255.0).round() as u8, // i=5 → 159
            (6.0_f64 / 8.0 * 255.0).round() as u8, // i=6 → 191
            (7.0_f64 / 8.0 * 255.0).round() as u8, // i=7 → 223
            255u8,                                 // i=8, v=U → 255
        ];
        assert_eq!(
            out, expected,
            "apply_slice output must match DICOM PS 3.3 §C.7.6.3.1.5 formula"
        );
    }

    // ── SliceRenderer shape tests ─────────────────────────────────────────────

    /// Axial slice (axis=0) at depth index `d` must produce a ColorImage of
    /// egui size `[cols, rows]` (width = C, height = R).
    ///
    /// Analytical: volume [D=4, R=5, C=6], d=2 → width=6, height=5.
    #[test]
    fn test_slice_render_axial_shape() {
        let vol = make_volume(4, 5, 6);
        let wl = WindowLevel::new(12.0, 24.0);
        let img = SliceRenderer::render(&vol, 0, 2, wl, Colormap::Grayscale);
        // egui::ColorImage size convention: [width, height] = [cols, rows].
        assert_eq!(
            img.size,
            [6, 5],
            "axial slice of [D=4,R=5,C=6] at d=2 must have egui size [cols=6, rows=5]"
        );
        assert_eq!(
            img.pixels.len(),
            5 * 6,
            "axial pixel count must equal rows × cols = 30"
        );
    }

    /// Coronal slice (axis=1) at row index `r` must produce a ColorImage of
    /// egui size `[cols, depth]` (width = C, height = D).
    ///
    /// Analytical: volume [D=4, R=5, C=6], r=2 → width=6, height=4.
    #[test]
    fn test_slice_render_coronal_shape() {
        let vol = make_volume(4, 5, 6);
        let wl = WindowLevel::new(12.0, 24.0);
        let img = SliceRenderer::render(&vol, 1, 2, wl, Colormap::Grayscale);
        // egui::ColorImage size convention: [width, height] = [cols, depth].
        assert_eq!(
            img.size,
            [6, 4],
            "coronal slice of [D=4,R=5,C=6] at r=2 must have egui size [cols=6, depth=4]"
        );
        assert_eq!(
            img.pixels.len(),
            4 * 6,
            "coronal pixel count must equal depth × cols = 24"
        );
    }

    /// Sagittal slice (axis=2) at column index `c` must produce a ColorImage
    /// of egui size `[rows, depth]` (width = R, height = D).
    ///
    /// Analytical: volume [D=4, R=5, C=6], c=1 → width=5, height=4.
    #[test]
    fn test_slice_render_sagittal_shape() {
        let vol = make_volume(4, 5, 6);
        let wl = WindowLevel::new(12.0, 24.0);
        let img = SliceRenderer::render(&vol, 2, 1, wl, Colormap::Grayscale);
        // egui::ColorImage size convention: [width, height] = [rows, depth].
        assert_eq!(
            img.size,
            [5, 4],
            "sagittal slice of [D=4,R=5,C=6] at c=1 must have egui size [rows=5, depth=4]"
        );
        assert_eq!(
            img.pixels.len(),
            4 * 5,
            "sagittal pixel count must equal depth × rows = 20"
        );
    }

    /// Verify axial pixel values against the analytically derived WL formula.
    ///
    /// Volume [D=2, R=3, C=3]: pixel at (d,r,c) = d×9 + r×3 + c.
    /// Axial slice d=0: pixels 0..=8 in row-major order.
    /// WL(centre=4, width=8) → L=0, U=8.
    /// With Grayscale colormap the R channel equals the WL output exactly.
    #[test]
    fn test_slice_render_axial_pixel_values() {
        let vol = make_volume(2, 3, 3);
        let wl = WindowLevel::new(4.0, 8.0);
        let img = SliceRenderer::render(&vol, 0, 0, wl, Colormap::Grayscale);
        // Extract the red channel (= green = blue for Grayscale).
        let actual: Vec<u8> = img.pixels.iter().map(|p| p.r()).collect();
        // Analytically derived expected values (see test_window_level_apply_slice_analytic).
        let expected: Vec<u8> = vec![
            0,
            (1.0_f64 / 8.0 * 255.0).round() as u8,
            (2.0_f64 / 8.0 * 255.0).round() as u8,
            (3.0_f64 / 8.0 * 255.0).round() as u8,
            (4.0_f64 / 8.0 * 255.0).round() as u8,
            (5.0_f64 / 8.0 * 255.0).round() as u8,
            (6.0_f64 / 8.0 * 255.0).round() as u8,
            (7.0_f64 / 8.0 * 255.0).round() as u8,
            255,
        ];
        assert_eq!(
            actual, expected,
            "axial slice d=0 pixel values must match WL formula output"
        );
    }
}
