//! 2-D slice rendering pipeline for medical volume display.
//!
//! # Mathematical specification
//!
//! ## Window/Level (DICOM PS 3.3 C.11.2)
//!
//! Given centre `c` and width `w`, define:
//! ```text
//! L = c − 0.5 − (w − 1)/2        (lower bound)
//! U = c − 0.5 + (w − 1)/2        (upper bound)
//! ```
//! For pixel value `v`:
//! ```text
//! output = 0                                      if v ≤ L
//! output = 255                                    if v > U
//! output = round(((v − (c − 0.5))/(w − 1) + 0.5) × 255) otherwise
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
//! Scalar slices are converted through the WL LUT and then through a
//! [`NamedColorMap`]. RGB slices preserve their three decoded channels and
//! bypass scalar windowing, producing an [`egui::ColorImage`] of size
//! `[width, height]` = `[cols, rows]` in egui convention.

use super::buffer_pool::RenderBufferPool;
use super::{GrayscalePresentation, NamedColorMap, WindowLevel};
use crate::LoadedVolume;
use iris::color::{ColorMap, Normalized};

// ── SliceRenderer ─────────────────────────────────────────────────────────────

/// Renders a single 2-D slice from a [`LoadedVolume`] to an [`egui::ColorImage`].
///
/// Scalar volumes use DICOM window/level and the selected Iris colormap. RGB
/// volumes preserve decoded red, green, and blue channels without scalar
/// windowing or colormap mapping.
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
    /// - `wl`       — DICOM window/level parameters for scalar intensity mapping.
    /// - `colormap` — colormap applied after scalar WL normalisation; ignored for RGB.
    ///
    /// # Returns
    /// An [`egui::ColorImage`] of size `[width, height]` (see table above)
    /// containing RGB pixels ready for GPU upload.
    pub fn render(
        volume: &LoadedVolume,
        axis: usize,
        index: usize,
        wl: WindowLevel,
        colormap: NamedColorMap,
    ) -> egui::ColorImage {
        if volume.channels == 3 {
            let (samples, width, height) = volume.extract_slice_channels(axis, index);
            return render_rgb_slice(&samples, width, height);
        }
        if volume.channels != 1 {
            return invalid_channel_image(volume.channels);
        }

        let presentation = match GrayscalePresentation::for_volume(volume) {
            Ok(presentation) => presentation,
            Err(error) => {
                tracing::error!(%error, "invalid DICOM grayscale presentation metadata");
                return invalid_image();
            }
        };
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
            let byte = presentation.apply(wl, f64::from(p));
            let value = Normalized::from_u8(byte);
            rgba.extend_from_slice(&colormap.sample(value).to_rgba8());
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
        colormap: NamedColorMap,
    ) -> egui::ColorImage {
        if volume.channels == 3 {
            let (width, height) =
                volume.extract_slice_channels_into(&mut pool.pixel_f32, axis, index);
            if width == 0 || height == 0 {
                return egui::ColorImage::from_rgb([1, 1], &[0_u8, 0, 0]);
            }
            pool.resize_pixel_bytes(width * height * 4);
            let valid = write_rgb_rgba(&pool.pixel_f32, &mut pool.rgba_u8);
            if !valid {
                return invalid_channel_image(volume.channels);
            }
            return egui::ColorImage::from_rgba_unmultiplied([width, height], &pool.rgba_u8);
        }
        if volume.channels != 1 {
            return invalid_channel_image(volume.channels);
        }

        let presentation = match GrayscalePresentation::for_volume(volume) {
            Ok(presentation) => presentation,
            Err(error) => {
                tracing::error!(%error, "invalid DICOM grayscale presentation metadata");
                return invalid_image();
            }
        };
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
            let byte = presentation.apply(wl, f64::from(p));
            let value = Normalized::from_u8(byte);
            let [r, g, b, alpha] = colormap.sample(value).to_rgba8();
            let base = i * 4;
            rgba[base] = r;
            rgba[base + 1] = g;
            rgba[base + 2] = b;
            rgba[base + 3] = alpha;
        }
        egui::ColorImage::from_rgba_unmultiplied([width, height], &pool.rgba_u8)
    }
}

fn render_rgb_slice(samples: &[f32], width: usize, height: usize) -> egui::ColorImage {
    if width == 0 || height == 0 {
        return egui::ColorImage::from_rgb([1, 1], &[0_u8, 0, 0]);
    }
    let mut rgba = vec![0_u8; width * height * 4];
    if !write_rgb_rgba(samples, &mut rgba) {
        return invalid_channel_image(3);
    }
    egui::ColorImage::from_rgba_unmultiplied([width, height], &rgba)
}

fn write_rgb_rgba(samples: &[f32], rgba: &mut [u8]) -> bool {
    let mut valid = samples.len() == rgba.len() / 4 * 3;
    for (channels, pixel) in samples.chunks_exact(3).zip(rgba.chunks_exact_mut(4)) {
        let Some(red) = rgb_component(channels[0]) else {
            valid = false;
            break;
        };
        let Some(green) = rgb_component(channels[1]) else {
            valid = false;
            break;
        };
        let Some(blue) = rgb_component(channels[2]) else {
            valid = false;
            break;
        };
        pixel.copy_from_slice(&[red, green, blue, 255]);
    }
    valid && samples.chunks_exact(3).remainder().is_empty()
}

fn rgb_component(value: f32) -> Option<u8> {
    if !value.is_finite() || !(0.0..=255.0).contains(&value) || value.fract() != 0.0 {
        return None;
    }
    // DICOM RGB decoding admits unsigned 8-bit integer samples, represented
    // exactly as f32 by the IO boundary; this narrowing preserves that value.
    #[expect(clippy::cast_possible_truncation, reason = "validated DICOM RGB byte")]
    Some(value as u8)
}

fn invalid_channel_image(channels: u8) -> egui::ColorImage {
    tracing::error!(channels, "slice rendering requires scalar or RGB channels");
    invalid_image()
}

fn invalid_image() -> egui::ColorImage {
    egui::ColorImage::from_rgb([1, 1], &[255_u8, 0, 255])
}

#[cfg(test)]
#[path = "tests_slice_render.rs"]
mod tests;
