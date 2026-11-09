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
//! bypass scalar windowing. The canonical output is a bounded RGBA image.

use super::{GrayscalePresentation, NamedColorMap, WindowLevel};
use crate::LoadedVolume;
use iris::color::{ColorMap, Normalized};

// ── SliceRenderer ─────────────────────────────────────────────────────────────

/// A bounded row-major RGBA image produced by the RITK display pipeline.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbaImage {
    size: [usize; 2],
    pixels: Box<[u8]>,
}

/// Reusable scratch storage for format-neutral presentation frames.
///
/// The extraction buffer is aligned by the provider's existing image boundary;
/// the RGBA buffer retains its capacity between renders. A frame swaps its
/// completed storage with this scratch value, so the next render writes into
/// the storage returned by the previous frame without allocating a second
/// full-image buffer.
#[derive(Debug, Default)]
pub(crate) struct FrameRenderScratch {
    pub(crate) pixel_f32: mnemosyne::AlignedVec<f32>,
    pub(crate) rgba: Vec<u8>,
}

#[cfg(not(target_arch = "wasm32"))]
impl RgbaImage {
    /// Wraps row-major RGBA bytes for an image of `[width, height]` pixels.
    ///
    /// # Panics
    /// Panics when `pixels` does not hold exactly four bytes per pixel.
    #[must_use]
    pub fn new(size: [usize; 2], pixels: Vec<u8>) -> Self {
        assert_eq!(
            pixels.len(),
            size[0].saturating_mul(size[1]).saturating_mul(4),
            "RGBA storage must hold four bytes per pixel"
        );
        Self {
            size,
            pixels: pixels.into_boxed_slice(),
        }
    }

    #[cfg(all(windows, test))]
    pub(crate) fn into_parts(self) -> ([usize; 2], Box<[u8]>) {
        (self.size, self.pixels)
    }

    /// Builds an opaque image from row-major RGB bytes.
    #[must_use]
    pub fn from_rgb(size: [usize; 2], rgb: &[u8]) -> Self {
        let mut pixels = Vec::with_capacity(rgb.len() / 3 * 4);
        for channels in rgb.chunks_exact(3) {
            pixels.extend_from_slice(channels);
            pixels.push(u8::MAX);
        }
        Self::new(size, pixels)
    }

    /// Image extent as `[width, height]`.
    #[must_use]
    pub const fn size(&self) -> [usize; 2] {
        self.size
    }

    /// Row-major RGBA bytes.
    #[must_use]
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// Pixels as RGBA quadruples in row-major order.
    pub fn rgba_pixels(&self) -> impl Iterator<Item = [u8; 4]> + '_ {
        self.pixels
            .chunks_exact(4)
            .map(|pixel| [pixel[0], pixel[1], pixel[2], pixel[3]])
    }
}

/// Renders a single 2-D slice from a [`LoadedVolume`].
///
/// Scalar volumes use DICOM window/level and the selected Iris colormap. RGB
/// volumes preserve decoded red, green, and blue channels without scalar
/// windowing or colormap mapping.
///
/// # Coordinate conventions
///
/// | `axis` | Name     | Fixed index | Output: `[width, height]` in pixels |
/// |--------|----------|-------------|-----------------------------------|
/// | 0      | Axial    | depth `d`   | `[cols, rows]`                    |
/// | 1      | Coronal  | row `r`     | `[cols, depth]`                   |
/// | 2      | Sagittal | column `c`  | `[rows, depth]`                   |
///
/// An out-of-range `index` is silently clamped. An unknown `axis` yields a
/// 1×1 black image rather than a panic.
pub struct SliceRenderer;

impl SliceRenderer {
    /// Extracts and renders one slice into the neutral RGBA carrier used by
    /// Métis hosts.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn render(
        volume: &LoadedVolume,
        axis: usize,
        index: usize,
        wl: WindowLevel,
        colormap: NamedColorMap,
    ) -> RgbaImage {
        let mut scratch = FrameRenderScratch::default();
        let size = Self::render_rgba_into(&mut scratch, volume, axis, index, wl, colormap);
        RgbaImage::new(size, scratch.rgba)
    }

    /// Extracts and renders one slice into reusable caller-owned storage.
    ///
    /// The returned dimensions are `[width, height]`. The RGBA bytes in
    /// `scratch.rgba` are valid for exactly that extent. The method preserves
    /// the same invalid-channel and malformed-grayscale sentinel images as
    /// [`Self::render`], while reusing both extraction and pixel buffers.
    pub(crate) fn render_rgba_into(
        scratch: &mut FrameRenderScratch,
        volume: &LoadedVolume,
        axis: usize,
        index: usize,
        wl: WindowLevel,
        colormap: NamedColorMap,
    ) -> [usize; 2] {
        if volume.channels == 3 {
            let (width, height) =
                volume.extract_slice_channels_into(&mut scratch.pixel_f32, axis, index);
            if width == 0 || height == 0 {
                write_invalid_image(&mut scratch.rgba);
                return [1, 1];
            }
            let Some(byte_count) = width
                .checked_mul(height)
                .and_then(|pixels| pixels.checked_mul(4))
            else {
                write_invalid_image(&mut scratch.rgba);
                return [1, 1];
            };
            scratch.rgba.resize(byte_count, 0);
            if !write_rgb_rgba(&scratch.pixel_f32, &mut scratch.rgba) {
                write_invalid_image(&mut scratch.rgba);
                return [1, 1];
            }
            return [width, height];
        }
        if volume.channels != 1 {
            tracing::error!(
                channels = volume.channels,
                "slice rendering requires scalar or RGB channels"
            );
            write_invalid_image(&mut scratch.rgba);
            return [1, 1];
        }

        let presentation = match GrayscalePresentation::for_volume(volume) {
            Ok(presentation) => presentation,
            Err(error) => {
                tracing::error!(%error, "invalid DICOM grayscale presentation metadata");
                write_invalid_image(&mut scratch.rgba);
                return [1, 1];
            }
        };
        let (width, height) = volume.extract_slice_into(&mut scratch.pixel_f32, axis, index);
        if width == 0 || height == 0 {
            // Return a minimal valid image rather than panic; callers can detect
            // the degenerate case by checking image.size.
            scratch.rgba.clear();
            scratch.rgba.extend_from_slice(&[0, 0, 0, 255]);
            return [1, 1];
        }

        // Fused WL+colormap single pass: apply window/level per pixel, then
        // map the normalised result through the colormap directly, eliminating
        // the intermediate `wl_bytes` allocation.
        let Some(byte_count) = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(4))
        else {
            write_invalid_image(&mut scratch.rgba);
            return [1, 1];
        };
        scratch.rgba.resize(byte_count, 0);
        for (index, &p) in scratch.pixel_f32.iter().enumerate() {
            let byte = presentation.apply(wl, f64::from(p));
            let value = Normalized::from_u8(byte);
            let base = index * 4;
            scratch.rgba[base..base + 4].copy_from_slice(&colormap.sample(value).to_rgba8());
        }
        [width, height]
    }
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

fn write_invalid_image(rgba: &mut Vec<u8>) {
    rgba.clear();
    rgba.extend_from_slice(&[255, 0, 255, 255]);
}

#[cfg(test)]
#[path = "tests_slice_render.rs"]
mod tests;
