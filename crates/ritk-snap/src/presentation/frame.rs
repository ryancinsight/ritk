//! Validated RGBA frame produced by the RITK presentation boundary.

use crate::render::{NamedColorMap, RgbaImage, SliceRenderer, WindowLevel};
use crate::LoadedVolume;
use anyhow::{anyhow, bail, Result};
use metis_platform::framebuffer::MAX_PIXELS;

/// Validated row and column sample distances for a presentation frame.
///
/// Values use the physical unit carried by the source volume geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PresentationSpacing {
    row: f64,
    column: f64,
}

impl PresentationSpacing {
    pub(crate) fn try_new(row: f64, column: f64) -> Result<Self> {
        if row.is_finite() && row > 0.0 && column.is_finite() && column > 0.0 {
            Ok(Self { row, column })
        } else {
            bail!("presentation frame display spacing must be finite and positive")
        }
    }

    const fn unit() -> Self {
        Self {
            row: 1.0,
            column: 1.0,
        }
    }

    /// Returns row and column sample distances in display order.
    #[must_use]
    pub const fn values(self) -> [f64; 2] {
        [self.row, self.column]
    }

    #[cfg(windows)]
    pub(crate) const fn swapped(self) -> Self {
        Self {
            row: self.column,
            column: self.row,
        }
    }
}

/// A bounded, row-major RGBA frame with validated display geometry.
///
/// The frame is the only value that crosses from RITK's display pipeline to a
/// host renderer. Its dimensions, byte count and row/column sample distances
/// are validated at construction; DICOM identifiers, paths and volume storage
/// never enter the value.
#[derive(Clone, Debug, PartialEq)]
pub struct PresentationFrame {
    width: u32,
    height: u32,
    rgba: Box<[u8]>,
    display_spacing: PresentationSpacing,
}

impl PresentationFrame {
    /// Renders one validated volume slice into a host-neutral frame.
    ///
    /// `axis` follows [`SliceRenderer`] (0 axial, 1 coronal, 2 sagittal), and
    /// `index` is clamped by the existing RITK renderer. Window/level and
    /// colormap semantics therefore remain RITK-owned.
    ///
    /// # Errors
    /// Returns an error if the rendered image has invalid dimensions or pixel
    /// storage, or if the bounded RGBA allocation cannot be reserved.
    pub fn from_slice(
        volume: &LoadedVolume,
        axis: usize,
        index: usize,
        window_level: WindowLevel,
        colormap: NamedColorMap,
    ) -> Result<Self> {
        let display_spacing = slice_display_spacing(volume.spacing, axis)?;
        let image = SliceRenderer::render_rgba(volume, axis, index, window_level, colormap);
        Ok(Self::from_rgba_image(image)?.with_display_spacing(display_spacing))
    }

    /// Renders the three orthogonal volume slices for a multi-viewport host.
    ///
    /// The indices are ordered axial, coronal, sagittal. The returned frames
    /// retain the slice renderer's axis-specific dimensions and all display
    /// semantics remain in RITK before the host boundary is crossed.
    ///
    /// # Errors
    /// Returns an error if any rendered slice violates the bounded frame
    /// contract or cannot be allocated.
    #[cfg(any(target_arch = "wasm32", test))]
    pub(crate) fn from_orthogonal_slices(
        volume: &LoadedVolume,
        indices: [usize; 3],
        window_level: WindowLevel,
        colormap: NamedColorMap,
    ) -> Result<[Self; 3]> {
        let axial = Self::from_slice(volume, 0, indices[0], window_level, colormap)?;
        let coronal = Self::from_slice(volume, 1, indices[1], window_level, colormap)?;
        let sagittal = Self::from_slice(volume, 2, indices[2], window_level, colormap)?;
        Ok([axial, coronal, sagittal])
    }

    /// Copies validated row-major RGBA bytes into the presentation boundary.
    ///
    /// This is a presentation adapter only. It does not inspect or retain the
    /// source image's viewer metadata.
    ///
    /// # Errors
    /// Returns an error for zero dimensions, dimensions outside the host's
    /// signed-coordinate range, inconsistent pixel storage, or allocation
    /// failure.
    pub fn from_rgba(width: u32, height: u32, rgba: &[u8]) -> Result<Self> {
        let pixel_count = Self::validate_dimensions(width, height)?;
        let byte_count = pixel_count
            .checked_mul(4)
            .ok_or_else(|| anyhow!("presentation frame byte count overflows usize"))?;
        if rgba.len() != byte_count {
            bail!(
                "presentation frame byte count {} does not match {}x{} RGBA storage",
                rgba.len(),
                width,
                height
            );
        }
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(byte_count)
            .map_err(|_| anyhow!("unable to reserve presentation frame bytes"))?;
        owned.extend_from_slice(rgba);
        Self::from_rgba_storage(width, height, owned.into_boxed_slice())
    }

    fn from_rgba_image(image: RgbaImage) -> Result<Self> {
        let [width, height] = image.size();
        let width = u32::try_from(width).map_err(|_| anyhow!("frame width exceeds u32"))?;
        let height = u32::try_from(height).map_err(|_| anyhow!("frame height exceeds u32"))?;
        let (_, rgba) = image.into_parts();
        Self::from_rgba_storage(width, height, rgba)
    }

    pub(crate) fn from_rgba_storage(width: u32, height: u32, rgba: Box<[u8]>) -> Result<Self> {
        let pixel_count = Self::validate_dimensions(width, height)?;
        let byte_count = pixel_count
            .checked_mul(4)
            .ok_or_else(|| anyhow!("presentation frame byte count overflows usize"))?;
        if rgba.len() != byte_count {
            bail!(
                "presentation frame byte count {} does not match {}x{} RGBA storage",
                rgba.len(),
                width,
                height
            );
        }
        Ok(Self {
            width,
            height,
            rgba,
            display_spacing: PresentationSpacing::unit(),
        })
    }

    /// Replaces the physical row and column sample distances.
    ///
    /// The values are ordered for the rendered frame, so a quarter-turn
    /// transform must swap them with the transformed pixel dimensions.
    pub(crate) fn with_display_spacing(mut self, spacing: PresentationSpacing) -> Self {
        self.display_spacing = spacing;
        self
    }

    #[cfg(windows)]
    pub(crate) fn into_rgba_parts(self) -> (u32, u32, Box<[u8]>) {
        (self.width, self.height, self.rgba)
    }

    fn validate_dimensions(width: u32, height: u32) -> Result<usize> {
        if width == 0 || height == 0 {
            bail!("presentation frame dimensions must be nonzero");
        }
        if width > i32::MAX as u32 || height > i32::MAX as u32 {
            bail!("presentation frame dimensions exceed host coordinate limits");
        }
        let pixel_count = usize::try_from(u64::from(width) * u64::from(height))
            .map_err(|_| anyhow!("presentation frame pixel count exceeds usize"))?;
        if pixel_count > MAX_PIXELS {
            bail!(
                "presentation frame pixel count {} exceeds host limit {}",
                pixel_count,
                MAX_PIXELS
            );
        }
        Ok(pixel_count)
    }

    /// Horizontal pixel count.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Vertical pixel count.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Contiguous straight-alpha RGBA bytes in row-major order.
    #[must_use]
    pub fn rgba(&self) -> &[u8] {
        &self.rgba
    }

    /// Returns row and column sample distances for the rendered display frame.
    #[must_use]
    pub const fn display_spacing(&self) -> PresentationSpacing {
        self.display_spacing
    }
}

fn slice_display_spacing(spacing: [f64; 3], axis: usize) -> Result<PresentationSpacing> {
    let [dz, dy, dx] = spacing;
    let [row, column] = match axis {
        0 => [dy, dx],
        1 => [dz, dx],
        2 => [dz, dy],
        _ => return Err(anyhow!("presentation frame axis {axis} is outside 0..=2")),
    };
    PresentationSpacing::try_new(row, column)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrayvec::ArrayString;
    use std::sync::Arc;

    fn test_volume() -> LoadedVolume {
        LoadedVolume {
            data: Arc::new(vec![0.0, 255.0]),
            shape: [1, 1, 2],
            channels: 1,
            spacing: [1.0, 1.0, 1.0],
            origin: [0.0, 0.0, 0.0],
            direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            metadata: None,
            source: None,
            modality: Some(ArrayString::from("CT").expect("bounded modality")),
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

    #[test]
    fn frame_preserves_rendered_pixel_order_and_alpha() {
        let frame = PresentationFrame::from_rgba(2, 1, &[0, 0, 0, 0, 200, 150, 100, 255])
            .expect("valid frame");
        assert_eq!(frame.width(), 2);
        assert_eq!(frame.height(), 1);
        assert_eq!(frame.rgba(), &[0, 0, 0, 0, 200, 150, 100, 255]);
        assert_eq!(frame.display_spacing().values(), [1.0, 1.0]);
    }

    #[test]
    fn frame_rejects_inconsistent_rgba_storage() {
        let error = PresentationFrame::from_rgba(2, 1, &[255; 4]).expect_err("mismatched bytes");
        assert!(error.to_string().contains("byte count"));
    }

    #[test]
    fn frame_rejects_zero_dimensions() {
        let error = PresentationFrame::from_rgba(0, 1, &[]).expect_err("zero width");
        assert!(error.to_string().contains("dimensions must be nonzero"));
    }

    #[test]
    fn frame_rejects_host_oversized_storage_before_copying_pixels() {
        let width = u32::try_from(MAX_PIXELS + 1).expect("test width fits u32");
        let error = PresentationFrame::from_rgba(width, 1, &[]).expect_err("oversized frame");
        assert!(error.to_string().contains("exceeds host limit"));
    }

    #[test]
    fn frame_uses_ritk_slice_display_semantics() {
        let frame = PresentationFrame::from_slice(
            &test_volume(),
            0,
            0,
            WindowLevel::new(127.5, 255.0),
            NamedColorMap::Grayscale,
        )
        .expect("rendered slice frame");
        assert_eq!(frame.width(), 2);
        assert_eq!(frame.height(), 1);
        assert_eq!(frame.rgba(), &[0, 0, 0, 255, 255, 255, 255, 255]);
        assert_eq!(frame.display_spacing().values(), [1.0, 1.0]);
    }

    #[test]
    fn orthogonal_frames_preserve_axis_order_and_dimensions() {
        let frames = PresentationFrame::from_orthogonal_slices(
            &test_volume(),
            [0, 0, 0],
            WindowLevel::new(127.5, 255.0),
            NamedColorMap::Grayscale,
        )
        .expect("orthogonal frames");

        assert_eq!(frames[0].width(), 2);
        assert_eq!(frames[0].height(), 1);
        assert_eq!(frames[1].width(), 2);
        assert_eq!(frames[1].height(), 1);
        assert_eq!(frames[2].width(), 1);
        assert_eq!(frames[2].height(), 1);
        assert_eq!(frames[0].rgba(), frames[1].rgba());
        assert_eq!(frames[2].rgba(), &[0, 0, 0, 255]);
        assert_eq!(frames[0].display_spacing().values(), [1.0, 1.0]);
        assert_eq!(frames[1].display_spacing().values(), [1.0, 1.0]);
        assert_eq!(frames[2].display_spacing().values(), [1.0, 1.0]);
    }

    #[test]
    fn slice_frame_carries_axis_specific_spacing() {
        let volume = LoadedVolume {
            spacing: [2.0, 3.0, 5.0],
            ..test_volume()
        };
        let frames = [0_usize, 1, 2].map(|axis| {
            PresentationFrame::from_slice(
                &volume,
                axis,
                0,
                WindowLevel::new(127.5, 255.0),
                NamedColorMap::Grayscale,
            )
            .expect("rendered spacing frame")
        });
        assert_eq!(frames[0].display_spacing().values(), [3.0, 5.0]);
        assert_eq!(frames[1].display_spacing().values(), [2.0, 5.0]);
        assert_eq!(frames[2].display_spacing().values(), [2.0, 3.0]);
    }

    #[test]
    fn malformed_display_spacing_is_rejected() {
        for [row, column] in [
            [0.0, 1.0],
            [-1.0, 1.0],
            [f64::NAN, 1.0],
            [1.0, f64::INFINITY],
        ] {
            let error =
                PresentationSpacing::try_new(row, column).expect_err("invalid display spacing");
            assert!(error.to_string().contains("display spacing"));
        }
    }

    #[test]
    fn slice_rejects_invalid_volume_spacing() {
        let volume = LoadedVolume {
            spacing: [1.0, 0.0, 1.0],
            ..test_volume()
        };
        let error = PresentationFrame::from_slice(
            &volume,
            0,
            0,
            WindowLevel::new(127.5, 255.0),
            NamedColorMap::Grayscale,
        )
        .expect_err("invalid volume spacing");
        assert!(error.to_string().contains("display spacing"));
    }
}
