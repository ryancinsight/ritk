//! Validated RGBA frame produced by the RITK presentation boundary.

use crate::render::{NamedColorMap, SliceRenderer, WindowLevel};
use crate::LoadedVolume;
use anyhow::{anyhow, bail, Result};
use metis_platform::framebuffer::MAX_PIXELS;

/// A bounded, row-major RGBA frame with no format or viewer metadata.
///
/// The frame is the only value that crosses from RITK's display pipeline to a
/// host renderer. Its dimensions and byte count are validated at construction;
/// DICOM identifiers, paths and volume storage never enter the value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PresentationFrame {
    width: u32,
    height: u32,
    rgba: Box<[u8]>,
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
        let image = SliceRenderer::render(volume, axis, index, window_level, colormap);
        Self::from_color_image(&image)
    }

    /// Copies an egui image into the format-neutral RGBA representation.
    ///
    /// This is a presentation adapter only. It does not inspect or retain the
    /// source image's viewer metadata.
    ///
    /// # Errors
    /// Returns an error for zero dimensions, dimensions outside the host's
    /// signed-coordinate range, inconsistent pixel storage, or allocation
    /// failure.
    pub fn from_color_image(image: &egui::ColorImage) -> Result<Self> {
        let [width, height] = image.size;
        let width = u32::try_from(width).map_err(|_| anyhow!("frame width exceeds u32"))?;
        let height = u32::try_from(height).map_err(|_| anyhow!("frame height exceeds u32"))?;
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
        if image.pixels.len() != pixel_count {
            bail!(
                "presentation frame pixel count {} does not match {}x{}",
                image.pixels.len(),
                width,
                height
            );
        }
        let byte_count = pixel_count
            .checked_mul(4)
            .ok_or_else(|| anyhow!("presentation frame byte count overflows usize"))?;
        let mut rgba = Vec::new();
        rgba.try_reserve_exact(byte_count)
            .map_err(|_| anyhow!("unable to reserve presentation frame bytes"))?;
        for pixel in &image.pixels {
            rgba.extend_from_slice(&pixel.to_srgba_unmultiplied());
        }
        Ok(Self {
            width,
            height,
            rgba: rgba.into_boxed_slice(),
        })
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
        let image =
            egui::ColorImage::from_rgba_unmultiplied([2, 1], &[0, 0, 0, 0, 200, 150, 100, 255]);
        let frame = PresentationFrame::from_color_image(&image).expect("valid frame");
        assert_eq!(frame.width(), 2);
        assert_eq!(frame.height(), 1);
        assert_eq!(frame.rgba(), &[0, 0, 0, 0, 200, 150, 100, 255]);
    }

    #[test]
    fn frame_rejects_inconsistent_image_storage() {
        let image = egui::ColorImage {
            size: [2, 1],
            pixels: vec![egui::Color32::WHITE],
        };
        let error = PresentationFrame::from_color_image(&image).expect_err("mismatched pixels");
        assert!(error.to_string().contains("pixel count"));
    }

    #[test]
    fn frame_rejects_zero_dimensions() {
        let image = egui::ColorImage {
            size: [0, 1],
            pixels: Vec::new(),
        };
        let error = PresentationFrame::from_color_image(&image).expect_err("zero width");
        assert!(error.to_string().contains("dimensions must be nonzero"));
    }

    #[test]
    fn frame_rejects_host_oversized_storage_before_copying_pixels() {
        let image = egui::ColorImage {
            size: [MAX_PIXELS + 1, 1],
            pixels: Vec::new(),
        };
        let error = PresentationFrame::from_color_image(&image).expect_err("oversized frame");
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
    }
}
