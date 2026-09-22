//! Validated RGBA frame produced by the RITK presentation boundary.

use crate::render::{FrameRenderScratch, NamedColorMap, SliceRenderer, WindowLevel};
#[cfg(target_arch = "wasm32")]
use crate::tools::interaction::ViewportOffset;
#[cfg(windows)]
use crate::ui::{apply_to_rgba_into, ViewTransform};
use crate::LoadedVolume;
#[cfg(windows)]
use anyhow::Context;
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
    rgba: Vec<u8>,
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
        let mut scratch = FrameRenderScratch::default();
        let mut frame = Self::empty();
        frame.render_slice_into(volume, axis, index, window_level, colormap, &mut scratch)?;
        Ok(frame)
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
    #[cfg(test)]
    pub(crate) fn from_orthogonal_slices(
        volume: &LoadedVolume,
        indices: [usize; 3],
        window_level: WindowLevel,
        colormap: NamedColorMap,
    ) -> Result<[Self; 3]> {
        let mut scratch = FrameRenderScratch::default();
        let mut frames = [Self::empty(), Self::empty(), Self::empty()];
        for (axis, frame) in frames.iter_mut().enumerate() {
            frame.render_slice_into(
                volume,
                axis,
                indices[axis],
                window_level,
                colormap,
                &mut scratch,
            )?;
        }
        Ok(frames)
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
        Self::from_rgba_storage(width, height, owned)
    }

    pub(crate) fn from_rgba_storage(width: u32, height: u32, rgba: Vec<u8>) -> Result<Self> {
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

    pub(crate) fn empty() -> Self {
        Self {
            width: 1,
            height: 1,
            rgba: vec![0, 0, 0, 255],
            display_spacing: PresentationSpacing::unit(),
        }
    }

    pub(crate) fn render_slice_into(
        &mut self,
        volume: &LoadedVolume,
        axis: usize,
        index: usize,
        window_level: WindowLevel,
        colormap: NamedColorMap,
        scratch: &mut FrameRenderScratch,
    ) -> Result<()> {
        let display_spacing = slice_display_spacing(volume.spacing, axis)?;
        let [width, height] =
            SliceRenderer::render_rgba_into(scratch, volume, axis, index, window_level, colormap);
        let width = u32::try_from(width).map_err(|_| anyhow!("frame width exceeds u32"))?;
        let height = u32::try_from(height).map_err(|_| anyhow!("frame height exceeds u32"))?;
        self.replace_rgba_storage(width, height, display_spacing, &mut scratch.rgba)
    }

    /// Applies the native display orientation using reusable caller-owned
    /// storage and updates the frame's display geometry in place.
    #[cfg(windows)]
    pub(crate) fn apply_view_transform(
        &mut self,
        transform: ViewTransform,
        display_spacing: PresentationSpacing,
        scratch: &mut FrameRenderScratch,
    ) -> Result<()> {
        if transform.is_identity() {
            self.display_spacing = display_spacing;
            return Ok(());
        }
        let source_size = [
            usize::try_from(self.width).map_err(|_| anyhow!("frame width exceeds usize"))?,
            usize::try_from(self.height).map_err(|_| anyhow!("frame height exceeds usize"))?,
        ];
        let output_size = apply_to_rgba_into(source_size, &self.rgba, transform, &mut scratch.rgba)
            .context("apply native viewport orientation into reusable RGBA storage")?;
        let output_width =
            u32::try_from(output_size[0]).map_err(|_| anyhow!("frame width exceeds u32"))?;
        let output_height =
            u32::try_from(output_size[1]).map_err(|_| anyhow!("frame height exceeds u32"))?;
        self.replace_rgba_storage(
            output_width,
            output_height,
            display_spacing,
            &mut scratch.rgba,
        )
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn apply_zoom_pan(
        &mut self,
        zoom: f32,
        pan: ViewportOffset,
        storage: &mut Vec<u8>,
    ) -> Result<()> {
        if zoom == 1.0 && pan == ViewportOffset::new(0.0, 0.0) {
            return Ok(());
        }
        super::viewport::transform_rgba(
            [
                usize::try_from(self.width).map_err(|_| anyhow!("frame width exceeds usize"))?,
                usize::try_from(self.height).map_err(|_| anyhow!("frame height exceeds usize"))?,
            ],
            &self.rgba,
            zoom,
            pan,
            storage,
        )?;
        std::mem::swap(&mut self.rgba, storage);
        Ok(())
    }

    /// Replaces this frame's RGBA storage with a validated caller-owned
    /// buffer, preserving the previous allocation in `storage` for reuse.
    pub(crate) fn replace_rgba_storage(
        &mut self,
        width: u32,
        height: u32,
        display_spacing: PresentationSpacing,
        storage: &mut Vec<u8>,
    ) -> Result<()> {
        let pixel_count = Self::validate_dimensions(width, height)?;
        let byte_count = pixel_count
            .checked_mul(4)
            .ok_or_else(|| anyhow!("presentation frame byte count overflows usize"))?;
        if storage.len() != byte_count {
            bail!(
                "presentation frame byte count {} does not match {}x{} RGBA storage",
                storage.len(),
                width,
                height
            );
        }
        std::mem::swap(&mut self.rgba, storage);
        self.width = width;
        self.height = height;
        self.display_spacing = display_spacing;
        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn reclaim_storage(&mut self, scratch: &mut FrameRenderScratch) {
        std::mem::swap(&mut self.rgba, &mut scratch.rgba);
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
mod tests;
