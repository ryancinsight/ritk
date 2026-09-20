//! Browser-specific frame rendering for RITK presentation surfaces.

use super::SnapApp;
use crate::presentation::PresentationFrame;
use crate::render::{
    map_scalar_value, FrameRenderScratch, GrayscalePresentation, ProjectionStatistic,
    SlabProjection,
};
impl SnapApp {
    pub(super) fn render_browser_frame_into(
        &self,
        frame: &mut PresentationFrame,
        scratch: &mut FrameRenderScratch,
    ) -> anyhow::Result<()> {
        let Some(volume) = self.loaded.as_ref() else {
            return Ok(());
        };
        let window_level = self.browser_window_level();
        let slice_index = match self.axis {
            0 => self.viewer_state.slice_index,
            1 => self.coronal_slice,
            _ => self.sagittal_slice,
        };
        frame.render_slice_into(
            volume,
            self.axis,
            slice_index,
            window_level,
            self.colormap,
            scratch,
        )
    }

    pub(super) fn render_browser_frames_into(
        &self,
        frames: &mut [PresentationFrame],
        scratch: &mut FrameRenderScratch,
    ) -> anyhow::Result<()> {
        let Some(volume) = self.loaded.as_ref() else {
            return Ok(());
        };
        if frames.len() != 3 {
            return Err(anyhow::anyhow!(
                "browser orthogonal presentation requires three frames"
            ));
        }
        let indices = [0_usize, 1, 2].map(|axis| self.axis_slice_info(axis).0);
        for (axis, frame) in frames.iter_mut().enumerate() {
            frame.render_slice_into(
                volume,
                axis,
                indices[axis],
                self.browser_window_level(),
                self.colormap,
                scratch,
            )?;
        }
        Ok(())
    }

    pub(super) fn render_browser_projection_into(
        &self,
        frame: &mut PresentationFrame,
        statistic: ProjectionStatistic,
        projection_pixels: &mut Vec<f32>,
        scratch: &mut FrameRenderScratch,
    ) -> anyhow::Result<()> {
        let volume = self
            .loaded
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("browser projection has no loaded RITK volume"))?;
        if volume.channels != 1 {
            return Err(anyhow::anyhow!(
                "browser scalar projection requires one channel"
            ));
        }
        let depth = volume
            .shape
            .first()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("browser projection volume has no depth axis"))?;
        let center = depth
            .checked_sub(1)
            .ok_or_else(|| anyhow::anyhow!("browser projection volume has no depth samples"))?
            / 2;
        let request = SlabProjection::try_new(volume, 0, center, center)
            .map_err(|error| anyhow::anyhow!("validate browser slab projection: {error}"))?;
        let dimensions = request
            .compute_into(volume, statistic, projection_pixels)
            .map_err(|error| anyhow::anyhow!("compute browser slab projection: {error}"))?;
        let presentation = GrayscalePresentation::for_volume(volume)
            .map_err(|error| anyhow::anyhow!("validate browser grayscale metadata: {error}"))?;
        let byte_len = dimensions[0]
            .checked_mul(dimensions[1])
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| anyhow::anyhow!("browser projection byte count overflows usize"))?;
        scratch.rgba.resize(byte_len, 0);
        for (index, &value) in projection_pixels.iter().enumerate() {
            let rgba = map_scalar_value(
                value,
                presentation,
                self.browser_window_level(),
                self.colormap,
            );
            let offset = index.checked_mul(4).ok_or_else(|| {
                anyhow::anyhow!("browser projection pixel offset overflows usize")
            })?;
            scratch.rgba[offset..offset + 4].copy_from_slice(&rgba);
        }
        let width = u32::try_from(dimensions[0])
            .map_err(|_| anyhow::anyhow!("browser projection width exceeds u32"))?;
        let height = u32::try_from(dimensions[1])
            .map_err(|_| anyhow::anyhow!("browser projection height exceeds u32"))?;
        let [_, row_spacing, column_spacing] = volume.spacing;
        let spacing =
            crate::presentation::PresentationSpacing::try_new(row_spacing, column_spacing)
                .map_err(|error| anyhow::anyhow!("validate browser projection spacing: {error}"))?;
        frame.replace_rgba_storage(width, height, spacing, &mut scratch.rgba)
    }

    pub(super) fn browser_window_level(&self) -> crate::render::WindowLevel {
        let window_center = self
            .viewer_state
            .window_center
            .unwrap_or(crate::viewer::DEFAULT_WINDOW_CENTER);
        let window_width = self
            .viewer_state
            .window_width
            .unwrap_or(crate::viewer::DEFAULT_WINDOW_WIDTH)
            .max(1.0);
        crate::render::WindowLevel::new(f64::from(window_center), f64::from(window_width))
    }
}
