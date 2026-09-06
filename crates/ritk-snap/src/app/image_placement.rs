//! Physical sizing shared by primary and comparison slice viewports.

use crate::ui::{RotationSteps, ViewTransform};
use crate::LoadedVolume;
use egui::Vec2;

pub(super) struct ImagePlacement {
    pub(super) response: egui::Response,
    pub(super) texel_size: Vec2,
    pub(super) row_col_spacing: [f32; 2],
}

impl ImagePlacement {
    /// Fit, allocate, and paint one physical slice with a shared hit rectangle.
    ///
    /// Textures already contain the requested flip/rotation. Quarter turns swap
    /// their row/column sampling distances as well as their pixel dimensions;
    /// flips and half turns preserve both distances. Layout mode is immaterial.
    /// Positive finite loaded geometry is used without a minimum-spacing clamp.
    /// Returns an error when the final screen extents or aspect cannot be
    /// represented as positive finite egui coordinates, including after translation
    /// to the allocated screen position. Paint and texel mapping use that exact
    /// rectangle, so layout rounding cannot detach interaction from the image.
    pub(super) fn show(
        ui: &mut egui::Ui,
        texture: egui::load::SizedTexture,
        volume: &LoadedVolume,
        axis: usize,
        transform: ViewTransform,
        zoom: f32,
        sense: egui::Sense,
    ) -> anyhow::Result<Self> {
        let [dz, dy, dx] = volume.spacing;
        let [row_mm, col_mm] = match axis {
            0 => [dy, dx],
            1 => [dz, dx],
            _ => [dz, dy],
        };
        let [row_mm, col_mm] = match transform.rotation {
            RotationSteps::Ninety | RotationSteps::TwoSeventy => [col_mm, row_mm],
            RotationSteps::Zero | RotationSteps::OneEighty => [row_mm, col_mm],
        };
        anyhow::ensure!(
            row_mm.is_finite() && col_mm.is_finite() && row_mm > 0.0 && col_mm > 0.0,
            "slice sample distances must be positive and finite"
        );
        let pixels = texture.size;
        let available = ui.available_size();
        // A common distance unit cancels from the fit. Normalize before the
        // f64 geometry crosses into egui's f32 coordinate space, preserving
        // aspect even when all sample distances exceed f32's exponent range.
        let reference_mm = row_mm.max(col_mm);
        let relative_spacing = Vec2::new(
            (col_mm / reference_mm) as f32,
            (row_mm / reference_mm) as f32,
        );
        let physical = pixels * relative_spacing;
        let fit = (available.x / physical.x).min(available.y / physical.y);
        let texel_size = relative_spacing * (fit * zoom);
        let size = pixels * texel_size;
        let aspect = size.x / size.y;
        anyhow::ensure!(
            size.is_finite()
                && texel_size.is_finite()
                && size.min_elem() > 0.0
                && texel_size.min_elem() > 0.0
                && aspect.is_finite()
                && aspect > 0.0,
            "physical slice geometry cannot be represented in viewport coordinates"
        );
        let (id, allocated) = ui.allocate_space(size);
        let rect = ui.layout().align_size_within_rect(size, allocated);
        let texel_size = rect.size() / pixels;
        let aspect = rect.width() / rect.height();
        anyhow::ensure!(
            rect.is_finite()
                && rect.width() > 0.0
                && rect.height() > 0.0
                && texel_size.is_finite()
                && texel_size.min_elem() > 0.0
                && aspect.is_finite()
                && aspect > 0.0,
            "physical slice rectangle collapses at screen coordinates"
        );
        let response = ui.interact(rect, id, sense);
        if ui.is_rect_visible(rect) {
            egui::Image::new(texture).paint_at(ui, rect);
        }
        Ok(Self {
            response,
            texel_size,
            row_col_spacing: [row_mm as f32, col_mm as f32],
        })
    }
}
