//! Secondary / fused-compare viewport renderer for [`SnapApp`].
//!
//! Handles the side-by-side or fused overlay comparison viewport that
//! displays a secondary volume against the primary, including fused-slice
//! rendering via [`render_fused_slice`] and standard fit-scale / zoom logic.

use super::image_placement::ImagePlacement;
use super::state::SnapApp;
use super::viewport_render::{OVERLAY_LABEL_COLOR, OVERLAY_LABEL_FONT_SIZE, OVERLAY_LABEL_INSET};
use crate::render::fusion::{render_fused_slice, secondary_slice_for_primary, FusedSliceParams};
use crate::render::slice_render::WindowLevel;
use crate::ui::apply_to_image_into;
use crate::viewer::{DEFAULT_WINDOW_CENTER, DEFAULT_WINDOW_WIDTH};

impl SnapApp {
    /// Render the secondary (compare / fused-overlay) viewport.
    ///
    /// When `compare_fused_overlay` is active the two volumes are
    /// alpha-blended into a single texture; otherwise the secondary
    /// volume is displayed independently with its own window/level.
    pub(crate) fn render_secondary_compare_viewport(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        primary_axis: usize,
        secondary_axis: usize,
    ) {
        let Some(secondary) = self.loaded_secondary.as_ref() else {
            ui.centered_and_justified(|ui| {
                ui.label("Select a series for Secondary to compare.");
            });
            return;
        };

        let primary_total = self.axis_slice_info(primary_axis).1.max(1);
        let primary_idx = self.axis_slice_info(primary_axis).0;
        let secondary_total = Self::axis_extent_for_volume(secondary, secondary_axis).max(1);
        let secondary_idx = if self.compare_fused_overlay {
            let mapped = match (self.loaded.as_ref(), self.loaded_secondary.as_ref()) {
                (Some(primary), Some(secondary)) => secondary_slice_for_primary(
                    primary,
                    primary_axis,
                    primary_idx,
                    secondary,
                    secondary_axis,
                ),
                _ => return,
            };
            match mapped {
                Ok(index) => index,
                Err(error) => {
                    self.secondary_texture = None;
                    self.status_message = format!("Fused comparison unavailable: {error}");
                    ui.centered_and_justified(|ui| {
                        ui.label(&self.status_message);
                    });
                    return;
                }
            }
        } else {
            Self::map_slice_index_between_volumes(primary_idx, primary_total, secondary_total)
        };

        let needs_rebuild = if self.compare_fused_overlay {
            true
        } else {
            self.secondary_texture_dirty
                || self.secondary_texture.is_none()
                || self.secondary_texture_axis != secondary_axis
                || self.secondary_texture_slice != secondary_idx
        };

        if needs_rebuild {
            if self.compare_fused_overlay {
                let (color_image, tex_name) = {
                    let Some(primary) = self.loaded.as_ref() else {
                        return;
                    };

                    let primary_wc =
                        self.viewer_state
                            .window_center
                            .unwrap_or(DEFAULT_WINDOW_CENTER) as f64;
                    let primary_ww = self
                        .viewer_state
                        .window_width
                        .unwrap_or(DEFAULT_WINDOW_WIDTH)
                        .max(1.0) as f64;
                    let secondary_wc =
                        self.secondary_window_center
                            .unwrap_or(DEFAULT_WINDOW_CENTER) as f64;
                    let secondary_ww = self
                        .secondary_window_width
                        .unwrap_or(DEFAULT_WINDOW_WIDTH)
                        .max(1.0) as f64;

                    let color_image = match render_fused_slice(
                        FusedSliceParams {
                            volume: primary,
                            axis: primary_axis,
                            slice: primary_idx,
                            wl: WindowLevel::new(primary_wc, primary_ww),
                            colormap: self.colormap,
                        },
                        FusedSliceParams {
                            volume: secondary,
                            axis: secondary_axis,
                            slice: secondary_idx,
                            wl: WindowLevel::new(secondary_wc, secondary_ww),
                            colormap: self.secondary_colormap,
                        },
                        self.compare_fusion_alpha,
                    ) {
                        Ok(image) => image,
                        Err(error) => {
                            self.secondary_texture = None;
                            self.status_message = format!("Fused comparison unavailable: {error}");
                            ui.centered_and_justified(|ui| {
                                ui.label(&self.status_message);
                            });
                            return;
                        }
                    };
                    let color_image = apply_to_image_into(
                        &mut self.render_buffer_pool,
                        &color_image,
                        self.view_transform,
                    );
                    let tex_name = "slice_tex_fused";
                    (color_image, tex_name)
                };
                self.secondary_texture =
                    Some(ctx.load_texture(tex_name, color_image, egui::TextureOptions::LINEAR));
                self.secondary_texture_axis = secondary_axis;
                self.secondary_texture_slice = secondary_idx;
                self.secondary_texture_dirty = false;
            } else {
                self.rebuild_secondary_texture(ctx, secondary_axis, secondary_idx);
            }
        }

        let Some(tex) = self.secondary_texture.as_ref() else {
            return;
        };
        // Fusion emits the primary sampling grid; independent comparison emits
        // the selected secondary grid. Placement follows that output contract.
        let (volume, axis) = match (
            self.compare_fused_overlay,
            self.loaded.as_ref(),
            self.loaded_secondary.as_ref(),
        ) {
            (true, Some(primary), _) => (primary, primary_axis),
            (false, _, Some(secondary)) => (secondary, secondary_axis),
            _ => return,
        };
        let placement = match ImagePlacement::show(
            ui,
            egui::load::SizedTexture::from_handle(tex),
            volume,
            axis,
            self.view_transform,
            self.zoom,
            egui::Sense::hover(),
        ) {
            Ok(placement) => placement,
            Err(error) => {
                self.status_message = format!("Image placement failed: {error}");
                ui.label(&self.status_message);
                return;
            }
        };

        let response = placement.response;

        let painter = ui.painter_at(response.rect);
        painter.text(
            response.rect.min + egui::vec2(OVERLAY_LABEL_INSET, OVERLAY_LABEL_INSET),
            egui::Align2::LEFT_TOP,
            if self.compare_fused_overlay {
                "Fused"
            } else {
                "Secondary"
            },
            egui::FontId::proportional(OVERLAY_LABEL_FONT_SIZE),
            OVERLAY_LABEL_COLOR,
        );
    }
}
