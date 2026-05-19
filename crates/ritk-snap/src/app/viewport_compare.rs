//! Secondary / fused-compare viewport renderer for [`SnapApp`].
//!
//! Handles the side-by-side or fused overlay comparison viewport that
//! displays a secondary volume against the primary, including fused-slice
//! rendering via [`render_fused_slice`] and standard fit-scale / zoom logic.

use super::state::SnapApp;
use crate::render::fusion::{render_fused_slice, FusedSliceParams};
use crate::render::slice_render::WindowLevel;
use crate::ui::apply_to_image_into;

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
        let secondary_idx =
            Self::map_slice_index_between_volumes(primary_idx, primary_total, secondary_total);

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

                    let primary_wc = self.viewer_state.window_center.unwrap_or(128.0) as f64;
                    let primary_ww =
                        self.viewer_state.window_width.unwrap_or(256.0).max(1.0) as f64;
                    let secondary_wc = self.secondary_window_center.unwrap_or(128.0) as f64;
                    let secondary_ww = self.secondary_window_width.unwrap_or(256.0).max(1.0) as f64;

                    let color_image = render_fused_slice(
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
                    );
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
        let [w, h] = tex.size();
        let tex_w = w as f32;
        let tex_h = h as f32;
        let avail = ui.available_size();
        let fit = if tex_w > 0.0 && tex_h > 0.0 {
            (avail.x / tex_w).min(avail.y / tex_h)
        } else {
            1.0
        };
        let size = egui::vec2(tex_w * fit * self.zoom, tex_h * fit * self.zoom);

        let response = ui.add(
            egui::Image::new(egui::load::SizedTexture::new(tex.id(), size))
                .sense(egui::Sense::hover()),
        );

        let painter = ui.painter_at(response.rect);
        painter.text(
            response.rect.min + egui::vec2(6.0, 6.0),
            egui::Align2::LEFT_TOP,
            if self.compare_fused_overlay {
                "Fused"
            } else {
                "Secondary"
            },
            egui::FontId::proportional(12.0),
            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 210),
        );
    }
}
