//! Per-axis viewport renderer for [`SnapApp`].
//
//! Builds or refreshes GPU textures, computes spacing-aware fit scales,
//! paints overlays (DICOM 4-corner, crosshair, annotations), and dispatches
//! pointer / wheel events to the active tool.
//
//! The secondary / fused-compare viewport lives in [`super::viewport_compare`].
use super::image_placement::ImagePlacement;
use super::state::SnapApp;
use crate::render::slice_render::WindowLevel;
use crate::tools::kind::ToolKind;
use crate::ui::overlay::{OverlayContext, OverlayRenderer};
use crate::viewer::{DEFAULT_WINDOW_CENTER, DEFAULT_WINDOW_WIDTH};
// ── Overlay label constants ──────────────────────────────────────────────────

/// Pixel inset from the viewport corner for overlay text labels.
pub(crate) const OVERLAY_LABEL_INSET: f32 = 6.0;

/// Font size for viewport overlay text labels (proportional points).
pub(crate) const OVERLAY_LABEL_FONT_SIZE: f32 = 12.0;

/// Colour for viewport overlay text labels (white @ 82 % opacity, premultiplied).
pub(crate) const OVERLAY_LABEL_COLOR: egui::Color32 =
    egui::Color32::from_rgba_premultiplied(210, 210, 210, 210);

use crate::ui::{should_zoom_with_scroll, zoom_from_scroll};

impl SnapApp {
    /// Render one MPR viewport for the given `axis` into `ui`.
    ///
    /// # Responsibilities
    ///
    /// 1. Rebuild the texture for this axis if dirty or absent.
    /// 2. Fit physical slice extents to available space, then apply zoom.
    /// 3. Display the image widget with click-and-drag sensing.
    /// 4. Draw compact axis and slice labels when the full overlay is disabled.
    /// 5. Draw the DICOM 4-corner overlay when `show_overlay` is set.
    /// 6. Draw crosshair lines when `show_crosshair` is set.
    /// 7. Handle wheel input: Ctrl/Cmd+wheel zooms, plain wheel steps slices.
    /// 8. Dispatch pointer events to the active tool handler.
    pub(crate) fn render_axis_viewport(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        axis: usize,
    ) {
        // ── 1. Rebuild texture if stale ────────────────────────────────────────
        let needs_rebuild = match axis {
            0 => self.texture_dirty || self.texture.is_none(),
            1 => self.coronal_dirty || self.coronal_tex.is_none(),
            _ => self.sagittal_dirty || self.sagittal_tex.is_none(),
        };

        if needs_rebuild && self.loaded.is_some() {
            self.rebuild_texture_for_axis(ctx, axis);
            match axis {
                0 => self.texture_dirty = false,
                1 => self.coronal_dirty = false,
                _ => self.sagittal_dirty = false,
            }
        }

        // ── 2. Extract texture ID and size (copy, releases borrow) ─────────────
        let tex_info: Option<(egui::TextureId, [usize; 2])> = match axis {
            0 => self.texture.as_ref().map(|t| (t.id(), t.size())),
            1 => self.coronal_tex.as_ref().map(|t| (t.id(), t.size())),
            _ => self.sagittal_tex.as_ref().map(|t| (t.id(), t.size())),
        };

        let (tex_id, [tex_w_usize, tex_h_usize]) = match tex_info {
            Some(info) => info,
            None => {
                ui.centered_and_justified(|ui| {
                    let label = self.axis_label(axis);
                    ui.label(format!("{label} — open a volume to begin"));
                });
                return;
            }
        };

        // ── 3. Compute spacing-aware fit and render image ─────────────────────
        let Some(volume) = self.loaded.as_ref() else {
            return;
        };
        let placement = match ImagePlacement::show(
            ui,
            egui::load::SizedTexture::new(
                tex_id,
                egui::vec2(tex_w_usize as f32, tex_h_usize as f32),
            ),
            volume,
            axis,
            self.view_transform,
            self.zoom,
            egui::Sense::click_and_drag(),
        ) {
            Ok(placement) => placement,
            Err(error) => {
                self.status_message = format!("Image placement failed: {error}");
                ui.label(&self.status_message);
                return;
            }
        };
        let scale_x = placement.texel_size.x;
        let scale_y = placement.texel_size.y;
        let response = placement.response;

        // Track which axis is currently hovered for status/info display
        if response.hovered() || response.has_focus() || response.clicked() {
            self.status_axis = axis;
        }

        // ── 4–6. Overlay text, DICOM overlay, crosshair ────────────────────────
        // Painter::new clones the Arc<Context>; it does not hold a borrow on ui.
        let painter = ui.painter_at(response.rect);

        let (slice_idx, total) = self.axis_slice_info(axis);
        if !self.show_overlay {
            OverlayRenderer::draw_text_anchored(
                &painter,
                response.rect,
                egui::Align2::LEFT_TOP,
                self.axis_label(axis),
                OVERLAY_LABEL_COLOR,
            );
            OverlayRenderer::draw_text_anchored(
                &painter,
                response.rect,
                egui::Align2::RIGHT_TOP,
                &format!("{}/{}", slice_idx + 1, total),
                OVERLAY_LABEL_COLOR,
            );
        }

        // DICOM 4-corner overlay.
        if self.show_overlay {
            if let Some(vol) = &self.loaded {
                let wc = self
                    .viewer_state
                    .window_center
                    .unwrap_or(DEFAULT_WINDOW_CENTER) as f64;
                let ww = self
                    .viewer_state
                    .window_width
                    .unwrap_or(DEFAULT_WINDOW_WIDTH)
                    .max(1.0) as f64;
                let wl = WindowLevel::new(wc, ww);

                let cursor_value = self.current_cursor_value();

                let details = OverlayRenderer::draw(
                    &painter,
                    response.rect,
                    vol,
                    OverlayContext {
                        axis,
                        slice_index: slice_idx,
                        wl,
                        zoom: self.zoom,
                        cursor_value,
                        pointer_intensity: self.pointer_intensity,
                        cursor_suv: self.current_cursor_suv(),
                        pointer_suv: self.pointer_suv,
                    },
                );
                if let Some(details) = details {
                    OverlayRenderer::show_details(ui, response.rect, &details);
                }
            }
        }

        if self.show_label_overlay {
            self.draw_label_overlay(&painter, response.rect, axis);
        }
        if self.show_rt_struct_overlay {
            self.draw_rt_struct_overlay(&painter, response.rect, axis, tex_h_usize, tex_w_usize);
        }
        if self.show_rt_dose_overlay {
            self.draw_rt_dose_overlay(&painter, response.rect, axis, slice_idx);
        }

        // Crosshair at the linked study-coordinate cursor.
        if self.show_crosshair {
            if let (Some(vol), Some(cursor)) = (&self.loaded, self.linked_cursor) {
                if let Some(crosshair) = cursor.viewport_crosshair(vol.shape, axis, response.rect) {
                    let color = egui::Color32::from_rgba_unmultiplied(255, 255, 0, 120);
                    painter.line_segment(
                        [
                            egui::pos2(response.rect.min.x, crosshair.y),
                            egui::pos2(response.rect.max.x, crosshair.y),
                        ],
                        egui::Stroke::new(1.0_f32, color),
                    );
                    painter.line_segment(
                        [
                            egui::pos2(crosshair.x, response.rect.min.y),
                            egui::pos2(crosshair.x, response.rect.max.y),
                        ],
                        egui::Stroke::new(1.0_f32, color),
                    );
                }
            }
        }

        // ── 7. Measurement annotations and live tool preview ───────────────────
        //
        // Mathematical mapping:
        //   screen_px = rect.min + img_px × scale
        //   img_px    = (screen_px − rect.min) / scale
        //
        // Each texture axis uses its physical sample distance times the shared
        // physical fit scale and zoom.
        // The image widget occupies exactly response.rect (egui places it top-left).
        {
            // img_to_screen: image-pixel Pos2 { x: col, y: row } → screen Pos2
            let origin = response.rect.min;
            let img_to_screen =
                |p: egui::Pos2| egui::pos2(origin.x + p.x * scale_x, origin.y + p.y * scale_y);

            let spacing_2d = placement.row_col_spacing;

            // Cursor in image-pixel coords for live preview labels.
            let cursor_img_opt = if scale_x > 0.0 && scale_y > 0.0 {
                response
                    .hover_pos()
                    .map(|s| egui::pos2((s.x - origin.x) / scale_x, (s.y - origin.y) / scale_y))
            } else {
                None
            };

            // Re-acquire the painter (drop(painter) has not been called yet —
            // this block replaces it; the original painter is consumed below).
            let meas_painter = ui.painter_at(response.rect);

            crate::ui::measurements::MeasurementLayer::draw_annotations(
                &meas_painter,
                &self.annotations,
                img_to_screen,
            );
            crate::ui::measurements::MeasurementLayer::draw_in_progress(
                &meas_painter,
                &self.tool_state,
                response.hover_pos(),
                cursor_img_opt,
                spacing_2d,
                img_to_screen,
            );
        } // painter is dropped here; no longer borrows ui.
        drop(painter);

        // ── 7. Wheel input: zoom or slice navigation ───────────────────────────
        let (scroll_y, ctrl_or_cmd) = ctx.input(|i| {
            (
                i.smooth_scroll_delta.y,
                i.modifiers.ctrl || i.modifiers.command,
            )
        });

        if response.hovered() && scroll_y != 0.0 {
            if should_zoom_with_scroll(ctrl_or_cmd) {
                self.zoom = zoom_from_scroll(self.zoom, scroll_y);
                self.status_message = format!("Zoom: {:.0}%", self.zoom * 100.0);
            } else {
                let step = if scroll_y > 0.0 { -1i32 } else { 1 };
                self.step_slice_for_axis(axis, step);
            }
        }

        // ── 8. Pointer events ──────────────────────────────────────────────────
        // Update pointer intensity whenever pointer is over the viewport
        if response.hovered() || response.dragged() || response.interact_pointer_pos().is_some() {
            self.update_pointer_intensity(axis, response.interact_pointer_pos(), response.rect);
        }

        if response.drag_started() {
            if self.active_tool == ToolKind::LabelPaint || self.active_tool == ToolKind::LabelErase
            {
                self.apply_label_at_pointer(axis, response.interact_pointer_pos(), response.rect);
            }

            // Map screen to image-pixel coordinates for tool event
            let img_pos = response.interact_pointer_pos().map(|s| {
                egui::pos2(
                    (s.x - response.rect.min.x) / scale_x,
                    (s.y - response.rect.min.y) / scale_y,
                )
            });
            self.on_drag_start(img_pos);
        }

        if response.dragged() {
            if self.active_tool == ToolKind::LabelPaint || self.active_tool == ToolKind::LabelErase
            {
                self.apply_label_at_pointer(axis, response.interact_pointer_pos(), response.rect);
            }

            let img_pos = response.interact_pointer_pos().map(|s| {
                egui::pos2(
                    (s.x - response.rect.min.x) / scale_x,
                    (s.y - response.rect.min.y) / scale_y,
                )
            });
            self.on_drag(img_pos);
        }

        if response.drag_stopped() {
            let img_pos = response.interact_pointer_pos().map(|s| {
                egui::pos2(
                    (s.x - response.rect.min.x) / scale_x,
                    (s.y - response.rect.min.y) / scale_y,
                )
            });
            self.on_drag_end(img_pos);
        }

        if response.clicked() {
            self.update_linked_cursor_from_pointer(
                axis,
                response.interact_pointer_pos(),
                response.rect,
            );

            if self.active_tool == ToolKind::LabelPaint || self.active_tool == ToolKind::LabelErase
            {
                self.apply_label_at_pointer(axis, response.interact_pointer_pos(), response.rect);
            }

            let img_pos = response.interact_pointer_pos().map(|s| {
                egui::pos2(
                    (s.x - response.rect.min.x) / scale_x,
                    (s.y - response.rect.min.y) / scale_y,
                )
            });
            self.on_click(img_pos);
        }
    }
}
