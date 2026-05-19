//! Per-axis viewport renderer for [`SnapApp`].
//
//! Builds or refreshes GPU textures, computes spacing-aware fit scales,
//! paints overlays (DICOM 4-corner, crosshair, annotations), and dispatches
//! pointer / wheel events to the active tool.
//
//! The secondary / fused-compare viewport lives in [`super::viewport_compare`].
use super::state::SnapApp;
use crate::render::slice_render::WindowLevel;
use crate::tools::kind::ToolKind;
use crate::ui::overlay::{OverlayContext, OverlayRenderer};
use crate::ui::{should_zoom_with_scroll, zoom_from_scroll};

impl SnapApp {
    /// Render one MPR viewport for the given `axis` into `ui`.
    ///
    /// # Responsibilities
    ///
    /// 1. Rebuild the texture for this axis if dirty or absent.
    /// 2. Compute fit scale: `min(avail_w / tex_w, avail_h / tex_h) × zoom`.
    /// 3. Display the image widget with click-and-drag sensing.
    /// 4. Draw the axis label and slice counter as overlay text.
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
        let tex_w = tex_w_usize as f32;
        let tex_h = tex_h_usize as f32;
        let available = ui.available_size();

        let (row_mm, col_mm) = if let Some(vol) = &self.loaded {
            let [dz, dy, dx] = vol.spacing.map(|s| s as f32);
            match axis {
                0 => (dy.max(1e-6), dx.max(1e-6)),
                1 => (dz.max(1e-6), dx.max(1e-6)),
                _ => (dz.max(1e-6), dy.max(1e-6)),
            }
        } else {
            (1.0, 1.0)
        };

        let phys_w = tex_w * col_mm;
        let phys_h = tex_h * row_mm;

        let use_uniform_fit = self.multi_planar || self.dual_plane || self.compare_side_by_side;

        let fit_scale = if use_uniform_fit {
            if tex_w > 0.0 && tex_h > 0.0 {
                (available.x / tex_w).min(available.y / tex_h)
            } else {
                1.0
            }
        } else if phys_w > 0.0 && phys_h > 0.0 {
            (available.x / phys_w).min(available.y / phys_h)
        } else {
            1.0
        };

        let (scale_x, scale_y) = if use_uniform_fit {
            let s = fit_scale * self.zoom;
            (s, s)
        } else {
            (
                fit_scale * self.zoom * col_mm,
                fit_scale * self.zoom * row_mm,
            )
        };

        let display_size = egui::vec2(tex_w * scale_x, tex_h * scale_y);
        let image_widget = egui::Image::new(egui::load::SizedTexture::new(tex_id, display_size))
            .sense(egui::Sense::click_and_drag());
        let response = ui.add(image_widget);

        // Track which axis is currently hovered for status/info display
        if response.hovered() || response.has_focus() || response.clicked() {
            self.status_axis = axis;
        }

        // ── 4–6. Overlay text, DICOM overlay, crosshair ────────────────────────
        // Painter::new clones the Arc<Context>; it does not hold a borrow on ui.
        let painter = ui.painter_at(response.rect);

        let axis_name = self.axis_label(axis);
        let label_color = egui::Color32::from_rgba_unmultiplied(255, 255, 255, 210);

        painter.text(
            response.rect.min + egui::vec2(6.0, 6.0),
            egui::Align2::LEFT_TOP,
            axis_name,
            egui::FontId::proportional(12.0),
            label_color,
        );

        let (slice_idx, total) = self.axis_slice_info(axis);
        painter.text(
            egui::pos2(response.rect.max.x - 6.0, response.rect.min.y + 6.0),
            egui::Align2::RIGHT_TOP,
            format!("{}/{}", slice_idx + 1, total),
            egui::FontId::proportional(11.0),
            label_color,
        );

        // DICOM 4-corner overlay.
        if self.show_overlay {
            if let Some(vol) = &self.loaded {
                let wc = self.viewer_state.window_center.unwrap_or(128.0) as f64;
                let ww = self.viewer_state.window_width.unwrap_or(256.0).max(1.0) as f64;
                let wl = WindowLevel::new(wc, ww);

                let cursor_value = self.current_cursor_value();

                OverlayRenderer::draw(
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
                OverlayRenderer::draw_orientation_labels(
                    &painter,
                    response.rect,
                    axis,
                    &vol.direction,
                );
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
        // `scale = fit_scale × zoom` where fit_scale = min(avail_w/tex_w, avail_h/tex_h).
        // The image widget occupies exactly response.rect (egui places it top-left).
        {
            // img_to_screen: image-pixel Pos2 { x: col, y: row } → screen Pos2
            let origin = response.rect.min;
            let img_to_screen =
                |p: egui::Pos2| egui::pos2(origin.x + p.x * scale_x, origin.y + p.y * scale_y);

            // Per-axis 2-D spacing: [row_mm_per_px, col_mm_per_px]
            //   axis 0 axial  → dy, dx
            //   axis 1 coronal → dz, dx
            //   axis 2 sagittal→ dz, dy
            let spacing_2d: [f32; 2] = if let Some(vol) = &self.loaded {
                let [dz, dy, dx] = vol.spacing.map(|s| s as f32);
                match axis {
                    0 => [dy, dx],
                    1 => [dz, dx],
                    _ => [dz, dy],
                }
            } else {
                [1.0, 1.0]
            };

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
