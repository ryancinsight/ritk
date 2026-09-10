//! Per-axis viewport renderer for [`SnapApp`].
//
//! Builds or refreshes GPU textures, computes spacing-aware fit scales,
//! paints overlays (DICOM 4-corner, crosshair, annotations), and dispatches
//! pointer / wheel events to the active tool.
//
//! The secondary / fused-compare viewport lives in [`super::viewport_compare`].
use super::image_placement::ImagePlacement;
use super::state::SnapApp;
use crate::render::WindowLevel;
use crate::tools::interaction::Annotation;
use crate::ui::overlay::{OverlayContext, OverlayRenderer};
use crate::viewer::{DEFAULT_WINDOW_CENTER, DEFAULT_WINDOW_WIDTH};
use crate::{
    app::action_adapter::ViewerViewport,
    presentation::{PointerButton, PresentationEvent, PresentationModifiers, ViewportPoint},
};
// ── Overlay label constants ──────────────────────────────────────────────────

/// Pixel inset from the viewport corner for overlay text labels.
pub(crate) const OVERLAY_LABEL_INSET: f32 = 6.0;

/// Font size for viewport overlay text labels (proportional points).
pub(crate) const OVERLAY_LABEL_FONT_SIZE: f32 = 12.0;

/// Colour for viewport overlay text labels (white @ 82 % opacity, premultiplied).
pub(crate) const OVERLAY_LABEL_COLOR: egui::Color32 =
    egui::Color32::from_rgba_premultiplied(210, 210, 210, 210);

use crate::ui::{axis_slice_dimensions, ViewTransform};

fn source_point_to_screen(
    point: egui::Pos2,
    origin: egui::Pos2,
    texel_size: egui::Vec2,
    transform: ViewTransform,
    source_size: [usize; 2],
) -> egui::Pos2 {
    let output = transform.source_to_output(point, source_size);
    egui::pos2(
        origin.x + output.x * texel_size.x,
        origin.y + output.y * texel_size.y,
    )
}

fn screen_point_to_source(
    point: egui::Pos2,
    origin: egui::Pos2,
    texel_size: egui::Vec2,
    transform: ViewTransform,
    source_size: [usize; 2],
    output_size: [usize; 2],
) -> egui::Pos2 {
    let output = egui::pos2(
        ((point.x - origin.x) / texel_size.x).clamp(0.0, output_size[0] as f32 * 0.999_999),
        ((point.y - origin.y) / texel_size.y).clamp(0.0, output_size[1] as f32 * 0.999_999),
    );
    transform.output_to_source(output, source_size)
}

fn client_point(point: egui::Pos2) -> Option<ViewportPoint> {
    if !point.x.is_finite() || !point.y.is_finite() {
        return None;
    }
    Some(ViewportPoint::new(f64::from(point.x), f64::from(point.y)))
}

fn map_input_modifiers(input: egui::Modifiers) -> PresentationModifiers {
    // `command` is egui's logical shortcut alias: it mirrors Ctrl on
    // Windows/Linux and Command on macOS. The RITK contract carries the
    // physical Meta state separately, so only `mac_cmd` maps to `meta`.
    PresentationModifiers::new(input.ctrl, input.shift, input.alt, input.mac_cmd)
}

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
    /// 7. Dispatch pointer and wheel events through the shared action adapter.
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
        let Some((source_width, source_height)) = axis_slice_dimensions(volume.shape, axis) else {
            return;
        };
        let source_size = [source_width, source_height];
        let view_transform = self.view_transform;
        let output_size = view_transform.output_size(source_size);
        debug_assert_eq!(output_size, [tex_w_usize, tex_h_usize]);
        let origin = response.rect.min;
        let source_to_screen = |point| {
            source_point_to_screen(
                point,
                origin,
                egui::vec2(scale_x, scale_y),
                view_transform,
                source_size,
            )
        };
        let screen_to_source = |point| {
            screen_point_to_source(
                point,
                origin,
                egui::vec2(scale_x, scale_y),
                view_transform,
                source_size,
                output_size,
            )
        };

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
                        view_transform,
                    },
                );
                if let Some(details) = details {
                    OverlayRenderer::show_details(ui, response.rect, &details);
                }
            }
        }

        if self.show_label_overlay {
            self.draw_label_overlay(&painter, response.rect, axis, view_transform);
        }
        if self.show_rt_struct_overlay {
            self.draw_rt_struct_overlay(&painter, response.rect, axis, view_transform);
        }
        if self.show_rt_dose_overlay {
            self.draw_rt_dose_overlay(&painter, response.rect, axis, slice_idx);
        }

        // Crosshair at the linked study-coordinate cursor.
        if self.show_crosshair {
            if let (Some(vol), Some(cursor)) = (&self.loaded, self.linked_cursor) {
                if let Some(crosshair) =
                    cursor.viewport_crosshair(vol.shape, axis, response.rect, view_transform)
                {
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
        // Source image points are mapped through the same discrete transform
        // used to build the texture. Measurements remain in source voxel
        // coordinates while their screen projection follows the displayed
        // orientation.
        //
        // Each texture axis uses its physical sample distance times the shared
        // physical fit scale and zoom.
        // The image widget occupies exactly response.rect (egui places it top-left).
        {
            let spacing_2d = self
                .slice_plane_spacing()
                .ok()
                .and_then(|spacing| Annotation::validate_spacing(spacing).ok());

            // Cursor in image-pixel coords for live preview labels.
            let cursor_img_opt = if scale_x > 0.0 && scale_y > 0.0 {
                response.hover_pos().map(screen_to_source)
            } else {
                None
            };

            // Re-acquire the painter (drop(painter) has not been called yet —
            // this block replaces it; the original painter is consumed below).
            let meas_painter = ui.painter_at(response.rect);

            crate::ui::measurements::MeasurementLayer::draw_annotations(
                &meas_painter,
                &self.annotations,
                source_to_screen,
            );
            crate::ui::measurements::MeasurementLayer::draw_in_progress(
                &meas_painter,
                &self.tool_state,
                response.hover_pos(),
                cursor_img_opt,
                spacing_2d,
                source_to_screen,
            );
        } // painter is dropped here; no longer borrows ui.
        drop(painter);

        // ── 7. Pointer and wheel events ────────────────────────────────────────
        // The egui response is translated into the same bounded presentation
        // event sequence used by native and browser hosts. RITK then reduces
        // and applies it through the action adapter, so the viewer state does
        // not depend on egui's response methods.
        let action_viewport = match ViewerViewport::new(
            axis,
            origin,
            egui::vec2(scale_x, scale_y),
            source_size,
            view_transform,
        ) {
            Ok(viewport) => viewport,
            Err(error) => {
                self.status_message = format!("Presentation viewport rejected: {error}");
                return;
            }
        };
        let pointer = response
            .interact_pointer_pos()
            .or_else(|| response.hover_pos());
        if let Some(pointer) = pointer.and_then(client_point) {
            let (scroll_delta, input_modifiers) =
                ctx.input(|input| (input.smooth_scroll_delta, input.modifiers));
            let mut events = arrayvec::ArrayVec::<PresentationEvent, 4>::new();
            if response.hovered() && (scroll_delta.x != 0.0 || scroll_delta.y != 0.0) {
                events.push(PresentationEvent::PointerWheel {
                    x: pointer.x(),
                    y: pointer.y(),
                    delta_x: f64::from(scroll_delta.x),
                    delta_y: f64::from(scroll_delta.y),
                    modifiers: map_input_modifiers(input_modifiers),
                });
            }
            if response.drag_started() {
                events.push(PresentationEvent::PointerDown {
                    x: pointer.x(),
                    y: pointer.y(),
                    button: PointerButton::Left,
                });
            }
            if response.dragged() {
                events.push(PresentationEvent::PointerMove {
                    x: pointer.x(),
                    y: pointer.y(),
                });
            }
            if response.drag_stopped() {
                events.push(PresentationEvent::PointerUp {
                    x: pointer.x(),
                    y: pointer.y(),
                    button: PointerButton::Left,
                });
            } else if response.clicked() && !response.drag_started() {
                events.push(PresentationEvent::PointerDown {
                    x: pointer.x(),
                    y: pointer.y(),
                    button: PointerButton::Left,
                });
                events.push(PresentationEvent::PointerUp {
                    x: pointer.x(),
                    y: pointer.y(),
                    button: PointerButton::Left,
                });
            } else if events.is_empty() && response.hovered() {
                events.push(PresentationEvent::PointerMove {
                    x: pointer.x(),
                    y: pointer.y(),
                });
            }
            if !events.is_empty() {
                if let Err(error) = self.apply_presentation_events(&events, Some(&action_viewport))
                {
                    self.status_message = format!("Presentation input rejected: {error}");
                    tracing::error!(error = %error, "RITK presentation event batch rejected");
                }
            }
        } else if response.drag_stopped() {
            // A host can end a drag after its final pointer coordinate has
            // disappeared. Clear the reducer and viewer gesture state so a
            // later press is not rejected as a duplicate pointer.
            self.cancel_presentation_gesture();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::map_input_modifiers;
    use crate::presentation::PresentationModifiers;

    #[test]
    fn maps_logical_shortcut_alias_without_physical_meta() {
        let input = egui::Modifiers {
            ctrl: true,
            command: true,
            ..egui::Modifiers::NONE
        };

        assert_eq!(
            map_input_modifiers(input),
            PresentationModifiers::new(true, false, false, false)
        );
    }

    #[test]
    fn preserves_physical_command_state() {
        let input = egui::Modifiers {
            mac_cmd: true,
            command: true,
            ..egui::Modifiers::NONE
        };

        assert_eq!(
            map_input_modifiers(input),
            PresentationModifiers::new(false, false, false, true)
        );
    }
}
