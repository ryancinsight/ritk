//! ViewportPanel struct — constructor and rendering.

use egui::{pos2, vec2, Color32, Pos2, Rect, Sense, Stroke, TextureOptions, Ui, Vec2};

use super::super::state::{
    draw_crosshair, img_to_screen, screen_to_img, screen_to_img_f32, slice_dims, ViewportPanel,
    ViewportRenderMode, ViewportState,
};
use crate::{
    render::{
        mip_vr::{render_mip_axial, render_vr_axial},
        slice_render::SliceRenderer,
    },
    tools::interaction::ToolState,
    ui::{
        measurements::MeasurementLayer,
        overlay::{OverlayContext, OverlayRenderer},
        zoom::fit_view_transform,
    },
    LoadedVolume,
};

impl<'a> ViewportPanel<'a> {
    /// Construct a viewport panel.
    pub fn new(
        id: egui::Id,
        volume: Option<&'a LoadedVolume>,
        state: &'a mut ViewportState,
        tool: crate::tools::kind::ToolKind,
    ) -> Self {
        Self {
            id,
            volume,
            state,
            active_tool: tool,
        }
    }

    /// Render the viewport into `ui`.
    ///
    /// # Returns
    /// `Some([d, r, c])` when the Crosshair tool was used to set a linked
    /// crosshair position in volume-voxel coordinates. `None` otherwise.
    ///
    /// # Steps
    /// 1. Allocate a fill-available rectangle and obtain a painter + response.
    /// 2. Fill background.
    /// 3. If a volume is loaded:
    ///    a. Ensure the slice texture is up-to-date.
    ///    b. Paint the texture scaled to fit (respecting zoom + pan).
    ///    c. Draw orientation labels when overlay is enabled.
    ///    d. Draw the 4-corner DICOM overlay when enabled.
    ///    e. Draw the crosshair when enabled.
    ///    f. Draw completed annotations and in-progress tool state.
    /// 4. Handle pointer events.
    /// 5. Draw a thin border around the viewport.
    pub fn show(&mut self, ui: &mut Ui, pointer_intensity: f32) -> Option<[usize; 3]> {
        let available = ui.available_rect_before_wrap();
        let (response, painter) = ui.allocate_painter(available.size(), Sense::click_and_drag());
        let rect = response.rect;

        // Background
        painter.rect_filled(rect, 0.0, Color32::BLACK);

        let mut crosshair_result: Option<[usize; 3]> = None;

        if let Some(volume) = self.volume {
            // ── ensure slice texture is current ───────────────────────────
            let render_key = match self.state.render_mode {
                ViewportRenderMode::Slice => (self.state.axis, self.state.slice_index),
                // Volume projections do not vary with slice index.
                ViewportRenderMode::Mip | ViewportRenderMode::Vr => (self.state.axis, 0),
            };
            let needs_render =
                self.state.texture_slice_key != Some(render_key) || self.state.texture.is_none();
            if needs_render {
                let img = match self.state.render_mode {
                    ViewportRenderMode::Slice => SliceRenderer::render(
                        volume,
                        self.state.axis,
                        self.state.slice_index,
                        self.state.wl,
                        self.state.colormap,
                    ),
                    ViewportRenderMode::Mip => {
                        render_mip_axial(volume, self.state.wl, self.state.colormap)
                    }
                    ViewportRenderMode::Vr => {
                        render_vr_axial(volume, self.state.wl, self.state.colormap, 0.06)
                    }
                };
                let handle = ui.ctx().load_texture(
                    format!("vp_{:?}", self.id),
                    img,
                    TextureOptions::default(),
                );
                self.state.texture = Some(handle);
                self.state.texture_slice_key = Some(render_key);
            }

            // ── derive image dimensions from slice ────────────────────────
            let (img_w, img_h) = slice_dims(volume, self.state.axis);
            let (offset, scale) = self.state.image_transform(rect, img_w, img_h);
            let map_img_to_screen = |p: Pos2| -> Pos2 { img_to_screen(p, offset, scale) };

            // ── paint texture ─────────────────────────────────────────────
            if let Some(texture) = &self.state.texture {
                let img_rect = Rect::from_min_size(
                    pos2(offset.x, offset.y),
                    vec2(img_w as f32 * scale, img_h as f32 * scale),
                );
                painter.image(
                    texture.id(),
                    img_rect,
                    Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                    Color32::WHITE,
                );
            }

            // ── orientation labels ────────────────────────────────────────
            if self.state.show_overlay {
                OverlayRenderer::draw_orientation_labels(
                    &painter,
                    rect,
                    self.state.axis,
                    &volume.direction,
                );
            }

            // ── DICOM 4-corner overlay ────────────────────────────────────
            if self.state.show_overlay {
                // Compute cursor HU value if the cursor is inside the viewport.
                let cursor_hu = response
                    .hover_pos()
                    .and_then(|cursor| screen_to_img(cursor, offset, scale, img_w, img_h))
                    .map(|(col, row)| volume.pixel_at(self.state.slice_index, row, col));
                OverlayRenderer::draw(
                    &painter,
                    rect,
                    volume,
                    OverlayContext {
                        axis: self.state.axis,
                        slice_index: self.state.slice_index,
                        wl: self.state.wl,
                        zoom: self.state.zoom,
                        cursor_value: cursor_hu,
                        pointer_intensity,
                        cursor_suv: None,
                        pointer_suv: None,
                    },
                );
            }

            // ── crosshair ─────────────────────────────────────────────────
            // (Also used for linked crosshair position display.)
            if self.state.show_crosshair {
                draw_crosshair(&painter, rect, Color32::from_rgb(0, 200, 255));
            }

            // ── measurements ─────────────────────────────────────────────
            MeasurementLayer::draw_annotations(
                &painter,
                &self.state.annotations,
                map_img_to_screen,
            );
            // Compute cursor in image coordinates for live measurement labels.
            let cursor_img_opt = response
                .hover_pos()
                .and_then(|s| screen_to_img_f32(s, offset, scale))
                .map(|(col, row)| egui::pos2(col, row));
            let sp = volume.spacing;
            let spacing_2d: [f32; 2] = match self.state.axis {
                0 => [sp[1] as f32, sp[2] as f32],
                1 => [sp[0] as f32, sp[2] as f32],
                _ => [sp[0] as f32, sp[1] as f32],
            };
            MeasurementLayer::draw_in_progress(
                &painter,
                &self.state.tool_state,
                response.hover_pos(),
                cursor_img_opt,
                spacing_2d,
                map_img_to_screen,
            );

            // ── pointer event handling ────────────────────────────────────
            crosshair_result =
                self.handle_pointer(&response, volume, rect, offset, scale, img_w, img_h);
        } else {
            // No volume loaded — draw placeholder text.
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Drop a DICOM folder or NIfTI file here\nor use File → Open",
                egui::FontId::proportional(14.0),
                Color32::from_rgb(120, 120, 120),
            );
        }

        // ── viewport border ───────────────────────────────────────────────
        painter.rect_stroke(
            rect,
            0.0,
            Stroke::new(1.0_f32, Color32::from_rgb(60, 60, 60)),
        );

        // ── context menu ──────────────────────────────────────────────────
        response.context_menu(|ui| {
            if ui.button("Clear annotations").clicked() {
                self.state.annotations.clear();
                self.state.tool_state = ToolState::Idle;
                ui.close_menu();
            }
            ui.separator();
            if ui.button("Reset zoom & pan").clicked() {
                let (pan_offset, zoom) = fit_view_transform();
                self.state.zoom = zoom;
                self.state.pan_offset = Vec2::new(pan_offset[0], pan_offset[1]);
                ui.close_menu();
            }
            ui.separator();
            let overlay_label = if self.state.show_overlay {
                "Hide overlay"
            } else {
                "Show overlay"
            };
            if ui.button(overlay_label).clicked() {
                self.state.show_overlay = !self.state.show_overlay;
                ui.close_menu();
            }
            let xhair_label = if self.state.show_crosshair {
                "Hide crosshair"
            } else {
                "Show crosshair"
            };
            if ui.button(xhair_label).clicked() {
                self.state.show_crosshair = !self.state.show_crosshair;
                ui.close_menu();
            }
        });

        crosshair_result
    }
}
