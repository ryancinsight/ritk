//! Per-viewport MPR slice display widget.
//!
//! # Architecture
//!
//! Each visible viewport is owned by a [`ViewportState`] (serialisable,
//! frame-independent) plus an ephemeral [`ViewportPanel`] that borrows state
//! references for one `update()` call.
//!
//! ## Image-to-screen transform
//!
//! The transform is a uniform scale followed by a translation:
//!
//! ```text
//! screen = offset + scale × img_pos
//! ```
//!
//! where
//! - `img_pos` is in image-pixel coordinates `(col, row)`,
//! - `scale`   is derived from `zoom` and the fit-to-viewport base scale,
//! - `offset`  incorporates the fit centering and `pan_offset`.
//!
//! ## Slice index bounds
//!
//! | axis | dimension          | valid index range |
//! |------|--------------------|-------------------|
//! | 0    | `shape[0]` (depth) | `[0, depth−1]`    |
//! | 1    | `shape[1]` (rows)  | `[0, rows−1]`     |
//! | 2    | `shape[2]` (cols)  | `[0, cols−1]`     |
//!
//! Out-of-range indices are silently clamped by [`ViewportState::clamp_slice_index`].

use egui::{
    pos2, vec2, Color32, Id, Pos2, Rect, Response, Sense, Stroke, TextureHandle, TextureOptions,
    Ui, Vec2,
};

use crate::{
    render::{
        colormap::Colormap,
        slice_render::{SliceRenderer, WindowLevel},
    },
    tools::{
        interaction::{Annotation, RoiKind, ToolState},
        kind::ToolKind,
    },
    ui::{measurements::MeasurementLayer, overlay::OverlayRenderer},
    LoadedVolume,
};

// ── ViewportState ─────────────────────────────────────────────────────────────

/// Persistent per-viewport display state, independent of the egui frame.
///
/// One instance is stored per active viewport slot in [`crate::app::SnapApp`].
pub struct ViewportState {
    /// MPR axis: 0 = axial (fixed depth), 1 = coronal (fixed row),
    /// 2 = sagittal (fixed column).
    pub axis: usize,
    /// Currently displayed slice index along `axis`.
    pub slice_index: usize,
    /// Zoom factor relative to fit-to-viewport (1.0 = fit).
    pub zoom: f32,
    /// Pan offset in screen pixels, applied after fit centering.
    pub pan_offset: Vec2,
    /// Current window/level settings.
    pub wl: WindowLevel,
    /// Active colormap.
    pub colormap: Colormap,
    /// Completed measurement annotations for this viewport.
    pub annotations: Vec<Annotation>,
    /// In-progress tool state.
    pub tool_state: ToolState,
    /// Whether to draw the 4-corner DICOM text overlay.
    pub show_overlay: bool,
    /// Whether to draw the crosshair lines.
    pub show_crosshair: bool,
    /// Cached texture for the current slice (None when the slice has changed).
    pub texture: Option<TextureHandle>,
    /// Key identifying the currently cached texture `(axis, slice_index)`.
    /// `None` means the texture must be (re-)rendered.
    pub texture_slice_key: Option<(usize, usize)>,
}

impl ViewportState {
    /// Construct a viewport state for the given `axis` and initial WL.
    pub fn new(axis: usize, wl: WindowLevel) -> Self {
        Self {
            axis,
            slice_index: 0,
            zoom: 1.0,
            pan_offset: Vec2::ZERO,
            wl,
            colormap: Colormap::Grayscale,
            annotations: Vec::new(),
            tool_state: ToolState::Idle,
            show_overlay: true,
            show_crosshair: true,
            texture: None,
            texture_slice_key: None,
        }
    }

    /// Axial viewport (axis = 0).
    pub fn for_axial(wl: WindowLevel) -> Self {
        Self::new(0, wl)
    }

    /// Coronal viewport (axis = 1).
    pub fn for_coronal(wl: WindowLevel) -> Self {
        Self::new(1, wl)
    }

    /// Sagittal viewport (axis = 2).
    pub fn for_sagittal(wl: WindowLevel) -> Self {
        Self::new(2, wl)
    }

    /// 3-D / MIP viewport — uses axial data by convention; rendering
    /// may differ in future when a full MIP renderer is available.
    pub fn for_mip(wl: WindowLevel) -> Self {
        Self::new(0, wl)
    }

    /// Clamp `slice_index` to the valid range `[0, dim − 1]` for the
    /// current `axis` and `volume`.
    ///
    /// No-op when `volume.shape[axis] == 0` (degenerate volume).
    pub fn clamp_slice_index(&mut self, volume: &LoadedVolume) {
        let dim = volume.shape[self.axis.min(2)];
        if dim > 0 {
            self.slice_index = self.slice_index.min(dim - 1);
        }
    }

    /// Invalidate the cached texture so it is re-rendered on the next frame.
    pub fn invalidate_texture(&mut self) {
        self.texture = None;
        self.texture_slice_key = None;
    }

    /// Compute the image-to-screen transform for a viewport of `viewport_rect`
    /// displaying an image of `(img_w, img_h)` pixels.
    ///
    /// Returns `(offset, scale)` such that:
    /// ```text
    /// screen_pos = offset + scale × img_pos
    /// ```
    /// where `img_pos = Pos2 { x: col as f32, y: row as f32 }`.
    ///
    /// # Algorithm
    /// 1. Compute `base_scale` = min(vp_w / img_w, vp_h / img_h) (fit-to-viewport).
    /// 2. Apply zoom: `scale = base_scale × zoom`.
    /// 3. Centre the scaled image in the viewport.
    /// 4. Add `pan_offset`.
    pub fn image_transform(&self, viewport_rect: Rect, img_w: usize, img_h: usize) -> (Vec2, f32) {
        let vp_w = viewport_rect.width();
        let vp_h = viewport_rect.height();

        let base_scale = if img_w == 0 || img_h == 0 || vp_w == 0.0 || vp_h == 0.0 {
            1.0_f32
        } else {
            (vp_w / img_w as f32).min(vp_h / img_h as f32)
        };

        let scale = base_scale * self.zoom;
        let img_screen_w = img_w as f32 * scale;
        let img_screen_h = img_h as f32 * scale;

        // Centre offset: shift so the image occupies the middle of the viewport.
        let center_x = viewport_rect.min.x + (vp_w - img_screen_w) * 0.5;
        let center_y = viewport_rect.min.y + (vp_h - img_screen_h) * 0.5;

        let offset = Vec2::new(center_x, center_y) + self.pan_offset;
        (offset, scale)
    }
}

// ── ViewportPanel ─────────────────────────────────────────────────────────────

/// Ephemeral per-frame viewport widget.
///
/// Borrows mutable state for one `update()` call then is discarded.
pub struct ViewportPanel<'a> {
    /// Unique widget id (used for egui interaction tracking).
    pub id: Id,
    /// The volume to display, or `None` when no volume is loaded.
    pub volume: Option<&'a LoadedVolume>,
    /// Mutable viewport state.
    pub state: &'a mut ViewportState,
    /// Currently active tool, used to route pointer events.
    pub active_tool: ToolKind,
}

impl<'a> ViewportPanel<'a> {
    /// Construct a viewport panel.
    pub fn new(
        id: Id,
        volume: Option<&'a LoadedVolume>,
        state: &'a mut ViewportState,
        tool: ToolKind,
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
    /// crosshair position in volume-voxel coordinates.  `None` otherwise.
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
    pub fn show(&mut self, ui: &mut Ui) -> Option<[usize; 3]> {
        let available = ui.available_rect_before_wrap();
        let (response, painter) = ui.allocate_painter(available.size(), Sense::click_and_drag());
        let rect = response.rect;

        // Background
        painter.rect_filled(rect, 0.0, Color32::BLACK);

        let mut crosshair_result: Option<[usize; 3]> = None;

        if let Some(volume) = self.volume {
            // ── ensure slice texture is current ───────────────────────────
            let needs_render = self.state.texture_slice_key
                != Some((self.state.axis, self.state.slice_index))
                || self.state.texture.is_none();

            if needs_render {
                let img = SliceRenderer::render(
                    volume,
                    self.state.axis,
                    self.state.slice_index,
                    self.state.wl,
                    self.state.colormap,
                );
                let handle = ui.ctx().load_texture(
                    format!("vp_{:?}", self.id),
                    img,
                    TextureOptions::default(),
                );
                self.state.texture = Some(handle);
                self.state.texture_slice_key = Some((self.state.axis, self.state.slice_index));
            }

            // ── derive image dimensions from slice ────────────────────────
            let (img_w, img_h) = slice_dims(volume, self.state.axis);
            let (offset, scale) = self.state.image_transform(rect, img_w, img_h);

            let img_to_screen =
                |p: Pos2| -> Pos2 { pos2(offset.x + p.x * scale, offset.y + p.y * scale) };

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
                    self.state.axis,
                    self.state.slice_index,
                    self.state.wl,
                    self.state.zoom,
                    cursor_hu,
                );
            }

            // ── crosshair ─────────────────────────────────────────────────
            // (Also used for linked crosshair position display.)
            if self.state.show_crosshair {
                draw_crosshair(&painter, rect, Color32::from_rgb(0, 200, 255));
            }

            // ── measurements ─────────────────────────────────────────────
            MeasurementLayer::draw_annotations(&painter, &self.state.annotations, &img_to_screen);
            MeasurementLayer::draw_in_progress(
                &painter,
                &self.state.tool_state,
                response.hover_pos(),
                &img_to_screen,
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
        painter.rect_stroke(rect, 0.0, Stroke::new(1.0, Color32::from_rgb(60, 60, 60)));

        // ── context menu ──────────────────────────────────────────────────
        response.context_menu(|ui| {
            if ui.button("Clear annotations").clicked() {
                self.state.annotations.clear();
                self.state.tool_state = ToolState::Idle;
                ui.close_menu();
            }
            ui.separator();
            if ui.button("Reset zoom & pan").clicked() {
                self.state.zoom = 1.0;
                self.state.pan_offset = Vec2::ZERO;
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

    // ── private: event handling ───────────────────────────────────────────────

    /// Route pointer events to the active tool and return a crosshair update
    /// when the Crosshair tool is used.
    fn handle_pointer(
        &mut self,
        response: &Response,
        volume: &LoadedVolume,
        rect: Rect,
        offset: Vec2,
        scale: f32,
        img_w: usize,
        img_h: usize,
    ) -> Option<[usize; 3]> {
        let _ = rect; // used by context menu, acknowledged
        let mut crosshair_result = None;

        // ── scroll wheel: change slice (or zoom with Ctrl) ────────────────
        if response.hovered() {
            let scroll = response.ctx.input(|i| i.smooth_scroll_delta);
            let ctrl = response.ctx.input(|i| i.modifiers.ctrl);
            if scroll.y != 0.0 {
                if ctrl {
                    // Zoom: each notch changes zoom by ±10 %.
                    let factor = if scroll.y > 0.0 { 1.1_f32 } else { 1.0 / 1.1 };
                    self.state.zoom = (self.state.zoom * factor).clamp(0.05, 32.0);
                    self.state.invalidate_texture();
                } else {
                    // Scroll through slices.
                    let dim = volume.shape[self.state.axis.min(2)];
                    if dim > 0 {
                        let delta = if scroll.y > 0.0 { 1_isize } else { -1 };
                        self.state.slice_index = (self.state.slice_index as isize + delta)
                            .clamp(0, dim as isize - 1)
                            as usize;
                    }
                }
            }
        }

        // ── drag: tool-dependent ──────────────────────────────────────────
        if response.drag_started_by(egui::PointerButton::Primary) {
            match self.active_tool {
                ToolKind::Pan => {
                    if let Some(pos) = response.interact_pointer_pos() {
                        self.state.tool_state = ToolState::Panning {
                            start: pos,
                            viewport_origin: egui::pos2(
                                self.state.pan_offset.x,
                                self.state.pan_offset.y,
                            ),
                        };
                    }
                }
                ToolKind::WindowLevel => {
                    if let Some(pos) = response.interact_pointer_pos() {
                        self.state.tool_state = ToolState::WindowLevelDrag {
                            start: pos,
                            original_center: self.state.wl.center,
                            original_width: self.state.wl.width,
                        };
                    }
                }
                ToolKind::MeasureLength => {
                    // First click of a two-click length measurement.
                    if let Some(cursor) = response.interact_pointer_pos() {
                        if let Some((col, row)) = screen_to_img_f32(cursor, offset, scale) {
                            match &self.state.tool_state {
                                ToolState::Idle => {
                                    self.state.tool_state = ToolState::MeasureLength1 {
                                        p1: Pos2::new(col, row),
                                    };
                                }
                                ToolState::MeasureLength1 { p1 } => {
                                    let p1 = *p1;
                                    let p2 = Pos2::new(col, row);
                                    // Convert Pos2{x=col, y=row} → [row, col]
                                    let p1_arr = [p1.y, p1.x];
                                    let p2_arr = [p2.y, p2.x];
                                    let sp = volume.spacing;
                                    let spacing_2d = match self.state.axis {
                                        0 => [sp[1] as f32, sp[2] as f32],
                                        1 => [sp[0] as f32, sp[2] as f32],
                                        _ => [sp[0] as f32, sp[1] as f32],
                                    };
                                    let length_mm =
                                        Annotation::compute_length(p1_arr, p2_arr, spacing_2d);
                                    self.state.annotations.push(Annotation::Length {
                                        p1: p1_arr,
                                        p2: p2_arr,
                                        length_mm,
                                    });
                                    self.state.tool_state = ToolState::Idle;
                                }
                                _ => {
                                    self.state.tool_state = ToolState::Idle;
                                }
                            }
                        }
                    }
                }
                ToolKind::MeasureAngle => {
                    if let Some(cursor) = response.interact_pointer_pos() {
                        if let Some((col, row)) = screen_to_img_f32(cursor, offset, scale) {
                            let pt = Pos2::new(col, row);
                            match &self.state.tool_state {
                                ToolState::Idle => {
                                    self.state.tool_state = ToolState::MeasureLength1 { p1: pt };
                                }
                                ToolState::MeasureLength1 { p1 } => {
                                    let p1 = *p1;
                                    self.state.tool_state = ToolState::MeasureAngle2 { p1, p2: pt };
                                }
                                ToolState::MeasureAngle2 { p1, p2 } => {
                                    let p1 = *p1;
                                    let p2 = *p2;
                                    // Convert Pos2{x=col, y=row} → [row, col]
                                    let p1_arr = [p1.y, p1.x];
                                    let p2_arr = [p2.y, p2.x];
                                    let p3_arr = [pt.y, pt.x];
                                    let angle = Annotation::compute_angle(p1_arr, p2_arr, p3_arr);
                                    self.state.annotations.push(Annotation::Angle {
                                        p1: p1_arr,
                                        p2: p2_arr,
                                        p3: p3_arr,
                                        angle_deg: angle,
                                    });
                                    self.state.tool_state = ToolState::Idle;
                                }
                                _ => {
                                    self.state.tool_state = ToolState::Idle;
                                }
                            }
                        }
                    }
                }
                ToolKind::RoiRect | ToolKind::RoiEllipse => {
                    if let Some(cursor) = response.interact_pointer_pos() {
                        if let Some((col, row)) = screen_to_img_f32(cursor, offset, scale) {
                            let kind = if self.active_tool == ToolKind::RoiRect {
                                RoiKind::Rect
                            } else {
                                RoiKind::Ellipse
                            };
                            self.state.tool_state = ToolState::RoiDrag {
                                start: Pos2::new(col, row),
                                current: Pos2::new(col, row),
                                kind,
                            };
                        }
                    }
                }
                ToolKind::PointHu => {
                    if let Some(cursor) = response.interact_pointer_pos() {
                        if let Some((col, row)) = screen_to_img_f32(cursor, offset, scale) {
                            let col_i = col.round() as usize;
                            let row_i = row.round() as usize;
                            let value = match self.state.axis {
                                0 => volume.pixel_at(self.state.slice_index, row_i, col_i),
                                1 => volume.pixel_at(row_i, self.state.slice_index, col_i),
                                _ => volume.pixel_at(row_i, col_i, self.state.slice_index),
                            };
                            self.state.annotations.push(Annotation::HuPoint {
                                pos: [row, col],
                                value,
                            });
                        }
                    }
                }
                _ => {}
            }
        }

        // ── drag update ───────────────────────────────────────────────────
        if response.dragged_by(egui::PointerButton::Primary) {
            match self.state.tool_state.clone() {
                ToolState::Panning {
                    start,
                    viewport_origin,
                } => {
                    if let Some(current_pos) = response.interact_pointer_pos() {
                        self.state.pan_offset = (viewport_origin + (current_pos - start)).to_vec2();
                    }
                }
                ToolState::WindowLevelDrag {
                    start,
                    original_center,
                    original_width,
                } => {
                    if let Some(current_pos) = response.interact_pointer_pos() {
                        let delta = current_pos - start;
                        // Horizontal drag → window width; vertical drag → centre.
                        // 2 HU per pixel is a clinically comfortable sensitivity.
                        self.state.wl.width = (original_width + delta.x as f64 * 2.0).max(1.0);
                        self.state.wl.center = original_center - delta.y as f64 * 2.0;
                        self.state.invalidate_texture();
                    }
                }
                ToolState::RoiDrag { start, kind, .. } => {
                    if let Some(cursor) = response.interact_pointer_pos() {
                        if let Some((col, row)) = screen_to_img_f32(cursor, offset, scale) {
                            self.state.tool_state = ToolState::RoiDrag {
                                start,
                                current: Pos2::new(col, row),
                                kind,
                            };
                        }
                    }
                }
                _ => {}
            }
        }

        // ── drag released ─────────────────────────────────────────────────
        if response.drag_stopped_by(egui::PointerButton::Primary) {
            match &self.state.tool_state.clone() {
                ToolState::Panning { .. } => {
                    self.state.tool_state = ToolState::Idle;
                }
                ToolState::WindowLevelDrag { .. } => {
                    self.state.tool_state = ToolState::Idle;
                }
                ToolState::RoiDrag {
                    start,
                    current,
                    kind: RoiKind::Rect,
                } => {
                    let tl = Pos2::new(start.x.min(current.x), start.y.min(current.y));
                    let br = Pos2::new(start.x.max(current.x), start.y.max(current.y));
                    if (br.x - tl.x) > 0.5 && (br.y - tl.y) > 0.5 {
                        // Convert Pos2{x=col,y=row} → [row, col]
                        let tl_arr = [tl.y, tl.x];
                        let br_arr = [br.y, br.x];
                        let sp = volume.spacing;
                        let spacing_2d = match self.state.axis {
                            0 => [sp[1] as f32, sp[2] as f32],
                            1 => [sp[0] as f32, sp[2] as f32],
                            _ => [sp[0] as f32, sp[1] as f32],
                        };
                        let (pixels, pix_w, pix_h) =
                            volume.extract_slice(self.state.axis, self.state.slice_index);
                        let area_mm2 = (br.x - tl.x).abs()
                            * spacing_2d[1]
                            * (br.y - tl.y).abs()
                            * spacing_2d[0];
                        let stats = Annotation::compute_roi_rect_stats(
                            tl_arr, br_arr, &pixels, pix_w, pix_h, spacing_2d,
                        );
                        self.state.annotations.push(Annotation::RoiRect {
                            top_left: tl_arr,
                            bottom_right: br_arr,
                            mean: stats.0,
                            std_dev: stats.1,
                            min: stats.2,
                            max: stats.3,
                            area_mm2,
                        });
                    }
                    self.state.tool_state = ToolState::Idle;
                }
                ToolState::RoiDrag {
                    start,
                    current,
                    kind: RoiKind::Ellipse,
                } => {
                    // For ellipse ROI, store as RoiRect (bounding box) for now.
                    let tl = Pos2::new(start.x.min(current.x), start.y.min(current.y));
                    let br = Pos2::new(start.x.max(current.x), start.y.max(current.y));
                    if (br.x - tl.x) > 0.5 && (br.y - tl.y) > 0.5 {
                        let tl_arr = [tl.y, tl.x];
                        let br_arr = [br.y, br.x];
                        let sp = volume.spacing;
                        let spacing_2d = match self.state.axis {
                            0 => [sp[1] as f32, sp[2] as f32],
                            1 => [sp[0] as f32, sp[2] as f32],
                            _ => [sp[0] as f32, sp[1] as f32],
                        };
                        let (pixels, pix_w, pix_h) =
                            volume.extract_slice(self.state.axis, self.state.slice_index);
                        let rx = (br.x - tl.x).abs() * 0.5 * spacing_2d[1];
                        let ry = (br.y - tl.y).abs() * 0.5 * spacing_2d[0];
                        let area_mm2 = std::f32::consts::PI * rx * ry;
                        let stats = Annotation::compute_roi_rect_stats(
                            tl_arr, br_arr, &pixels, pix_w, pix_h, spacing_2d,
                        );
                        self.state.annotations.push(Annotation::RoiRect {
                            top_left: tl_arr,
                            bottom_right: br_arr,
                            mean: stats.0,
                            std_dev: stats.1,
                            min: stats.2,
                            max: stats.3,
                            area_mm2,
                        });
                    }
                    self.state.tool_state = ToolState::Idle;
                }
                _ => {}
            }
        }

        // ── Zoom tool: scroll-wheel zooms ─────────────────────────────────
        if self.active_tool == ToolKind::Zoom && response.hovered() {
            // Right-drag for continuous zoom.
            if response.dragged_by(egui::PointerButton::Secondary) {
                let delta = response.drag_delta();
                let factor = 1.0 + delta.y * 0.005;
                self.state.zoom = (self.state.zoom * factor).clamp(0.05, 32.0);
                self.state.invalidate_texture();
            }
        }

        // ── Crosshair tool: click to place linked position ────────────────
        if self.active_tool == ToolKind::Crosshair && response.clicked() {
            if let Some(cursor) = response.interact_pointer_pos() {
                if let Some((col, row)) = screen_to_img(cursor, offset, scale, img_w, img_h) {
                    crosshair_result = Some(img_to_volume(
                        col,
                        row,
                        self.state.slice_index,
                        self.state.axis,
                    ));
                }
            }
        }

        crosshair_result
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Return the `(width, height)` of the 2-D slice image for the given axis.
///
/// Matches the convention in [`LoadedVolume::extract_slice`]:
/// - axis 0 (axial):   `(cols, rows)`
/// - axis 1 (coronal): `(cols, depth)`
/// - axis 2 (sagittal):`(rows, depth)`
fn slice_dims(volume: &LoadedVolume, axis: usize) -> (usize, usize) {
    let [depth, rows, cols] = volume.shape;
    match axis {
        0 => (cols, rows),
        1 => (cols, depth),
        _ => (rows, depth),
    }
}

/// Convert a screen position to integer image coordinates `(col, row)`.
///
/// Returns `None` when the position is outside the image bounds.
fn screen_to_img(
    screen: Pos2,
    offset: Vec2,
    scale: f32,
    img_w: usize,
    img_h: usize,
) -> Option<(usize, usize)> {
    if scale <= 0.0 {
        return None;
    }
    let col_f = (screen.x - offset.x) / scale;
    let row_f = (screen.y - offset.y) / scale;
    if col_f < 0.0 || row_f < 0.0 {
        return None;
    }
    let col = col_f as usize;
    let row = row_f as usize;
    if col >= img_w || row >= img_h {
        return None;
    }
    Some((col, row))
}

/// Convert a screen position to floating-point image coordinates `(col, row)`.
///
/// No bounds check; returns `None` only when `scale <= 0`.
fn screen_to_img_f32(screen: Pos2, offset: Vec2, scale: f32) -> Option<(f32, f32)> {
    if scale <= 0.0 {
        return None;
    }
    Some(((screen.x - offset.x) / scale, (screen.y - offset.y) / scale))
}

/// Map image-pixel coordinates `(col, row)` on a given `axis` slice to
/// volume-voxel coordinates `[depth, row, col]`.
///
/// | axis | slice   | img_col  | img_row  | volume coords              |
/// |------|---------|----------|----------|----------------------------|
/// | 0    | fixed d | col      | row      | `[slice, img_row, img_col]` |
/// | 1    | fixed r | col      | depth    | `[img_row, slice, img_col]` |
/// | 2    | fixed c | row      | depth    | `[img_row, img_col, slice]` |
fn img_to_volume(img_col: usize, img_row: usize, slice: usize, axis: usize) -> [usize; 3] {
    match axis {
        0 => [slice, img_row, img_col],
        1 => [img_row, slice, img_col],
        _ => [img_row, img_col, slice],
    }
}

/// Draw a thin crosshair (full-length horizontal + vertical lines) through
/// the centre of `rect`.
fn draw_crosshair(painter: &egui::Painter, rect: Rect, color: Color32) {
    let cx = rect.center().x;
    let cy = rect.center().y;
    let stroke = Stroke::new(0.75, color);
    // Horizontal line
    painter.line_segment([pos2(rect.min.x, cy), pos2(rect.max.x, cy)], stroke);
    // Vertical line
    painter.line_segment([pos2(cx, rect.min.y), pos2(cx, rect.max.y)], stroke);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_volume(depth: usize, rows: usize, cols: usize) -> LoadedVolume {
        let n = depth * rows * cols;
        LoadedVolume {
            data: std::sync::Arc::new((0..n).map(|i| i as f32).collect()),
            shape: [depth, rows, cols],
            spacing: [1.0, 1.0, 1.0],
            origin: [0.0, 0.0, 0.0],
            direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            metadata: None,
            source: None,
            modality: None,
            patient_name: None,
            patient_id: None,
            study_date: None,
            series_description: None,
        }
    }

    // ── image_transform ───────────────────────────────────────────────────────

    /// For zoom=1.0 and a square image fitting exactly in the viewport,
    /// `scale` must equal `vp_size / img_size`.
    ///
    /// Analytical: viewport 100×100, image 50×50 → base_scale = 2.0,
    /// zoom = 1.0 → scale = 2.0.
    #[test]
    fn test_image_transform_scale_fit_square() {
        let state = ViewportState::new(0, WindowLevel::new(0.0, 1.0));
        let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(100.0, 100.0));
        let (_, scale) = state.image_transform(rect, 50, 50);
        assert!(
            (scale - 2.0).abs() < 1e-4,
            "scale must be 2.0 for 100×100 viewport and 50×50 image, got {scale}"
        );
    }

    /// For a landscape viewport (200×100) and a square image (50×50),
    /// the constraining dimension is the height, so `scale = 100/50 = 2.0`.
    #[test]
    fn test_image_transform_scale_fit_landscape_viewport() {
        let state = ViewportState::new(0, WindowLevel::new(0.0, 1.0));
        let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(200.0, 100.0));
        let (_, scale) = state.image_transform(rect, 50, 50);
        // min(200/50, 100/50) = min(4.0, 2.0) = 2.0
        assert!(
            (scale - 2.0).abs() < 1e-4,
            "scale must be 2.0 (height-constrained), got {scale}"
        );
    }

    /// Zoom doubles the scale.
    #[test]
    fn test_image_transform_zoom_doubles_scale() {
        let mut state = ViewportState::new(0, WindowLevel::new(0.0, 1.0));
        state.zoom = 2.0;
        let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(100.0, 100.0));
        let (_, scale) = state.image_transform(rect, 50, 50);
        // base=2.0, zoom=2.0 → scale=4.0
        assert!(
            (scale - 4.0).abs() < 1e-4,
            "zoom=2.0 must double scale to 4.0, got {scale}"
        );
    }

    /// For a zero-size image `image_transform` must return scale=1.0 without
    /// panic (defensive path).
    #[test]
    fn test_image_transform_zero_image_no_panic() {
        let state = ViewportState::new(0, WindowLevel::new(0.0, 1.0));
        let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(100.0, 100.0));
        let (_, scale) = state.image_transform(rect, 0, 0);
        assert_eq!(scale, 1.0, "zero-size image must return scale=1.0");
    }

    // ── slice_dims ────────────────────────────────────────────────────────────

    /// Axial slice of [D=4, R=5, C=6] must have (width, height) = (6, 5).
    #[test]
    fn test_slice_dims_axial() {
        let vol = make_volume(4, 5, 6);
        let (w, h) = slice_dims(&vol, 0);
        assert_eq!(w, 6, "axial width must equal cols=6");
        assert_eq!(h, 5, "axial height must equal rows=5");
    }

    /// Coronal slice of [D=4, R=5, C=6] must have (width, height) = (6, 4).
    #[test]
    fn test_slice_dims_coronal() {
        let vol = make_volume(4, 5, 6);
        let (w, h) = slice_dims(&vol, 1);
        assert_eq!(w, 6, "coronal width must equal cols=6");
        assert_eq!(h, 4, "coronal height must equal depth=4");
    }

    /// Sagittal slice of [D=4, R=5, C=6] must have (width, height) = (5, 4).
    #[test]
    fn test_slice_dims_sagittal() {
        let vol = make_volume(4, 5, 6);
        let (w, h) = slice_dims(&vol, 2);
        assert_eq!(w, 5, "sagittal width must equal rows=5");
        assert_eq!(h, 4, "sagittal height must equal depth=4");
    }

    // ── img_to_volume ─────────────────────────────────────────────────────────

    /// Axial axis: img_col → volume col, img_row → volume row,
    /// slice → volume depth.
    #[test]
    fn test_img_to_volume_axial() {
        let vox = img_to_volume(3, 7, 10, 0);
        assert_eq!(vox, [10, 7, 3], "axial: [slice, img_row, img_col]");
    }

    /// Coronal axis: img_col → volume col, img_row → volume depth,
    /// slice → volume row.
    #[test]
    fn test_img_to_volume_coronal() {
        let vox = img_to_volume(3, 7, 10, 1);
        assert_eq!(vox, [7, 10, 3], "coronal: [img_row, slice, img_col]");
    }

    /// Sagittal axis: img_col → volume row, img_row → volume depth,
    /// slice → volume col.
    #[test]
    fn test_img_to_volume_sagittal() {
        let vox = img_to_volume(3, 7, 10, 2);
        assert_eq!(vox, [7, 3, 10], "sagittal: [img_row, img_col, slice]");
    }

    // ── screen_to_img ─────────────────────────────────────────────────────────

    /// screen_to_img must return `None` for positions outside the image
    /// bounds.
    #[test]
    fn test_screen_to_img_out_of_bounds() {
        let offset = Vec2::new(0.0, 0.0);
        // Position at col=10, row=0 with img_w=8 is out of bounds.
        let result = screen_to_img(pos2(10.0, 0.0), offset, 1.0, 8, 8);
        assert!(
            result.is_none(),
            "position col=10 must be out of bounds for img_w=8"
        );
    }

    /// screen_to_img must correctly round-down to integer coordinates for
    /// an in-bounds position.
    ///
    /// Analytical: offset=(0,0), scale=2.0, screen=(7.9, 5.0)
    /// → col_f = 3.95 → col = 3; row_f = 2.5 → row = 2.
    #[test]
    fn test_screen_to_img_in_bounds() {
        let offset = Vec2::new(0.0, 0.0);
        let result = screen_to_img(pos2(7.9, 5.0), offset, 2.0, 10, 10);
        assert!(result.is_some(), "position must be in bounds");
        let (col, row) = result.unwrap();
        assert_eq!(col, 3, "col must be floor(7.9 / 2.0) = 3");
        assert_eq!(row, 2, "row must be floor(5.0 / 2.0) = 2");
    }

    // ── clamp_slice_index ─────────────────────────────────────────────────────

    /// `clamp_slice_index` must reduce an out-of-range index to `dim − 1`.
    #[test]
    fn test_clamp_slice_index() {
        let vol = make_volume(10, 20, 30);
        let mut state = ViewportState::new(0, WindowLevel::new(0.0, 100.0));
        state.slice_index = 999;
        state.clamp_slice_index(&vol);
        assert_eq!(
            state.slice_index, 9,
            "clamp must reduce 999 to depth-1=9 for axial axis"
        );
    }

    // ── invalidate_texture ────────────────────────────────────────────────────

    /// `invalidate_texture` must clear both `texture` and `texture_slice_key`.
    #[test]
    fn test_invalidate_texture_clears_key() {
        let mut state = ViewportState::new(0, WindowLevel::new(0.0, 1.0));
        state.texture_slice_key = Some((0, 5));
        state.invalidate_texture();
        assert!(
            state.texture.is_none(),
            "texture must be None after invalidation"
        );
        assert!(
            state.texture_slice_key.is_none(),
            "texture_slice_key must be None after invalidation"
        );
    }
}
