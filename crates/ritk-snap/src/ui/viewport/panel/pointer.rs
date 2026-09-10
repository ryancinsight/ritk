//! ViewportPanel pointer event handling.

use egui::{Rect, Response, Vec2};

use super::super::state::{img_to_volume, screen_to_img, screen_to_img_exact, ViewportPanel};
use crate::{
    tools::{
        interaction::{Annotation, ImagePoint, RoiKind, ToolState, ViewportOffset},
        kind::ToolKind,
    },
    LoadedVolume,
};

impl<'a> ViewportPanel<'a> {
    /// Route pointer events to the active tool and return a crosshair update
    /// when the Crosshair tool is used.
    pub(super) fn handle_pointer(
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
                            start: ImagePoint::new(pos.x, pos.y),
                            viewport_origin: ViewportOffset::new(
                                self.state.pan_offset.x,
                                self.state.pan_offset.y,
                            ),
                        };
                    }
                }
                ToolKind::WindowLevel => {
                    if let Some(pos) = response.interact_pointer_pos() {
                        self.state.tool_state = ToolState::WindowLevelDrag {
                            start: ImagePoint::new(pos.x, pos.y),
                            original_center: self.state.wl.center,
                            original_width: self.state.wl.width,
                        };
                    }
                }
                ToolKind::MeasureLength => {
                    // First click of a two-click length measurement.
                    if let Some(cursor) = response.interact_pointer_pos() {
                        if let Some((col, row)) = screen_to_img_exact(cursor, offset, scale) {
                            match &self.state.tool_state {
                                ToolState::Idle => {
                                    self.state.tool_state = ToolState::MeasureLength1 {
                                        p1: ImagePoint::new(col, row),
                                    };
                                }
                                ToolState::MeasureLength1 { p1 } => {
                                    let p1 = *p1;
                                    let p2 = ImagePoint::new(col, row);
                                    // Convert display point {x=col, y=row} → [row, col]
                                    let p1_arr = [p1.y(), p1.x()];
                                    let p2_arr = [p2.y(), p2.x()];
                                    let sp = volume.spacing;
                                    let spacing_2d = match self.state.axis {
                                        0 => [sp[1], sp[2]],
                                        1 => [sp[0], sp[2]],
                                        _ => [sp[0], sp[1]],
                                    };
                                    let Ok(length_mm) = Annotation::compute_length_checked(
                                        p1_arr, p2_arr, spacing_2d,
                                    ) else {
                                        self.state.tool_state = ToolState::Idle;
                                        return None;
                                    };
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
                        if let Some((col, row)) = screen_to_img_exact(cursor, offset, scale) {
                            let pt = ImagePoint::new(col, row);
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
                                    // Convert display point {x=col, y=row} → [row, col]
                                    let p1_arr = [p1.y(), p1.x()];
                                    let p2_arr = [p2.y(), p2.x()];
                                    let p3_arr = [pt.y(), pt.x()];
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
                        if let Some((col, row)) = screen_to_img_exact(cursor, offset, scale) {
                            let kind = if self.active_tool == ToolKind::RoiRect {
                                RoiKind::Rect
                            } else {
                                RoiKind::Ellipse
                            };
                            self.state.tool_state = ToolState::RoiDrag {
                                start: ImagePoint::new(col, row),
                                current: ImagePoint::new(col, row),
                                kind,
                            };
                        }
                    }
                }
                ToolKind::PointHu => {
                    if let Some(cursor) = response.interact_pointer_pos() {
                        if let Some((col, row)) = screen_to_img_exact(cursor, offset, scale) {
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
                        self.state.pan_offset = egui::vec2(
                            viewport_origin.x() + current_pos.x - start.x(),
                            viewport_origin.y() + current_pos.y - start.y(),
                        );
                    }
                }
                ToolState::WindowLevelDrag {
                    start,
                    original_center,
                    original_width,
                } => {
                    if let Some(current_pos) = response.interact_pointer_pos() {
                        let delta_x = current_pos.x - start.x();
                        let delta_y = current_pos.y - start.y();
                        // Horizontal drag → window width; vertical drag → centre.
                        // 2 HU per pixel is a clinically comfortable sensitivity.
                        self.state.wl.width = (original_width + f64::from(delta_x) * 2.0).max(1.0);
                        self.state.wl.center = original_center - f64::from(delta_y) * 2.0;
                        self.state.invalidate_texture();
                    }
                }
                ToolState::RoiDrag { start, kind, .. } => {
                    if let Some(cursor) = response.interact_pointer_pos() {
                        if let Some((col, row)) = screen_to_img_exact(cursor, offset, scale) {
                            self.state.tool_state = ToolState::RoiDrag {
                                start,
                                current: ImagePoint::new(col, row),
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
                    let tl =
                        ImagePoint::new(start.x().min(current.x()), start.y().min(current.y()));
                    let br =
                        ImagePoint::new(start.x().max(current.x()), start.y().max(current.y()));
                    if (br.x() - tl.x()) > 0.5 && (br.y() - tl.y()) > 0.5 {
                        // Convert display point {x=col,y=row} → [row, col]
                        let tl_arr = [tl.y(), tl.x()];
                        let br_arr = [br.y(), br.x()];
                        let sp = volume.spacing;
                        let spacing_2d = match self.state.axis {
                            0 => [sp[1], sp[2]],
                            1 => [sp[0], sp[2]],
                            _ => [sp[0], sp[1]],
                        };
                        let (pixels, pix_w, pix_h) =
                            volume.extract_slice(self.state.axis, self.state.slice_index);
                        let Ok(stats) = Annotation::compute_roi_rect_stats_checked(
                            tl_arr, br_arr, &pixels, pix_w, pix_h, spacing_2d,
                        ) else {
                            self.state.tool_state = ToolState::Idle;
                            return None;
                        };
                        self.state.annotations.push(Annotation::RoiRect {
                            top_left: tl_arr,
                            bottom_right: br_arr,
                            mean: stats.0,
                            std_dev: stats.1,
                            min: stats.2,
                            max: stats.3,
                            area_mm2: stats.4,
                        });
                    }
                    self.state.tool_state = ToolState::Idle;
                }
                ToolState::RoiDrag {
                    start,
                    current,
                    kind: RoiKind::Ellipse,
                } => {
                    // Use compute_roi_ellipse_stats for correct ellipse pixel-mask statistics.
                    let tl =
                        ImagePoint::new(start.x().min(current.x()), start.y().min(current.y()));
                    let br =
                        ImagePoint::new(start.x().max(current.x()), start.y().max(current.y()));
                    if (br.x() - tl.x()) > 0.5 && (br.y() - tl.y()) > 0.5 {
                        let tl_arr = [tl.y(), tl.x()];
                        let br_arr = [br.y(), br.x()];
                        let sp = volume.spacing;
                        let spacing_2d = match self.state.axis {
                            0 => [sp[1], sp[2]],
                            1 => [sp[0], sp[2]],
                            _ => [sp[0], sp[1]],
                        };
                        let (pixels, pix_w, pix_h) =
                            volume.extract_slice(self.state.axis, self.state.slice_index);
                        let Ok((center, radii, mean, std_dev, min, max, area_mm2)) =
                            Annotation::compute_roi_ellipse_stats_checked(
                                tl_arr, br_arr, &pixels, pix_w, pix_h, spacing_2d,
                            )
                        else {
                            self.state.tool_state = ToolState::Idle;
                            return None;
                        };
                        self.state.annotations.push(Annotation::RoiEllipse {
                            center,
                            radii,
                            mean,
                            std_dev,
                            min,
                            max,
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
