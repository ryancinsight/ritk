use super::SnapApp;
use crate::tools::interaction::{ImagePoint, ToolState};
use crate::tools::kind::ToolKind;
use crate::ui::{
    pan_from_drag_delta, window_level_from_drag_delta, zoom_from_drag_delta,
    WINDOW_LEVEL_SENSITIVITY,
};
use eunomia::FloatElement;
use metis_ui_lang::PathPush;

impl SnapApp {
    pub(crate) fn on_drag(&mut self, pos: Option<ImagePoint>) {
        let Some(pos) = pos else { return };
        if let ToolState::PencilDrag { capture } = &mut self.tool_state {
            if matches!(capture.push(pos), PathPush::Full) {
                self.status_message = "Pencil sample limit reached".to_owned();
            }
            return;
        }
        match self.tool_state.clone() {
            ToolState::Panning {
                start,
                viewport_origin,
            } => {
                let displacement = crate::tools::interaction::ViewportOffset::new(
                    pos.x() - start.x(),
                    pos.y() - start.y(),
                );
                match pan_from_drag_delta(viewport_origin, displacement) {
                    Ok(offset) => self.pan_offset = offset,
                    Err(error) => {
                        tracing::warn!(%error, "pan update rejected");
                        self.status_message = format!("Navigation rejected: {error}");
                    }
                }
            }
            ToolState::Zooming {
                start,
                original_zoom,
            } => {
                let drag_delta_y = pos.y() - start.y();
                self.zoom = zoom_from_drag_delta(original_zoom, drag_delta_y);
                self.status_message = format!("Zoom: {:.0}%", self.zoom * 100.0);
            }
            ToolState::WindowLevelDrag {
                start,
                original_center,
                original_width,
            } => {
                let (new_center, new_width) = window_level_from_drag_delta(
                    original_center,
                    original_width,
                    pos.x() - start.x(),
                    pos.y() - start.y(),
                    WINDOW_LEVEL_SENSITIVITY,
                );
                self.viewer_state.window_center = Some(FloatElement::from_f64(new_center));
                self.viewer_state.window_width = Some(FloatElement::from_f64(new_width));
                self.bump_visual_revision();
            }
            ToolState::RoiDrag { start, kind, .. } => {
                self.tool_state = ToolState::RoiDrag {
                    start,
                    current: pos,
                    kind,
                };
            }
            ToolState::LengthDrag { start, .. } => {
                self.tool_state = ToolState::LengthDrag {
                    start,
                    current: pos,
                };
            }
            ToolState::MeasureLength1 { p1 }
                if self.active_tool == ToolKind::MeasureLength
                    && self.pointer.pending_measurement_location
                        == Some(self.annotation_location()) =>
            {
                self.pointer.drag_measurement_location =
                    self.pointer.pending_measurement_location.take();
                self.tool_state = ToolState::LengthDrag {
                    start: p1,
                    current: pos,
                };
            }
            ToolState::DeviationDrag { start, .. } => {
                self.tool_state = ToolState::DeviationDrag {
                    start,
                    current: pos,
                };
            }
            ToolState::ArrowDrag { start, .. } => {
                self.tool_state = ToolState::ArrowDrag {
                    start,
                    current: pos,
                };
            }
            ToolState::CobbDrag { first, start, .. } => {
                self.tool_state = ToolState::CobbDrag {
                    first,
                    start,
                    current: pos,
                };
            }
            _ => {}
        }
    }
}
