//! Multi-click and drawing annotation gestures.

use super::SnapApp;
use crate::tools::interaction::{Annotation, ImagePoint, PolygonKind, ToolState};
use crate::tools::kind::ToolKind;
use crate::ui::image_point_to_voxel;

impl SnapApp {
    pub(crate) fn on_annotation_click(&mut self, pos: ImagePoint) {
        let location = self.annotation_location();
        match self.active_tool {
            ToolKind::MeasureAngle => match self.tool_state.clone() {
                ToolState::MeasureAngle2 { p1, p2 } => {
                    let a = [p1.y(), p1.x()];
                    let b = [p2.y(), p2.x()];
                    let c = [pos.y(), pos.x()];
                    let angle_deg = match self
                        .slice_plane_spacing()
                        .and_then(Annotation::validate_spacing)
                    {
                        Ok(spacing) => Annotation::compute_angle_with_spacing(a, b, c, spacing),
                        Err(error) => {
                            self.status_message = format!("Measurement rejected: {error}");
                            self.tool_state = ToolState::Idle;
                            return;
                        }
                    };
                    self.push_annotation(Annotation::Angle {
                        location: None,
                        p1: a,
                        p2: b,
                        p3: c,
                        angle_deg,
                    });
                    self.tool_state = ToolState::Idle;
                }
                ToolState::MeasureLength1 { p1 } => {
                    self.tool_state = ToolState::MeasureAngle2 { p1, p2: pos }
                }
                _ => self.tool_state = ToolState::MeasureLength1 { p1: pos },
            },
            ToolKind::RoiClosedPolygon | ToolKind::RoiOpenPolygon => {
                let kind = if self.active_tool == ToolKind::RoiClosedPolygon {
                    PolygonKind::Closed
                } else {
                    PolygonKind::Open
                };
                match self.tool_state.clone() {
                    ToolState::Polygon {
                        mut points,
                        kind: current,
                    } if current == kind => {
                        let near = |point: &ImagePoint| {
                            let Some([width, height]) = self.pointer.annotation_pixel_size else {
                                return *point == pos;
                            };
                            const NODE_PICK_RADIUS: f64 = 6.0;
                            let dx = (f64::from(point.x()) - f64::from(pos.x())) * width;
                            let dy = (f64::from(point.y()) - f64::from(pos.y())) * height;
                            dx.hypot(dy) <= NODE_PICK_RADIUS
                        };
                        let minimum = match kind {
                            PolygonKind::Closed => 3,
                            PolygonKind::Open => 2,
                        };
                        let finish = points.len() >= minimum
                            && (points.last().is_some_and(near)
                                || (kind == PolygonKind::Closed
                                    && points.first().is_some_and(near)));
                        if finish {
                            self.finalise_polygon(points, kind);
                            self.tool_state = ToolState::Idle;
                        } else {
                            points.push(pos);
                            self.tool_state = ToolState::Polygon { points, kind };
                        }
                    }
                    _ => {
                        self.tool_state = ToolState::Polygon {
                            points: vec![pos],
                            kind,
                        }
                    }
                }
            }
            ToolKind::PointHu => {
                if let Some(vol) = &self.loaded {
                    let Some(voxel) =
                        image_point_to_voxel(vol.shape, self.axis, location.slice, pos)
                    else {
                        return;
                    };
                    let value = crate::ui::intensity_at_voxel(vol, voxel);
                    self.push_annotation(Annotation::HuPoint {
                        location: None,
                        pos: [pos.y(), pos.x()],
                        value,
                    });
                    let [depth, row, col] = voxel;
                    self.status_message = format!("Value at [{depth},{row},{col}]: {value:.0}");
                }
            }
            _ => {}
        }
    }
}
