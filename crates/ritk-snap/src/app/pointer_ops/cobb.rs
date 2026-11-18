use super::SnapApp;
use crate::tools::interaction::{Annotation, AnnotationLocation, ImagePoint, ToolState};

impl SnapApp {
    pub(crate) fn cobb_location(&self) -> Option<AnnotationLocation> {
        match self.tool_state {
            ToolState::CobbDrag { .. } => self.pointer.drag_measurement_location,
            ToolState::CobbPending { .. } => self.pointer.pending_measurement_location,
            _ if matches!(
                self.pointer.suspended_measurement,
                Some(ToolState::CobbPending { .. })
            ) =>
            {
                self.pointer.pending_measurement_location
            }
            _ => None,
        }
    }

    pub(super) fn finish_cobb(
        &mut self,
        first: Option<[ImagePoint; 2]>,
        second: [ImagePoint; 2],
        location: Option<AnnotationLocation>,
    ) {
        self.pointer.suspended_measurement = None;
        let Some(first) = first else {
            self.tool_state = ToolState::CobbPending { segment: second };
            self.pointer.pending_measurement_location = location;
            return;
        };
        let [p1, p2, p3, p4] =
            [first[0], first[1], second[0], second[1]].map(|point| [point.y(), point.x()]);
        let angle = self
            .slice_plane_spacing()
            .and_then(|spacing| Annotation::compute_cobb_angle(p1, p2, p3, p4, spacing));
        match angle {
            Ok(angle_deg) => self.push_annotation(Annotation::CobbAngle {
                points: [p1, p2, p3, p4],
                location,
                angle_deg,
            }),
            Err(error) => self.status_message = format!("Measurement rejected: {error}"),
        }
        self.tool_state = ToolState::Idle;
        self.pointer.pending_measurement_location = None;
    }
}
