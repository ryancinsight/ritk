//! Projection of mutable viewer state into the host-neutral presentation contract.

use super::SnapApp;
use crate::presentation::{AnnotationKind, AnnotationSummary, PresentationSnapshot};
use crate::render::WindowLevel;
use crate::tools::interaction::Annotation;
use crate::tools::kind::ToolKind;
use crate::viewer::{DEFAULT_WINDOW_CENTER, DEFAULT_WINDOW_WIDTH};

impl SnapApp {
    /// Project viewer state into the host-neutral presentation contract.
    pub(crate) fn presentation_snapshot(&self) -> PresentationSnapshot {
        let slice_indices = std::array::from_fn(|axis| self.axis_slice_info(axis).0);
        let slice_counts = std::array::from_fn(|axis| self.axis_slice_info(axis).1);
        let window_level = WindowLevel::new(
            f64::from(
                self.viewer_state
                    .window_center
                    .unwrap_or(DEFAULT_WINDOW_CENTER),
            ),
            f64::from(
                self.viewer_state
                    .window_width
                    .unwrap_or(DEFAULT_WINDOW_WIDTH)
                    .max(1.0),
            ),
        );
        let active_tool_index = ToolKind::all()
            .iter()
            .position(|tool| *tool == self.active_tool)
            .expect("invariant: active tool belongs to ToolKind::all");
        let last_annotation = self.annotations.last().map(|annotation| match annotation {
            Annotation::Length { length_mm, .. } => {
                AnnotationSummary::new(AnnotationKind::Length, f64::from(*length_mm))
            }
            Annotation::PatientLength(measurement) => {
                AnnotationSummary::new(AnnotationKind::PatientLength, measurement.length_mm())
            }
            Annotation::Angle { angle_deg, .. } => {
                AnnotationSummary::new(AnnotationKind::Angle, f64::from(*angle_deg))
            }
            Annotation::RoiRect { area_mm2, .. } => {
                AnnotationSummary::new(AnnotationKind::RoiRect, f64::from(*area_mm2))
            }
            Annotation::RoiEllipse { area_mm2, .. } => {
                AnnotationSummary::new(AnnotationKind::RoiEllipse, f64::from(*area_mm2))
            }
            Annotation::HuPoint { value, .. } => {
                AnnotationSummary::new(AnnotationKind::HuPoint, f64::from(*value))
            }
        });
        PresentationSnapshot::from_parts(
            self.visual_revision,
            self.loaded.is_some(),
            self.axis,
            slice_indices,
            slice_counts,
            window_level,
            self.cine.enabled,
            self.cine.fps,
            self.zoom,
            self.pan_offset,
            self.view_transform,
            self.show_crosshair,
            self.linked_cursor.map(|cursor| cursor.voxel()),
            None,
            active_tool_index,
            self.active_tool.label(),
            self.annotations.len(),
            last_annotation,
        )
    }
}
