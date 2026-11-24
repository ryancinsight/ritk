//! Projection of mutable viewer state into the host-neutral presentation contract.

use super::SnapApp;
use crate::presentation::{AnnotationKind, AnnotationSummary, PresentationSnapshot};
use crate::render::WindowLevel;
use crate::tools::interaction::{Annotation, ImageMeasurementScale};
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
        let summary = |kind, value| Some(AnnotationSummary::new(kind, value));
        let last_annotation = self.annotations.last().and_then(|annotation| {
            #[cfg(any(windows, target_arch = "wasm32"))]
            let image_scale = ImageMeasurementScale::new(self.length_scale(annotation.location()));
            #[cfg(not(any(windows, target_arch = "wasm32")))]
            let image_scale = ImageMeasurementScale::new(1.0);
            match annotation {
                Annotation::Length { length_mm, .. } => {
                    summary(AnnotationKind::Length, image_scale.length(*length_mm))
                }
                Annotation::PatientLength { length, .. } => {
                    summary(AnnotationKind::PatientLength, length.length_mm())
                }
                Annotation::Angle { angle_deg, .. } => {
                    summary(AnnotationKind::Angle, f64::from(*angle_deg))
                }
                Annotation::Deviation { distance_mm, .. } => {
                    summary(AnnotationKind::Deviation, image_scale.length(*distance_mm))
                }
                Annotation::RoiRect { area_mm2, .. } => {
                    summary(AnnotationKind::RoiRect, image_scale.area(*area_mm2))
                }
                Annotation::RoiEllipse { area_mm2, .. } => {
                    summary(AnnotationKind::RoiEllipse, image_scale.area(*area_mm2))
                }
                Annotation::HuPoint { value, .. } => {
                    summary(AnnotationKind::HuPoint, f64::from(*value))
                }
                Annotation::ClosedPolygon { area_mm2, .. } => {
                    summary(AnnotationKind::ClosedPolygon, image_scale.area(*area_mm2))
                }
                Annotation::OpenPolygon { length_mm, .. } => {
                    summary(AnnotationKind::OpenPolygon, image_scale.length(*length_mm))
                }
                Annotation::Arrow { .. } => None,
                Annotation::Pencil { .. } => None,
                Annotation::CobbAngle { angle_deg, .. } => Some(AnnotationSummary::new(
                    AnnotationKind::Angle,
                    f64::from(*angle_deg),
                )),
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
