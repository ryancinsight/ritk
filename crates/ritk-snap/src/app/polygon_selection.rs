//! Selection of completed polygon marks on their owning slice.

use super::viewer_viewport::ViewerViewport;
use super::SnapApp;
use crate::presentation::ViewportPoint;
use crate::tools::interaction::{Annotation, AnnotationLocation};
use metis_ui_lang::{nearest_polyline_segment, PathTopology};

pub(super) const PICK_RADIUS_PIXELS: f64 = 6.0;

impl SnapApp {
    pub(super) fn pick_polygon(&mut self, viewport: ViewerViewport, point: ViewportPoint) {
        let plane = viewport.axis();
        let location = AnnotationLocation {
            plane,
            slice: self.axis_slice_info(plane).0,
        };
        self.pointer.selected_annotation = self
            .annotations
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, annotation)| annotation.location() == Some(location))
            .filter_map(|(index, annotation)| {
                let (vertices, topology) = polygon_path(annotation)?;
                match nearest_polyline_segment(
                    vertices
                        .iter()
                        .copied()
                        .map(|point| viewport.project(point)),
                    [point.x(), point.y()],
                    topology,
                ) {
                    Ok(hit) => Some((index, hit.distance())),
                    Err(error) => {
                        tracing::debug!(%error, "cannot pick polygon");
                        None
                    }
                }
            })
            .filter(|(_, distance)| *distance <= PICK_RADIUS_PIXELS)
            .min_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index);
    }

    pub(super) fn delete_selected_polygon(&mut self) -> bool {
        let Some(index) = self.pointer.selected_annotation.take() else {
            return false;
        };
        let plane = self.axis;
        let location = AnnotationLocation {
            plane,
            slice: self.axis_slice_info(plane).0,
        };
        if !self.annotations.get(index).is_some_and(|annotation| {
            annotation.location() == Some(location) && polygon_path(annotation).is_some()
        }) {
            return false;
        }
        self.annotations.remove(index);
        self.bump_visual_revision();
        true
    }
}

pub(super) fn polygon_path(annotation: &Annotation) -> Option<(&[[f32; 2]], PathTopology)> {
    match annotation {
        Annotation::ClosedPolygon { points, .. } => Some((points, PathTopology::Closed)),
        Annotation::OpenPolygon { points, .. } => Some((points, PathTopology::Open)),
        _ => None,
    }
}
