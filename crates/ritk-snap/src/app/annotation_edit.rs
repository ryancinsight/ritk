//! Native annotation selection and deletion state transitions.

use super::state::SnapApp;
use crate::tools::interaction::{Annotation, AnnotationLocation, ImagePoint};

impl SnapApp {
    /// Select a completed annotation by its stable index in the session list.
    pub(crate) fn select_annotation(&mut self, index: usize) -> bool {
        if index >= self.annotations.len() {
            return false;
        }
        self.pointer.selected_annotation = Some(index);
        true
    }

    /// Delete the selected annotation when it belongs to the displayed image.
    pub(crate) fn delete_selected_annotation(&mut self) -> bool {
        let Some(index) = self.pointer.selected_annotation else {
            return false;
        };
        let Some(annotation) = self.annotations.get(index) else {
            self.pointer.selected_annotation = None;
            return false;
        };
        if !annotation_matches_location(annotation, self.annotation_location()) {
            return false;
        }
        self.annotations.remove(index);
        self.pointer.selected_annotation = None;
        true
    }

    /// Delete all completed annotations anchored to the displayed image.
    pub(crate) fn delete_annotations_on_current_image(&mut self) -> usize {
        let location = self.annotation_location();
        let before = self.annotations.len();
        self.annotations
            .retain(|annotation| !annotation_matches_location(annotation, location));
        self.pointer.selected_annotation = None;
        before.saturating_sub(self.annotations.len())
    }

    /// Select the nearest visible annotation under an image-space pointer.
    ///
    /// The tolerance is expressed in image pixels so selection remains
    /// deterministic across native hosts; the nearest mark wins when labels
    /// overlap.
    pub(crate) fn select_annotation_at(&mut self, point: ImagePoint) -> bool {
        let location = self.annotation_location();
        let point = [point.x(), point.y()];
        let selected = self
            .annotations
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, annotation)| annotation_matches_location(annotation, location))
            .filter_map(|(index, annotation)| {
                annotation_distance(annotation, point).map(|distance| (index, distance))
            })
            .min_by(|(_, left), (_, right)| left.total_cmp(right))
            .filter(|(_, distance)| *distance <= 12.0)
            .map(|(index, _)| index);
        self.pointer.selected_annotation = selected;
        selected.is_some()
    }
}

fn annotation_matches_location(annotation: &Annotation, location: AnnotationLocation) -> bool {
    match annotation.location() {
        Some(recorded) => recorded == location,
        None => match annotation {
            Annotation::Length { plane, .. } | Annotation::PatientLength { plane, .. } => {
                *plane == location.plane
            }
            Annotation::Angle { .. }
            | Annotation::RoiRect { .. }
            | Annotation::RoiEllipse { .. }
            | Annotation::HuPoint { .. } => false,
        },
    }
}

fn annotation_distance(annotation: &Annotation, point: [f32; 2]) -> Option<f32> {
    let distance = match annotation {
        Annotation::Length { p1, p2, .. } => {
            segment_distance(point, [p1[1], p1[0]], [p2[1], p2[0]])
        }
        Annotation::PatientLength { .. } => return None,
        Annotation::Angle { p1, p2, p3, .. } => segment_distance(
            point,
            [p1[1], p1[0]],
            [p2[1], p2[0]],
        )
        .min(segment_distance(point, [p2[1], p2[0]], [p3[1], p3[0]])),
        Annotation::RoiRect {
            top_left,
            bottom_right,
            ..
        } => {
            let [left, top] = [top_left[1], top_left[0]];
            let [right, bottom] = [bottom_right[1], bottom_right[0]];
            segment_distance(point, [left, top], [right, top])
                .min(segment_distance(point, [right, top], [right, bottom]))
                .min(segment_distance(point, [right, bottom], [left, bottom]))
                .min(segment_distance(point, [left, bottom], [left, top]))
        }
        Annotation::RoiEllipse { center, radii, .. } => {
            let dx = point[0] - center[1];
            let dy = point[1] - center[0];
            let radial = (dx * dx + dy * dy).sqrt();
            let boundary = radii[1].max(radii[0]);
            (radial - boundary).abs()
        }
        Annotation::HuPoint { pos, .. } => {
            let dx = point[0] - pos[1];
            let dy = point[1] - pos[0];
            (dx * dx + dy * dy).sqrt()
        }
    };
    distance.is_finite().then_some(distance)
}

fn segment_distance(point: [f32; 2], start: [f32; 2], end: [f32; 2]) -> f32 {
    let [dx, dy] = [end[0] - start[0], end[1] - start[1]];
    let length_squared = dx.mul_add(dx, dy * dy);
    if length_squared <= f32::EPSILON {
        let [px, py] = [point[0] - start[0], point[1] - start[1]];
        return px.mul_add(px, py * py).sqrt();
    }
    let t = ((point[0] - start[0]).mul_add(dx, (point[1] - start[1]) * dy) / length_squared)
        .clamp(0.0, 1.0);
    let [x, y] = [start[0] + t * dx, start[1] + t * dy];
    let [px, py] = [point[0] - x, point[1] - y];
    px.mul_add(px, py * py).sqrt()
}

#[cfg(test)]
mod tests {
    use super::{annotation_distance, annotation_matches_location};
    use crate::tools::interaction::{Annotation, AnnotationLocation};

    #[test]
    fn anchored_annotations_are_scoped_to_plane_and_slice() {
        let annotation = Annotation::HuPoint {
            location: Some(AnnotationLocation { plane: 1, slice: 4 }),
            pos: [1.0, 2.0],
            value: 3.0,
        };
        assert!(annotation_matches_location(
            &annotation,
            AnnotationLocation { plane: 1, slice: 4 }
        ));
        assert!(!annotation_matches_location(
            &annotation,
            AnnotationLocation { plane: 1, slice: 5 }
        ));
    }

    #[test]
    fn line_hit_distance_prefers_geometry_over_label_order() {
        let annotation = Annotation::Length {
            plane: 0,
            location: Some(AnnotationLocation { plane: 0, slice: 2 }),
            p1: [10.0, 10.0],
            p2: [10.0, 30.0],
            length_mm: 20.0,
        };
        let distance = annotation_distance(&annotation, [20.0, 10.0])
            .expect("invariant: finite line geometry produces a distance");
        assert_eq!(distance, 0.0);
    }
}
