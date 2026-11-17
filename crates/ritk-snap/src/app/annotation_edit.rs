//! Native annotation selection and deletion state transitions.

use super::state::SnapApp;
use crate::tools::interaction::{Annotation, ImagePoint};
use crate::ui::{image_point_to_voxel, intensity_at_voxel};
use metis_ui_lang::{polyline_distance, PathTopology};

impl SnapApp {
    #[cfg(windows)]
    pub(crate) fn has_selected_length(&self) -> bool {
        matches!(
            self.pointer.selected_annotation.and_then(|index| self.annotations.get(index)),
            Some(Annotation::Length { location: Some(location), .. })
                if *location == self.annotation_location()
        )
    }

    /// Delete the selected annotation when it belongs to the displayed image.
    #[cfg(windows)]
    pub(crate) fn delete_selected_annotation(&mut self) -> bool {
        let Some(index) = self.pointer.selected_annotation else {
            return false;
        };
        let Some(annotation) = self.annotations.get(index) else {
            self.pointer.selected_annotation = None;
            return false;
        };
        if !annotation.visible_on_image(self.annotation_location()) {
            return false;
        }
        self.annotations.remove(index);
        self.pointer.selected_annotation = None;
        true
    }

    /// Delete all completed annotations anchored to the displayed image.
    #[cfg(windows)]
    pub(crate) fn delete_annotations_on_current_image(&mut self) -> usize {
        let location = self.annotation_location();
        let before = self.annotations.len();
        self.annotations
            .retain(|annotation| !annotation.visible_on_image(location));
        self.pointer.selected_annotation = None;
        before.saturating_sub(self.annotations.len())
    }

    /// Select the nearest visible annotation under an image-space pointer.
    ///
    pub(crate) fn select_annotation_at(&mut self, point: ImagePoint, pixel_size: [f64; 2]) -> bool {
        const PICK_RADIUS_PIXELS: f64 = 6.0;
        let location = self.annotation_location();
        let selected = self
            .annotations
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, annotation)| annotation.visible_on_image(location))
            .filter_map(|(index, annotation)| {
                annotation_distance(annotation, [point.y(), point.x()], pixel_size)
                    .map(|distance| (index, distance))
            })
            .min_by(|(_, left), (_, right)| left.total_cmp(right))
            .filter(|(_, distance)| *distance <= PICK_RADIUS_PIXELS)
            .map(|(index, _)| index);
        self.pointer.selected_annotation = selected;
        selected.is_some()
    }

    /// Start moving the selected image-coordinate mark from `point`.
    pub(crate) fn begin_annotation_move(&mut self, point: ImagePoint) -> bool {
        let Some(index) = self.pointer.selected_annotation else {
            return false;
        };
        let Some(annotation) = self.annotations.get(index) else {
            return false;
        };
        if !annotation.visible_on_image(self.annotation_location()) || !is_movable(annotation) {
            return false;
        }
        self.pointer.moving_annotation = Some((index, point));
        true
    }

    /// Translate the selected image-coordinate mark and refresh ROI statistics.
    pub(crate) fn move_annotation_to(&mut self, point: ImagePoint) -> bool {
        let Some((index, previous)) = self.pointer.moving_annotation else {
            return false;
        };
        let delta = [point.y() - previous.y(), point.x() - previous.x()];
        if delta == [0.0; 2] || !delta.iter().all(|value| value.is_finite()) {
            return false;
        }
        let Some(original) = self.annotations.get(index) else {
            return false;
        };
        if !original.visible_on_image(self.annotation_location()) {
            return false;
        }
        let mut candidate = original.clone();
        translate_annotation(&mut candidate, delta);
        if !self.refresh_moved_annotation(&mut candidate) {
            return false;
        }
        let Some(annotation) = self.annotations.get_mut(index) else {
            return false;
        };
        *annotation = candidate;
        self.pointer.moving_annotation = Some((index, point));
        true
    }

    fn refresh_moved_annotation(&self, annotation: &mut Annotation) -> bool {
        if let Annotation::HuPoint { pos, value, .. } = annotation {
            let Some(volume) = self.loaded.as_ref() else {
                return false;
            };
            let Some(voxel) = image_point_to_voxel(
                volume.shape,
                self.axis,
                self.axis_slice_info(self.axis).0,
                ImagePoint::new(pos[1], pos[0]),
            ) else {
                return false;
            };
            *value = intensity_at_voxel(volume, voxel);
            return value.is_finite();
        }
        if !matches!(
            annotation,
            Annotation::RoiRect { .. } | Annotation::RoiEllipse { .. }
        ) {
            return true;
        }
        let Some(volume) = self.loaded.as_ref() else {
            return false;
        };
        let Ok(spacing) = self.slice_plane_spacing() else {
            return false;
        };
        let (pixels, width, height) =
            volume.extract_slice(self.axis, self.axis_slice_info(self.axis).0);
        match annotation {
            Annotation::RoiRect {
                top_left,
                bottom_right,
                mean,
                std_dev,
                min,
                max,
                area_mm2,
                ..
            } => {
                if let Ok((new_mean, new_std, new_min, new_max, new_area)) =
                    Annotation::compute_roi_rect_stats_checked(
                        *top_left,
                        *bottom_right,
                        &pixels,
                        width,
                        height,
                        spacing,
                    )
                {
                    (*mean, *std_dev, *min, *max, *area_mm2) =
                        (new_mean, new_std, new_min, new_max, new_area);
                } else {
                    return false;
                }
            }
            Annotation::RoiEllipse {
                center,
                radii,
                mean,
                std_dev,
                min,
                max,
                area_mm2,
                ..
            } => {
                let p1 = [center[0] - radii[0], center[1] - radii[1]];
                let p2 = [center[0] + radii[0], center[1] + radii[1]];
                if let Ok((_, _, new_mean, new_std, new_min, new_max, new_area)) =
                    Annotation::compute_roi_ellipse_stats_checked(
                        p1, p2, &pixels, width, height, spacing,
                    )
                {
                    (*mean, *std_dev, *min, *max, *area_mm2) =
                        (new_mean, new_std, new_min, new_max, new_area);
                } else {
                    return false;
                }
            }
            _ => {}
        }
        true
    }
}

fn is_movable(annotation: &Annotation) -> bool {
    !matches!(annotation, Annotation::PatientLength { .. })
}

fn translate_annotation(annotation: &mut Annotation, [row, col]: [f32; 2]) {
    let shift = |point: &mut [f32; 2]| {
        point[0] += row;
        point[1] += col;
    };
    match annotation {
        Annotation::Length { p1, p2, .. } => {
            shift(p1);
            shift(p2);
        }
        Annotation::PatientLength { .. } => {}
        Annotation::Angle { p1, p2, p3, .. } => {
            shift(p1);
            shift(p2);
            shift(p3);
        }
        Annotation::RoiRect {
            top_left,
            bottom_right,
            ..
        } => {
            shift(top_left);
            shift(bottom_right);
        }
        Annotation::RoiEllipse { center, .. } => shift(center),
        Annotation::HuPoint { pos, .. } => shift(pos),
        Annotation::ClosedPolygon { points, .. }
        | Annotation::OpenPolygon { points, .. }
        | Annotation::Pencil { points, .. } => points.iter_mut().for_each(shift),
        Annotation::CobbAngle { p1, p2, p3, p4, .. } => {
            shift(p1);
            shift(p2);
            shift(p3);
            shift(p4);
        }
        Annotation::Deviation {
            start,
            end,
            reference,
            ..
        } => {
            shift(start);
            shift(end);
            shift(reference);
        }
        Annotation::Arrow { tail, head, .. } => {
            shift(tail);
            shift(head);
        }
    }
}

fn annotation_distance(
    annotation: &Annotation,
    point: [f32; 2],
    [width, height]: [f64; 2],
) -> Option<f64> {
    let relative = |[row, col]: [f32; 2]| {
        [
            (f64::from(col) - f64::from(point[1])) * width,
            (f64::from(row) - f64::from(point[0])) * height,
        ]
    };
    let path = |points: &[[f32; 2]], topology| {
        // An invalid or incomplete saved path cannot be a pointer hit.
        polyline_distance(points.iter().copied().map(relative), [0.0; 2], topology).ok()
    };
    let segment = |start, end| path(&[start, end], PathTopology::Open);
    match annotation {
        Annotation::Length { p1, p2, .. } => segment(*p1, *p2),
        Annotation::PatientLength { .. } => None,
        Annotation::Angle { p1, p2, p3, .. } => path(&[*p1, *p2, *p3], PathTopology::Open),
        Annotation::RoiRect {
            top_left,
            bottom_right,
            ..
        } => {
            let top_right = [top_left[0], bottom_right[1]];
            let bottom_left = [bottom_right[0], top_left[1]];
            path(
                &[*top_left, top_right, *bottom_right, bottom_left],
                PathTopology::Closed,
            )
        }
        Annotation::RoiEllipse { center, radii, .. } => {
            let row = (point[0] - center[0]) / radii[0];
            let col = (point[1] - center[1]) / radii[1];
            (row * row + col * col <= 1.0).then_some(0.0)
        }
        Annotation::HuPoint { pos, .. } => {
            let [x, y] = relative(*pos);
            let distance = x.hypot(y);
            distance.is_finite().then_some(distance)
        }
        Annotation::ClosedPolygon { points, .. } => path(points, PathTopology::Closed),
        Annotation::OpenPolygon { points, .. } | Annotation::Pencil { points, .. } => {
            path(points, PathTopology::Open)
        }
        Annotation::CobbAngle { p1, p2, p3, p4, .. } => {
            Some(segment(*p1, *p2)?.min(segment(*p3, *p4)?))
        }
        Annotation::Deviation {
            start,
            end,
            reference,
            ..
        } => path(&[*end, *start, *reference], PathTopology::Open),
        Annotation::Arrow { tail, head, .. } => segment(*tail, *head),
    }
}

#[cfg(test)]
mod tests {
    use super::{annotation_distance, translate_annotation};
    use crate::tools::interaction::{Annotation, AnnotationLocation};

    #[test]
    fn anchored_annotations_are_scoped_to_plane_and_slice() {
        let annotation = Annotation::HuPoint {
            location: Some(AnnotationLocation { plane: 1, slice: 4 }),
            pos: [1.0, 2.0],
            value: 3.0,
        };
        assert!(annotation.visible_on_image(AnnotationLocation { plane: 1, slice: 4 }));
        assert!(!annotation.visible_on_image(AnnotationLocation { plane: 1, slice: 5 }));
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
        let distance = annotation_distance(&annotation, [10.0, 20.0], [2.0, 3.0])
            .expect("invariant: finite line geometry produces a distance");
        assert_eq!(distance, 0.0);
    }

    #[test]
    fn deviation_only_picks_the_two_drawn_segments() {
        let annotation = Annotation::Deviation {
            location: None,
            start: [0.0, 0.0],
            end: [4.0, 4.0],
            reference: [0.0, 4.0],
            distance_mm: 4.0,
            angle_deg: 45.0,
        };
        assert_eq!(
            annotation_distance(&annotation, [4.0, 4.0], [1.0, 1.0]),
            Some(0.0)
        );
        assert_eq!(
            annotation_distance(&annotation, [0.0, 2.0], [1.0, 1.0]),
            Some(0.0)
        );
        assert_eq!(
            annotation_distance(&annotation, [2.0, 4.0], [1.0, 1.0])
                .map(|distance| distance >= 1.0),
            Some(true)
        );
    }

    #[test]
    fn moving_length_translates_both_endpoints_without_changing_distance() {
        let mut annotation = Annotation::Length {
            plane: 0,
            location: Some(AnnotationLocation { plane: 0, slice: 2 }),
            p1: [10.0, 10.0],
            p2: [10.0, 30.0],
            length_mm: 20.0,
        };
        translate_annotation(&mut annotation, [3.0, -4.0]);
        assert_eq!(
            annotation,
            Annotation::Length {
                plane: 0,
                location: Some(AnnotationLocation { plane: 0, slice: 2 }),
                p1: [13.0, 6.0],
                p2: [13.0, 26.0],
                length_mm: 20.0,
            }
        );
    }
}
