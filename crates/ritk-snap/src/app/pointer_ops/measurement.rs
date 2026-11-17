//! Completed measurements and slice geometry.

use super::SnapApp;
use crate::tools::interaction::{
    Annotation, AnnotationLocation, ImagePoint, MeasurementError, PolygonKind,
};

impl SnapApp {
    pub(super) fn finalise_polygon(&mut self, points: Vec<ImagePoint>, kind: PolygonKind) {
        let spacing = match self
            .slice_plane_spacing()
            .and_then(Annotation::validate_spacing)
        {
            Ok(spacing) => spacing,
            Err(error) => {
                self.status_message = format!("Measurement rejected: {error}");
                return;
            }
        };
        let points: Vec<[f32; 2]> = points.iter().map(|p| [p.y(), p.x()]).collect();
        let Some(first) = points.first().copied() else {
            return;
        };
        let Some(last) = points.last().copied() else {
            return;
        };
        let area_mm2 = Annotation::compute_polygon_area(&points, spacing);
        let perimeter_mm = Annotation::compute_polyline_length(&points, spacing)
            + Annotation::compute_length(last, first, spacing);
        if !area_mm2.is_finite() || !perimeter_mm.is_finite() {
            self.status_message = "Polygon measurement exceeds the supported range".into();
            return;
        }
        match kind {
            PolygonKind::Closed => self.push_annotation(Annotation::ClosedPolygon {
                location: None,
                points,
                area_mm2,
                perimeter_mm,
            }),
        }
    }

    pub(super) fn push_annotation(&mut self, annotation: Annotation) {
        self.annotations
            .push(annotation.with_location(self.annotation_location()));
    }

    pub(super) fn annotation_location(&self) -> AnnotationLocation {
        AnnotationLocation {
            plane: self.axis,
            slice: self.axis_slice_info(self.axis).0,
        }
    }

    /// Record the physical length between two points on the current image.
    pub(super) fn record_length(&mut self, start: ImagePoint, end: ImagePoint) {
        let p1 = [start.y(), start.x()];
        let p2 = [end.y(), end.x()];
        let measured = self
            .slice_plane_spacing()
            .and_then(|spacing| Annotation::compute_length_checked(p1, p2, spacing));
        match measured {
            Ok(length_mm) => {
                self.push_annotation(Annotation::Length {
                    plane: self.axis,
                    location: None,
                    p1,
                    p2,
                    length_mm,
                });
                self.status_message = format!("Length: {length_mm:.2} mm");
            }
            Err(error) => {
                self.status_message = format!("Measurement rejected: {error}");
            }
        }
    }

    /// Record a one-drag deviation from the nearer image axis.
    pub(super) fn finalise_deviation(&mut self, start: ImagePoint, end: ImagePoint) {
        let spacing = match self
            .slice_plane_spacing()
            .and_then(Annotation::validate_spacing)
        {
            Ok(spacing) => spacing,
            Err(error) => {
                self.status_message = format!("Measurement rejected: {error}");
                return;
            }
        };
        let start = [start.y(), start.x()];
        let end = [end.y(), end.x()];
        let (reference, distance_mm, angle_deg) =
            match Annotation::compute_deviation_with_angle(start, end, spacing) {
                Ok(measured) => measured,
                Err(error) => {
                    self.status_message = format!("Measurement rejected: {error}");
                    return;
                }
            };
        self.push_annotation(Annotation::Deviation {
            location: None,
            start,
            end,
            reference,
            distance_mm,
            angle_deg,
        });
        self.status_message = format!("Deviation: {distance_mm:.2} mm at {angle_deg:.1} deg");
    }

    /// Compute ROI rect statistics for the pixel region between `start` and
    /// `end` (screen-space corners) on the current primary-axis slice.
    pub(super) fn finalise_roi_rect(&mut self, start: ImagePoint, end: ImagePoint) {
        let spacing = match self.slice_plane_spacing() {
            Ok(spacing) => spacing,
            Err(error) => {
                self.status_message = format!("Measurement rejected: {error}");
                return;
            }
        };
        let Some(vol) = &self.loaded else { return };
        let p1 = [start.y(), start.x()];
        let p2 = [end.y(), end.x()];
        let (pixels, width, height) =
            vol.extract_slice(self.axis, self.axis_slice_info(self.axis).0);
        let (mean, std_dev, min, max, area_mm2) = match Annotation::compute_roi_rect_stats_checked(
            p1, p2, &pixels, width, height, spacing,
        ) {
            Ok(stats) => stats,
            Err(error) => {
                self.status_message = format!("Measurement rejected: {error}");
                return;
            }
        };
        self.push_annotation(Annotation::RoiRect {
            location: None,
            top_left: [p1[0].min(p2[0]), p1[1].min(p2[1])],
            bottom_right: [p1[0].max(p2[0]), p1[1].max(p2[1])],
            mean,
            std_dev,
            min,
            max,
            area_mm2,
        });
        self.status_message =
            format!("ROI: \u{03bc}={mean:.1} \u{03c3}={std_dev:.1} [{min:.0}, {max:.0}] {area_mm2:.1} mm\u{b2}");
    }

    pub(super) fn finalise_roi_ellipse(&mut self, start: ImagePoint, end: ImagePoint) {
        let spacing = match self.slice_plane_spacing() {
            Ok(spacing) => spacing,
            Err(error) => {
                self.status_message = format!("Measurement rejected: {error}");
                return;
            }
        };
        let Some(vol) = &self.loaded else { return };
        let p1 = [start.y(), start.x()];
        let p2 = [end.y(), end.x()];
        let (pixels, width, height) =
            vol.extract_slice(self.axis, self.axis_slice_info(self.axis).0);
        let (center, radii, mean, std_dev, min, max, area_mm2) =
            match Annotation::compute_roi_ellipse_stats_checked(
                p1, p2, &pixels, width, height, spacing,
            ) {
                Ok(stats) => stats,
                Err(error) => {
                    self.status_message = format!("Measurement rejected: {error}");
                    return;
                }
            };
        self.push_annotation(Annotation::RoiEllipse {
            location: None,
            center,
            radii,
            mean,
            std_dev,
            min,
            max,
            area_mm2,
        });
        self.status_message = format!(
            "Ellipse ROI: \u{03bc}={mean:.1} \u{03c3}={std_dev:.1} [{min:.0}, {max:.0}] {area_mm2:.1} mm\u{b2}"
        );
    }

    /// Per-axis 2-D pixel spacing `[row_spacing, col_spacing]` in mm/pixel.
    ///
    /// | axis | row spacing | col spacing |
    /// |------|-------------|-------------|
    /// | 0 axial   | dy | dx |
    /// | 1 coronal | dz | dx |
    /// | 2 sagittal| dz | dy |
    pub(crate) fn slice_plane_spacing(&self) -> Result<[f64; 2], MeasurementError> {
        let Some(vol) = &self.loaded else {
            return Err(MeasurementError::MissingVolume);
        };
        let [dz, dy, dx] = vol.spacing;
        let spacing = match self.axis {
            0 => [dy, dx],
            1 => [dz, dx],
            _ => [dz, dy],
        };
        Annotation::validate_spacing(spacing)?;
        Ok(spacing)
    }
}
