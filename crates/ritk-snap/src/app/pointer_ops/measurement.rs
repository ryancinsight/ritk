//! Completed measurements and slice geometry.

use super::SnapApp;
use crate::tools::interaction::{Annotation, AnnotationLocation, ImagePoint, MeasurementError};

impl SnapApp {
    pub(super) fn finalise_polygon(
        &mut self,
        points: Vec<ImagePoint>,
        kind: crate::tools::interaction::PolygonKind,
    ) {
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
        let converted: Vec<[f32; 2]> = points.iter().map(|p| [p.y(), p.x()]).collect();
        match kind {
            crate::tools::interaction::PolygonKind::Closed => {
                let area_mm2 = Annotation::compute_polygon_area(&converted, spacing);
                let Some(first) = converted.first() else {
                    return;
                };
                let Some(last) = converted.last() else { return };
                let perimeter_mm = Annotation::compute_polyline_length(&converted, spacing)
                    + Annotation::compute_length(*last, *first, spacing);
                if !area_mm2.is_finite() || !perimeter_mm.is_finite() {
                    self.status_message = "Polygon measurement exceeds the supported range".into();
                    return;
                }
                self.push_annotation(Annotation::ClosedPolygon {
                    location: None,
                    points: converted,
                    area_mm2,
                    perimeter_mm,
                });
            }
            crate::tools::interaction::PolygonKind::Open => {
                let length_mm = Annotation::compute_polyline_length(&converted, spacing);
                if !length_mm.is_finite() {
                    self.status_message = "Path measurement exceeds the supported range".into();
                    return;
                }
                self.push_annotation(Annotation::OpenPolygon {
                    location: None,
                    points: converted,
                    length_mm,
                });
            }
        }
    }

    pub(super) fn finalise_cobb(&mut self, points: &[ImagePoint]) {
        let Ok(spacing) = self
            .slice_plane_spacing()
            .and_then(Annotation::validate_spacing)
        else {
            return;
        };
        let p: Vec<[f32; 2]> = points.iter().map(|v| [v.y(), v.x()]).collect();
        let angle_deg = Annotation::compute_cobb_angle(p[0], p[1], p[2], p[3], spacing);
        self.push_annotation(Annotation::CobbAngle {
            location: None,
            p1: p[0],
            p2: p[1],
            p3: p[2],
            p4: p[3],
            angle_deg,
        });
    }

    pub(super) fn finalise_deviation(&mut self, from: ImagePoint, to: ImagePoint) {
        let Ok(spacing) = self
            .slice_plane_spacing()
            .and_then(Annotation::validate_spacing)
        else {
            return;
        };
        let start = [from.y(), from.x()];
        let end = [to.y(), to.x()];
        let (reference, distance_mm, angle_deg) =
            Annotation::compute_deviation_with_angle(start, end, spacing);
        self.push_annotation(Annotation::Deviation {
            location: None,
            start,
            end,
            reference,
            distance_mm,
            angle_deg,
        });
    }

    pub(super) fn push_annotation(&mut self, annotation: Annotation) {
        self.annotations
            .push(annotation.with_location(self.annotation_location()));
    }

    pub(in crate::app) fn annotation_location(&self) -> AnnotationLocation {
        AnnotationLocation {
            plane: self.axis,
            slice: self.axis_slice_info(self.axis).0,
        }
    }

    /// Record the physical length between two points on the current image.
    pub(super) fn record_length(&mut self, start: ImagePoint, end: ImagePoint) {
        let p1 = [start.y(), start.x()];
        let p2 = [end.y(), end.x()];
        let measured = self.measured_length(p1, p2);
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

    pub(crate) fn measured_length(
        &self,
        p1: [f32; 2],
        p2: [f32; 2],
    ) -> Result<f32, MeasurementError> {
        let measured = self
            .slice_plane_spacing()
            .and_then(|spacing| Annotation::compute_length_checked(p1, p2, spacing))?;
        let location = self.annotation_location();
        self.length_calibrations
            .iter()
            .find(|calibration| calibration.applies_to(location))
            .map_or(Some(measured), |calibration| calibration.apply(measured))
            .ok_or(MeasurementError::NonFiniteResult {
                kind: "calibrated length",
            })
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
