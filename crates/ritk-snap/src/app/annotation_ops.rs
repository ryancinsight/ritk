//! Completed measurements: computing their values and recording them.

use super::state::SnapApp;
use crate::tools::interaction::{Annotation, ImagePoint};

impl SnapApp {
    /// Record the patient-space length between two image points.
    pub(super) fn record_length(&mut self, start: ImagePoint, end: ImagePoint) {
        let p1 = [start.y(), start.x()];
        let p2 = [end.y(), end.x()];
        let measured = self
            .slice_plane_spacing()
            .and_then(|spacing| Annotation::compute_length_checked(p1, p2, spacing));
        match measured {
            Ok(length_mm) => {
                self.annotations
                    .push(Annotation::Length { p1, p2, length_mm });
                self.status_message = format!("Length: {length_mm:.2} mm");
            }
            Err(error) => {
                self.status_message = format!("Measurement rejected: {error}");
            }
        }
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
        let (pixels, width, height) = vol.extract_slice(self.axis, self.viewer_state.slice_index);
        let (mean, std_dev, min, max, area_mm2) = match Annotation::compute_roi_rect_stats_checked(
            p1, p2, &pixels, width, height, spacing,
        ) {
            Ok(stats) => stats,
            Err(error) => {
                self.status_message = format!("Measurement rejected: {error}");
                return;
            }
        };
        self.annotations.push(Annotation::RoiRect {
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
        let (pixels, width, height) = vol.extract_slice(self.axis, self.viewer_state.slice_index);
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
        self.annotations.push(Annotation::RoiEllipse {
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
}
