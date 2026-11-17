//! Placement and spacing of image-space annotations.

use super::{Annotation, AnnotationLocation};

impl Annotation {
    pub(crate) fn visible_on_image(&self, image: AnnotationLocation) -> bool {
        match self.location() {
            Some(recorded) => recorded == image,
            None => matches!(
                self,
                Self::Length { plane, .. } | Self::PatientLength { plane, .. } if *plane == image.plane
            ),
        }
    }

    pub(crate) fn location(&self) -> Option<AnnotationLocation> {
        match self {
            Self::Length { location, .. }
            | Self::PatientLength { location, .. }
            | Self::Angle { location, .. }
            | Self::RoiRect { location, .. }
            | Self::RoiEllipse { location, .. }
            | Self::HuPoint { location, .. } => *location,
            Self::ClosedPolygon { location, .. }
            | Self::OpenPolygon { location, .. }
            | Self::CobbAngle { location, .. }
            | Self::Deviation { location, .. }
            | Self::Arrow { location, .. }
            | Self::Pencil { location, .. } => *location,
        }
    }

    /// Associates a newly created annotation with its source pane location.
    pub(crate) fn with_location(mut self, location: AnnotationLocation) -> Self {
        match &mut self {
            Self::Length {
                location: value, ..
            }
            | Self::PatientLength {
                location: value, ..
            }
            | Self::Angle {
                location: value, ..
            }
            | Self::RoiRect {
                location: value, ..
            }
            | Self::RoiEllipse {
                location: value, ..
            }
            | Self::HuPoint {
                location: value, ..
            } => *value = Some(location),
            Self::ClosedPolygon {
                location: value, ..
            }
            | Self::OpenPolygon {
                location: value, ..
            }
            | Self::CobbAngle {
                location: value, ..
            }
            | Self::Deviation {
                location: value, ..
            }
            | Self::Arrow {
                location: value, ..
            }
            | Self::Pencil {
                location: value, ..
            } => *value = Some(location),
        }
        self
    }

    /// Compute the angle between rays scaled by the row and column pixel spacing.
    pub fn compute_angle_with_spacing(
        p1: [f32; 2],
        p2: [f32; 2],
        p3: [f32; 2],
        spacing: [f32; 2],
    ) -> f32 {
        let ray = |point: [f32; 2]| {
            [
                (point[0] - p2[0]) * spacing[0],
                (point[1] - p2[1]) * spacing[1],
            ]
        };
        let v1 = ray(p1);
        let v2 = ray(p3);
        let dot = v1[0] * v2[0] + v1[1] * v2[1];
        let mag1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt();
        let mag2 = (v2[0] * v2[0] + v2[1] * v2[1]).sqrt();
        if mag1 < f32::EPSILON || mag2 < f32::EPSILON {
            // Degenerate: one or both rays have zero length; angle is undefined.
            return 0.0;
        }
        let cos_theta = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
        cos_theta.acos().to_degrees()
    }
}
