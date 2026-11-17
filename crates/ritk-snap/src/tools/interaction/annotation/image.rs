//! Placement and spacing of image-space annotations.

use super::{Annotation, AnnotationLocation, MeasurementError};

impl Annotation {
    pub(crate) fn location(&self) -> Option<AnnotationLocation> {
        match self {
            Self::Length { location, .. }
            | Self::PatientLength { location, .. }
            | Self::Angle { location, .. }
            | Self::Deviation { location, .. }
            | Self::RoiRect { location, .. }
            | Self::RoiEllipse { location, .. }
            | Self::HuPoint { location, .. }
            | Self::ClosedPolygon { location, .. }
            | Self::OpenPolygon { location, .. } => *location,
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
            | Self::Deviation {
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
            }
            | Self::ClosedPolygon {
                location: value, ..
            }
            | Self::OpenPolygon {
                location: value, ..
            } => *value = Some(location),
        }
        self
    }

    /// Compute a finite length after validating physical spacing.
    pub fn compute_length_checked(
        p1: [f32; 2],
        p2: [f32; 2],
        spacing: [f64; 2],
    ) -> Result<f32, MeasurementError> {
        let spacing = Self::validate_spacing(spacing)?;
        let result = Self::compute_length(p1, p2, spacing);
        result
            .is_finite()
            .then_some(result)
            .ok_or(MeasurementError::NonFiniteResult { kind: "length" })
    }

    /// Compute the Euclidean distance between two image points in physical space.
    ///
    /// # Parameters
    /// - `p1`, `p2` — image coordinates `[row, col]` in pixels.
    /// - `spacing` — pixel spacing `[row_spacing, col_spacing]` in mm/pixel.
    ///
    /// # Formula
    /// ```text
    /// length = √( ((p2[0] − p1[0]) · spacing[0])²
    ///            + ((p2[1] − p1[1]) · spacing[1])² )
    /// ```
    pub fn compute_length(p1: [f32; 2], p2: [f32; 2], spacing: [f32; 2]) -> f32 {
        let dr = (p2[0] - p1[0]) * spacing[0];
        let dc = (p2[1] - p1[1]) * spacing[1];
        (dr * dr + dc * dc).sqrt()
    }

    /// Compute the physical area of a closed polygon by the shoelace formula.
    pub fn compute_polygon_area(points: &[[f32; 2]], spacing: [f32; 2]) -> f32 {
        let [origin, vertices @ ..] = points else {
            return 0.0;
        };
        let sum = vertices
            .windows(2)
            .map(|pair| {
                let [a, b] = pair else {
                    unreachable!("invariant: windows(2) contains two vertices")
                };
                let ar = a[0] - origin[0];
                let ac = a[1] - origin[1];
                let br = b[0] - origin[0];
                let bc = b[1] - origin[1];
                ar * bc - br * ac
            })
            .sum::<f32>();
        0.5 * sum.abs() * spacing[0] * spacing[1]
    }

    /// Compute the physical length of a polyline.
    pub fn compute_polyline_length(points: &[[f32; 2]], spacing: [f32; 2]) -> f32 {
        points
            .windows(2)
            .map(|pair| Self::compute_length(pair[0], pair[1], spacing))
            .sum()
    }

    /// Compute the angle at vertex `p2` between the rays `p2→p1` and `p2→p3`.
    ///
    /// Returns the angle in degrees in the range `[0°, 180°]`.
    ///
    /// Returns `0.0` when either input ray has zero length (degenerate case:
    /// two coincident points), avoiding division by zero without panicking.
    ///
    /// # Formula
    /// ```text
    /// v₁ = p₁ − p₂, v₂ = p₃ − p₂
    /// cos θ = clamp( (v₁ · v₂) / (|v₁| · |v₂|), −1, 1 )
    /// θ = arccos(cos θ) [converted to degrees]
    /// ```
    pub fn compute_angle(p1: [f32; 2], p2: [f32; 2], p3: [f32; 2]) -> f32 {
        Self::compute_angle_with_spacing(p1, p2, p3, [1.0, 1.0])
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
