//! Completed measurement annotations with computed physical values.
//!
//! # Coordinate system
//!
//! Orthogonal image-space coordinates are stored as `[row, col]` in pixel
//! units. Physical distances are computed by multiplying pixel deltas by the
//! corresponding voxel spacing components. Oblique lengths instead retain
//! validated `[x, y, z]` endpoints in DICOM patient millimetres.
//!
//! # Mathematical specifications
//!
//! ## Length (Euclidean distance in physical space)
//!
//! Given two image points p₁ = (r₁, c₁) and p₂ = (r₂, c₂) and pixel spacing
//! s = (s_r, s_c) in mm/pixel:
//!
//! ```text
//! length_mm = √( ((r₂ − r₁) · s_r)² + ((c₂ − c₁) · s_c)² )
//! ```
//!
//! ## Angle (at vertex p₂, between vectors p₂→p₁ and p₂→p₃)
//!
//! ```text
//! v₁ = p₁ − p₂, v₂ = p₃ − p₂
//! cos θ = (v₁ · v₂) / (|v₁| · |v₂|)
//! θ = arccos(clamp(cos θ, −1, 1)) [degrees]
//! ```
//!
//! ## ROI rectangle statistics
//!
//! Pixels whose row index lies in [min_r, max_r] and column index lies in
//! [min_c, max_c] (inclusive, integer bounds derived from p₁ and p₂) are
//! collected into a sample set S.
//!
//! ```text
//! mean = (1/|S|) Σ v
//! std_dev = √( (1/|S|) Σ (v − mean)² ) [population std dev]
//! min = min S
//! max = max S
//! area = (max_r − min_r + 1) · s_r × (max_c − min_c + 1) · s_c [mm²]
//! ```

mod paths;
mod patient_length;
mod regions;

pub use patient_length::PatientLength;

/// Failure raised when physical measurement inputs or results cannot be
/// represented without changing their meaning.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum MeasurementError {
    /// A spacing component is not finite and strictly positive.
    #[error("voxel spacing component {index} is not finite and positive: {value}")]
    InvalidSpacing { index: usize, value: f64 },
    /// A valid physical spacing component cannot be represented by the
    /// f32-backed annotation storage used by the viewer.
    #[error("voxel spacing component {index} cannot be represented as f32: {value}")]
    UnrepresentableSpacing { index: usize, value: f64 },
    /// A computed physical value overflowed or became non-finite.
    #[error("{kind} measurement result is not finite")]
    NonFiniteResult { kind: &'static str },
    /// A measurement was requested without a loaded volume.
    #[error("measurement requires a loaded volume")]
    MissingVolume,
}

// ── Completed annotations ─────────────────────────────────────────────────────

/// Optional creation location for a completed annotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AnnotationLocation {
    /// Orthogonal pane axis.
    pub plane: usize,
    /// Zero-based slice index within the plane.
    pub slice: usize,
}

/// A completed measurement annotation stored on a viewport.
///
/// Orthogonal positions are stored as `[row, col]` in image pixel coordinates.
/// Patient-space lengths retain `[x, y, z]` millimetre endpoints. Computed
/// values use physical units (mm, degrees, HU) and remain derivable from the
/// stored coordinates and applicable geometry.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum Annotation {
    /// Straight-line distance between two image points.
    Length {
        /// Orthogonal pane axis (`0` axial, `1` coronal, `2` sagittal) the
        /// gesture was placed on; the overlay draws the length on that pane.
        #[serde(default)]
        plane: usize,
        /// Creation location; absent on records written before slice anchoring.
        #[serde(default)]
        location: Option<AnnotationLocation>,
        /// Start point as `[row, col]` in image pixels.
        p1: [f32; 2],
        /// End point as `[row, col]` in image pixels.
        p2: [f32; 2],
        /// Euclidean distance in millimetres.
        length_mm: f32,
    },

    /// Straight-line distance between two points in DICOM patient space.
    PatientLength {
        /// Orthogonal pane axis the measurement belongs to; the overlay
        /// projects the patient endpoints onto that pane.
        #[serde(default)]
        plane: usize,
        /// Creation location; absent on records written before slice anchoring.
        #[serde(default)]
        location: Option<AnnotationLocation>,
        /// Validated patient-space endpoints and distance.
        length: PatientLength,
    },

    /// Angle at vertex `p2` between the rays `p2→p1` and `p2→p3`.
    Angle {
        /// Creation location; absent on legacy records.
        #[serde(default)]
        location: Option<AnnotationLocation>,
        /// First ray endpoint as `[row, col]` in image pixels.
        p1: [f32; 2],
        /// Vertex as `[row, col]` in image pixels.
        p2: [f32; 2],
        /// Second ray endpoint as `[row, col]` in image pixels.
        p3: [f32; 2],
        /// Included angle in degrees.
        angle_deg: f32,
    },

    /// Axis-aligned rectangle ROI with intensity statistics.
    RoiRect {
        /// Creation location; absent on legacy records.
        #[serde(default)]
        location: Option<AnnotationLocation>,
        /// Top-left corner as `[row, col]` in image pixels.
        top_left: [f32; 2],
        /// Bottom-right corner as `[row, col]` in image pixels.
        bottom_right: [f32; 2],
        /// Mean intensity (HU or relative) within the ROI.
        mean: f32,
        /// Population standard deviation of intensities within the ROI.
        std_dev: f32,
        /// Minimum intensity within the ROI.
        min: f32,
        /// Maximum intensity within the ROI.
        max: f32,
        /// ROI area in mm².
        area_mm2: f32,
    },

    /// Ellipse ROI with intensity statistics computed over pixels whose centres
    /// lie inside the ellipse mask.
    ///
    /// # Ellipse membership condition
    ///
    /// A pixel at integer coordinates `(r, c)` is included when:
    ///
    /// ```text
    /// ((r − center[0]) / radii[0])² + ((c − center[1]) / radii[1])² ≤ 1
    /// ```
    ///
    /// When `radii[0]` or `radii[1]` is zero (degenerate ellipse), no pixels
    /// are collected and all statistics return `0.0`.
    ///
    /// # Physical area
    ///
    /// ```text
    /// area_mm² = π × radii[0] × spacing[0] × radii[1] × spacing[1]
    /// ```
    RoiEllipse {
        /// Creation location; absent on legacy records.
        #[serde(default)]
        location: Option<AnnotationLocation>,
        /// Centre of the ellipse as `[row, col]` in image pixels.
        center: [f32; 2],
        /// Semi-axis radii as `[row_radius, col_radius]` in image pixels.
        radii: [f32; 2],
        /// Mean intensity within the ellipse mask.
        mean: f32,
        /// Population standard deviation of intensities within the ellipse mask.
        std_dev: f32,
        /// Minimum intensity within the ellipse mask.
        min: f32,
        /// Maximum intensity within the ellipse mask.
        max: f32,
        /// Physical area of the ellipse in mm².
        area_mm2: f32,
    },

    /// Single-point HU measurement.
    HuPoint {
        /// Creation location; absent on legacy records.
        #[serde(default)]
        location: Option<AnnotationLocation>,
        /// Point location as `[row, col]` in image pixels.
        pos: [f32; 2],
        /// Intensity value at the point (HU or relative).
        value: f32,
    },

    /// Closed polygon ROI with physical area.
    ClosedPolygon {
        location: Option<AnnotationLocation>,
        points: Vec<[f32; 2]>,
        area_mm2: f32,
        perimeter_mm: f32,
    },
    /// Open polyline annotation.
    OpenPolygon {
        location: Option<AnnotationLocation>,
        points: Vec<[f32; 2]>,
        length_mm: f32,
    },
    /// Two-line Cobb angle, in degrees.
    CobbAngle {
        location: Option<AnnotationLocation>,
        p1: [f32; 2],
        p2: [f32; 2],
        p3: [f32; 2],
        p4: [f32; 2],
        angle_deg: f32,
    },
    /// Offset and angle of a segment from the nearer image axis.
    Deviation {
        location: Option<AnnotationLocation>,
        start: [f32; 2],
        end: [f32; 2],
        reference: [f32; 2],
        distance_mm: f32,
        angle_deg: f32,
    },
    /// Directional arrow annotation.
    Arrow {
        location: Option<AnnotationLocation>,
        tail: [f32; 2],
        head: [f32; 2],
    },
    /// Freehand pencil annotation.
    Pencil {
        location: Option<AnnotationLocation>,
        points: Vec<[f32; 2]>,
    },
}

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

    /// Validate physical spacing and its conversion to annotation storage.
    ///
    /// The viewer persists derived values as `f32`; accepting a finite `f64`
    /// that narrows to zero or infinity would create an invalid displayed
    /// measurement, so the boundary rejects it explicitly.
    pub fn validate_spacing(spacing: [f64; 2]) -> Result<[f32; 2], MeasurementError> {
        let mut converted = [0.0_f32; 2];
        for (index, value) in spacing.into_iter().enumerate() {
            if !value.is_finite() || value <= 0.0 {
                return Err(MeasurementError::InvalidSpacing { index, value });
            }
            let narrowed = value as f32;
            if !narrowed.is_finite() || narrowed <= 0.0 {
                return Err(MeasurementError::UnrepresentableSpacing { index, value });
            }
            converted[index] = narrowed;
        }
        Ok(converted)
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
