//! Persisted annotation data definitions.

use super::patient_length::PatientLength;

/// Optional creation location for a completed annotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AnnotationLocation {
    /// Orthogonal pane axis.
    pub plane: usize,
    /// Zero-based slice index within the plane.
    pub slice: usize,
}

/// A completed measurement or drawing annotation stored on a viewport.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum Annotation {
    /// Straight-line distance between two image points.
    Length {
        /// Orthogonal pane axis.
        #[serde(default)]
        plane: usize,
        /// Creation location.
        #[serde(default)]
        location: Option<AnnotationLocation>,
        /// Start point as `[row, col]` pixels.
        p1: [f32; 2],
        /// End point as `[row, col]` pixels.
        p2: [f32; 2],
        /// Physical distance in millimetres.
        length_mm: f32,
    },
    /// Straight-line distance between two DICOM patient-space points.
    PatientLength {
        /// Orthogonal pane axis.
        #[serde(default)]
        plane: usize,
        /// Creation location.
        #[serde(default)]
        location: Option<AnnotationLocation>,
        /// Validated patient-space length.
        length: PatientLength,
    },
    /// Angle at vertex `p2` between two image-space rays.
    Angle {
        /// Creation location.
        #[serde(default)]
        location: Option<AnnotationLocation>,
        /// First ray endpoint.
        p1: [f32; 2],
        /// Vertex.
        p2: [f32; 2],
        /// Second ray endpoint.
        p3: [f32; 2],
        /// Included angle in degrees.
        angle_deg: f32,
    },
    /// Axis-aligned rectangle ROI with intensity statistics.
    RoiRect {
        /// Creation location.
        #[serde(default)]
        location: Option<AnnotationLocation>,
        /// Top-left corner.
        top_left: [f32; 2],
        /// Bottom-right corner.
        bottom_right: [f32; 2],
        /// Mean intensity.
        mean: f32,
        /// Population standard deviation.
        std_dev: f32,
        /// Minimum intensity.
        min: f32,
        /// Maximum intensity.
        max: f32,
        /// Physical area in square millimetres.
        area_mm2: f32,
    },
    /// Ellipse ROI with intensity statistics.
    RoiEllipse {
        /// Creation location.
        #[serde(default)]
        location: Option<AnnotationLocation>,
        /// Centre point.
        center: [f32; 2],
        /// Semi-axis radii.
        radii: [f32; 2],
        /// Mean intensity.
        mean: f32,
        /// Population standard deviation.
        std_dev: f32,
        /// Minimum intensity.
        min: f32,
        /// Maximum intensity.
        max: f32,
        /// Physical area in square millimetres.
        area_mm2: f32,
    },
    /// Single-point HU measurement.
    HuPoint {
        /// Creation location.
        #[serde(default)]
        location: Option<AnnotationLocation>,
        /// Point as `[row, col]` pixels.
        pos: [f32; 2],
        /// Intensity value.
        value: f32,
    },
    /// Directional arrow with image-space tail and head points.
    Arrow {
        /// Creation location.
        #[serde(default)]
        location: Option<AnnotationLocation>,
        /// Tail as `[row, col]` pixels.
        tail: [f32; 2],
        /// Head as `[row, col]` pixels.
        head: [f32; 2],
    },
    /// Freehand pencil path in image-space points.
    Pencil {
        /// Creation location.
        #[serde(default)]
        location: Option<AnnotationLocation>,
        /// Path vertices as `[row, col]` pixels.
        points: Vec<[f32; 2]>,
    },
}
