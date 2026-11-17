//! Completed annotations and their source images.

use super::PatientLength;

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
