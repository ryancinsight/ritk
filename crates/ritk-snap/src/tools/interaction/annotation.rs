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

mod image;
mod patient_length;

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

    /// Closed polygon ROI with physical area and perimeter.
    ClosedPolygon {
        /// Creation location on the source image plane.
        #[serde(default)]
        location: Option<AnnotationLocation>,
        /// Polygon vertices as `[row, col]` image pixels.
        points: Vec<[f32; 2]>,
        /// Physical enclosed area in square millimetres.
        area_mm2: f32,
        /// Physical perimeter in millimetres.
        perimeter_mm: f32,
    },
}

impl Annotation {
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

    /// Compute intensity statistics for pixels within the axis-aligned rectangle
    /// defined by corners `p1` and `p2` (order-independent).
    ///
    /// # Parameters
    /// - `p1`, `p2` — opposite corners of the ROI as `[row, col]` in pixels.
    /// - `pixels` — flat row-major pixel buffer of the slice.
    /// - `width` — number of columns in the slice.
    /// - `height` — number of rows in the slice.
    /// - `spacing` — pixel spacing `[row_spacing, col_spacing]` in mm/pixel.
    ///
    /// # Returns
    /// `(mean, std_dev, min, max, area_mm2)` using population statistics.
    ///
    /// Returns `(0.0, 0.0, 0.0, 0.0, 0.0)` when the ROI contains no pixels
    /// (e.g., the rectangle lies fully outside the image bounds).
    ///
    /// # Formula
    /// ```text
    /// S = { pixels[r × width + c] | r ∈ [min_r, max_r], c ∈ [min_c, max_c] }
    /// mean = Σ S / |S|
    /// std_dev = √( Σ (v − mean)² / |S| )
    /// area = (max_r − min_r + 1) · spacing[0]
    ///       × (max_c − min_c + 1) · spacing[1]
    /// ```
    pub fn compute_roi_rect_stats(
        p1: [f32; 2],
        p2: [f32; 2],
        pixels: &[f32],
        width: usize,
        height: usize,
        spacing: [f32; 2],
    ) -> (f32, f32, f32, f32, f32) {
        // Derive integer row/col bounds from the two corner points, clamped to
        // the valid image extent.
        let r_min_f = p1[0].min(p2[0]).floor();
        let r_max_f = p1[0].max(p2[0]).ceil();
        let c_min_f = p1[1].min(p2[1]).floor();
        let c_max_f = p1[1].max(p2[1]).ceil();

        // Guard against negative coordinates before casting to usize.
        let r_min = (r_min_f.max(0.0) as usize).min(height.saturating_sub(1));
        let r_max = (r_max_f.max(0.0) as usize).min(height.saturating_sub(1));
        let c_min = (c_min_f.max(0.0) as usize).min(width.saturating_sub(1));
        let c_max = (c_max_f.max(0.0) as usize).min(width.saturating_sub(1));

        // Collect valid sample values.
        let mut vals: Vec<f32> = Vec::with_capacity((r_max - r_min + 1) * (c_max - c_min + 1));
        for r in r_min..=r_max {
            for c in c_min..=c_max {
                let idx = r * width + c;
                if idx < pixels.len() {
                    vals.push(pixels[idx]);
                }
            }
        }

        if vals.is_empty() {
            return (0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let n = vals.len() as f32;
        let mean = vals.iter().copied().sum::<f32>() / n;
        // Population standard deviation.
        let variance = vals.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let std_dev = variance.sqrt();
        let min = vals.iter().copied().fold(f32::INFINITY, f32::min);
        let max = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Physical area: integer pixel counts multiplied by per-axis spacing.
        let h_px = (r_max - r_min + 1) as f32;
        let w_px = (c_max - c_min + 1) as f32;
        let area_mm2 = h_px * spacing[0] * w_px * spacing[1];

        (mean, std_dev, min, max, area_mm2)
    }

    /// Compute rectangle statistics after validating physical spacing and the
    /// f32-backed derived area.
    pub fn compute_roi_rect_stats_checked(
        p1: [f32; 2],
        p2: [f32; 2],
        pixels: &[f32],
        width: usize,
        height: usize,
        spacing: [f64; 2],
    ) -> Result<(f32, f32, f32, f32, f32), MeasurementError> {
        let spacing = Self::validate_spacing(spacing)?;
        let result = Self::compute_roi_rect_stats(p1, p2, pixels, width, height, spacing);
        let finite = result.0.is_finite()
            && result.1.is_finite()
            && result.2.is_finite()
            && result.3.is_finite()
            && result.4.is_finite();
        finite
            .then_some(result)
            .ok_or(MeasurementError::NonFiniteResult {
                kind: "rectangle ROI",
            })
    }

    /// Compute intensity statistics for pixels whose centres lie inside the
    /// ellipse defined by two opposite corners `p1` and `p2`.
    ///
    /// # Parameters
    /// - `p1`, `p2` — opposite corners of the bounding rectangle as `[row, col]`.
    /// - `pixels` — flat row-major pixel buffer of the slice.
    /// - `width` — number of columns in the slice.
    /// - `height` — number of rows in the slice.
    /// - `spacing` — pixel spacing `[row_spacing, col_spacing]` in mm/pixel.
    ///
    /// # Returns
    /// `(center, radii, mean, std_dev, min, max, area_mm2)`.
    ///
    /// Returns `([0,0], [0,0], 0, 0, 0, 0, 0)` when the ellipse contains no
    /// pixels (degenerate or fully outside the image bounds).
    ///
    /// # Ellipse membership
    ///
    /// ```text
    /// cy = (p1[0] + p2[0]) / 2
    /// cx = (p1[1] + p2[1]) / 2
    /// a = |p2[0] − p1[0]| / 2   (row semi-axis)
    /// b = |p2[1] − p1[1]| / 2   (col semi-axis)
    ///
    /// pixel (r,c) is inside ⟺ ((r−cy)/a)² + ((c−cx)/b)² ≤ 1
    ///
    /// area_mm² = π × a × spacing[0] × b × spacing[1]
    /// ```
    pub fn compute_roi_ellipse_stats(
        p1: [f32; 2],
        p2: [f32; 2],
        pixels: &[f32],
        width: usize,
        height: usize,
        spacing: [f32; 2],
    ) -> ([f32; 2], [f32; 2], f32, f32, f32, f32, f32) {
        let cy = (p1[0] + p2[0]) * 0.5;
        let cx = (p1[1] + p2[1]) * 0.5;
        let a = (p2[0] - p1[0]).abs() * 0.5; // row semi-axis
        let b = (p2[1] - p1[1]).abs() * 0.5; // col semi-axis
        let center = [cy, cx];
        let radii = [a, b];

        // Degenerate ellipse: one or both semi-axes are zero.
        if a < f32::EPSILON || b < f32::EPSILON {
            return (center, radii, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        // Scan the bounding box and apply the ellipse membership test.
        let r_min = ((cy - a).floor().max(0.0) as usize).min(height.saturating_sub(1));
        let r_max = ((cy + a).ceil().max(0.0) as usize).min(height.saturating_sub(1));
        let c_min = ((cx - b).floor().max(0.0) as usize).min(width.saturating_sub(1));
        let c_max = ((cx + b).ceil().max(0.0) as usize).min(width.saturating_sub(1));

        let mut vals: Vec<f32> = Vec::with_capacity((r_max - r_min + 1) * (c_max - c_min + 1));
        for r in r_min..=r_max {
            let dr = (r as f32 - cy) / a;
            for c in c_min..=c_max {
                let dc = (c as f32 - cx) / b;
                // Membership: (dr)² + (dc)² ≤ 1
                if dr * dr + dc * dc <= 1.0 {
                    let idx = r * width + c;
                    if idx < pixels.len() {
                        vals.push(pixels[idx]);
                    }
                }
            }
        }

        if vals.is_empty() {
            return (center, radii, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let n = vals.len() as f32;
        let mean = vals.iter().copied().sum::<f32>() / n;
        let variance = vals.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let std_dev = variance.sqrt();
        let min = vals.iter().copied().fold(f32::INFINITY, f32::min);
        let max = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Physical area of the ellipse: π × a × dr × b × dc
        let area_mm2 = std::f32::consts::PI * a * spacing[0] * b * spacing[1];

        (center, radii, mean, std_dev, min, max, area_mm2)
    }

    /// Compute ellipse statistics after validating physical spacing and the
    /// f32-backed derived area.
    #[expect(
        clippy::type_complexity,
        reason = "the checked API mirrors the established ellipse statistics tuple"
    )]
    pub fn compute_roi_ellipse_stats_checked(
        p1: [f32; 2],
        p2: [f32; 2],
        pixels: &[f32],
        width: usize,
        height: usize,
        spacing: [f64; 2],
    ) -> Result<([f32; 2], [f32; 2], f32, f32, f32, f32, f32), MeasurementError> {
        let spacing = Self::validate_spacing(spacing)?;
        let result = Self::compute_roi_ellipse_stats(p1, p2, pixels, width, height, spacing);
        let finite = result
            .0
            .iter()
            .chain(result.1.iter())
            .all(|value| value.is_finite())
            && result.2.is_finite()
            && result.3.is_finite()
            && result.4.is_finite()
            && result.5.is_finite()
            && result.6.is_finite();
        finite
            .then_some(result)
            .ok_or(MeasurementError::NonFiniteResult {
                kind: "ellipse ROI",
            })
    }
}
