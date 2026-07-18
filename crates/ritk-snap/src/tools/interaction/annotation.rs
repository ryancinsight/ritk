//! Completed measurement annotations with computed physical values.
//!
//! # Coordinate system
//!
//! All image-space coordinates are stored as `[row, col]` in pixel units.
//! Physical distances are computed by multiplying pixel deltas by the
//! corresponding voxel spacing components.
//!
//! # Mathematical specifications
//!
//! ## Length (Euclidean distance in physical space)
//!
//! Given two image points pâ‚ = (râ‚, câ‚) and pâ‚‚ = (râ‚‚, câ‚‚) and pixel spacing
//! s = (s_r, s_c) in mm/pixel:
//!
//! ```text
//! length_mm = âˆš( ((râ‚‚ âˆ’ râ‚) Â· s_r)Â² + ((câ‚‚ âˆ’ câ‚) Â· s_c)Â² )
//! ```
//!
//! ## Angle (at vertex pâ‚‚, between vectors pâ‚‚â†’pâ‚ and pâ‚‚â†’pâ‚ƒ)
//!
//! ```text
//! vâ‚ = pâ‚ âˆ’ pâ‚‚, vâ‚‚ = pâ‚ƒ âˆ’ pâ‚‚
//! cos Î¸ = (vâ‚ Â· vâ‚‚) / (|vâ‚| Â· |vâ‚‚|)
//! Î¸ = arccos(clamp(cos Î¸, âˆ’1, 1)) [degrees]
//! ```
//!
//! ## ROI rectangle statistics
//!
//! Pixels whose row index lies in [min_r, max_r] and column index lies in
//! [min_c, max_c] (inclusive, integer bounds derived from pâ‚ and pâ‚‚) are
//! collected into a sample set S.
//!
//! ```text
//! mean = (1/|S|) Î£ v
//! std_dev = âˆš( (1/|S|) Î£ (v âˆ’ mean)Â² ) [population std dev]
//! min = min S
//! max = max S
//! area = (max_r âˆ’ min_r + 1) Â· s_r Ã— (max_c âˆ’ min_c + 1) Â· s_c [mmÂ²]
//! ```

// â”€â”€ Completed annotations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// A completed measurement annotation stored on a viewport.
///
/// Positions are stored as `[row, col]` in image pixel coordinates. Computed
/// values (lengths, angles, statistics) are stored in physical units (mm,
/// degrees, HU) and are derived quantities â€” they can be recomputed from the
/// position data and the volume spacing at any time.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Annotation {
    /// Straight-line distance between two image points.
    Length {
        /// Start point as `[row, col]` in image pixels.
        p1: [f32; 2],
        /// End point as `[row, col]` in image pixels.
        p2: [f32; 2],
        /// Euclidean distance in millimetres.
        length_mm: f32 },

    /// Angle at vertex `p2` between the rays `p2â†’p1` and `p2â†’p3`.
    Angle {
        /// First ray endpoint as `[row, col]` in image pixels.
        p1: [f32; 2],
        /// Vertex as `[row, col]` in image pixels.
        p2: [f32; 2],
        /// Second ray endpoint as `[row, col]` in image pixels.
        p3: [f32; 2],
        /// Included angle in degrees.
        angle_deg: f32 },

    /// Axis-aligned rectangle ROI with intensity statistics.
    RoiRect {
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
        /// ROI area in mmÂ².
        area_mm2: f32 },

    /// Ellipse ROI with intensity statistics computed over pixels whose centres
    /// lie inside the ellipse mask.
    ///
    /// # Ellipse membership condition
    ///
    /// A pixel at integer coordinates `(r, c)` is included when:
    ///
    /// ```text
    /// ((r âˆ’ center[0]) / radii[0])Â² + ((c âˆ’ center[1]) / radii[1])Â² â‰¤ 1
    /// ```
    ///
    /// When `radii[0]` or `radii[1]` is zero (degenerate ellipse), no pixels
    /// are collected and all statistics return `0.0`.
    ///
    /// # Physical area
    ///
    /// ```text
    /// area_mmÂ² = Ï€ Ã— radii[0] Ã— spacing[0] Ã— radii[1] Ã— spacing[1]
    /// ```
    RoiEllipse {
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
        /// Physical area of the ellipse in mmÂ².
        area_mm2: f32 },

    /// Single-point HU measurement.
    HuPoint {
        /// Point location as `[row, col]` in image pixels.
        pos: [f32; 2],
        /// Intensity value at the point (HU or relative).
        value: f32 } }

impl Annotation {
    /// Compute the Euclidean distance between two image points in physical space.
    ///
    /// # Parameters
    /// - `p1`, `p2` â€” image coordinates `[row, col]` in pixels.
    /// - `spacing` â€” pixel spacing `[row_spacing, col_spacing]` in mm/pixel.
    ///
    /// # Formula
    /// ```text
    /// length = âˆš( ((p2[0] âˆ’ p1[0]) Â· spacing[0])Â²
    ///            + ((p2[1] âˆ’ p1[1]) Â· spacing[1])Â² )
    /// ```
    pub fn compute_length(p1: [f32; 2], p2: [f32; 2], spacing: [f32; 2]) -> f32 {
        let dr = (p2[0] - p1[0]) * spacing[0];
        let dc = (p2[1] - p1[1]) * spacing[1];
        (dr * dr + dc * dc).sqrt()
    }

    /// Compute the angle at vertex `p2` between the rays `p2â†’p1` and `p2â†’p3`.
    ///
    /// Returns the angle in degrees in the range `[0Â°, 180Â°]`.
    ///
    /// Returns `0.0` when either input ray has zero length (degenerate case:
    /// two coincident points), avoiding division by zero without panicking.
    ///
    /// # Formula
    /// ```text
    /// vâ‚ = pâ‚ âˆ’ pâ‚‚, vâ‚‚ = pâ‚ƒ âˆ’ pâ‚‚
    /// cos Î¸ = clamp( (vâ‚ Â· vâ‚‚) / (|vâ‚| Â· |vâ‚‚|), âˆ’1, 1 )
    /// Î¸ = arccos(cos Î¸) [converted to degrees]
    /// ```
    pub fn compute_angle(p1: [f32; 2], p2: [f32; 2], p3: [f32; 2]) -> f32 {
        let v1 = [p1[0] - p2[0], p1[1] - p2[1]];
        let v2 = [p3[0] - p2[0], p3[1] - p2[1]];
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

    /// Compute intensity statistics for pixels within the axis-aligned rectangle
    /// defined by corners `p1` and `p2` (order-independent).
    ///
    /// # Parameters
    /// - `p1`, `p2` â€” opposite corners of the ROI as `[row, col]` in pixels.
    /// - `pixels` â€” flat row-major pixel buffer of the slice.
    /// - `width` â€” number of columns in the slice.
    /// - `height` â€” number of rows in the slice.
    /// - `spacing` â€” pixel spacing `[row_spacing, col_spacing]` in mm/pixel.
    ///
    /// # Returns
    /// `(mean, std_dev, min, max, area_mm2)` using population statistics.
    ///
    /// Returns `(0.0, 0.0, 0.0, 0.0, 0.0)` when the ROI contains no pixels
    /// (e.g., the rectangle lies fully outside the image bounds).
    ///
    /// # Formula
    /// ```text
    /// S = { pixels[r Ã— width + c] | r âˆˆ [min_r, max_r], c âˆˆ [min_c, max_c] }
    /// mean = Î£ S / |S|
    /// std_dev = âˆš( Î£ (v âˆ’ mean)Â² / |S| )
    /// area = (max_r âˆ’ min_r + 1) Â· spacing[0]
    ///       Ã— (max_c âˆ’ min_c + 1) Â· spacing[1]
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

    /// Compute intensity statistics for pixels whose centres lie inside the
    /// ellipse defined by two opposite corners `p1` and `p2`.
    ///
    /// # Parameters
    /// - `p1`, `p2` â€” opposite corners of the bounding rectangle as `[row, col]`.
    /// - `pixels` â€” flat row-major pixel buffer of the slice.
    /// - `width` â€” number of columns in the slice.
    /// - `height` â€” number of rows in the slice.
    /// - `spacing` â€” pixel spacing `[row_spacing, col_spacing]` in mm/pixel.
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
    /// a = |p2[0] âˆ’ p1[0]| / 2   (row semi-axis)
    /// b = |p2[1] âˆ’ p1[1]| / 2   (col semi-axis)
    ///
    /// pixel (r,c) is inside âŸº ((râˆ’cy)/a)Â² + ((câˆ’cx)/b)Â² â‰¤ 1
    ///
    /// area_mmÂ² = Ï€ Ã— a Ã— spacing[0] Ã— b Ã— spacing[1]
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
                // Membership: (dr)Â² + (dc)Â² â‰¤ 1
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

        // Physical area of the ellipse: Ï€ Ã— a Ã— dr Ã— b Ã— dc
        let area_mm2 = std::f32::consts::PI * a * spacing[0] * b * spacing[1];

        (center, radii, mean, std_dev, min, max, area_mm2)
    }
}
