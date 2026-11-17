//! Region-of-interest statistics in image and physical coordinates.

use super::{Annotation, MeasurementError};

impl Annotation {
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
