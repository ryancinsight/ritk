//! Live measurement preview computation.
//!
//! Provides pure functions for computing live distance and angle
//! during in-progress rubber-band rendering, enabling ITK-SNAP-parity
//! real-time measurement feedback while the user drags.
//!
//! # Coordinate convention
//!
//! All point arguments are in **image pixel coordinates** `[row, col]`.
//! Physical spacing `[row_mm_per_px, col_mm_per_px]` converts pixel
//! distances to millimetres.
//!
//! # Mathematical specifications
//!
//! ## `live_length_mm`
//!
//! For points `p1 = [r₁, c₁]` and `p2 = [r₂, c₂]` with spacing `[dr, dc]`:
//!
//! ```text
//! d = √( (Δr × dr)² + (Δc × dc)² )
//! ```
//!
//! This is the standard Euclidean distance after converting pixel deltas
//! to physical millimetres via anisotropic voxel spacing.
//!
//! ## `live_angle_deg`
//!
//! For vectors `v₁ = p1 − vertex` and `v₃ = p3 − vertex`:
//!
//! ```text
//! cos θ = (v₁ · v₃) / (|v₁| × |v₃|)
//! θ = arccos(clamp(cos θ, −1, 1))  [degrees]
//! ```
//!
//! Physical spacing is not required: the angle between pixel-space
//! direction vectors is invariant under uniform scaling. Returns 0.0
//! when either ray has zero length (degenerate input).

/// Compute the live Euclidean distance in mm between two image-space points.
///
/// # Arguments
/// - `p1`: `[row, col]` of the first point in image pixels.
/// - `p2`: `[row, col]` of the second point in image pixels.
/// - `spacing`: `[row_mm_per_px, col_mm_per_px]` anisotropic voxel spacing.
///
/// # Returns
/// Physical distance in mm: `√((Δrow × spacing[0])² + (Δcol × spacing[1])²)`.
///
/// Returns 0.0 when `p1 == p2`.
#[inline]
pub fn live_length_mm(p1: [f32; 2], p2: [f32; 2], spacing: [f32; 2]) -> f32 {
    let dr = (p2[0] - p1[0]) * spacing[0];
    let dc = (p2[1] - p1[1]) * spacing[1];
    (dr * dr + dc * dc).sqrt()
}

/// Compute the live angle in degrees at `vertex` between rays `vertex→p1`
/// and `vertex→p3`.
///
/// # Arguments
/// - `p1`:     `[row, col]` of the first endpoint (start of first ray).
/// - `vertex`: `[row, col]` of the angle vertex.
/// - `p3`:     `[row, col]` of the third endpoint (start of second ray).
///
/// # Returns
/// Angle θ in degrees `[0°, 180°]`, computed from the normalized dot product.
/// Returns 0.0 when either `p1 == vertex` or `p3 == vertex` (degenerate ray).
#[inline]
pub fn live_angle_deg(p1: [f32; 2], vertex: [f32; 2], p3: [f32; 2]) -> f32 {
    let v1 = [p1[0] - vertex[0], p1[1] - vertex[1]];
    let v3 = [p3[0] - vertex[0], p3[1] - vertex[1]];
    let len1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt();
    let len3 = (v3[0] * v3[0] + v3[1] * v3[1]).sqrt();
    if len1 < f32::EPSILON || len3 < f32::EPSILON {
        return 0.0;
    }
    let cos_theta = ((v1[0] * v3[0] + v1[1] * v3[1]) / (len1 * len3)).clamp(-1.0, 1.0);
    cos_theta.acos().to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── live_length_mm ────────────────────────────────────────────────────────

    #[test]
    fn live_length_horizontal_unit_spacing_matches_pixel_distance() {
        // p1=[0,0] p2=[0,3], spacing=[1,1] → Δcol=3, d=3.0 mm
        let mm = live_length_mm([0.0, 0.0], [0.0, 3.0], [1.0, 1.0]);
        assert!((mm - 3.0).abs() < 1e-5, "expected 3.0 mm, got {mm}");
    }

    #[test]
    fn live_length_vertical_unit_spacing_matches_pixel_distance() {
        // p1=[0,0] p2=[4,0], spacing=[1,1] → Δrow=4, d=4.0 mm
        let mm = live_length_mm([0.0, 0.0], [4.0, 0.0], [1.0, 1.0]);
        assert!((mm - 4.0).abs() < 1e-5, "expected 4.0 mm, got {mm}");
    }

    #[test]
    fn live_length_anisotropic_spacing_scales_per_axis() {
        // p1=[0,0] p2=[3,4], spacing=[2.0, 0.5]
        // dr=3×2=6, dc=4×0.5=2 → √(36+4) = √40 ≈ 6.3246
        let mm = live_length_mm([0.0, 0.0], [3.0, 4.0], [2.0, 0.5]);
        let expected = (36.0_f32 + 4.0_f32).sqrt();
        assert!(
            (mm - expected).abs() < 1e-4,
            "expected {expected:.6} mm, got {mm:.6}"
        );
    }

    #[test]
    fn live_length_zero_delta_returns_zero() {
        // Same point → distance = 0
        let mm = live_length_mm([5.0, 3.0], [5.0, 3.0], [1.5, 0.8]);
        assert!(mm.abs() < 1e-6, "expected 0.0 mm, got {mm}");
    }

    #[test]
    fn live_length_diagonal_pythagorean_triple() {
        // p1=[0,0] p2=[3,4], spacing=[1,1] → d=√(9+16)=5.0 mm  (3-4-5)
        let mm = live_length_mm([0.0, 0.0], [3.0, 4.0], [1.0, 1.0]);
        assert!((mm - 5.0).abs() < 1e-5, "expected 5.0 mm, got {mm}");
    }

    // ── live_angle_deg ────────────────────────────────────────────────────────

    #[test]
    fn live_angle_right_angle_returns_90_degrees() {
        // vertex at origin, p1 in +row direction, p3 in +col direction
        // v1=[1,0], v3=[0,1] → dot=0 → θ=90°
        let deg = live_angle_deg([1.0, 0.0], [0.0, 0.0], [0.0, 1.0]);
        assert!((deg - 90.0).abs() < 1e-4, "expected 90.0°, got {deg}");
    }

    #[test]
    fn live_angle_straight_line_returns_180_degrees() {
        // vertex at origin, p1=[1,0], p3=[-1,0]: opposite collinear vectors
        // v1=[1,0], v3=[-1,0] → dot=-1 → θ=180°
        let deg = live_angle_deg([1.0, 0.0], [0.0, 0.0], [-1.0, 0.0]);
        assert!((deg - 180.0).abs() < 1e-4, "expected 180.0°, got {deg}");
    }

    #[test]
    fn live_angle_45_degrees_analytical() {
        // vertex at origin, p1=[1,0], p3=[1,1]
        // v1=[1,0], v3=[1,1], dot=1, |v1|=1, |v3|=√2
        // cos θ = 1/√2 → θ = 45°
        let deg = live_angle_deg([1.0, 0.0], [0.0, 0.0], [1.0, 1.0]);
        assert!((deg - 45.0).abs() < 1e-4, "expected 45.0°, got {deg}");
    }

    #[test]
    fn live_angle_degenerate_p1_equals_vertex_returns_zero() {
        // p1 == vertex → zero-length ray → returns 0
        let deg = live_angle_deg([0.0, 0.0], [0.0, 0.0], [1.0, 0.0]);
        assert!(
            deg.abs() < 1e-6,
            "expected 0.0° for degenerate input, got {deg}"
        );
    }

    #[test]
    fn live_angle_60_degrees_equilateral_analytical() {
        // Equilateral triangle: vertex at origin, p1=[1,0], p3=[0.5, 0.866025]
        // v1=[1,0], v3=[0.5,0.866025]: dot=0.5, |v1|=1, |v3|=1 → cos θ=0.5 → θ=60°
        let deg = live_angle_deg([1.0, 0.0], [0.0, 0.0], [0.5, 0.866_025_4]);
        assert!((deg - 60.0).abs() < 1e-3, "expected 60.0°, got {deg}");
    }
}
