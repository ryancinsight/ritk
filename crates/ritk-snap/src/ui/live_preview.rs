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
#[path = "tests_live_preview.rs"]
mod tests;
