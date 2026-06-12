//! Scalar cubic B-spline basis functions and per-axis pre-computation.
//!
//! Implements Rueckert (1999) uniform cubic B-spline basis:
//!
//! ```text
//! β₃₀(t) = (1 − t)³ / 6
//! β₃₁(t) = (3t³ − 6t² + 4) / 6
//! β₃₂(t) = (−3t³ + 3t² + 3t + 1) / 6
//! β₃₃(t) = t³ / 6
//! ```

/// Evaluate the four cubic B-spline basis values at parameter `t ∈ [0, 1]`.
///
/// Returns `[β₃₀(t), β₃₁(t), β₃₂(t), β₃₃(t)]`. These sum to 1.0 (partition
/// of unity) and are non-negative on `[0, 1]`.
#[inline]
pub fn cubic_bspline_basis(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    let omt = 1.0 - t;
    let omt3 = omt * omt * omt;

    [
        omt3 / 6.0,
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
        t3 / 6.0,
    ]
}

/// Pre-computed B-spline basis data for one axis.
///
/// For each image coordinate `i ∈ [0, dim)`, stores:
/// - `k`: the first control-point index (kz, ky, or kx)
/// - `b`: the four cubic basis values `[β₃₀(t), β₃₁(t), β₃₂(t), β₃₃(t)]`
///
/// This eliminates per-voxel `cubic_bspline_1d` calls (hot path is
/// lookup-only) and hoists `k` computation out of the inner loop.
#[derive(Clone)]
pub struct AxisBasis {
    /// `k[i]` = first control-point index for image coordinate i.
    pub k: Vec<isize>,
    /// `b[i]` = `[β₃₀(t_i), β₃₁(t_i), β₃₂(t_i), β₃₃(t_i)]`.
    pub b: Vec<[f64; 4]>,
}

impl AxisBasis {
    /// Pre-compute basis data for `dim` coordinates with the given control spacing.
    pub fn new(dim: usize, ctrl_spacing: f64) -> Self {
        let mut k = Vec::with_capacity(dim);
        let mut b = Vec::with_capacity(dim);
        for i in 0..dim {
            let u = i as f64 / ctrl_spacing + 1.0;
            let ki = u.floor() as isize - 1;
            let t = u - (ki + 1) as f64;
            k.push(ki);
            b.push(cubic_bspline_basis(t));
        }
        Self { k, b }
    }
}
