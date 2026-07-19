//! Scalar cubic B-spline basis functions and per-axis pre-computation.
//!
//! Implements Rueckert (1999) uniform cubic B-spline basis:
//!
//! ```text
//! Î²â‚ƒâ‚€(t) = (1 âˆ’ t)Â³ / 6
//! Î²â‚ƒâ‚(t) = (3tÂ³ âˆ’ 6tÂ² + 4) / 6
//! Î²â‚ƒâ‚‚(t) = (âˆ’3tÂ³ + 3tÂ² + 3t + 1) / 6
//! Î²â‚ƒâ‚ƒ(t) = tÂ³ / 6
//! ```

/// Evaluate the four cubic B-spline basis values at parameter `t âˆˆ [0, 1]`.
///
/// Returns `[Î²â‚ƒâ‚€(t), Î²â‚ƒâ‚(t), Î²â‚ƒâ‚‚(t), Î²â‚ƒâ‚ƒ(t)]`. These sum to 1.0 (partition
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
/// For each image coordinate `i âˆˆ [0, dim)`, stores:
/// - `k`: the first control-point index (kz, ky, or kx)
/// - `b`: the four cubic basis values `[Î²â‚ƒâ‚€(t), Î²â‚ƒâ‚(t), Î²â‚ƒâ‚‚(t), Î²â‚ƒâ‚ƒ(t)]`
///
/// This eliminates per-voxel `cubic_bspline_1d` calls (hot path is
/// lookup-only) and hoists `k` computation out of the inner loop.
#[derive(Clone)]
pub struct AxisBasis {
    /// `k[i]` = first control-point index for image coordinate i.
    pub k: Vec<isize>,
    /// `b[i]` = `[Î²â‚ƒâ‚€(t_i), Î²â‚ƒâ‚(t_i), Î²â‚ƒâ‚‚(t_i), Î²â‚ƒâ‚ƒ(t_i)]`.
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
