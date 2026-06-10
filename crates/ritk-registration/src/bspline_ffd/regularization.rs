//! Bending energy regularizer for the B-spline displacement field.
//!
//! Implements the Rueckert (1999) bending energy:
//!
//! ```text
//! R(φ) = (1/|Ω|) Σ_x [ (∂²φ/∂z²)² + (∂²φ/∂y²)² + (∂²φ/∂x²)²
//!                      + 2(∂²φ/∂z∂y)² + 2(∂²φ/∂z∂x)² + 2(∂²φ/∂y∂x)² ]
//! ```
//!
//! Approximated via finite differences on the control-point lattice.

use crate::deformable_field_ops::{flat, VelocityField};

/// Compute the bending energy of the displacement field defined by the control
/// points.
///
/// The bending energy is computed directly on the control-point lattice using
/// finite differences of the control-point displacements:
///
/// ```text
/// R = (1/N) Σᵢ [ (Δ²_z cᵢ)² + (Δ²_y cᵢ)² + (Δ²_x cᵢ)²
///              + 2(Δ_zy cᵢ)² + 2(Δ_zx cᵢ)² + 2(Δ_yx cᵢ)² ]
/// ```
///
/// where `Δ²_d` denotes the second-order central difference along axis `d`
/// and `Δ_ab` denotes the cross second difference.
#[inline]
pub fn bending_energy(
    cp_z: &[f32],
    cp_y: &[f32],
    cp_x: &[f32],
    ctrl_dims: &[usize; 3],
    ctrl_spacing: &[f64; 3],
) -> f64 {
    let [cnz, cny, cnx] = *ctrl_dims;
    // ACCUMULATOR: f64 prevents catastrophic cancellation when summing many small
    // f32² second-derivative terms. The f32→f64 widening is intentional here.
    let mut energy = 0.0_f64;
    let mut count = 0usize;

    // Compute squared spacings in f64 then cast to f32: this order (rather than
    // casting first, squaring second) minimises rounding in the coefficient.
    let sz2 = (ctrl_spacing[0] * ctrl_spacing[0]) as f32;
    let sy2 = (ctrl_spacing[1] * ctrl_spacing[1]) as f32;
    let sx2 = (ctrl_spacing[2] * ctrl_spacing[2]) as f32;
    let szy = (ctrl_spacing[0] * ctrl_spacing[1]) as f32;
    let szx = (ctrl_spacing[0] * ctrl_spacing[2]) as f32;
    let syx = (ctrl_spacing[1] * ctrl_spacing[2]) as f32;

    for comp in [cp_z, cp_y, cp_x] {
        for iz in 1..cnz.saturating_sub(1) {
            for iy in 1..cny.saturating_sub(1) {
                for ix in 1..cnx.saturating_sub(1) {
                    let c = comp[flat(iz, iy, ix, cny, cnx)];

                    // Pure second derivatives.
                    let dzz = (comp[flat(iz + 1, iy, ix, cny, cnx)] - 2.0 * c
                        + comp[flat(iz - 1, iy, ix, cny, cnx)])
                        / sz2;
                    let dyy = (comp[flat(iz, iy + 1, ix, cny, cnx)] - 2.0 * c
                        + comp[flat(iz, iy - 1, ix, cny, cnx)])
                        / sy2;
                    let dxx = (comp[flat(iz, iy, ix + 1, cny, cnx)] - 2.0 * c
                        + comp[flat(iz, iy, ix - 1, cny, cnx)])
                        / sx2;

                    // Cross second derivatives.
                    let dzy = (comp[flat(iz + 1, iy + 1, ix, cny, cnx)]
                        - comp[flat(iz + 1, iy - 1, ix, cny, cnx)]
                        - comp[flat(iz - 1, iy + 1, ix, cny, cnx)]
                        + comp[flat(iz - 1, iy - 1, ix, cny, cnx)])
                        / (4.0 * szy);
                    let dzx = (comp[flat(iz + 1, iy, ix + 1, cny, cnx)]
                        - comp[flat(iz + 1, iy, ix - 1, cny, cnx)]
                        - comp[flat(iz - 1, iy, ix + 1, cny, cnx)]
                        + comp[flat(iz - 1, iy, ix - 1, cny, cnx)])
                        / (4.0 * szx);
                    let dyx = (comp[flat(iz, iy + 1, ix + 1, cny, cnx)]
                        - comp[flat(iz, iy + 1, ix - 1, cny, cnx)]
                        - comp[flat(iz, iy - 1, ix + 1, cny, cnx)]
                        + comp[flat(iz, iy - 1, ix - 1, cny, cnx)])
                        / (4.0 * syx);

                    energy += (dzz * dzz
                        + dyy * dyy
                        + dxx * dxx
                        + 2.0 * dzy * dzy
                        + 2.0 * dzx * dzx
                        + 2.0 * dyx * dyx) as f64;
                    count += 1;
                }
            }
        }
    }

    if count > 0 {
        energy / count as f64
    } else {
        0.0
    }
}

/// Compute the gradient of the bending energy w.r.t. control-point displacements.
///
/// Derivative of `bending_energy` w.r.t. each control-point component,
/// computed via the chain rule on the finite-difference operators.
/// Applies the discretized biharmonic operator (Laplacian of Laplacian) to
/// each control point as an efficient approximation.
#[inline]
pub(super) fn bending_energy_gradient(
    cp_z: &[f32],
    cp_y: &[f32],
    cp_x: &[f32],
    ctrl_dims: &[usize; 3],
    ctrl_spacing: &[f64; 3],
) -> VelocityField {
    let [cnz, cny, cnx] = *ctrl_dims;
    let cn = cnz * cny * cnx;
    let count = count_interior(ctrl_dims);
    let norm = if count > 0 { 2.0 / count as f64 } else { 0.0 };

    let sz2 = ctrl_spacing[0] * ctrl_spacing[0];
    let sy2 = ctrl_spacing[1] * ctrl_spacing[1];
    let sx2 = ctrl_spacing[2] * ctrl_spacing[2];

    let mut out_z = vec![0.0_f32; cn];
    let mut out_y = vec![0.0_f32; cn];
    let mut out_x = vec![0.0_f32; cn];

    for (comp, out) in [(cp_z, &mut out_z), (cp_y, &mut out_y), (cp_x, &mut out_x)] {
        // Apply the biharmonic stencil: gradient of ∫ (∂²c/∂d²)² dd is
        // the fourth-order finite difference operator. Compute as the
        // Laplacian of the Laplacian on the interior.
        let mut lap = vec![0.0_f64; cn];
        for iz in 1..cnz.saturating_sub(1) {
            for iy in 1..cny.saturating_sub(1) {
                for ix in 1..cnx.saturating_sub(1) {
                    let ci = flat(iz, iy, ix, cny, cnx);
                    let c = comp[ci] as f64;
                    let dzz = (comp[flat(iz + 1, iy, ix, cny, cnx)] as f64 - 2.0 * c
                        + comp[flat(iz - 1, iy, ix, cny, cnx)] as f64)
                        / sz2;
                    let dyy = (comp[flat(iz, iy + 1, ix, cny, cnx)] as f64 - 2.0 * c
                        + comp[flat(iz, iy - 1, ix, cny, cnx)] as f64)
                        / sy2;
                    let dxx = (comp[flat(iz, iy, ix + 1, cny, cnx)] as f64 - 2.0 * c
                        + comp[flat(iz, iy, ix - 1, cny, cnx)] as f64)
                        / sx2;
                    lap[ci] = dzz + dyy + dxx;
                }
            }
        }
        // Gradient = Laplacian of Laplacian (biharmonic approximation).
        for iz in 2..cnz.saturating_sub(2) {
            for iy in 2..cny.saturating_sub(2) {
                for ix in 2..cnx.saturating_sub(2) {
                    let ci = flat(iz, iy, ix, cny, cnx);
                    let l = lap[ci];
                    let lzz = (lap[flat(iz + 1, iy, ix, cny, cnx)] - 2.0 * l
                        + lap[flat(iz - 1, iy, ix, cny, cnx)])
                        / sz2;
                    let lyy = (lap[flat(iz, iy + 1, ix, cny, cnx)] - 2.0 * l
                        + lap[flat(iz, iy - 1, ix, cny, cnx)])
                        / sy2;
                    let lxx = (lap[flat(iz, iy, ix + 1, cny, cnx)] - 2.0 * l
                        + lap[flat(iz, iy, ix - 1, cny, cnx)])
                        / sx2;
                    out[ci] = (norm * (lzz + lyy + lxx)) as f32;
                }
            }
        }
    }

    VelocityField {
        z: out_z,
        y: out_y,
        x: out_x,
    }
}

/// Count interior control points (those with at least 1 neighbor in each
/// direction).
#[inline]
fn count_interior(ctrl_dims: &[usize; 3]) -> usize {
    let w = |d: usize| if d >= 3 { d - 2 } else { 0 };
    w(ctrl_dims[0]) * w(ctrl_dims[1]) * w(ctrl_dims[2])
}
