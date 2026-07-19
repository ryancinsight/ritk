//! Bending energy regularizer for the B-spline displacement field.
//!
//! Implements the Rueckert (1999) bending energy:
//!
//! ```text
//! R(Ï†) = (1/|Î©|) Î£_x [ (âˆ‚Â²Ï†/âˆ‚zÂ²)Â² + (âˆ‚Â²Ï†/âˆ‚yÂ²)Â² + (âˆ‚Â²Ï†/âˆ‚xÂ²)Â²
//!                      + 2(âˆ‚Â²Ï†/âˆ‚zâˆ‚y)Â² + 2(âˆ‚Â²Ï†/âˆ‚zâˆ‚x)Â² + 2(âˆ‚Â²Ï†/âˆ‚yâˆ‚x)Â² ]
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
/// R = (1/N) Î£áµ¢ [ (Î”Â²_z cáµ¢)Â² + (Î”Â²_y cáµ¢)Â² + (Î”Â²_x cáµ¢)Â²
///              + 2(Î”_zy cáµ¢)Â² + 2(Î”_zx cáµ¢)Â² + 2(Î”_yx cáµ¢)Â² ]
/// ```
///
/// where `Î”Â²_d` denotes the second-order central difference along axis `d`
/// and `Î”_ab` denotes the cross second difference.
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
    // f32Â² second-derivative terms. The f32â†’f64 widening is intentional here.
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

/// Pre-allocated scratch buffers for [`bending_energy_gradient_into`].
///
/// Eliminates 3 per-iteration heap allocations of `Vec<f64>` temporary Laplacian buffers
/// inside `bending_energy_gradient`.
#[derive(Clone, Debug)]
pub(super) struct BendingEnergyScratch {
    /// Temporary buffer for the Laplacian computation `[cn]`.
    pub lap: Vec<f64>,
}

impl BendingEnergyScratch {
    /// Allocate scratch buffers for a control grid with `cn` nodes.
    pub fn new(cn: usize) -> Self {
        Self {
            lap: vec![0.0_f64; cn],
        }
    }

    /// Resize scratch buffers when the control grid changes.
    pub fn resize(&mut self, cn: usize) {
        self.lap.resize(cn, 0.0);
    }
}

/// Compute the gradient of the bending energy w.r.t. control-point displacements,
/// writing the result into a pre-allocated `VelocityField`.
///
/// Zero-allocation variant of [`bending_energy_gradient`].
#[inline]
pub(super) fn bending_energy_gradient_into(
    cp_z: &[f32],
    cp_y: &[f32],
    cp_x: &[f32],
    ctrl_dims: &[usize; 3],
    ctrl_spacing: &[f64; 3],
    scratch: &mut BendingEnergyScratch,
    be_grad: &mut VelocityField,
) {
    let [cnz, cny, cnx] = *ctrl_dims;
    let cn = cnz * cny * cnx;
    let count = count_interior(ctrl_dims);
    let norm = if count > 0 { 2.0 / count as f64 } else { 0.0 };

    let sz2 = ctrl_spacing[0] * ctrl_spacing[0];
    let sy2 = ctrl_spacing[1] * ctrl_spacing[1];
    let sx2 = ctrl_spacing[2] * ctrl_spacing[2];

    be_grad.z.fill(0.0);
    be_grad.y.fill(0.0);
    be_grad.x.fill(0.0);

    debug_assert_eq!(be_grad.z.len(), cn);
    debug_assert_eq!(be_grad.y.len(), cn);
    debug_assert_eq!(be_grad.x.len(), cn);

    for (comp, out) in [
        (cp_z, &mut be_grad.z),
        (cp_y, &mut be_grad.y),
        (cp_x, &mut be_grad.x),
    ] {
        scratch.lap.fill(0.0);
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
                    scratch.lap[ci] = dzz + dyy + dxx;
                }
            }
        }
        // Gradient = Laplacian of Laplacian (biharmonic approximation).
        for iz in 2..cnz.saturating_sub(2) {
            for iy in 2..cny.saturating_sub(2) {
                for ix in 2..cnx.saturating_sub(2) {
                    let ci = flat(iz, iy, ix, cny, cnx);
                    let l = scratch.lap[ci];
                    let lzz = (scratch.lap[flat(iz + 1, iy, ix, cny, cnx)] - 2.0 * l
                        + scratch.lap[flat(iz - 1, iy, ix, cny, cnx)])
                        / sz2;
                    let lyy = (scratch.lap[flat(iz, iy + 1, ix, cny, cnx)] - 2.0 * l
                        + scratch.lap[flat(iz, iy - 1, ix, cny, cnx)])
                        / sy2;
                    let lxx = (scratch.lap[flat(iz, iy, ix + 1, cny, cnx)] - 2.0 * l
                        + scratch.lap[flat(iz, iy - 1, ix, cny, cnx)])
                        / sx2;
                    out[ci] = (norm * (lzz + lyy + lxx)) as f32;
                }
            }
        }
    }
}

/// Compute the gradient of the bending energy w.r.t. control-point displacements.
///
/// Derivative of `bending_energy` w.r.t. each control-point component,
/// computed via the chain rule on the finite-difference operators.
/// Applies the discretized biharmonic operator (Laplacian of Laplacian) to
/// each control point as an efficient approximation.
#[allow(dead_code)]
#[inline]
pub(super) fn bending_energy_gradient(
    cp_z: &[f32],
    cp_y: &[f32],
    cp_x: &[f32],
    ctrl_dims: &[usize; 3],
    ctrl_spacing: &[f64; 3],
) -> VelocityField {
    let cn = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];
    let mut scratch = BendingEnergyScratch::new(cn);
    let mut be_grad = VelocityField::zeros(cn);
    bending_energy_gradient_into(
        cp_z,
        cp_y,
        cp_x,
        ctrl_dims,
        ctrl_spacing,
        &mut scratch,
        &mut be_grad,
    );
    be_grad
}

/// Count interior control points (those with at least 1 neighbor in each
/// direction).
#[inline]
fn count_interior(ctrl_dims: &[usize; 3]) -> usize {
    let w = |d: usize| if d >= 3 { d - 2 } else { 0 };
    w(ctrl_dims[0]) * w(ctrl_dims[1]) * w(ctrl_dims[2])
}
