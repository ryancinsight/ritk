//! NCC similarity metric and its gradient w.r.t. control-point displacements.

use super::basis::cubic_bspline_1d;
use crate::deformable_field_ops::{compute_gradient, flat};

/// Compute global normalized cross-correlation between two images.
///
/// ```text
/// NCC = Σ (F − μ_F)(M − μ_M) / sqrt(Σ (F − μ_F)² · Σ (M − μ_M)²)
/// ```
///
/// Returns a value in `[-1, 1]` where 1.0 indicates identical images up to
/// affine intensity scaling.
pub(super) fn compute_ncc(fixed: &[f32], warped: &[f32]) -> f64 {
    let n = fixed.len() as f64;
    if n < 1.0 {
        return 0.0;
    }

    let mean_f: f64 = fixed.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mean_w: f64 = warped.iter().map(|&v| v as f64).sum::<f64>() / n;

    let mut sum_fw = 0.0_f64;
    let mut sum_ff = 0.0_f64;
    let mut sum_ww = 0.0_f64;

    for i in 0..fixed.len() {
        let f = fixed[i] as f64 - mean_f;
        let w = warped[i] as f64 - mean_w;
        sum_fw += f * w;
        sum_ff += f * f;
        sum_ww += w * w;
    }

    let denom = (sum_ff * sum_ww).sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    sum_fw / denom
}

/// Compute the gradient of NCC w.r.t. control-point displacements.
///
/// Uses the chain rule:
///
/// ```text
/// ∂NCC/∂cᵢ = Σ_x (∂NCC/∂φ(x)) · (∂φ(x)/∂cᵢ)
/// ```
///
/// where `∂NCC/∂φ(x)` is the voxel-wise NCC gradient propagated through the
/// spatial gradient of the warped moving image, and `∂φ(x)/∂cᵢ = β₃((x−xᵢ)/δ)`.
///
/// # NCC Gradient Derivation
///
/// For the global NCC `ρ = Σ f̃ w̃ / sqrt(Σ f̃² · Σ w̃²)` where `f̃ = F − μ_F`
/// and `w̃ = W − μ_W`:
///
/// ```text
/// ∂ρ/∂W(x) = (1/σ_W) [ (f̃(x) / (n · σ_F)) − ρ · (w̃(x) / (n · σ_W)) ]
/// ```
///
/// and the displacement gradient at voxel x is:
///
/// ```text
/// ∂NCC/∂φ_d(x) = (∂ρ/∂W(x)) · ∇_d M_warped(x)
/// ```
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_metric_gradient(
    fixed: &[f32],
    _moving: &[f32],
    warped: &[f32],
    _disp_z: &[f32],
    _disp_y: &[f32],
    _disp_x: &[f32],
    ctrl_dims: &[usize; 3],
    ctrl_spacing: &[f64; 3],
    dims: [usize; 3],
    spacing: [f64; 3],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let [cnz, cny, cnx] = *ctrl_dims;
    let cn = cnz * cny * cnx;

    // Compute NCC statistics.
    let nf = n as f64;
    let mean_f: f64 = fixed.iter().map(|&v| v as f64).sum::<f64>() / nf;
    let mean_w: f64 = warped.iter().map(|&v| v as f64).sum::<f64>() / nf;

    let mut sum_ff = 0.0_f64;
    let mut sum_ww = 0.0_f64;
    let mut sum_fw = 0.0_f64;
    for i in 0..n {
        let f = fixed[i] as f64 - mean_f;
        let w = warped[i] as f64 - mean_w;
        sum_ff += f * f;
        sum_ww += w * w;
        sum_fw += f * w;
    }
    let sigma_f = (sum_ff / nf).sqrt();
    let sigma_w = (sum_ww / nf).sqrt();
    let rho = if sigma_f * sigma_w > 1e-12 {
        sum_fw / (nf * sigma_f * sigma_w)
    } else {
        0.0
    };

    // Compute spatial gradient of the warped image.
    let (gw_z, gw_y, gw_x) = compute_gradient(warped, dims, spacing);

    // Accumulate gradient into control points.
    let mut grad_cz = vec![0.0_f64; cn];
    let mut grad_cy = vec![0.0_f64; cn];
    let mut grad_cx = vec![0.0_f64; cn];

    let inv_n_sigma_f = if sigma_f > 1e-12 {
        1.0 / (nf * sigma_f)
    } else {
        0.0
    };
    let inv_n_sigma_w = if sigma_w > 1e-12 {
        1.0 / (nf * sigma_w)
    } else {
        0.0
    };
    let inv_sigma_w = if sigma_w > 1e-12 { 1.0 / sigma_w } else { 0.0 };

    for iz in 0..nz {
        let uz = iz as f64 / ctrl_spacing[0] + 1.0;
        let kz = uz.floor() as isize - 1;
        let tz = uz - (kz + 1) as f64;
        let bz = cubic_bspline_1d(tz);

        for iy in 0..ny {
            let uy = iy as f64 / ctrl_spacing[1] + 1.0;
            let ky = uy.floor() as isize - 1;
            let ty = uy - (ky + 1) as f64;
            let by = cubic_bspline_1d(ty);

            for ix in 0..nx {
                let ux = ix as f64 / ctrl_spacing[2] + 1.0;
                let kx = ux.floor() as isize - 1;
                let tx = ux - (kx + 1) as f64;
                let bx = cubic_bspline_1d(tx);

                let fi = flat(iz, iy, ix, ny, nx);

                let f_tilde = fixed[fi] as f64 - mean_f;
                let w_tilde = warped[fi] as f64 - mean_w;

                // ∂ρ/∂W(x) for the global NCC.
                let drho_dw =
                    inv_sigma_w * (f_tilde * inv_n_sigma_f - rho * w_tilde * inv_n_sigma_w);

                // Voxel-wise displacement gradient = drho_dw * ∇(warped).
                let vg_z = drho_dw * gw_z[fi] as f64;
                let vg_y = drho_dw * gw_y[fi] as f64;
                let vg_x = drho_dw * gw_x[fi] as f64;

                // Splatting: accumulate onto control points weighted by basis.
                for az in 0..4isize {
                    let ciz = kz + az;
                    if ciz < 0 || ciz >= cnz as isize {
                        continue;
                    }
                    let ciz = ciz as usize;
                    let wz = bz[az as usize];

                    for ay in 0..4isize {
                        let ciy = ky + ay;
                        if ciy < 0 || ciy >= cny as isize {
                            continue;
                        }
                        let ciy = ciy as usize;
                        let wzy = wz * by[ay as usize];

                        for ax in 0..4isize {
                            let cix = kx + ax;
                            if cix < 0 || cix >= cnx as isize {
                                continue;
                            }
                            let cix = cix as usize;
                            let w = wzy * bx[ax as usize];

                            let ci = flat(ciz, ciy, cix, cny, cnx);
                            grad_cz[ci] += w * vg_z;
                            grad_cy[ci] += w * vg_y;
                            grad_cx[ci] += w * vg_x;
                        }
                    }
                }
            }
        }
    }

    // Convert to f32.
    let out_z: Vec<f32> = grad_cz.iter().map(|&v| v as f32).collect();
    let out_y: Vec<f32> = grad_cy.iter().map(|&v| v as f32).collect();
    let out_x: Vec<f32> = grad_cx.iter().map(|&v| v as f32).collect();

    (out_z, out_y, out_x)
}
