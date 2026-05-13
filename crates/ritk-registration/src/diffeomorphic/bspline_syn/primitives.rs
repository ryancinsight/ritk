//! Cubic B-spline primitives: basis evaluation, control-point layout,
//! dense-field synthesis, force accumulation, and Laplacian regularisation.
//!
//! # Cubic B-Spline Basis
//!
//! Uniform cubic B-spline basis over parameter `u ∈ [0,1)`:
//!   - `B₀(u) = (1 − u)³ / 6`
//!   - `B₁(u) = (3u³ − 6u² + 4) / 6`
//!   - `B₂(u) = (−3u³ + 3u² + 3u + 1) / 6`
//!   - `B₃(u) = u³ / 6`
//!
//! **Partition of unity**: `Σₖ Bₖ(u) = 1  ∀ u ∈ [0,1]`.

use crate::deformable_field_ops::flat;

// ── Basis evaluation ──────────────────────────────────────────────────────────

/// Evaluate the `k`-th uniform cubic B-spline basis function at parameter
/// `u ∈ [0, 1]`.
///
/// # Partition of unity
/// `Σ_{k=0}^{3} Bₖ(u) = 1` for all `u ∈ [0, 1]`.
#[inline]
pub(crate) fn bspline_basis(k: usize, u: f64) -> f64 {
    let u2 = u * u;
    let u3 = u2 * u;
    match k {
        0 => (1.0 - 3.0 * u + 3.0 * u2 - u3) / 6.0,
        1 => (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0,
        2 => (1.0 + 3.0 * u + 3.0 * u2 - 3.0 * u3) / 6.0,
        3 => u3 / 6.0,
        _ => 0.0,
    }
}

// ── Control-point layout ──────────────────────────────────────────────────────

/// Number of control points along one axis for image dimension `dim` and
/// control-point spacing `spacing`.
///
/// Formula: `⌊(dim − 1) / spacing⌋ + 4`, ensuring the cubic B-spline support
/// (4 CPs per knot span) covers the entire image domain.
#[inline]
pub(crate) fn cp_count(dim: usize, spacing: usize) -> usize {
    if dim <= 1 {
        return 4;
    }
    (dim - 1) / spacing + 4
}

// ── Dense field evaluation ────────────────────────────────────────────────────

/// Evaluate a single dense displacement-field component from its
/// control-point lattice via cubic B-spline evaluation.
///
/// For each voxel, the displacement is the tensor-product sum over the
/// 4×4×4 local CP neighbourhood weighted by the B-spline basis.
pub(crate) fn evaluate_dense(
    cp: &[f32],
    cp_dims: [usize; 3],
    dims: [usize; 3],
    control_spacing: [usize; 3],
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut out = vec![0.0_f32; n];

    for iz in 0..nz {
        let tz = iz as f64 / control_spacing[0] as f64;
        let sz = (tz.floor() as usize).min(cp_dims[0].saturating_sub(4));
        let uz = tz - sz as f64;

        for iy in 0..ny {
            let ty = iy as f64 / control_spacing[1] as f64;
            let sy = (ty.floor() as usize).min(cp_dims[1].saturating_sub(4));
            let uy = ty - sy as f64;

            for ix in 0..nx {
                let tx = ix as f64 / control_spacing[2] as f64;
                let sx = (tx.floor() as usize).min(cp_dims[2].saturating_sub(4));
                let ux = tx - sx as f64;

                let mut val = 0.0_f64;
                for kz in 0..4 {
                    let bz = bspline_basis(kz, uz);
                    for ky in 0..4 {
                        let bzy = bz * bspline_basis(ky, uy);
                        for kx in 0..4 {
                            let w = bzy * bspline_basis(kx, ux);
                            let ci = flat(sz + kz, sy + ky, sx + kx, cp_dims[1], cp_dims[2]);
                            val += w * cp[ci] as f64;
                        }
                    }
                }
                out[flat(iz, iy, ix, ny, nx)] = val as f32;
            }
        }
    }
    out
}

// ── Force accumulation ────────────────────────────────────────────────────────

/// Accumulate dense voxel-wise forces to the control-point lattice.
///
/// Each CP receives a weighted average of the forces from voxels in its
/// B-spline support region:
///
///   `cp_force[c] = Σ_v w(v,c) · force[v]  /  Σ_v w(v,c)`
///
/// where `w(v,c) = Bₗ(u_z) Bₘ(u_y) Bₙ(u_x)` is the tensor-product weight.
pub(crate) fn accumulate_to_cp(
    force: &[f32],
    dims: [usize; 3],
    cp_dims: [usize; 3],
    control_spacing: [usize; 3],
) -> Vec<f32> {
    let cp_n = cp_dims[0] * cp_dims[1] * cp_dims[2];
    let mut accum = vec![0.0_f64; cp_n];
    let mut weight = vec![0.0_f64; cp_n];
    let [nz, ny, nx] = dims;

    for iz in 0..nz {
        let tz = iz as f64 / control_spacing[0] as f64;
        let sz = (tz.floor() as usize).min(cp_dims[0].saturating_sub(4));
        let uz = tz - sz as f64;

        for iy in 0..ny {
            let ty = iy as f64 / control_spacing[1] as f64;
            let sy = (ty.floor() as usize).min(cp_dims[1].saturating_sub(4));
            let uy = ty - sy as f64;

            for ix in 0..nx {
                let tx = ix as f64 / control_spacing[2] as f64;
                let sx = (tx.floor() as usize).min(cp_dims[2].saturating_sub(4));
                let ux = tx - sx as f64;

                let fv = force[flat(iz, iy, ix, ny, nx)] as f64;

                for kz in 0..4 {
                    let bz = bspline_basis(kz, uz);
                    for ky in 0..4 {
                        let bzy = bz * bspline_basis(ky, uy);
                        for kx in 0..4 {
                            let w = bzy * bspline_basis(kx, ux);
                            let ci = flat(sz + kz, sy + ky, sx + kx, cp_dims[1], cp_dims[2]);
                            accum[ci] += w * fv;
                            weight[ci] += w;
                        }
                    }
                }
            }
        }
    }

    accum
        .iter()
        .zip(weight.iter())
        .map(|(&a, &w)| if w > 1e-12 { (a / w) as f32 } else { 0.0 })
        .collect()
}

// ── Laplacian regularisation ──────────────────────────────────────────────────

/// Discrete 6-connected Laplacian on the control-point lattice.
///
/// `Δcp[i,j,k] = Σ_face_neighbours cp[n] − count · cp[i,j,k]`
///
/// At boundaries, missing neighbours are omitted and `count` is reduced
/// accordingly (Neumann-like boundary condition).
pub(crate) fn cp_laplacian(cp: &[f32], cp_dims: [usize; 3]) -> Vec<f32> {
    let [cnz, cny, cnx] = cp_dims;
    let cn = cnz * cny * cnx;
    let mut lap = vec![0.0_f32; cn];

    for ci in 0..cnz {
        for cj in 0..cny {
            for ck in 0..cnx {
                let idx = flat(ci, cj, ck, cny, cnx);
                let c = cp[idx] as f64;
                let mut sum = 0.0_f64;
                let mut cnt = 0u32;

                if ci > 0 {
                    sum += cp[flat(ci - 1, cj, ck, cny, cnx)] as f64;
                    cnt += 1;
                }
                if ci + 1 < cnz {
                    sum += cp[flat(ci + 1, cj, ck, cny, cnx)] as f64;
                    cnt += 1;
                }
                if cj > 0 {
                    sum += cp[flat(ci, cj - 1, ck, cny, cnx)] as f64;
                    cnt += 1;
                }
                if cj + 1 < cny {
                    sum += cp[flat(ci, cj + 1, ck, cny, cnx)] as f64;
                    cnt += 1;
                }
                if ck > 0 {
                    sum += cp[flat(ci, cj, ck - 1, cny, cnx)] as f64;
                    cnt += 1;
                }
                if ck + 1 < cnx {
                    sum += cp[flat(ci, cj, ck + 1, cny, cnx)] as f64;
                    cnt += 1;
                }

                lap[idx] = (sum - cnt as f64 * c) as f32;
            }
        }
    }
    lap
}
