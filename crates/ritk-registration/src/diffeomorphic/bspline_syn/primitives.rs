//! Cubic B-spline primitives: basis evaluation, control-point layout,
//! dense-field synthesis, force accumulation, and Laplacian regularisation.
//!
//! # Cubic B-Spline Basis
//!
//! Uniform cubic B-spline basis over parameter `u ∈ [0,1)`:
//! - `B₀(u) = (1 − u)³ / 6`
//! - `B₁(u) = (3u³ − 6u² + 4) / 6`
//! - `B₂(u) = (−3u³ + 3u² + 3u + 1) / 6`
//! - `B₃(u) = u³ / 6`
//!
//! **Partition of unity**: `Σₖ Bₖ(u) = 1 ∀ u ∈ [0,1]`.

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

/// Zero-allocation variant of [`evaluate_dense`]: writes the dense displacement
/// field into a caller-provided buffer.
pub(crate) fn evaluate_dense_into(
    cp: &[f32],
    cp_dims: [usize; 3],
    dims: [usize; 3],
    control_spacing: [usize; 3],
    out: &mut [f32],
) {
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
}

/// Evaluate a single dense displacement-field component from its
/// control-point lattice via cubic B-spline evaluation.
///
/// Delegates to [`evaluate_dense_into`] with a freshly allocated buffer.
#[cfg(test)]
pub(crate) fn evaluate_dense(
    cp: &[f32],
    cp_dims: [usize; 3],
    dims: [usize; 3],
    control_spacing: [usize; 3],
) -> Vec<f32> {
    let n = dims[0] * dims[1] * dims[2];
    let mut out = vec![0.0_f32; n];
    evaluate_dense_into(cp, cp_dims, dims, control_spacing, &mut out);
    out
}

// ── Force accumulation ────────────────────────────────────────────────────────

/// Zero-allocation variant of [`accumulate_to_cp`]: reuses caller-provided
/// `accum` and `weight` temporaries and writes the result into `out`.
pub(crate) fn accumulate_to_cp_into(
    force: &[f32],
    dims: [usize; 3],
    cp_dims: [usize; 3],
    control_spacing: [usize; 3],
    accum: &mut [f64],
    weight: &mut [f64],
    out: &mut [f32],
) {
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
    for i in 0..out.len() {
        out[i] = if weight[i] > 1e-12 {
            (accum[i] / weight[i]) as f32
        } else {
            0.0
        };
    }
}

/// Accumulate dense voxel-wise forces to the control-point lattice.
///
/// Delegates to [`accumulate_to_cp_into`] with freshly allocated temporaries.
#[cfg(test)]
pub(crate) fn accumulate_to_cp(
    force: &[f32],
    dims: [usize; 3],
    cp_dims: [usize; 3],
    control_spacing: [usize; 3],
) -> Vec<f32> {
    let cp_n = cp_dims[0] * cp_dims[1] * cp_dims[2];
    let mut accum = vec![0.0_f64; cp_n];
    let mut weight = vec![0.0_f64; cp_n];
    let mut out = vec![0.0_f32; cp_n];
    accumulate_to_cp_into(
        force,
        dims,
        cp_dims,
        control_spacing,
        &mut accum,
        &mut weight,
        &mut out,
    );
    out
}

// ── Laplacian regularisation ──────────────────────────────────────────────────

/// Zero-allocation variant of [`cp_laplacian`]: writes the Laplacian into a
/// caller-provided buffer.
pub(crate) fn cp_laplacian_into(cp: &[f32], cp_dims: [usize; 3], out: &mut [f32]) {
    let [cnz, cny, cnx] = cp_dims;
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
                out[idx] = (sum - cnt as f64 * c) as f32;
            }
        }
    }
}

/// Discrete 6-connected Laplacian on the control-point lattice.
///
/// Delegates to [`cp_laplacian_into`] with a freshly allocated buffer.
#[cfg(test)]
pub(crate) fn cp_laplacian(cp: &[f32], cp_dims: [usize; 3]) -> Vec<f32> {
    let cn = cp_dims[0] * cp_dims[1] * cp_dims[2];
    let mut out = vec![0.0_f32; cn];
    cp_laplacian_into(cp, cp_dims, &mut out);
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: small 3D grid dimensions and control spacing for tests.
    fn test_params() -> ([usize; 3], [usize; 3], [usize; 3]) {
        let dims = [4, 4, 4];
        let spacing = [2, 2, 2];
        let cp_d = [
            cp_count(dims[0], spacing[0]),
            cp_count(dims[1], spacing[1]),
            cp_count(dims[2], spacing[2]),
        ];
        (dims, spacing, cp_d)
    }

    #[test]
    fn evaluate_dense_into_matches_evaluate_dense() {
        let (dims, spacing, cp_d) = test_params();
        let cp_n = cp_d[0] * cp_d[1] * cp_d[2];
        let cp: Vec<f32> = (0..cp_n).map(|i| i as f32 * 0.1 - 1.0).collect();
        let expected = evaluate_dense(&cp, cp_d, dims, spacing);
        let n = dims[0] * dims[1] * dims[2];
        let mut out = vec![0.0_f32; n];
        evaluate_dense_into(&cp, cp_d, dims, spacing, &mut out);
        for i in 0..n {
            let diff = (out[i] - expected[i]).abs();
            assert!(
                diff <= 1e-6,
                "mismatch at {}: {} vs {}",
                i,
                out[i],
                expected[i]
            );
        }
    }

    #[test]
    fn accumulate_to_cp_into_matches_accumulate_to_cp() {
        let (dims, spacing, cp_d) = test_params();
        let n = dims[0] * dims[1] * dims[2];
        let cp_n = cp_d[0] * cp_d[1] * cp_d[2];
        let force: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let expected = accumulate_to_cp(&force, dims, cp_d, spacing);
        let mut accum = vec![0.0_f64; cp_n];
        let mut weight = vec![0.0_f64; cp_n];
        let mut out = vec![0.0_f32; cp_n];
        accumulate_to_cp_into(
            &force,
            dims,
            cp_d,
            spacing,
            &mut accum,
            &mut weight,
            &mut out,
        );
        for i in 0..cp_n {
            let diff = (out[i] - expected[i]).abs();
            assert!(
                diff <= 1e-6,
                "mismatch at {}: {} vs {}",
                i,
                out[i],
                expected[i]
            );
        }
    }

    #[test]
    fn cp_laplacian_into_matches_cp_laplacian() {
        let (_, _, cp_d) = test_params();
        let cp_n = cp_d[0] * cp_d[1] * cp_d[2];
        let cp: Vec<f32> = (0..cp_n).map(|i| i as f32 * 0.5 - 2.0).collect();
        let expected = cp_laplacian(&cp, cp_d);
        let mut out = vec![0.0_f32; cp_n];
        cp_laplacian_into(&cp, cp_d, &mut out);
        for i in 0..cp_n {
            let diff = (out[i] - expected[i]).abs();
            assert!(
                diff <= 1e-6,
                "mismatch at {}: {} vs {}",
                i,
                out[i],
                expected[i]
            );
        }
    }
}
