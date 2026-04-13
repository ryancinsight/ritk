//! Cubic B-spline surface fitting and evaluation for bias field estimation.
//!
//! # Mathematical Specification
//!
//! Uniform cubic B-spline with 4 basis functions per span (de Boor, "A Practical
//! Guide to Splines", 1978).
//!
//! **Parameterization**: voxel index `i` in grid of size `n` with `c` control points:
//!   t(i)  = i · (c − 3) / max(n − 1, 1)
//!   k(t)  = clamp(⌊t⌋, 0, c − 4)          (span index)
//!   u(t)  = t − k(t)                        (local parameter ∈ [0, 1))
//!
//! **Cubic B-spline basis** (partition of unity):
//!   B₀(u) = (1 − u)³ / 6
//!   B₁(u) = (3u³ − 6u² + 4) / 6
//!   B₂(u) = (−3u³ + 3u² + 3u + 1) / 6
//!   B₃(u) = u³ / 6
//!
//!   Partition of unity proof (B₀+B₁+B₂+B₃ = 1 for all u ∈ [0,1]):
//!     Sum = [(1−u)³ + 3u³−6u²+4 − 3u³+3u²+3u+1 + u³] / 6
//!         = [1 − 3u + 3u² − u³ + 3u³ − 6u² + 4 − 3u³ + 3u² + 3u + 1 + u³] / 6
//!         = [6 + 0·u + 0·u² + 0·u³] / 6 = 1. ∎
//!
//! **Trivariate surface value** at voxel (iz, iy, ix):
//!   s = Σ_{a,b,c ∈ 0..4} Bz[a]·By[b]·Bx[c]·C[(kz+a)·cy·cx + (ky+b)·cx + (kx+c)]
//!
//! **Fitting**: minimise ‖Ac − r‖² + λ‖c‖² with λ = 1e-6 via Tikhonov-regularised
//! normal equations solved by nalgebra full-pivoting LU decomposition.
//! Each row of A contains exactly 64 nonzero entries (4³ tensor-product evaluations).

use nalgebra::{DMatrix, DVector};

// ── Public API ─────────────────────────────────────────────────────────────────

/// Evaluate cubic B-spline surface at all `nz·ny·nx` voxel positions.
///
/// # Arguments
/// * `control_points` — z-major flat array, length `cz·cy·cx`.
/// * `ctrl_grid`      — `[cz, cy, cx]`, each ≥ 4.
/// * `image_dims`     — `[nz, ny, nx]` voxel dimensions.
///
/// # Returns
/// Length `nz·ny·nx` values in z-major order (matching `image_dims` layout).
pub fn bspline_evaluate(
    control_points: &[f64],
    ctrl_grid: [usize; 3],
    image_dims: [usize; 3],
) -> Vec<f32> {
    let [nz, ny, nx] = image_dims;
    let [cz, cy, cx] = ctrl_grid;
    debug_assert_eq!(
        control_points.len(),
        cz * cy * cx,
        "control_points length must equal cz*cy*cx"
    );

    let mut result = vec![0.0f32; nz * ny * nx];

    for iz in 0..nz {
        let (kz, bz) = basis_and_span(iz, nz, cz);
        for iy in 0..ny {
            let (ky, by) = basis_and_span(iy, ny, cy);
            for ix in 0..nx {
                let (kx, bx) = basis_and_span(ix, nx, cx);
                let mut val = 0.0f64;
                for a in 0..4usize {
                    for b in 0..4usize {
                        for c in 0..4usize {
                            let cp = (kz + a) * cy * cx + (ky + b) * cx + (kx + c);
                            val += bz[a] * by[b] * bx[c] * control_points[cp];
                        }
                    }
                }
                result[iz * ny * nx + iy * nx + ix] = val as f32;
            }
        }
    }

    result
}

/// Fit a cubic B-spline surface to `residuals` via subsampled Tikhonov-regularised
/// normal equations.
///
/// Solves `(AᵀA + λI)c = Aᵀr` with `λ = 1e-6` using nalgebra full-pivoting LU.
/// Uniform subsampling with step = max(1, n / max_fitting_points).
/// Each row of A has exactly 64 nonzero entries (4³ tensor-product basis values).
///
/// # Arguments
/// * `residuals`          — z-major flat slice, length `nz·ny·nx`.
/// * `image_dims`         — `[nz, ny, nx]`.
/// * `ctrl_grid`          — `[cz, cy, cx]`, each ≥ 4.
/// * `max_fitting_points` — uniform subsampling target (upper bound on rows of A).
pub fn bspline_fit(
    residuals: &[f32],
    image_dims: [usize; 3],
    ctrl_grid: [usize; 3],
    max_fitting_points: usize,
) -> anyhow::Result<Vec<f64>> {
    let [nz, ny, nx] = image_dims;
    let [cz, cy, cx] = ctrl_grid;
    let n_total = nz * ny * nx;
    let n_cp = cz * cy * cx;
    debug_assert_eq!(residuals.len(), n_total, "residuals length mismatch");

    let step = (n_total / max_fitting_points.max(1)).max(1);
    let samples: Vec<usize> = (0..n_total).step_by(step).collect();
    let n_s = samples.len();

    if n_s == 0 {
        return Ok(vec![0.0f64; n_cp]);
    }

    // Build design matrix A ∈ ℝ^{n_s × n_cp} (row-major) and target vector r.
    let mut a_data = vec![0.0f64; n_s * n_cp];
    let mut r_data = vec![0.0f64; n_s];

    for (si, &vi) in samples.iter().enumerate() {
        let iz = vi / (ny * nx);
        let iy = (vi % (ny * nx)) / nx;
        let ix = vi % nx;
        r_data[si] = residuals[vi] as f64;

        let (kz, bz) = basis_and_span(iz, nz, cz);
        let (ky, by) = basis_and_span(iy, ny, cy);
        let (kx, bx) = basis_and_span(ix, nx, cx);

        for a in 0..4usize {
            for b in 0..4usize {
                for c in 0..4usize {
                    let cp = (kz + a) * cy * cx + (ky + b) * cx + (kx + c);
                    // cp ∈ [0, n_cp) by construction (kz+a ≤ cz-1, etc.)
                    a_data[si * n_cp + cp] += bz[a] * by[b] * bx[c];
                }
            }
        }
    }

    let a = DMatrix::from_row_slice(n_s, n_cp, &a_data);
    let r = DVector::from_vec(r_data);

    // Tikhonov-regularised normal equations: (AᵀA + λI)c = Aᵀr
    let ata = a.tr_mul(&a);
    let atr = a.tr_mul(&r);
    let lhs = ata + DMatrix::<f64>::identity(n_cp, n_cp) * 1e-6;

    let solution = lhs
        .lu()
        .solve(&atr)
        .ok_or_else(|| anyhow::anyhow!("B-spline normal equations singular; LU solve failed"))?;

    Ok(solution.as_slice().to_vec())
}

// ── Private helpers ────────────────────────────────────────────────────────────

/// Compute span index `k` and cubic B-spline basis `[B₀, B₁, B₂, B₃]` for
/// voxel index `i` in a grid of size `n` with `c` control points.
///
/// Invariants enforced:
/// - k ∈ [0, c − 4]
/// - u ∈ [0, 1]
/// - Σⱼ basis[j] = 1 (partition of unity)
#[inline]
fn basis_and_span(i: usize, n: usize, c: usize) -> (usize, [f64; 4]) {
    // Avoid usize underflow: cast to f64 before subtracting 1.
    let denom = (n as f64 - 1.0).max(1.0);
    let t = i as f64 * (c as f64 - 3.0) / denom;
    let k = (t.floor() as i64).clamp(0, (c as i64) - 4) as usize;
    let u = (t - k as f64).clamp(0.0, 1.0);
    (k, cubic_bspline_basis(u))
}

/// Cubic B-spline basis values at local parameter u ∈ [0, 1].
///
/// Partition of unity: B₀(u) + B₁(u) + B₂(u) + B₃(u) = 1 for all u.
#[inline]
fn cubic_bspline_basis(u: f64) -> [f64; 4] {
    let u2 = u * u;
    let u3 = u2 * u;
    [
        (1.0 - u).powi(3) / 6.0,
        (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0,
        (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0,
        u3 / 6.0,
    ]
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Partition of unity: if all control points = 1.0, every evaluated voxel = 1.0.
    ///
    /// Proof: s = Σ_{a,b,c} Bz[a]·By[b]·Bx[c]·1 = (ΣBz)·(ΣBy)·(ΣBx) = 1³ = 1. ∎
    #[test]
    fn constant_control_points_partition_of_unity() {
        let ctrl = vec![1.0f64; 4 * 4 * 4];
        let cg = [4usize, 4, 4];
        let dims = [10usize, 10, 10];

        let result = bspline_evaluate(&ctrl, cg, dims);

        assert_eq!(result.len(), 10 * 10 * 10);
        for (vi, &v) in result.iter().enumerate() {
            assert!(
                (v - 1.0_f32).abs() < 1e-5,
                "voxel {vi}: expected 1.0, got {v:.8}"
            );
        }
    }

    /// Partition of unity holds with larger control grid and non-cubic dimensions.
    #[test]
    fn partition_of_unity_larger_grid() {
        let cg = [6usize, 5, 7];
        let dims = [15usize, 12, 20];
        let ctrl = vec![1.0f64; 6 * 5 * 7];

        let result = bspline_evaluate(&ctrl, cg, dims);

        assert_eq!(result.len(), 15 * 12 * 20);
        for (vi, &v) in result.iter().enumerate() {
            assert!(
                (v - 1.0_f32).abs() < 1e-5,
                "voxel {vi}: expected 1.0, got {v:.8}"
            );
        }
    }

    /// Basis values sum to 1 for a dense sweep of u ∈ [0, 1].
    #[test]
    fn basis_sums_to_one_over_unit_interval() {
        for i in 0..=1000 {
            let u = i as f64 / 1000.0;
            let b = cubic_bspline_basis(u);
            let sum = b[0] + b[1] + b[2] + b[3];
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "u={u:.4}: basis sum = {sum:.15} ≠ 1"
            );
        }
    }

    /// Round-trip: fit a trilinear field (in the B-spline span) then re-evaluate.
    /// A trilinear function is in the span of the cubic B-spline basis, so the
    /// Tikhonov-regularised fit (λ=1e-6) must recover it to RMS < 0.1.
    #[test]
    fn round_trip_linear_field_rms_below_threshold() {
        let dims = [10usize, 10, 10];
        let cg = [6usize, 6, 6];
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;

        // Trilinear field: f(iz,iy,ix) ∈ [0, 0.65].
        let field: Vec<f32> = (0..n)
            .map(|vi| {
                let iz = vi / (ny * nx);
                let iy = (vi % (ny * nx)) / nx;
                let ix = vi % nx;
                0.30 * (iz as f32 / (nz - 1) as f32)
                    + 0.20 * (iy as f32 / (ny - 1) as f32)
                    + 0.15 * (ix as f32 / (nx - 1) as f32)
            })
            .collect();

        let ctrl = bspline_fit(&field, dims, cg, 10_000).expect("bspline_fit failed");
        let approx = bspline_evaluate(&ctrl, cg, dims);

        let rms = (field
            .iter()
            .zip(approx.iter())
            .map(|(&r, &a)| ((r - a) as f64).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();

        assert!(rms < 0.1, "round-trip RMS {rms:.6} ≥ 0.1");
    }

    /// Fit to a constant field yields control points that reconstruct to ≈ constant.
    #[test]
    fn round_trip_constant_field() {
        let dims = [8usize, 8, 8];
        let cg = [4usize, 4, 4];
        let n = 8 * 8 * 8;
        let field = vec![2.5f32; n];

        let ctrl = bspline_fit(&field, dims, cg, 10_000).expect("bspline_fit failed");
        let approx = bspline_evaluate(&ctrl, cg, dims);

        for (vi, &v) in approx.iter().enumerate() {
            assert!(
                (v - 2.5_f32).abs() < 0.05,
                "voxel {vi}: expected ~2.5, got {v:.6}"
            );
        }
    }
}
