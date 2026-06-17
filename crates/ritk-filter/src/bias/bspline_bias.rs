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
//! Partition of unity proof (B₀+B₁+B₂+B₃ = 1 for all u ∈ \[0,1\]):
//!     Sum = [(1−u)³ + 3u³−6u²+4 − 3u³+3u²+3u+1 + u³] / 6
//!         = [1 − 3u + 3u² − u³ + 3u³ − 6u² + 4 − 3u³ + 3u² + 3u + 1 + u³] / 6
//!         = [6 + 0·u + 0·u² + 0·u³] / 6 = 1. ∎
//!
//! **Trivariate surface value** at voxel (iz, iy, ix):
//! s = Σ_{a,b,c ∈ 0..4} Bz\[a\]·By\[b\]·Bx\[c\]·C\[(kz+a)·cy·cx + (ky+b)·cx + (kx+c)\]
//!
//! **Fitting**: Lee–Wolberg–Shin multilevel B-spline (scattered-data)
//! approximation — a single O(N·4³) pass distributing each data value to its 64
//! surrounding control points, with no global linear solve (the kernel ITK's
//! `BSplineScatteredDataPointSetToImageFilter` uses for N4). See [`bspline_fit`].

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
                for (a, &bza) in bz.iter().enumerate() {
                    for (b, &byb) in by.iter().enumerate() {
                        for (c, &bxc) in bx.iter().enumerate() {
                            let cp = (kz + a) * cy * cx + (ky + b) * cx + (kx + c);
                            val += bza * byb * bxc * control_points[cp];
                        }
                    }
                }
                result[iz * ny * nx + iy * nx + ix] = val as f32;
            }
        }
    }

    result
}

/// Denominator floor below which a control point is treated as unconstrained
/// (no data in its support) and assigned 0.
const MIN_SUPPORT_WEIGHT: f64 = 1e-12;

/// Fit a cubic B-spline surface to `residuals` via the Lee–Wolberg–Shin
/// multilevel B-spline (scattered-data) approximation.
///
/// This is the fitting kernel used by ITK's
/// `BSplineScatteredDataPointSetToImageFilter` (and hence by N4/ANTs): a single
/// O(N·4³) pass with no global linear solve. For each data point the value is
/// distributed to its 64 surrounding control points; the lattice value is the
/// support-weighted average of the local estimates.
///
/// # Algorithm
/// For data point `p` with value `f` and tensor-product basis weights
/// `B_c = Bz·By·Bx` over the 64 neighbouring control points `c`
/// (`W² = Σ_c B_c²`):
///   `numerator[c]   += B_c³ · f / W²`
///   `denominator[c] += B_c²`
///   `control[c]      = numerator[c] / denominator[c]`  (0 if no support).
/// Lee, S., Wolberg, G., Shin, S.Y. (1997). *Scattered Data Interpolation with
/// Multilevel B-Splines.* IEEE TVCG 3(3):228–244.
///
/// # Arguments
/// * `residuals` — z-major flat slice, length `nz·ny·nx`.
/// * `image_dims` — `[nz, ny, nx]`.
/// * `ctrl_grid` — `[cz, cy, cx]`, each ≥ 4.
pub fn bspline_fit(
    residuals: &[f32],
    image_dims: [usize; 3],
    ctrl_grid: [usize; 3],
) -> anyhow::Result<Vec<f64>> {
    let [nz, ny, nx] = image_dims;
    let [cz, cy, cx] = ctrl_grid;
    let n_total = nz * ny * nx;
    let n_cp = cz * cy * cx;
    debug_assert_eq!(residuals.len(), n_total, "residuals length mismatch");

    let mut numerator = vec![0.0f64; n_cp];
    let mut denominator = vec![0.0f64; n_cp];

    for vi in 0..n_total {
        let iz = vi / (ny * nx);
        let iy = (vi % (ny * nx)) / nx;
        let ix = vi % nx;
        let f = residuals[vi] as f64;

        let (kz, bz) = basis_and_span(iz, nz, cz);
        let (ky, by) = basis_and_span(iy, ny, cy);
        let (kx, bx) = basis_and_span(ix, nx, cx);

        // W² = Σ_c B_c² over the 4³ neighbourhood (separable: (Σbz²)(Σby²)(Σbx²)).
        let sz: f64 = bz.iter().map(|&b| b * b).sum();
        let sy: f64 = by.iter().map(|&b| b * b).sum();
        let sx: f64 = bx.iter().map(|&b| b * b).sum();
        let w2 = sz * sy * sx;
        if w2 < MIN_SUPPORT_WEIGHT {
            continue;
        }
        let inv_w2 = 1.0 / w2;

        for (a, &bza) in bz.iter().enumerate() {
            for (b, &byb) in by.iter().enumerate() {
                for (c, &bxc) in bx.iter().enumerate() {
                    let bc = bza * byb * bxc;
                    let bc2 = bc * bc;
                    let cp = (kz + a) * cy * cx + (ky + b) * cx + (kx + c);
                    // cp ∈ [0, n_cp) by construction (kz+a ≤ cz-1, etc.)
                    numerator[cp] += bc2 * bc * f * inv_w2;
                    denominator[cp] += bc2;
                }
            }
        }
    }

    let control = numerator
        .iter()
        .zip(denominator.iter())
        .map(|(&num, &den)| {
            if den > MIN_SUPPORT_WEIGHT {
                num / den
            } else {
                0.0
            }
        })
        .collect();

    Ok(control)
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
#[path = "tests_bspline_bias.rs"]
mod tests;
