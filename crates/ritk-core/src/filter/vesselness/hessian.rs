//! Finite-difference 3-D Hessian and analytic symmetric 3×3 eigendecomposition.
//!
//! # Coordinate convention
//! `dims = [nz, ny, nx]`, `spacing = [sz, sy, sx]`.
//! Flat index: `data[iz * ny * nx + iy * nx + ix]`.
//!
//! # Hessian component layout
//! Each output element is `[Hzz, Hzy, Hzx, Hyy, Hyx, Hxx]` — upper triangle,
//! row-major.  This encodes the symmetric matrix:
//!
//! ```text
//!   M = | Hzz  Hzy  Hzx |
//!       | Hzy  Hyy  Hyx |
//!       | Hzx  Hyx  Hxx |
//! ```

use std::f64::consts::PI;

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute all 6 independent components of the 3×3 Hessian matrix at every
/// voxel using second-order finite differences scaled by physical spacing.
///
/// # Arguments
/// - `data`    – flat slice stored in `[nz, ny, nx]` Z-major order.
/// - `dims`    – `[nz, ny, nx]`.
/// - `spacing` – `[sz, sy, sx]` physical distance between adjacent voxels (mm).
///
/// # Returns
/// `Vec<[f32; 6]>` of length `nz * ny * nx`.
/// Element layout per voxel: `[Hzz, Hzy, Hzx, Hyy, Hyx, Hxx]`.
///
/// # Stencils
/// **Diagonal** terms use the symmetric 3-point stencil
/// `(v₋ − 2v₀ + v₊) / h²`, exact for polynomials up to degree 3.
/// At boundary slice `i = 0` the stencil shifts to a one-sided forward
/// 2nd-difference `(v₀ − 2v₁ + v₂) / h²`; at `i = n−1` it shifts to the
/// mirror backward form.  Both one-sided variants are exact for quadratics.
///
/// **Cross** terms use the 4-point bilinear stencil
/// `(I[+a,+b] − I[+a,−b] − I[−a,+b] + I[−a,−b]) / (Δa · Δb)`.
/// At edges the `±` index is clamped to the nearest valid position and `Δa`,
/// `Δb` are derived from the actual index separation, yielding a valid
/// one-sided first-order approximation.
pub fn compute_hessian_3d(data: &[f32], dims: [usize; 3], spacing: [f64; 3]) -> Vec<[f32; 6]> {
    let [nz, ny, nx] = dims;
    let [sz, sy, sx] = spacing;

    // Pre-cast squared spacings to avoid repeated promotion in the inner loop.
    let sz2 = (sz * sz) as f32;
    let sy2 = (sy * sy) as f32;
    let sx2 = (sx * sx) as f32;

    let n_total = nz * ny * nx;
    let mut out = vec![[0.0f32; 6]; n_total];

    // Flat-index helper; captures ny and nx by copy (usize is Copy).
    let idx = |iz: usize, iy: usize, ix: usize| -> usize { iz * ny * nx + iy * nx + ix };

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                // ── Diagonal: Hzz ────────────────────────────────────────────
                let hzz = if nz <= 1 {
                    0.0f32
                } else {
                    let (a, b, c) = diag_triplet(nz, iz);
                    (data[idx(a, iy, ix)] - 2.0 * data[idx(b, iy, ix)] + data[idx(c, iy, ix)]) / sz2
                };

                // ── Diagonal: Hyy ────────────────────────────────────────────
                let hyy = if ny <= 1 {
                    0.0f32
                } else {
                    let (a, b, c) = diag_triplet(ny, iy);
                    (data[idx(iz, a, ix)] - 2.0 * data[idx(iz, b, ix)] + data[idx(iz, c, ix)]) / sy2
                };

                // ── Diagonal: Hxx ────────────────────────────────────────────
                let hxx = if nx <= 1 {
                    0.0f32
                } else {
                    let (a, b, c) = diag_triplet(nx, ix);
                    (data[idx(iz, iy, a)] - 2.0 * data[idx(iz, iy, b)] + data[idx(iz, iy, c)]) / sx2
                };

                // ── Cross: Hzy ───────────────────────────────────────────────
                let hzy = if nz <= 1 || ny <= 1 {
                    0.0f32
                } else {
                    let (izm, izp) = cross_clamp(nz, iz);
                    let (iym, iyp) = cross_clamp(ny, iy);
                    let dz = (izp - izm) as f64 * sz;
                    let dy = (iyp - iym) as f64 * sy;
                    ((data[idx(izp, iyp, ix)] - data[idx(izp, iym, ix)] - data[idx(izm, iyp, ix)]
                        + data[idx(izm, iym, ix)]) as f64
                        / (dz * dy)) as f32
                };

                // ── Cross: Hzx ───────────────────────────────────────────────
                let hzx = if nz <= 1 || nx <= 1 {
                    0.0f32
                } else {
                    let (izm, izp) = cross_clamp(nz, iz);
                    let (ixm, ixp) = cross_clamp(nx, ix);
                    let dz = (izp - izm) as f64 * sz;
                    let dx = (ixp - ixm) as f64 * sx;
                    ((data[idx(izp, iy, ixp)] - data[idx(izp, iy, ixm)] - data[idx(izm, iy, ixp)]
                        + data[idx(izm, iy, ixm)]) as f64
                        / (dz * dx)) as f32
                };

                // ── Cross: Hyx ───────────────────────────────────────────────
                let hyx = if ny <= 1 || nx <= 1 {
                    0.0f32
                } else {
                    let (iym, iyp) = cross_clamp(ny, iy);
                    let (ixm, ixp) = cross_clamp(nx, ix);
                    let dy = (iyp - iym) as f64 * sy;
                    let dx = (ixp - ixm) as f64 * sx;
                    ((data[idx(iz, iyp, ixp)] - data[idx(iz, iyp, ixm)] - data[idx(iz, iym, ixp)]
                        + data[idx(iz, iym, ixm)]) as f64
                        / (dy * dx)) as f32
                };

                out[idx(iz, iy, ix)] = [hzz, hzy, hzx, hyy, hyx, hxx];
            }
        }
    }

    out
}

/// Compute eigenvalues of a real symmetric 3×3 matrix analytically.
///
/// # Input layout
/// `h = [Hzz, Hzy, Hzx, Hyy, Hyx, Hxx]` — upper triangle, row-major.
///
/// ```text
///   M = | h[0]  h[1]  h[2] |
///       | h[1]  h[3]  h[4] |
///       | h[2]  h[4]  h[5] |
/// ```
///
/// # Output
/// Three eigenvalues sorted by **absolute value** ascending: |λ₁| ≤ |λ₂| ≤ |λ₃|.
///
/// # Algorithm
/// Closed-form trigonometric method (Smith 1961; Kopp 2008, arXiv:physics/0610206).
///
/// 1. If all off-diagonal elements are zero the matrix is diagonal and
///    eigenvalues are read directly.
/// 2. Otherwise:
///    - `q = trace(M) / 3`
///    - `p = sqrt(((M − q·I)² trace) / 6)`
///    - `r = det((M − q·I) / p) / 2`, clamped to `[−1, 1]`
///    - `φ = acos(r) / 3`
///    - `λ₁ = q + 2p·cos(φ)`,
///      `λ₃ = q + 2p·cos(φ + 2π/3)`,
///      `λ₂ = 3q − λ₁ − λ₃`
///
/// All arithmetic is performed in `f64` to minimise rounding error.
pub fn symmetric_3x3_eigenvalues(h: [f32; 6]) -> [f32; 3] {
    let m00 = h[0] as f64;
    let m01 = h[1] as f64;
    let m02 = h[2] as f64;
    let m11 = h[3] as f64;
    let m12 = h[4] as f64;
    let m22 = h[5] as f64;

    // Sum of squares of the three independent off-diagonal elements.
    let p1 = m01 * m01 + m02 * m02 + m12 * m12;

    let mut eigs: [f32; 3] = if p1 == 0.0 {
        // Diagonal matrix: eigenvalues are the diagonal entries.
        [m00 as f32, m11 as f32, m22 as f32]
    } else {
        let q = (m00 + m11 + m22) / 3.0;

        // p² = (1/6) · ||M − q·I||_F²
        let p2 = (m00 - q) * (m00 - q) + (m11 - q) * (m11 - q) + (m22 - q) * (m22 - q) + 2.0 * p1;
        let p = (p2 / 6.0).sqrt();

        if p < f64::EPSILON {
            // Matrix is numerically a scalar multiple of identity.
            return [q as f32, q as f32, q as f32];
        }

        // B = (M − q·I) / p  (symmetric, Frobenius norm ≈ √6)
        let b00 = (m00 - q) / p;
        let b01 = m01 / p;
        let b02 = m02 / p;
        let b11 = (m11 - q) / p;
        let b12 = m12 / p;
        let b22 = (m22 - q) / p;

        // det(B) via cofactor expansion along the first row.
        // det(B) = b00·(b11·b22 − b12²) − b01·(b01·b22 − b12·b02) + b02·(b01·b12 − b11·b02)
        let det_b = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02)
            + b02 * (b01 * b12 - b11 * b02);

        // r = det(B)/2; mathematically in [−1, 1]; clamp for numerical safety.
        let r = (det_b / 2.0).clamp(-1.0, 1.0);
        let phi = r.acos() / 3.0;

        let eig1 = q + 2.0 * p * phi.cos();
        let eig3 = q + 2.0 * p * (phi + 2.0 * PI / 3.0).cos();
        // Trace identity: eig2 = trace − eig1 − eig3 = 3q − eig1 − eig3.
        let eig2 = 3.0 * q - eig1 - eig3;

        [eig1 as f32, eig2 as f32, eig3 as f32]
    };

    // Sort by absolute value ascending: |λ₁| ≤ |λ₂| ≤ |λ₃|.
    eigs.sort_unstable_by(|a, b| {
        a.abs()
            .partial_cmp(&b.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    eigs
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Returns index triplet `(a, b, c)` for the 3-point 2nd-difference stencil
/// `(data[a] − 2·data[b] + data[c]) / h²` at position `i` in a dimension of
/// length `n ≥ 2`.
///
/// | Position   | Triplet (a, b, c)     | Stencil kind           |
/// |------------|-----------------------|------------------------|
/// | `i = 0`    | `(0, 1, 2)`           | Forward 2nd-difference |
/// | `0 < i < n−1` | `(i−1, i, i+1)`  | Central 2nd-difference |
/// | `i = n−1`  | `(n−3, n−2, n−1)`    | Backward 2nd-difference|
/// | `n = 2`, `i = 0` | `(0, 0, 1)`  | Degenerate clamped     |
/// | `n = 2`, `i = 1` | `(0, 1, 1)`  | Degenerate clamped     |
#[inline(always)]
fn diag_triplet(n: usize, i: usize) -> (usize, usize, usize) {
    debug_assert!(n >= 2, "diag_triplet requires n >= 2");
    if n == 2 {
        if i == 0 {
            (0, 0, 1)
        } else {
            (0, 1, 1)
        }
    } else if i == 0 {
        (0, 1, 2)
    } else if i == n - 1 {
        (n - 3, n - 2, n - 1)
    } else {
        (i - 1, i, i + 1)
    }
}

/// Returns `(i_minus, i_plus)` for the cross-term 4-point stencil, with
/// indices clamped to `[0, n−1]`.
///
/// The denominator for the cross term must be `(i_plus − i_minus) * spacing`
/// so that the formula degrades gracefully to a one-sided difference at edges.
#[inline(always)]
fn cross_clamp(n: usize, i: usize) -> (usize, usize) {
    (i.saturating_sub(1), (i + 1).min(n - 1))
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    // ── compute_hessian_3d ────────────────────────────────────────────────────

    /// Quadratic image `I[z, y, x] = x² + y² + z²`.
    ///
    /// Analytical second derivatives: Hxx = Hyy = Hzz = 2, all cross terms = 0.
    /// Central differences are exact for degree-2 polynomials.
    #[test]
    fn test_hessian_quadratic_separable() {
        const N: usize = 5;
        let mut data = vec![0.0f32; N * N * N];
        for iz in 0..N {
            for iy in 0..N {
                for ix in 0..N {
                    data[iz * N * N + iy * N + ix] = (ix * ix + iy * iy + iz * iz) as f32;
                }
            }
        }

        let h_out = compute_hessian_3d(&data, [N, N, N], [1.0, 1.0, 1.0]);

        // Interior voxel (iz=2, iy=2, ix=2): central differences apply.
        let voxel = 2 * N * N + 2 * N + 2;
        let [hzz, hzy, hzx, hyy, hyx, hxx] = h_out[voxel];

        assert!((hzz - 2.0).abs() < TOL, "Hzz: expected 2.0, got {hzz}");
        assert!((hyy - 2.0).abs() < TOL, "Hyy: expected 2.0, got {hyy}");
        assert!((hxx - 2.0).abs() < TOL, "Hxx: expected 2.0, got {hxx}");
        assert!(hzy.abs() < TOL, "Hzy: expected 0.0, got {hzy}");
        assert!(hzx.abs() < TOL, "Hzx: expected 0.0, got {hzx}");
        assert!(hyx.abs() < TOL, "Hyx: expected 0.0, got {hyx}");
    }

    /// Image `I[z, y, x] = x · y`.
    ///
    /// Analytical second derivatives: Hyx = 1, all others = 0.
    /// The 4-point bilinear stencil recovers the mixed derivative exactly.
    #[test]
    fn test_hessian_product_xy() {
        const N: usize = 5;
        let mut data = vec![0.0f32; N * N * N];
        for iz in 0..N {
            for iy in 0..N {
                for ix in 0..N {
                    data[iz * N * N + iy * N + ix] = (ix * iy) as f32;
                }
            }
        }

        let h_out = compute_hessian_3d(&data, [N, N, N], [1.0, 1.0, 1.0]);

        // Interior voxel (iz=2, iy=2, ix=2).
        let voxel = 2 * N * N + 2 * N + 2;
        let [hzz, hzy, hzx, hyy, hyx, hxx] = h_out[voxel];

        assert!((hyx - 1.0).abs() < TOL, "Hyx: expected 1.0, got {hyx}");
        assert!(hzz.abs() < TOL, "Hzz: expected 0.0, got {hzz}");
        assert!(hyy.abs() < TOL, "Hyy: expected 0.0, got {hyy}");
        assert!(hxx.abs() < TOL, "Hxx: expected 0.0, got {hxx}");
        assert!(hzy.abs() < TOL, "Hzy: expected 0.0, got {hzy}");
        assert!(hzx.abs() < TOL, "Hzx: expected 0.0, got {hzx}");
    }

    /// Non-unit spacing: `I[z,y,x] = x²` with spacing `[1.0, 1.0, 2.0]`.
    ///
    /// Hxx = 2.0 / sx² = 2.0 / 4.0 = 0.5 (physical second derivative).
    /// All other components = 0.
    #[test]
    fn test_hessian_nonunit_spacing() {
        const N: usize = 5;
        let mut data = vec![0.0f32; N * N * N];
        for iz in 0..N {
            for iy in 0..N {
                for ix in 0..N {
                    data[iz * N * N + iy * N + ix] = (ix * ix) as f32;
                }
            }
        }

        // spacing = [sz=1.0, sy=1.0, sx=2.0]
        let h_out = compute_hessian_3d(&data, [N, N, N], [1.0, 1.0, 2.0]);

        let voxel = 2 * N * N + 2 * N + 2;
        let [_hzz, _hzy, _hzx, _hyy, _hyx, hxx] = h_out[voxel];

        // Central diff gives (ix-1)² - 2*ix² + (ix+1)² = 2 (in voxels).
        // Physical Hxx = 2 / sx² = 2 / 4 = 0.5.
        assert!((hxx - 0.5).abs() < TOL, "Hxx: expected 0.5, got {hxx}");
    }

    // ── symmetric_3x3_eigenvalues ─────────────────────────────────────────────

    /// Diagonal matrix `[1, 0, 0, 2, 0, 3]` → sorted: `[1.0, 2.0, 3.0]`.
    #[test]
    fn test_eigenvalues_diagonal_positive() {
        let eigs = symmetric_3x3_eigenvalues([1.0f32, 0.0, 0.0, 2.0, 0.0, 3.0]);
        assert!(
            (eigs[0] - 1.0).abs() < TOL,
            "eig[0]: expected 1.0, got {}",
            eigs[0]
        );
        assert!(
            (eigs[1] - 2.0).abs() < TOL,
            "eig[1]: expected 2.0, got {}",
            eigs[1]
        );
        assert!(
            (eigs[2] - 3.0).abs() < TOL,
            "eig[2]: expected 3.0, got {}",
            eigs[2]
        );
    }

    /// Identity matrix → all eigenvalues 1.0.
    #[test]
    fn test_eigenvalues_identity() {
        let eigs = symmetric_3x3_eigenvalues([1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0]);
        for (i, &e) in eigs.iter().enumerate() {
            assert!((e - 1.0).abs() < TOL, "eig[{i}]: expected 1.0, got {e}");
        }
    }

    /// Negative diagonal `[-3, 0, 0, -1, 0, -2]` → sorted by |λ|: `[-1, -2, -3]`.
    #[test]
    fn test_eigenvalues_negative_diagonal() {
        let eigs = symmetric_3x3_eigenvalues([-3.0f32, 0.0, 0.0, -1.0, 0.0, -2.0]);
        assert!(
            (eigs[0] + 1.0).abs() < TOL,
            "eig[0]: expected -1.0, got {}",
            eigs[0]
        );
        assert!(
            (eigs[1] + 2.0).abs() < TOL,
            "eig[1]: expected -2.0, got {}",
            eigs[1]
        );
        assert!(
            (eigs[2] + 3.0).abs() < TOL,
            "eig[2]: expected -3.0, got {}",
            eigs[2]
        );
    }

    /// Non-trivial off-diagonal case.
    ///
    /// `M = [[2, 1, 0], [1, 2, 0], [0, 0, 3]]`  →  eigenvalues {1, 3, 3}.
    ///
    /// Input `h = [Hzz=2, Hzy=1, Hzx=0, Hyy=2, Hyx=0, Hxx=3]`.
    /// Sorted by |λ|: `[1.0, 3.0, 3.0]`.
    ///
    /// # Derivation
    /// The 2×2 block `[[2,1],[1,2]]` has eigenvalues `2±1 = {1, 3}`.
    /// The remaining diagonal element 3 is already decoupled.
    #[test]
    fn test_eigenvalues_offdiagonal_block() {
        let eigs = symmetric_3x3_eigenvalues([2.0f32, 1.0, 0.0, 2.0, 0.0, 3.0]);
        assert!(
            (eigs[0] - 1.0).abs() < TOL,
            "eig[0]: expected 1.0, got {}",
            eigs[0]
        );
        assert!(
            (eigs[1] - 3.0).abs() < TOL,
            "eig[1]: expected 3.0, got {}",
            eigs[1]
        );
        assert!(
            (eigs[2] - 3.0).abs() < TOL,
            "eig[2]: expected 3.0, got {}",
            eigs[2]
        );
    }

    /// Mixed-sign diagonal: sorted by absolute value, not by sign.
    ///
    /// `h = [-5, 0, 0, 2, 0, -1]` → sorted by |λ|: `[-1, 2, -5]`.
    #[test]
    fn test_eigenvalues_mixed_sign() {
        let eigs = symmetric_3x3_eigenvalues([-5.0f32, 0.0, 0.0, 2.0, 0.0, -1.0]);
        assert!(
            (eigs[0] + 1.0).abs() < TOL,
            "eig[0]: expected -1.0, got {}",
            eigs[0]
        );
        assert!(
            (eigs[1] - 2.0).abs() < TOL,
            "eig[1]: expected  2.0, got {}",
            eigs[1]
        );
        assert!(
            (eigs[2] + 5.0).abs() < TOL,
            "eig[2]: expected -5.0, got {}",
            eigs[2]
        );
    }
}
