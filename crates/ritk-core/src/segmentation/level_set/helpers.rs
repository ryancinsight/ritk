//! Single source of truth (SSOT) for level-set numerical primitives.
//!
//! Every finite-difference operator, smoothing kernel, and regularisation
//! function used by Chan–Vese, Geodesic Active Contour, Shape Detection,
//! and Threshold Level Set lives here.  All helpers operate on `f64`.
//! Boundary conditions are clamped (Neumann) throughout: out-of-bounds
//! indices are replaced by the nearest in-bounds neighbour.
//!
//! # Contents
//!
//! | Category | Functions |
//! |---|---|
//! | Indexing | [`idx_clamped`] |
//! | Differential geometry | [`compute_curvature_into`] |
//! | Gradient operators | [`compute_gradient_magnitude`], [`compute_field_gradient`] |
//! | Edge / speed functions | [`compute_edge_stopping`] |
//! | Smoothing | [`gaussian_smooth_3d`], [`build_gaussian_kernel_1d`] |
//! | Regularisation | [`regularised_heaviside`], [`regularised_dirac`] |

use std::f64::consts::PI;

// ── Indexing ────────────────────────────────────────────────────────────────────────

/// Clamped 3-D → linear index.
///
/// Each axis coordinate is clamped to `[0, dim-1]` before computing the
/// row-major linear index `z * ny * nx + y * nx + x`.
#[inline]
pub(crate) fn idx_clamped(
    z: isize,
    y: isize,
    x: isize,
    nz: usize,
    ny: usize,
    nx: usize,
) -> usize {
    let cz = z.clamp(0, nz as isize - 1) as usize;
    let cy = y.clamp(0, ny as isize - 1) as usize;
    let cx = x.clamp(0, nx as isize - 1) as usize;
    cz * ny * nx + cy * nx + cx
}

// ── Curvature ──────────────────────────────────────────────────────────────────────

/// Compute mean curvature κ = div(∇φ / |∇φ|) into a pre-allocated buffer.
///
/// Uses second-order central finite differences on a regular grid with
/// clamped (Neumann) boundary conditions.
///
/// # Formula
///
/// ```text
/// num = φ_xx·(φ_y² + φ_z²)
///     + φ_yy·(φ_x² + φ_z²)
///     + φ_zz·(φ_x² + φ_y²)
///     − 2·φ_x·φ_y·φ_xy
///     − 2·φ_x·φ_z·φ_xz
///     − 2·φ_y·φ_z·φ_yz
///
/// grad_sq      = φ_x² + φ_y² + φ_z²
/// grad_sq_safe = grad_sq + 1e-10
/// grad_mag     = sqrt(grad_sq_safe)
///
/// κ = num / (grad_sq_safe · grad_mag)
/// ```
///
/// The ε = 1e-10 additive term prevents division by zero in flat regions
/// and makes `κ → 0` as `|∇φ| → 0`.
pub(crate) fn compute_curvature_into(phi: &[f64], dims: [usize; 3], kappa: &mut [f64]) {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    debug_assert_eq!(phi.len(), n, "phi length mismatch: {} vs {}", phi.len(), n);
    debug_assert_eq!(
        kappa.len(),
        n,
        "kappa length mismatch: {} vs {}",
        kappa.len(),
        n
    );

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let i = iz * ny * nx + iy * nx + ix;
                let zz = iz as isize;
                let yy = iy as isize;
                let xx = ix as isize;

                // Neighbour values (clamped).
                let phi_xp = phi[idx_clamped(zz, yy, xx + 1, nz, ny, nx)];
                let phi_xm = phi[idx_clamped(zz, yy, xx - 1, nz, ny, nx)];
                let phi_yp = phi[idx_clamped(zz, yy + 1, xx, nz, ny, nx)];
                let phi_ym = phi[idx_clamped(zz, yy - 1, xx, nz, ny, nx)];
                let phi_zp = phi[idx_clamped(zz + 1, yy, xx, nz, ny, nx)];
                let phi_zm = phi[idx_clamped(zz - 1, yy, xx, nz, ny, nx)];

                // First derivatives (central).
                let phi_x = (phi_xp - phi_xm) * 0.5;
                let phi_y = (phi_yp - phi_ym) * 0.5;
                let phi_z = (phi_zp - phi_zm) * 0.5;

                // Second derivatives.
                let phi_xx = phi_xp - 2.0 * phi[i] + phi_xm;
                let phi_yy = phi_yp - 2.0 * phi[i] + phi_ym;
                let phi_zz = phi_zp - 2.0 * phi[i] + phi_zm;

                // Cross derivatives.
                let phi_xy = (phi[idx_clamped(zz, yy + 1, xx + 1, nz, ny, nx)]
                    - phi[idx_clamped(zz, yy + 1, xx - 1, nz, ny, nx)]
                    - phi[idx_clamped(zz, yy - 1, xx + 1, nz, ny, nx)]
                    + phi[idx_clamped(zz, yy - 1, xx - 1, nz, ny, nx)])
                    * 0.25;

                let phi_xz = (phi[idx_clamped(zz + 1, yy, xx + 1, nz, ny, nx)]
                    - phi[idx_clamped(zz + 1, yy, xx - 1, nz, ny, nx)]
                    - phi[idx_clamped(zz - 1, yy, xx + 1, nz, ny, nx)]
                    + phi[idx_clamped(zz - 1, yy, xx - 1, nz, ny, nx)])
                    * 0.25;

                let phi_yz = (phi[idx_clamped(zz + 1, yy + 1, xx, nz, ny, nx)]
                    - phi[idx_clamped(zz + 1, yy - 1, xx, nz, ny, nx)]
                    - phi[idx_clamped(zz - 1, yy + 1, xx, nz, ny, nx)]
                    + phi[idx_clamped(zz - 1, yy - 1, xx, nz, ny, nx)])
                    * 0.25;

                let grad_sq = phi_x * phi_x + phi_y * phi_y + phi_z * phi_z;
                let grad_sq_safe = grad_sq + 1e-10;
                let grad_mag = grad_sq_safe.sqrt();

                let numerator = phi_xx * (phi_y * phi_y + phi_z * phi_z)
                    + phi_yy * (phi_x * phi_x + phi_z * phi_z)
                    + phi_zz * (phi_x * phi_x + phi_y * phi_y)
                    - 2.0 * phi_x * phi_y * phi_xy
                    - 2.0 * phi_x * phi_z * phi_xz
                    - 2.0 * phi_y * phi_z * phi_yz;

                kappa[i] = numerator / (grad_sq_safe * grad_mag);
            }
        }
    }
}

// ── Gradient operators ─────────────────────────────────────────────────────────────

/// Gradient magnitude |∇f| via central finite differences with clamped
/// boundary conditions.
///
/// ```text
/// |∇f|_i = sqrt( (∂f/∂z)² + (∂f/∂y)² + (∂f/∂x)² )
/// ```
pub(crate) fn compute_gradient_magnitude(data: &[f64], dims: [usize; 3]) -> Vec<f64> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut grad = vec![0.0_f64; n];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let i = iz * ny * nx + iy * nx + ix;
                let zz = iz as isize;
                let yy = iy as isize;
                let xx = ix as isize;

                let dz = (data[idx_clamped(zz + 1, yy, xx, nz, ny, nx)]
                    - data[idx_clamped(zz - 1, yy, xx, nz, ny, nx)])
                    * 0.5;
                let dy = (data[idx_clamped(zz, yy + 1, xx, nz, ny, nx)]
                    - data[idx_clamped(zz, yy - 1, xx, nz, ny, nx)])
                    * 0.5;
                let dx = (data[idx_clamped(zz, yy, xx + 1, nz, ny, nx)]
                    - data[idx_clamped(zz, yy, xx - 1, nz, ny, nx)])
                    * 0.5;

                grad[i] = (dz * dz + dy * dy + dx * dx).sqrt();
            }
        }
    }

    grad
}

/// Component-wise gradient of a scalar field via central finite differences.
///
/// Returns `(∂f/∂z, ∂f/∂y, ∂f/∂x)` as three flat vectors in row-major order.
pub(crate) fn compute_field_gradient(
    data: &[f64],
    dims: [usize; 3],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut gz = vec![0.0_f64; n];
    let mut gy = vec![0.0_f64; n];
    let mut gx = vec![0.0_f64; n];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let i = iz * ny * nx + iy * nx + ix;
                let zz = iz as isize;
                let yy = iy as isize;
                let xx = ix as isize;

                gz[i] = (data[idx_clamped(zz + 1, yy, xx, nz, ny, nx)]
                    - data[idx_clamped(zz - 1, yy, xx, nz, ny, nx)])
                    * 0.5;
                gy[i] = (data[idx_clamped(zz, yy + 1, xx, nz, ny, nx)]
                    - data[idx_clamped(zz, yy - 1, xx, nz, ny, nx)])
                    * 0.5;
                gx[i] = (data[idx_clamped(zz, yy, xx + 1, nz, ny, nx)]
                    - data[idx_clamped(zz, yy, xx - 1, nz, ny, nx)])
                    * 0.5;
            }
        }
    }

    (gz, gy, gx)
}

// ── Edge / speed functions ─────────────────────────────────────────────────────────

/// Edge stopping function.
///
/// ```text
/// g(s) = 1 / (1 + (s / k)²)
/// ```
///
/// # Invariant
///
/// ∀ s ≥ 0 : 0 < g(s) ≤ 1,  g(0) = 1,  lim_{s→∞} g(s) = 0.
pub(crate) fn compute_edge_stopping(grad_mag: &[f64], k: f64) -> Vec<f64> {
    let k2 = k * k;
    grad_mag
        .iter()
        .map(|&s| 1.0 / (1.0 + (s * s) / k2))
        .collect()
}

// ── Gaussian smoothing ─────────────────────────────────────────────────────────────

/// Separable 3-D Gaussian smoothing with clamped boundary conditions.
///
/// Kernel radius = ⌈3σ⌉.  If σ ≤ 0, returns a copy of the input unchanged.
pub(crate) fn gaussian_smooth_3d(data: &[f64], dims: [usize; 3], sigma: f64) -> Vec<f64> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;

    if sigma <= 0.0 {
        return data.to_vec();
    }

    let radius = (3.0 * sigma).ceil() as usize;
    let kernel = build_gaussian_kernel_1d(sigma, radius);
    let r = radius as isize;

    // Separable: smooth along x, then y, then z.
    let mut buf = data.to_vec();
    let mut tmp = vec![0.0_f64; n];

    // Pass 1 — smooth along x (axis 2).
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut sum = 0.0_f64;
                for (ki, &w) in kernel.iter().enumerate() {
                    let dx = ki as isize - r;
                    let sx = (ix as isize + dx).clamp(0, nx as isize - 1) as usize;
                    sum += w * buf[iz * ny * nx + iy * nx + sx];
                }
                tmp[iz * ny * nx + iy * nx + ix] = sum;
            }
        }
    }
    std::mem::swap(&mut buf, &mut tmp);

    // Pass 2 — smooth along y (axis 1).
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut sum = 0.0_f64;
                for (ki, &w) in kernel.iter().enumerate() {
                    let dy = ki as isize - r;
                    let sy = (iy as isize + dy).clamp(0, ny as isize - 1) as usize;
                    sum += w * buf[iz * ny * nx + sy * nx + ix];
                }
                tmp[iz * ny * nx + iy * nx + ix] = sum;
            }
        }
    }
    std::mem::swap(&mut buf, &mut tmp);

    // Pass 3 — smooth along z (axis 0).
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut sum = 0.0_f64;
                for (ki, &w) in kernel.iter().enumerate() {
                    let dz = ki as isize - r;
                    let sz = (iz as isize + dz).clamp(0, nz as isize - 1) as usize;
                    sum += w * buf[sz * ny * nx + iy * nx + ix];
                }
                tmp[iz * ny * nx + iy * nx + ix] = sum;
            }
        }
    }

    tmp
}

/// Normalised 1-D Gaussian kernel.
///
/// Length = `2 * radius + 1`.  Weights sum to 1.
///
/// ```text
/// w_i = exp(−d² / (2σ²)) / Z,   d = i − radius
/// ```
pub(crate) fn build_gaussian_kernel_1d(sigma: f64, radius: usize) -> Vec<f64> {
    let len = 2 * radius + 1;
    let inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);
    let mut kernel = Vec::with_capacity(len);
    let mut sum = 0.0_f64;

    for i in 0..len {
        let d = i as f64 - radius as f64;
        let w = (-d * d * inv_2sigma2).exp();
        kernel.push(w);
        sum += w;
    }

    let inv_sum = 1.0 / sum;
    for w in &mut kernel {
        *w *= inv_sum;
    }

    kernel
}

// ── Regularised Heaviside / Dirac ──────────────────────────────────────────────────

/// Regularised Heaviside function.
///
/// ```text
/// H_ε(z) = 0.5 · (1 + (2/π) · arctan(z / ε))
/// ```
///
/// Smoothly transitions from 0 to 1 across `z = 0` with width controlled
/// by ε.
#[inline]
pub(crate) fn regularised_heaviside(z: f64, eps: f64) -> f64 {
    0.5 * (1.0 + (2.0 / PI) * (z / eps).atan())
}

/// Regularised Dirac delta.
///
/// ```text
/// δ_ε(z) = (ε / π) / (ε² + z²)
/// ```
///
/// This is the derivative of [`regularised_heaviside`] with respect to `z`.
/// Positive everywhere, symmetric about `z = 0`, peak value `1 / (πε)`.
#[inline]
pub(crate) fn regularised_dirac(z: f64, eps: f64) -> f64 {
    (eps / PI) / (eps * eps + z * z)
}

// ── Tests ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ── idx_clamped ────────────────────────────────────────────────────────────

    #[test]
    fn test_idx_clamped_interior() {
        assert_eq!(idx_clamped(1, 2, 3, 4, 5, 6), 1 * 5 * 6 + 2 * 6 + 3);
    }

    #[test]
    fn test_idx_clamped_negative() {
        assert_eq!(idx_clamped(-1, -2, -3, 4, 5, 6), 0);
    }

    #[test]
    fn test_idx_clamped_overflow() {
        assert_eq!(
            idx_clamped(10, 10, 10, 4, 5, 6),
            3 * 5 * 6 + 4 * 6 + 5
        );
    }

    // ── Gaussian kernel ────────────────────────────────────────────────────────

    #[test]
    fn test_gaussian_kernel_normalised() {
        let kernel = build_gaussian_kernel_1d(1.5, 5);
        let sum: f64 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "kernel sum = {sum}");
    }

    #[test]
    fn test_gaussian_kernel_length() {
        let kernel = build_gaussian_kernel_1d(1.0, 5);
        assert_eq!(kernel.len(), 11);
    }

    #[test]
    fn test_gaussian_kernel_symmetric() {
        let kernel = build_gaussian_kernel_1d(2.0, 6);
        let n = kernel.len();
        for i in 0..n {
            assert!(
                (kernel[i] - kernel[n - 1 - i]).abs() < 1e-15,
                "asymmetry at i={i}: {} vs {}",
                kernel[i],
                kernel[n - 1 - i]
            );
        }
    }

    #[test]
    fn test_gaussian_kernel_peak_at_center() {
        let kernel = build_gaussian_kernel_1d(1.0, 3);
        let center = kernel.len() / 2;
        for (i, &w) in kernel.iter().enumerate() {
            if i != center {
                assert!(
                    kernel[center] >= w,
                    "center {} < kernel[{i}] = {w}",
                    kernel[center]
                );
            }
        }
    }

    // ── Regularised Heaviside ──────────────────────────────────────────────────

    #[test]
    fn test_heaviside_at_zero() {
        assert!((regularised_heaviside(0.0, 1.0) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_heaviside_positive_large() {
        let h = regularised_heaviside(1e6, 1.0);
        assert!((h - 1.0).abs() < 1e-6, "H(1e6) = {h}");
    }

    #[test]
    fn test_heaviside_negative_large() {
        let h = regularised_heaviside(-1e6, 1.0);
        assert!(h.abs() < 1e-6, "H(-1e6) = {h}");
    }

    #[test]
    fn test_heaviside_monotone() {
        let vals: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.1).collect();
        for w in vals.windows(2) {
            assert!(
                regularised_heaviside(w[1], 1.0) >= regularised_heaviside(w[0], 1.0),
                "monotonicity violated at z = {}",
                w[0]
            );
        }
    }

    // ── Regularised Dirac ──────────────────────────────────────────────────────

    #[test]
    fn test_dirac_peak_at_zero() {
        let eps = 1.0;
        let expected = 1.0 / (PI * eps);
        let got = regularised_dirac(0.0, eps);
        assert!(
            (got - expected).abs() < 1e-15,
            "delta(0,1) = {got}, expected {expected}"
        );
    }

    #[test]
    fn test_dirac_positive_everywhere() {
        for i in -100..=100 {
            let z = i as f64 * 0.3;
            assert!(
                regularised_dirac(z, 1.0) > 0.0,
                "delta({z}) <= 0"
            );
        }
    }

    #[test]
    fn test_dirac_symmetric() {
        for i in 1..=50 {
            let z = i as f64 * 0.2;
            let d_pos = regularised_dirac(z, 1.0);
            let d_neg = regularised_dirac(-z, 1.0);
            assert!(
                (d_pos - d_neg).abs() < 1e-15,
                "asymmetry at z = {z}: {d_pos} vs {d_neg}"
            );
        }
    }

    // ── Curvature ──────────────────────────────────────────────────────────────

    #[test]
    fn test_curvature_flat_phi_is_zero() {
        let dims = [5, 5, 5];
        let n = 125;
        let phi = vec![7.0_f64; n];
        let mut kappa = vec![0.0_f64; n];
        compute_curvature_into(&phi, dims, &mut kappa);
        for (i, &k) in kappa.iter().enumerate() {
            assert!(k.abs() < 1e-6, "kappa[{i}] = {k} for constant phi");
        }
    }

    #[test]
    fn test_curvature_sphere_positive() {
        // Signed distance function for a sphere of radius R=3 centred in
        // an 11^3 grid.  Analytical mean curvature on the surface = 2/R.
        let side = 11_usize;
        let dims = [side, side, side];
        let n = side * side * side;
        let center = 5.0_f64;
        let r = 3.0_f64;

        let mut phi = vec![0.0_f64; n];
        for iz in 0..side {
            for iy in 0..side {
                for ix in 0..side {
                    let dz = iz as f64 - center;
                    let dy = iy as f64 - center;
                    let dx = ix as f64 - center;
                    phi[iz * side * side + iy * side + ix] =
                        (dz * dz + dy * dy + dx * dx).sqrt() - r;
                }
            }
        }

        let mut kappa = vec![0.0_f64; n];
        compute_curvature_into(&phi, dims, &mut kappa);

        let expected_kappa = 2.0 / r; // 0.6667
        // Check voxels on the surface (|phi| < 0.6).
        let mut checked = 0_usize;
        for iz in 1..side - 1 {
            for iy in 1..side - 1 {
                for ix in 1..side - 1 {
                    let idx = iz * side * side + iy * side + ix;
                    if phi[idx].abs() < 0.6 {
                        let err = (kappa[idx] - expected_kappa).abs();
                        assert!(
                            err < 0.15,
                            "kappa[{iz},{iy},{ix}] = {}, expected ~{expected_kappa}, err = {err}",
                            kappa[idx]
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 0, "no surface voxels found");
    }

    // ── Edge stopping ──────────────────────────────────────────────────────────

    #[test]
    fn test_edge_stopping_at_zero() {
        let g = compute_edge_stopping(&[0.0], 1.0);
        assert!((g[0] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_edge_stopping_large_gradient() {
        let g = compute_edge_stopping(&[1e6], 1.0);
        assert!(g[0] < 1e-10, "g(1e6) = {}", g[0]);
    }

    #[test]
    fn test_edge_stopping_bounded() {
        let vals: Vec<f64> = (0..200).map(|i| i as f64 * 0.5).collect();
        let g = compute_edge_stopping(&vals, 5.0);
        for (i, &gi) in g.iter().enumerate() {
            assert!(gi > 0.0, "g[{i}] = {gi} <= 0");
            assert!(gi <= 1.0, "g[{i}] = {gi} > 1");
        }
    }

    // ── Gaussian smoothing ─────────────────────────────────────────────────────

    #[test]
    fn test_gaussian_smooth_identity_for_constant() {
        let dims = [4, 5, 6];
        let n = 4 * 5 * 6;
        let data = vec![42.0_f64; n];
        let smoothed = gaussian_smooth_3d(&data, dims, 1.5);
        for (i, &v) in smoothed.iter().enumerate() {
            assert!(
                (v - 42.0).abs() < 1e-10,
                "smoothed[{i}] = {v}, expected 42.0"
            );
        }
    }

    #[test]
    fn test_gaussian_smooth_sigma_zero_is_identity() {
        let dims = [3, 3, 3];
        let data: Vec<f64> = (0..27).map(|i| i as f64).collect();
        let smoothed = gaussian_smooth_3d(&data, dims, 0.0);
        assert_eq!(smoothed, data);
    }

    // ── Gradient magnitude ─────────────────────────────────────────────────────

    #[test]
    fn test_gradient_magnitude_constant_is_zero() {
        let dims = [4, 5, 6];
        let n = 4 * 5 * 6;
        let data = vec![3.14_f64; n];
        let grad = compute_gradient_magnitude(&data, dims);
        for (i, &g) in grad.iter().enumerate() {
            assert!(g.abs() < 1e-15, "grad[{i}] = {g} for constant field");
        }
    }

    #[test]
    fn test_gradient_magnitude_linear_ramp() {
        // f(z, y, x) = x  ->  interior |grad f| = 1.0
        let (nz, ny, nx) = (5, 5, 8);
        let dims = [nz, ny, nx];
        let n = nz * ny * nx;
        let mut data = vec![0.0_f64; n];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    data[iz * ny * nx + iy * nx + ix] = ix as f64;
                }
            }
        }
        let grad = compute_gradient_magnitude(&data, dims);
        // Interior voxels (1 <= ix <= nx-2) have exact central diff = 1.0.
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 1..nx - 1 {
                    let idx = iz * ny * nx + iy * nx + ix;
                    assert!(
                        (grad[idx] - 1.0).abs() < 1e-12,
                        "grad[{iz},{iy},{ix}] = {}, expected 1.0",
                        grad[idx]
                    );
                }
            }
        }
    }

    // ── Field gradient ─────────────────────────────────────────────────────────

    #[test]
    fn test_field_gradient_constant() {
        let dims = [4, 5, 6];
        let n = 4 * 5 * 6;
        let data = vec![99.0_f64; n];
        let (gz, gy, gx) = compute_field_gradient(&data, dims);
        for i in 0..n {
            assert!(gz[i].abs() < 1e-15, "gz[{i}] = {}", gz[i]);
            assert!(gy[i].abs() < 1e-15, "gy[{i}] = {}", gy[i]);
            assert!(gx[i].abs() < 1e-15, "gx[{i}] = {}", gx[i]);
        }
    }
}
