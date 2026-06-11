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
//! | Smoothing | [`gaussian_smooth_3d`] |
//! | Regularisation | [`regularised_heaviside`], [`regularised_dirac`] |

use std::f64::consts::PI;

// ── Indexing ────────────────────────────────────────────────────────────────────────

/// Clamped 3-D → linear index.
///
/// Each axis coordinate is clamped to `[0, dim-1]` before computing the
/// row-major linear index `z * ny * nx + y * nx + x`.
#[inline]
pub(crate) fn idx_clamped(z: isize, y: isize, x: isize, nz: usize, ny: usize, nx: usize) -> usize {
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
    let kernel = crate::filter::gaussian_kernel_1d(sigma, Some(radius));
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
#[path = "tests_helpers.rs"]
mod tests;
