//! Private level-set numerical primitives shared by
//! [`super::scalar_chan_and_vese`] and [`super::canny_segmentation_level_set`].
//!
//! All helpers operate on `f64` with clamped (Neumann) boundary conditions.
//!
//! | Helper | Description |
//! |---|---|
//! | [`idx_clamped`] | Clamped 3-D → linear index |
//! | [`compute_curvature_into`] | κ = div(∇φ / \|∇φ\|) into a buffer |
//! | [`compute_gradient_magnitude`] | \|∇f\| via central finite differences |
//! | [`gaussian_smooth`] | Separable 3-D Gaussian smoothing |

// ── Indexing ─────────────────────────────────────────────────────────────────────

/// Clamped 3-D → linear index in row-major (Z×Y×X) order.
///
/// Each axis coordinate is clamped to `[0, dim-1]` before computing
/// `z * ny * nx + y * nx + x`.
#[inline]
pub(crate) fn idx_clamped(z: isize, y: isize, x: isize, nz: usize, ny: usize, nx: usize) -> usize {
    let cz = z.clamp(0, nz as isize - 1) as usize;
    let cy = y.clamp(0, ny as isize - 1) as usize;
    let cx = x.clamp(0, nx as isize - 1) as usize;
    cz * ny * nx + cy * nx + cx
}

// ── Curvature ─────────────────────────────────────────────────────────────────────

/// Compute mean curvature κ = div(∇φ / |∇φ|) into a pre-allocated buffer.
///
/// Uses second-order central finite differences with clamped boundary conditions.
///
/// ```text
/// num = φ_xx·(φ_y² + φ_z²) + φ_yy·(φ_x² + φ_z²) + φ_zz·(φ_x² + φ_y²)
///       − 2·φ_x·φ_y·φ_xy − 2·φ_x·φ_z·φ_xz − 2·φ_y·φ_z·φ_yz
/// κ = num / (|∇φ|² + ε) / sqrt(|∇φ|² + ε)
/// ```
///
/// The ε = 1e-10 guard prevents division by zero in flat regions.
pub(crate) fn compute_curvature_into(phi: &[f64], dims: [usize; 3], kappa: &mut [f64]) {
    let [nz, ny, nx] = dims;
    debug_assert_eq!(phi.len(), nz * ny * nx);
    debug_assert_eq!(kappa.len(), nz * ny * nx);

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let i = iz * ny * nx + iy * nx + ix;
                let (zz, yy, xx) = (iz as isize, iy as isize, ix as isize);

                let phi_xp = phi[idx_clamped(zz, yy, xx + 1, nz, ny, nx)];
                let phi_xm = phi[idx_clamped(zz, yy, xx - 1, nz, ny, nx)];
                let phi_yp = phi[idx_clamped(zz, yy + 1, xx, nz, ny, nx)];
                let phi_ym = phi[idx_clamped(zz, yy - 1, xx, nz, ny, nx)];
                let phi_zp = phi[idx_clamped(zz + 1, yy, xx, nz, ny, nx)];
                let phi_zm = phi[idx_clamped(zz - 1, yy, xx, nz, ny, nx)];

                let phi_x = (phi_xp - phi_xm) * 0.5;
                let phi_y = (phi_yp - phi_ym) * 0.5;
                let phi_z = (phi_zp - phi_zm) * 0.5;

                let phi_xx = phi_xp - 2.0 * phi[i] + phi_xm;
                let phi_yy = phi_yp - 2.0 * phi[i] + phi_ym;
                let phi_zz = phi_zp - 2.0 * phi[i] + phi_zm;

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

// ── Gradient operators ────────────────────────────────────────────────────────────

/// Gradient magnitude |∇f| via central finite differences with clamped boundaries.
pub(crate) fn compute_gradient_magnitude(data: &[f64], dims: [usize; 3]) -> Vec<f64> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut grad = vec![0.0_f64; n];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let i = iz * ny * nx + iy * nx + ix;
                let (zz, yy, xx) = (iz as isize, iy as isize, ix as isize);

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

/// Component-wise gradient (gz, gy, gx) via central finite differences.
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
                let (zz, yy, xx) = (iz as isize, iy as isize, ix as isize);

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

// ── Gaussian smoothing ────────────────────────────────────────────────────────────

/// Separable 3-D Gaussian smoothing with clamped boundary conditions.
///
/// Kernel radius = ⌈3σ⌉. Returns a copy of `data` unchanged when `sigma ≤ 0`.
pub(crate) fn gaussian_smooth(data: &[f64], dims: [usize; 3], sigma: f64) -> Vec<f64> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;

    if sigma <= 0.0 {
        return data.to_vec();
    }

    let radius = (3.0 * sigma).ceil() as usize;
    let kernel: Vec<f64> = ritk_tensor_ops::gaussian_kernel(sigma, Some(radius));
    let r = radius as isize;

    let mut buf = data.to_vec();
    let mut tmp = vec![0.0_f64; n];

    // Pass 1 — along x.
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut s = 0.0_f64;
                for (ki, &w) in kernel.iter().enumerate() {
                    let dx = ki as isize - r;
                    let sx = (ix as isize + dx).clamp(0, nx as isize - 1) as usize;
                    s += w * buf[iz * ny * nx + iy * nx + sx];
                }
                tmp[iz * ny * nx + iy * nx + ix] = s;
            }
        }
    }
    std::mem::swap(&mut buf, &mut tmp);

    // Pass 2 — along y.
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut s = 0.0_f64;
                for (ki, &w) in kernel.iter().enumerate() {
                    let dy = ki as isize - r;
                    let sy = (iy as isize + dy).clamp(0, ny as isize - 1) as usize;
                    s += w * buf[iz * ny * nx + sy * nx + ix];
                }
                tmp[iz * ny * nx + iy * nx + ix] = s;
            }
        }
    }
    std::mem::swap(&mut buf, &mut tmp);

    // Pass 3 — along z.
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut s = 0.0_f64;
                for (ki, &w) in kernel.iter().enumerate() {
                    let dz = ki as isize - r;
                    let sz = (iz as isize + dz).clamp(0, nz as isize - 1) as usize;
                    s += w * buf[sz * ny * nx + iy * nx + ix];
                }
                tmp[iz * ny * nx + iy * nx + ix] = s;
            }
        }
    }

    tmp
}
