//! Private level-set numerical primitives shared by
//! [`super::scalar_chan_and_vese`].
//!
//! All helpers operate on `f64` with clamped (Neumann) boundary conditions.
//!
//! | Helper | Description |
//! |---|---|
//! | [`idx_clamped`] | Clamped 3-D → linear index |
//! | [`compute_curvature_into`] | κ = div(∇φ / \|∇φ\|) into a buffer |

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

