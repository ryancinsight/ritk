use super::indexing::idx_clamped;

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

    let slice_len = ny * nx;
    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        kappa,
        slice_len,
        |iz, k_slice| {
            let base = iz * slice_len;
            let zz = iz as isize;
            for iy in 0..ny {
                for ix in 0..nx {
                    let local = iy * nx + ix;
                    let i = base + local;
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
                    let phi_c = phi[i];
                    let phi_xx = phi_xp - 2.0 * phi_c + phi_xm;
                    let phi_yy = phi_yp - 2.0 * phi_c + phi_ym;
                    let phi_zz = phi_zp - 2.0 * phi_c + phi_zm;

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

                    k_slice[local] = numerator / (grad_sq_safe * grad_mag);
                }
            }
        },
    );
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
/// Returns `(∂f/∂z, ∂f/∂y, ∂f/∂x)` as three freshly allocated flat vectors in
/// row-major order. Delegates to [`compute_field_gradient_into`].
pub(crate) fn compute_field_gradient(
    data: &[f64],
    dims: [usize; 3],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = dims[0] * dims[1] * dims[2];
    let mut gz = vec![0.0_f64; n];
    let mut gy = vec![0.0_f64; n];
    let mut gx = vec![0.0_f64; n];
    compute_field_gradient_into(data, dims, &mut gz, &mut gy, &mut gx);
    (gz, gy, gx)
}

/// Component-wise gradient of a scalar field, writing into pre-allocated output
/// buffers to eliminate per-call heap allocation.
///
/// Each output Vec is resized to `nz·ny·nx` before writing. Computation
/// parallelises over z-slices with Adaptive moirai dispatch.
///
/// # Output order
///
/// `(gz, gy, gx)` matching `(∂f/∂z, ∂f/∂y, ∂f/∂x)` in row-major layout.
pub(crate) fn compute_field_gradient_into(
    data: &[f64],
    dims: [usize; 3],
    gz: &mut Vec<f64>,
    gy: &mut Vec<f64>,
    gx: &mut Vec<f64>,
) {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    gz.resize(n, 0.0);
    gy.resize(n, 0.0);
    gx.resize(n, 0.0);

    let slice_len = ny * nx;
    let mut zipped: Vec<(&mut [f64], &mut [f64], &mut [f64])> = gz
        .chunks_exact_mut(slice_len)
        .zip(gy.chunks_exact_mut(slice_len))
        .zip(gx.chunks_exact_mut(slice_len))
        .map(|((z, y), x)| (z, y, x))
        .collect();

    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        &mut zipped,
        1,
        |iz, chunk| {
            let (gz_s, gy_s, gx_s) = &mut chunk[0];
            let zz = iz as isize;
            for iy in 0..ny {
                for ix in 0..nx {
                    let local = iy * nx + ix;
                    let yy = iy as isize;
                    let xx = ix as isize;
                    gz_s[local] = (data[idx_clamped(zz + 1, yy, xx, nz, ny, nx)]
                        - data[idx_clamped(zz - 1, yy, xx, nz, ny, nx)])
                        * 0.5;
                    gy_s[local] = (data[idx_clamped(zz, yy + 1, xx, nz, ny, nx)]
                        - data[idx_clamped(zz, yy - 1, xx, nz, ny, nx)])
                        * 0.5;
                    gx_s[local] = (data[idx_clamped(zz, yy, xx + 1, nz, ny, nx)]
                        - data[idx_clamped(zz, yy, xx - 1, nz, ny, nx)])
                        * 0.5;
                }
            }
        },
    );
}

/// Upwind discretisation of the level-set advection term `∇a·∇φ`.
///
/// Returns, per voxel, `Σ_d (∂a/∂x_d) · D_upwind^d φ` where the φ difference along
/// axis `d` is taken from the upwind side of the advection velocity `−∇a`
/// (Osher & Sethian): a forward difference where `∂a/∂x_d > 0` and a backward
/// difference where `∂a/∂x_d ≤ 0`. Boundaries clamp to the edge voxel.
///
/// The advection term transports the front (it is hyperbolic, not diffusive);
/// central differencing of it is unconditionally unstable and lets the contour
/// leak through edges. Callers add `+advection_weight · this` to `∂φ/∂t` so the
/// front is pulled toward minima of `a` (image edges). Delegates to
/// [`upwind_advection_into`].
pub(crate) fn upwind_advection(
    phi: &[f64],
    dims: [usize; 3],
    a_z: &[f64],
    a_y: &[f64],
    a_x: &[f64],
) -> Vec<f64> {
    let n = dims[0] * dims[1] * dims[2];
    let mut adv = vec![0.0_f64; n];
    upwind_advection_into(phi, dims, a_z, a_y, a_x, &mut adv);
    adv
}

/// Upwind discretisation of `∇a·∇φ`, writing into a pre-allocated output buffer.
///
/// `out` is resized to `nz·ny·nx` before writing. Parallelises over z-slices
/// with Adaptive moirai dispatch.
pub(crate) fn upwind_advection_into(
    phi: &[f64],
    dims: [usize; 3],
    a_z: &[f64],
    a_y: &[f64],
    a_x: &[f64],
    out: &mut Vec<f64>,
) {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    out.resize(n, 0.0);

    let slice_len = ny * nx;
    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        out,
        slice_len,
        |iz, out_s| {
            let base = iz * slice_len;
            let zz = iz as isize;
            for iy in 0..ny {
                for ix in 0..nx {
                    let local = iy * nx + ix;
                    let i = base + local;
                    let (z, y, x) = (zz, iy as isize, ix as isize);
                    let c = phi[i];

                    let az = a_z[i];
                    let dz = if az > 0.0 {
                        phi[idx_clamped(z + 1, y, x, nz, ny, nx)] - c
                    } else {
                        c - phi[idx_clamped(z - 1, y, x, nz, ny, nx)]
                    };

                    let ay = a_y[i];
                    let dy = if ay > 0.0 {
                        phi[idx_clamped(z, y + 1, x, nz, ny, nx)] - c
                    } else {
                        c - phi[idx_clamped(z, y - 1, x, nz, ny, nx)]
                    };

                    let ax = a_x[i];
                    let dx = if ax > 0.0 {
                        phi[idx_clamped(z, y, x + 1, nz, ny, nx)] - c
                    } else {
                        c - phi[idx_clamped(z, y, x - 1, nz, ny, nx)]
                    };

                    out_s[local] = az * dz + ay * dy + ax * dx;
                }
            }
        },
    );
}
