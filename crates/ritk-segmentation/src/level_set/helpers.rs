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
//! | Gradient operators | [`compute_gradient_magnitude`], [`compute_field_gradient`], [`compute_field_gradient_into`] |
//! | Advection | [`upwind_advection`], [`upwind_advection_into`] |
//! | Edge / speed functions | [`compute_edge_stopping`] |
//! | Smoothing | [`gaussian_smooth`], [`smooth_or_borrow`] |
//! | Regularisation | [`regularised_heaviside`], [`regularised_dirac`] |

use std::f64::consts::PI;

// ── Parallel write helper ─────────────────────────────────────────────────────────

/// Send+Sync wrapper enabling safe parallel writes to disjoint `f64` z-slice regions.
///
/// Each call to [`slice_mut`](Self::slice_mut) must produce a range disjoint from all
/// other live calls. Z-slice parallelism satisfies this by construction: thread `iz`
/// exclusively owns `[iz·ny·nx, (iz+1)·ny·nx)`.
struct CellSliceF64 {
    ptr: *mut f64,
    len: usize,
}

impl CellSliceF64 {
    fn from_mut(s: &mut [f64]) -> Self {
        Self {
            ptr: s.as_mut_ptr(),
            len: s.len(),
        }
    }

    /// Reconstruct a mutable slice at `offset` with length `chunk_len`.
    ///
    /// # Safety
    /// `[offset, offset + chunk_len)` must lie within `[0, self.len)` and no other
    /// reference to the same memory range may exist during the lifetime of the
    /// returned slice.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice_mut(&self, offset: usize, chunk_len: usize) -> &mut [f64] {
        debug_assert!(offset + chunk_len <= self.len);
        std::slice::from_raw_parts_mut(self.ptr.add(offset), chunk_len)
    }
}

// SAFETY: Used exclusively for disjoint z-slice parallel writes; no two threads
// access the same memory region via this wrapper.
unsafe impl Send for CellSliceF64 {}
unsafe impl Sync for CellSliceF64 {}

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

    let slice_len = ny * nx;
    let kappa_cell = CellSliceF64::from_mut(kappa);
    moirai::for_each_index_with::<moirai::Adaptive, _>(nz, |iz| {
        let base = iz * slice_len;
        let zz = iz as isize;
        // SAFETY: each iz writes to its own disjoint z-slice [base, base+slice_len)
        // of kappa; phi is an immutable shared reference safe to read from any thread.
        let k_slice = unsafe { kappa_cell.slice_mut(base, slice_len) };
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
    });
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
    let gz_cell = CellSliceF64::from_mut(gz);
    let gy_cell = CellSliceF64::from_mut(gy);
    let gx_cell = CellSliceF64::from_mut(gx);

    moirai::for_each_index_with::<moirai::Adaptive, _>(nz, |iz| {
        let base = iz * slice_len;
        let zz = iz as isize;
        // SAFETY: each iz writes to its own disjoint z-slice [base, base+slice_len)
        // across all three output buffers; data is an immutable shared reference.
        let gz_s = unsafe { gz_cell.slice_mut(base, slice_len) };
        let gy_s = unsafe { gy_cell.slice_mut(base, slice_len) };
        let gx_s = unsafe { gx_cell.slice_mut(base, slice_len) };
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
    });
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
    let out_cell = CellSliceF64::from_mut(out);

    moirai::for_each_index_with::<moirai::Adaptive, _>(nz, |iz| {
        let base = iz * slice_len;
        let zz = iz as isize;
        // SAFETY: each iz writes to its own disjoint z-slice [base, base+slice_len)
        // of out; phi, a_z, a_y, a_x are immutable shared references.
        let out_s = unsafe { out_cell.slice_mut(base, slice_len) };
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
    });
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
pub(crate) fn gaussian_smooth(data: &[f64], dims: [usize; 3], sigma: f64) -> Vec<f64> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;

    if sigma <= 0.0 {
        return data.to_vec();
    }

    let radius = (3.0 * sigma).ceil() as usize;
    let kernel = ritk_filter::gaussian_kernel(sigma, Some(radius));
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

// ── Cow helpers ─────────────────────────────────────────────────────────────────────

/// Smooth `data` if `sigma > 0`; otherwise borrow it zero-copy.
///
/// Returns a `Cow<[f64]>`:
/// - `sigma > 0` → `Cow::Owned(gaussian_smooth(data, dims, sigma))`
/// - `sigma ≤ 0` → `Cow::Borrowed(data)` (zero allocation)
///
/// This collapses the repeated `if sigma > 0 { smooth } else { data.to_vec() }` pattern
/// across Chan-Vese, Geodesic Active Contour, and Shape Detection level-set solvers.
#[inline]
pub(crate) fn smooth_or_borrow<'a>(
    data: &'a [f64],
    dims: [usize; 3],
    sigma: f64,
) -> std::borrow::Cow<'a, [f64]> {
    if sigma > 0.0 {
        std::borrow::Cow::Owned(gaussian_smooth(data, dims, sigma))
    } else {
        std::borrow::Cow::Borrowed(data)
    }
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
