use std::f64::consts::PI;

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
