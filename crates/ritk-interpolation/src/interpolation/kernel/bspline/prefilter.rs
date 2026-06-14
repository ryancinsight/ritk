//! Cubic B-spline coefficient prefiltering.
//!
//! True B-spline *interpolation* reconstructs a continuous function
//! `f(x) = Σ_k c_k · β³(x − k)` that passes through the samples `s_k` at the
//! integers. Because the cubic basis is `β³(0) = 2/3`, `β³(±1) = 1/6`, the
//! samples are the discrete convolution `s = b ⊛ c` with `b = [1/6, 2/3, 1/6]`;
//! the coefficients `c` must therefore be recovered by *inverting* that filter
//! before any basis convolution. Applying the basis directly to the samples (no
//! prefilter) yields a B-spline *approximation* (a smoothing), not interpolation.
//!
//! The inverse filter factors into one real pole `z₁ = √3 − 2` and is applied as
//! a causal + anti-causal recursive pass per axis, with whole-sample mirror
//! boundary conditions (matching ITK's `BSplineDecompositionImageFilter`).
//!
//! # Reference
//! Unser, M., Aldroubi, A. & Eden, M. (1991). "Fast B-spline transforms for
//! continuous image representation and interpolation." *IEEE TPAMI* 13(3),
//! 277–285. — and the direct-filter form in Unser (1999), "Splines: A perfect
//! fit for signal and image processing," *IEEE Signal Processing Magazine*.

/// The single cubic B-spline pole `z₁ = √3 − 2 ≈ −0.2679491924311227`.
const POLE: f64 = -0.267_949_192_431_122_7;

/// Recover cubic B-spline coefficients in place for one 1-D line (mirror
/// boundary). A line shorter than two samples is left unchanged — its single
/// coefficient already equals the sample.
fn prefilter_line(line: &mut [f64]) {
    let n = line.len();
    if n < 2 {
        return;
    }
    let z = POLE;

    // Overall gain for the single pole: (1 − z)(1 − 1/z), which is exactly 6 for
    // the cubic spline. Applied to every sample before the recursive passes.
    let lambda = (1.0 - z) * (1.0 - 1.0 / z);
    for v in line.iter_mut() {
        *v *= lambda;
    }

    // Causal (forward) pass with the exact mirror-boundary initial value.
    line[0] = initial_causal_coefficient(line, z);
    for k in 1..n {
        line[k] += z * line[k - 1];
    }

    // Anti-causal (backward) pass with the mirror-boundary initial value.
    line[n - 1] = (z / (z * z - 1.0)) * (z * line[n - 2] + line[n - 1]);
    for k in (0..n - 1).rev() {
        line[k] = z * (line[k + 1] - line[k]);
    }
}

/// Exact causal initial coefficient `c⁺[0]` for whole-sample mirror boundary
/// conditions (period `2n − 2`), matching ITK's full (non-truncated) branch.
fn initial_causal_coefficient(c: &[f64], z: f64) -> f64 {
    let n = c.len();
    let iz = 1.0 / z;
    let mut zn = z; // z^k, k = 1..
    let mut z2n = z.powi((n - 1) as i32); // z^(n-1)
    let mut sum = c[0] + z2n * c[n - 1];
    z2n = z2n * z2n * iz; // z^(2n-2) · (1/z) = z^(2n-3)
    for ck in c.iter().take(n - 1).skip(1) {
        sum += (zn + z2n) * ck;
        zn *= z;
        z2n *= iz;
    }
    // After the loop zn = z^(n-1); the normaliser is 1 − z^(2n-2).
    sum / (1.0 - zn * zn)
}

/// Row-major strides for a rank-`D` shape `dims` (`stride[D-1] = 1`).
fn strides(dims: &[usize]) -> Vec<usize> {
    let rank = dims.len();
    let mut stride = vec![1usize; rank];
    for a in (0..rank.saturating_sub(1)).rev() {
        stride[a] = stride[a + 1] * dims[a + 1];
    }
    stride
}

/// Flat offset of the first element of the `line_idx`-th line along `axis`.
///
/// Only rank 2 and 3 are instantiated (the interpolator asserts `D ∈ {2, 3}`).
fn line_base(line_idx: usize, axis: usize, dims: &[usize], stride: &[usize]) -> usize {
    match (dims.len(), axis) {
        // ── rank 3, dims = [d0, d1, d2] ──────────────────────────────────────
        (3, 0) => {
            // Line along axis 0; identified by (i1, i2): base = i1·s1 + i2.
            let i1 = line_idx / dims[2];
            let i2 = line_idx % dims[2];
            i1 * stride[1] + i2
        }
        (3, 1) => {
            // Line along axis 1; identified by (i0, i2): base = i0·s0 + i2.
            let i0 = line_idx / dims[2];
            let i2 = line_idx % dims[2];
            i0 * stride[0] + i2
        }
        (3, 2) => {
            // Line along axis 2; identified by (i0, i1): base = i0·s0 + i1·s1.
            let i0 = line_idx / dims[1];
            let i1 = line_idx % dims[1];
            i0 * stride[0] + i1 * stride[1]
        }
        // ── rank 2, dims = [d0, d1] ──────────────────────────────────────────
        (2, 0) => line_idx,             // base = i1
        (2, 1) => line_idx * stride[0], // base = i0·s0
        _ => unreachable!("B-spline prefilter only supports rank 2 and 3"),
    }
}

/// Recover the separable cubic B-spline coefficients for a flat row-major volume.
///
/// Prefilters along every axis whose length is ≥ 2 (a degenerate size-1 axis is
/// skipped — its coefficients equal the samples). The recursion runs in `f64`
/// for numerical stability and the result is returned as `f32`, matching the
/// interpolation pipeline.
pub(super) fn compute_coefficients(volume: &[f32], dims: &[usize]) -> Vec<f32> {
    let total: usize = dims.iter().product();
    debug_assert_eq!(total, volume.len(), "volume length must equal product(dims)");

    let mut coeffs: Vec<f64> = volume.iter().map(|&v| f64::from(v)).collect();
    let stride = strides(dims);

    for axis in 0..dims.len() {
        let n = dims[axis];
        if n < 2 {
            continue; // degenerate axis: coefficient = sample
        }
        let s = stride[axis];
        let num_lines = total / n;
        let mut line = vec![0.0f64; n];
        for line_idx in 0..num_lines {
            let base = line_base(line_idx, axis, dims, &stride);
            for (i, slot) in line.iter_mut().enumerate() {
                *slot = coeffs[base + i * s];
            }
            prefilter_line(&mut line);
            for (i, &val) in line.iter().enumerate() {
                coeffs[base + i * s] = val;
            }
        }
    }

    coeffs.iter().map(|&v| v as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reconstructing the basis convolution of the coefficients at the integers
    /// must return the original samples: `(1/6)c[k-1] + (2/3)c[k] + (1/6)c[k+1]
    /// = s[k]` with mirror boundary. Verifies the prefilter inverts the basis.
    #[test]
    fn prefilter_inverts_cubic_basis_1d() {
        let samples = [3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let mut c = samples.to_vec();
        prefilter_line(&mut c);
        let n = c.len() as isize;
        let mirror = |i: isize| -> usize {
            let period = 2 * (n - 1);
            let mut m = i % period;
            if m < 0 {
                m += period;
            }
            if m >= n {
                m = period - m;
            }
            m as usize
        };
        for k in 0..samples.len() {
            let ki = k as isize;
            let recon = (1.0 / 6.0) * c[mirror(ki - 1)]
                + (2.0 / 3.0) * c[k]
                + (1.0 / 6.0) * c[mirror(ki + 1)];
            assert!(
                (recon - samples[k]).abs() < 1e-9,
                "reconstruction at {k}: got {recon}, want {}",
                samples[k]
            );
        }
    }

    /// A degenerate (length-1) line is unchanged.
    #[test]
    fn prefilter_length_one_is_identity() {
        let mut c = [42.0_f64];
        prefilter_line(&mut c);
        assert_eq!(c[0], 42.0);
    }
}
