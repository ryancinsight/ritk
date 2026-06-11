//! DFT / IDFT helpers and Gaussian kernel for N4 histogram sharpening.

/// Real DFT of `data` zero-padded to length `n`, written into `out` (O(n²), acceptable for n ≤ 512).
///
/// Computes `n` complex coefficients X[k] = Σ_{j=0}^{n-1} x[j]·e^{−2πi·jk/n}.
/// `out` must have length ≥ `n`.
pub(crate) fn dft_real_into(data: &[f64], n: usize, out: &mut [(f64, f64)]) {
    debug_assert!(out.len() >= n);
    let len = data.len().min(n);
    let two_pi_n = -2.0 * std::f64::consts::PI / n as f64;
    for (k, out_k) in out[..n].iter_mut().enumerate() {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (j, &dj) in data[..len].iter().enumerate() {
            let angle = two_pi_n * k as f64 * j as f64;
            re += dj * angle.cos();
            im += dj * angle.sin();
        }
        *out_k = (re, im);
    }
}

/// Inverse real DFT of `freq` (length N), written into `out` (first `n` real-valued
/// samples): x[j] = (1/N) Σ_{k=0}^{N-1} X[k]·e^{2πi·jk/N} (real part only).
///
/// O(N²) — acceptable for N ≤ 512.
/// `out` must have length ≥ `n`.
pub(crate) fn idft_real_into(freq: &[(f64, f64)], n: usize, out: &mut [f64]) {
    let big_n = freq.len();
    if big_n == 0 {
        out[..n].fill(0.0);
        return;
    }
    let two_pi_n = 2.0 * std::f64::consts::PI / big_n as f64;
    for (j, out_j) in out[..n].iter_mut().enumerate() {
        let val: f64 = freq
            .iter()
            .enumerate()
            .map(|(k, &(re, im))| {
                let angle = two_pi_n * k as f64 * j as f64;
                re * angle.cos() - im * angle.sin()
            })
            .sum();
        *out_j = val / big_n as f64;
    }
}

/// Smallest power of two ≥ `n`.
pub(crate) fn next_pow2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}
