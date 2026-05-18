//! Separable 3-D Gaussian smoothing for displacement fields.

use super::flat;

/// Build a normalised 1-D Gaussian kernel with radius `⌈3σ⌉`.
///
/// The kernel sums to exactly 1.0 (probability-preserving convolution).
pub(super) fn gaussian_kernel_1d(sigma: f64) -> Vec<f64> {
    let radius = (3.0 * sigma).ceil() as usize;
    let two_sigma2 = 2.0 * sigma * sigma;
    let mut k: Vec<f64> = (0..=(2 * radius))
        .map(|i| {
            let x = i as f64 - radius as f64;
            (-x * x / two_sigma2).exp()
        })
        .collect();
    let sum: f64 = k.iter().sum();
    for v in &mut k {
        *v /= sum;
    }
    k
}

/// Convolve `data` along the Z axis with `kernel`; write result into `output`.
/// Uses replicate-border boundary condition.
pub(super) fn convolve_z(data: &[f32], dims: [usize; 3], kernel: &[f64], output: &mut [f32]) {
    let [nz, ny, nx] = dims;
    let r = kernel.len() / 2;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let fi = flat(iz, iy, ix, ny, nx);
                let mut acc = 0.0_f64;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let src = (iz as isize + ki as isize - r as isize)
                        .max(0)
                        .min(nz as isize - 1) as usize;
                    acc += kv * data[flat(src, iy, ix, ny, nx)] as f64;
                }
                output[fi] = acc as f32;
            }
        }
    }
}

/// Convolve `data` along the Y axis with `kernel`; write result into `output`.
pub(super) fn convolve_y(data: &[f32], dims: [usize; 3], kernel: &[f64], output: &mut [f32]) {
    let [nz, ny, nx] = dims;
    let r = kernel.len() / 2;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let fi = flat(iz, iy, ix, ny, nx);
                let mut acc = 0.0_f64;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let src = (iy as isize + ki as isize - r as isize)
                        .max(0)
                        .min(ny as isize - 1) as usize;
                    acc += kv * data[flat(iz, src, ix, ny, nx)] as f64;
                }
                output[fi] = acc as f32;
            }
        }
    }
}

/// Convolve `data` along the X axis with `kernel`; write result into `output`.
pub(super) fn convolve_x(data: &[f32], dims: [usize; 3], kernel: &[f64], output: &mut [f32]) {
    let [nz, ny, nx] = dims;
    let r = kernel.len() / 2;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let fi = flat(iz, iy, ix, ny, nx);
                let mut acc = 0.0_f64;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let src = (ix as isize + ki as isize - r as isize)
                        .max(0)
                        .min(nx as isize - 1) as usize;
                    acc += kv * data[flat(iz, iy, src, ny, nx)] as f64;
                }
                output[fi] = acc as f32;
            }
        }
    }
}

/// Apply separable 3-D Gaussian smoothing to `data` **in place**.
///
/// Convolves sequentially along Z, Y, then X. Uses a temporary buffer to
/// avoid read-after-write aliasing. A `sigma ≤ 0` is a no-op.
pub(crate) fn gaussian_smooth_inplace(data: &mut Vec<f32>, dims: [usize; 3], sigma: f64) {
    if sigma <= 0.0 {
        return;
    }
    let n = data.len();
    let mut tmp = vec![0.0_f32; n];
    gaussian_smooth_with_scratch(data.as_mut_slice(), dims, sigma, &mut tmp);
}

/// Apply separable 3-D Gaussian smoothing to `data` **in place** using a
/// caller-provided scratch buffer.
///
/// Equivalent to [`gaussian_smooth_inplace`] but performs zero heap allocation.
/// `scratch` must have the same length as `data`. A `sigma ≤ 0` is a no-op.
pub(crate) fn gaussian_smooth_with_scratch(
    data: &mut [f32],
    dims: [usize; 3],
    sigma: f64,
    scratch: &mut [f32],
) {
    if sigma <= 0.0 {
        return;
    }
    let kernel = gaussian_kernel_1d(sigma);
    convolve_z(data, dims, &kernel, scratch);
    data.copy_from_slice(scratch);
    convolve_y(data, dims, &kernel, scratch);
    data.copy_from_slice(scratch);
    convolve_x(data, dims, &kernel, scratch);
    data.copy_from_slice(scratch);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gaussian smoothing of a uniform field leaves the field unchanged.
    #[test]
    fn gaussian_smooth_uniform_unchanged() {
        let dims = [6usize, 6, 6];
        let n = 6 * 6 * 6;
        let mut data = vec![3.0_f32; n];
        gaussian_smooth_inplace(&mut data, dims, 1.5);
        for &v in &data {
            assert!(
                (v - 3.0).abs() < 1e-4,
                "expected 3.0 after smoothing, got {v}"
            );
        }
    }

    /// Gaussian smoothing reduces peak amplitude of a delta-like spike.
    #[test]
    fn gaussian_smooth_reduces_peak() {
        let dims = [9usize, 9, 9];
        let n = 9 * 9 * 9;
        let mut data = vec![0.0_f32; n];
        // Single spike in the centre.
        data[flat(4, 4, 4, 9, 9)] = 1.0;
        let peak_before = 1.0_f32;
        gaussian_smooth_inplace(&mut data, dims, 1.0);
        let peak_after = data.iter().cloned().fold(0.0_f32, f32::max);
        assert!(
            peak_after < peak_before,
            "peak should decrease after smoothing: {peak_after} >= {peak_before}"
        );
        // Total mass should be approximately conserved.
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "mass not conserved: sum = {sum}");
    }

    /// sigma ≤ 0 is a no-op.
    #[test]
    fn gaussian_smooth_zero_sigma_noop() {
        let dims = [4usize, 4, 4];
        let mut data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let orig = data.clone();
        gaussian_smooth_inplace(&mut data, dims, 0.0);
        assert_eq!(data, orig, "zero sigma should leave data unchanged");
    }

    /// `gaussian_smooth_with_scratch` produces identical results to `gaussian_smooth_inplace`.
    #[test]
    fn gaussian_smooth_with_scratch_matches_inplace() {
        let dims = [6usize, 6, 6];
        let n = 6 * 6 * 6;
        let mut data1: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut data2 = data1.clone();
        let mut scratch = vec![0.0_f32; n];
        gaussian_smooth_inplace(&mut data1, dims, 1.5);
        gaussian_smooth_with_scratch(&mut data2, dims, 1.5, &mut scratch);
        for i in 0..n {
            assert!(
                (data1[i] - data2[i]).abs() < 1e-6,
                "mismatch at {i}: inplace={}, scratch={}",
                data1[i],
                data2[i]
            );
        }
    }
}
