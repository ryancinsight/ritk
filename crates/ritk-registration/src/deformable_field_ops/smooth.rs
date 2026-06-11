//! Separable 3-D Gaussian smoothing for displacement fields.
//!
//! The three per-axis convolution passes are unified into a single
//! `convolve_axis<const AXIS: usize>` function. Because `AXIS` is a
//! compile-time constant, the compiler monomorphizes three distinct
//! instantiations (0, 1, 2) and dead-code-eliminates the two unreachable
//! `match` arms in each one, producing machine code identical to the
//! former hand-written `convolve_z / convolve_y / convolve_x` trio while
//! maintaining a single authoritative implementation.

use super::flat;
use crate::parallel::CellSlice;
use ritk_filter::gaussian_kernel_1d;
use ritk_spatial::VolumeDims;

/// Build a normalised 1-D Gaussian kernel with radius `⌈3σ⌉`.
///
/// The kernel sums to exactly 1.0 (probability-preserving convolution).
///
/// Delegates to [`ritk_filter::gaussian_kernel_1d`].
pub(super) fn gaussian_kernel_1d_f64(sigma: f64) -> Vec<f64> {
    gaussian_kernel_1d(sigma, None)
}

/// Convolve `data` along axis `AXIS` (0 = Z, 1 = Y, 2 = X) with `kernel`;
/// write the result into `output`.  Uses a replicate-border boundary condition.
///
/// `AXIS` is a `const` generic: the compiler emits three fully inlined,
/// optimised specialisations and eliminates the two unreachable `match` arms
/// in each, so runtime overhead relative to the former per-axis functions is
/// zero.
pub(super) fn convolve_axis<const AXIS: usize>(
    data: &[f32],
    dims: VolumeDims,
    kernel: &[f64],
    output: &mut [f32],
) {
    let [nz, ny, nx] = dims.0;
    let r = kernel.len() / 2;
    let max_coord = dims.0[AXIS];
    // Parallelize over z-slices: each slice writes to a disjoint contiguous
    // range in `output`; all reads are from the immutable `data` input
    // (including cross-slice reads along the Z axis).
    let slice_len = ny * nx;
    let output = CellSlice::from_mut(output);
    moirai::for_each_index_with::<moirai::Adaptive, _>(nz, |iz| {
        let base = iz * slice_len;
        // SAFETY: `output` has length nz*ny*nx; each thread writes only to
        // its own disjoint [base, base + slice_len) range.
        let out_s = unsafe { output.slice_mut(base, slice_len) };
        for iy in 0..ny {
            for ix in 0..nx {
                let local = iy * nx + ix;
                let coord = [iz, iy, ix][AXIS];
                let mut acc = 0.0_f64;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let src_coord = (coord as isize + ki as isize - r as isize)
                        .max(0)
                        .min(max_coord as isize - 1) as usize;
                    let src_fi = match AXIS {
                        0 => flat(src_coord, iy, ix, ny, nx),
                        1 => flat(iz, src_coord, ix, ny, nx),
                        _ => flat(iz, iy, src_coord, ny, nx),
                    };
                    acc += kv * data[src_fi] as f64;
                }
                out_s[local] = acc as f32;
            }
        }
    });
}

/// Apply separable 3-D Gaussian smoothing to `data` **in place**.
///
/// Convolves sequentially along Z, Y, then X. Uses a temporary buffer to
/// avoid read-after-write aliasing. A `sigma ≤ 0` is a no-op.
pub(crate) fn gaussian_smooth_inplace(data: &mut [f32], dims: VolumeDims, sigma: f64) {
    if sigma <= 0.0 {
        return;
    }
    let n = data.len();
    let mut tmp = vec![0.0_f32; n];
    gaussian_smooth_with_scratch(data, dims, sigma, &mut tmp);
}

/// Apply separable 3-D Gaussian smoothing to `data` **in place** using a
/// caller-provided scratch buffer.
///
/// Equivalent to [`gaussian_smooth_inplace`] but performs zero heap allocation.
/// `scratch` must have the same length as `data`. A `sigma ≤ 0` is a no-op.
pub(crate) fn gaussian_smooth_with_scratch(
    data: &mut [f32],
    dims: VolumeDims,
    sigma: f64,
    scratch: &mut [f32],
) {
    if sigma <= 0.0 {
        return;
    }
    let kernel = gaussian_kernel_1d_f64(sigma);
    convolve_axis::<0>(data, dims, &kernel, scratch);
    data.copy_from_slice(scratch);
    convolve_axis::<1>(data, dims, &kernel, scratch);
    data.copy_from_slice(scratch);
    convolve_axis::<2>(data, dims, &kernel, scratch);
    data.copy_from_slice(scratch);
}

/// Smooth all three components of a 3-D vector field in place.
///
/// Equivalent to calling [`gaussian_smooth_inplace`] on each component,
/// but reuses the same scratch buffer across all three calls.
#[cfg(test)]
pub(crate) fn gaussian_smooth_field_inplace(
    fz: &mut [f32],
    fy: &mut [f32],
    fx: &mut [f32],
    dims: VolumeDims,
    sigma: f64,
) {
    gaussian_smooth_inplace(fz, dims, sigma);
    gaussian_smooth_inplace(fy, dims, sigma);
    gaussian_smooth_inplace(fx, dims, sigma);
}

/// Smooth all three components with caller-provided scratch buffer.
pub(crate) fn gaussian_smooth_field_inplace_with_scratch(
    fz: &mut [f32],
    fy: &mut [f32],
    fx: &mut [f32],
    dims: VolumeDims,
    sigma: f64,
    scratch: &mut [f32],
) {
    gaussian_smooth_with_scratch(fz, dims, sigma, scratch);
    gaussian_smooth_with_scratch(fy, dims, sigma, scratch);
    gaussian_smooth_with_scratch(fx, dims, sigma, scratch);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gaussian smoothing of a uniform field leaves the field unchanged.
    #[test]
    fn gaussian_smooth_uniform_unchanged() {
        let dims = VolumeDims::new([6, 6, 6]);
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
        let dims = VolumeDims::new([9, 9, 9]);
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
        let dims = VolumeDims::new([4, 4, 4]);
        let mut data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let orig = data.clone();
        gaussian_smooth_inplace(&mut data, dims, 0.0);
        assert_eq!(data, orig, "zero sigma should leave data unchanged");
    }

    /// `gaussian_smooth_with_scratch` produces identical results to `gaussian_smooth_inplace`.
    #[test]
    fn gaussian_smooth_with_scratch_matches_inplace() {
        let dims = VolumeDims::new([6, 6, 6]);
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
