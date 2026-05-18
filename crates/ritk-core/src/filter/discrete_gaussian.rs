//! Discrete Gaussian smoothing filter (ITK DiscreteGaussianImageFilter parity).
//!
//! # Mathematical Specification
//!
//! Given variance v_d (physical units^2) and voxel spacing h_d:
//!   sigma_pixel = sqrt(v_d) / h_d
//!   w[k] = exp(-k^2 / (2*sigma_pixel^2))  for k in {-r,...,r}
//!   W = Sum_k w[k]; w_bar[k] = w[k] / W
//!   r_min = ceil(sqrt(-2*sigma_pixel^2 * ln(maximum_error)))
//!
//! # Boundary Conditions
//! Replicate (edge) padding is used for all convolutions. This preserves the
//! invariant conv(constant_image, normalized_kernel) = constant_image for
//! ALL voxels, including boundaries.
//!
//! # ITK Parity
//!   - Parameterized by variance (sigma^2), not sigma.
//!   - maximum_error truncation criterion (default 0.01).
//!   - use_image_spacing flag (default true).
//!
//! # References
//! - Young et al. (1995). Fundamentals of Image Processing. Delft UT.
//! - ITK Software Guide 2nd Ed., Sec 6.3.1.

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use rayon::prelude::*;
use std::marker::PhantomData;

/// Discrete Gaussian smoothing filter (ITK DiscreteGaussianImageFilter parity).
///
/// Applies separable 1-D Gaussian convolution along each image dimension.
/// Replicate (edge) padding preserves the uniform-image identity invariant.
pub struct DiscreteGaussianFilter<B: Backend> {
    variance: Vec<f64>,
    maximum_error: f64,
    use_image_spacing: bool,
    _b: PhantomData<B>,
}

impl<B: Backend> DiscreteGaussianFilter<B> {
    /// Create with per-dimension variances (physical units^2).
    /// Panics if variance is empty.
    pub fn new(variance: Vec<f64>) -> Self {
        assert!(!variance.is_empty(), "variance list must not be empty");
        Self {
            variance,
            maximum_error: 0.01,
            use_image_spacing: true,
            _b: PhantomData,
        }
    }

    /// Set maximum_error (default 0.01). Panics if not in (0,1).
    pub fn with_maximum_error(mut self, maximum_error: f64) -> Self {
        assert!(
            maximum_error > 0.0 && maximum_error < 1.0,
            "maximum_error must be in (0, 1), got {maximum_error}"
        );
        self.maximum_error = maximum_error;
        self
    }

    /// Set use_image_spacing (default true).
    pub fn with_use_image_spacing(mut self, use_image_spacing: bool) -> Self {
        self.use_image_spacing = use_image_spacing;
        self
    }

    /// Apply the filter; spatial metadata preserved.
    pub fn apply<const D: usize>(&self, image: &Image<B, D>) -> Image<B, D> {
        let smoothed = self.apply_inner(image.data().clone(), image.spacing());
        Image::new(
            smoothed,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        )
    }

    fn variance_for_dim(&self, d: usize) -> f64 {
        if d < self.variance.len() {
            self.variance[d]
        } else {
            *self.variance.last().unwrap()
        }
    }

    fn pixel_sigma_for_dim<const D: usize>(
        &self,
        d: usize,
        spacing: &crate::spatial::Spacing<D>,
    ) -> f64 {
        let v = self.variance_for_dim(d);
        if self.use_image_spacing {
            let h = spacing[d];
            v.max(0.0).sqrt() / h.max(1e-12)
        } else {
            v.max(0.0).sqrt()
        }
    }

    /// r_min = ceil(sqrt(-2*sigma^2*ln(maximum_error))), minimum 1.
    fn kernel_radius(&self, sigma_pixel: f64) -> usize {
        if sigma_pixel < 1e-9 {
            return 0;
        }
        let r = (-2.0 * sigma_pixel * sigma_pixel * self.maximum_error.ln())
            .sqrt()
            .ceil() as usize;
        r.max(1)
    }

    fn build_kernel(&self, sigma_pixel: f64, radius: usize) -> Vec<f32> {
        let width = 2 * radius + 1;
        let mut k = Vec::with_capacity(width);
        let two_s2 = 2.0 * sigma_pixel * sigma_pixel;
        let mut sum = 0.0f64;
        for i in 0..width {
            let x = i as f64 - radius as f64;
            let v = (-x * x / two_s2).exp();
            k.push(v as f32);
            sum += v;
        }
        for val in &mut k {
            *val /= sum as f32;
        }
        k
    }
    fn apply_inner<const D: usize>(
        &self,
        data: Tensor<B, D>,
        spacing: &crate::spatial::Spacing<D>,
    ) -> Tensor<B, D> {
        let device = data.device();
        let dims: [usize; D] = data.shape().dims();

        // Build kernels for each dimension (None = skip, identity).
        let mut kernels: [Option<Vec<f32>>; D] = std::array::from_fn(|_| None);
        for (d, k) in kernels.iter_mut().enumerate() {
            let sigma = self.pixel_sigma_for_dim::<D>(d, spacing);
            if sigma >= 1e-9 {
                let radius = self.kernel_radius(sigma);
                if radius > 0 {
                    *k = Some(self.build_kernel(sigma, radius));
                }
            }
        }

        // Check if any dimension needs filtering.
        if kernels.iter().all(|k| k.is_none()) {
            return data;
        }

        // Extract flat Vec<f32>.
        let flat: Vec<f32> = data.into_data().into_vec::<f32>().expect("f32 tensor data");

        // Apply separable convolution on the flat array.
        let result = convolve_separable(flat, dims, &kernels);

        // Reconstruct tensor.
        Tensor::<B, D>::from_data(TensorData::new(result, Shape::new(dims)), &device)
    }
}

// Flat-array separable convolution for DiscreteGaussianFilter.

/// Convolve a 1-D slice with replicate (edge) boundary padding.
///
/// Reconstructs the output via SAXPY accumulation: for each kernel position kj
/// (offset = kj - r from the kernel center), add w x input[clamped_pos] to the
/// corresponding output elements. Boundary regions are identified analytically to
/// eliminate per-element clamping in the interior, enabling LLVM vectorization.
///
/// # Invariant
/// conv(constant_c, normalised_kernel) = c for all positions.
#[inline]
fn conv1d_replicate(input: &[f32], kernel: &[f32], output: &mut [f32]) {
    let n = input.len();
    if n == 0 {
        return;
    }
    let ksz = kernel.len();
    let r = (ksz / 2) as isize;
    let n_i = n as isize;
    output.fill(0.0);
    for (kj, &w) in kernel.iter().enumerate() {
        let offset = kj as isize - r; // src_pos = output_i + offset
                                      // i_start: first output index where src_pos >= 0, i.e., i >= -offset.
        let i_start = ((-offset).max(0) as usize).min(n);
        // i_end: first output index where src_pos >= n, i.e., i >= n - offset.
        let i_end = ((n_i - offset).max(0).min(n_i) as usize).min(n);
        // Left boundary [0, i_start): src_pos < 0 - replicate input[0].
        if i_start > 0 {
            let left_val = input[0] * w;
            for o in &mut output[..i_start] {
                *o += left_val;
            }
        }
        // Interior [i_start, i_end): no clamping - sequential reads, LLVM-vectorizable.
        for i in i_start..i_end {
            output[i] += input[(i as isize + offset) as usize] * w;
        }
        // Right boundary [i_end, n): src_pos >= n - replicate input[n-1].
        if i_end < n {
            let right_val = input[n - 1] * w;
            for o in &mut output[i_end..] {
                *o += right_val;
            }
        }
    }
}

/// Convolve flat C-order [NZ,NY,NX] array along one axis. Rayon-parallel.
fn convolve3d_dim(
    src: &[f32],
    dst: &mut [f32],
    nz: usize,
    ny: usize,
    nx: usize,
    dim: usize,
    kernel: &[f32],
) {
    let nyx = ny * nx;
    match dim {
        2 => {
            dst.par_chunks_mut(nx)
                .zip(src.par_chunks(nx))
                .for_each(|(o, i)| conv1d_replicate(i, kernel, o));
        }
        1 => {
            // Parallel over Z-slabs. Within each slab reorder loops to
            // (kj, iy, ix): reads src[sy*nx..(sy+1)*nx] (contiguous row) and
            // writes os[iy*nx..(iy+1)*nx] (contiguous row) - LLVM-vectorizable.
            // Eliminates the per-slab intermediate buf allocation.
            let ksz = kernel.len();
            let r = (ksz / 2) as isize;
            let ny_i = ny as isize;
            dst.par_chunks_mut(nyx)
                .zip(src.par_chunks(nyx))
                .for_each(|(os, is)| {
                    os.fill(0.0);
                    for (kj, &w) in kernel.iter().enumerate() {
                        let r_offset = kj as isize - r;
                        for iy in 0..ny {
                            let sy = ((iy as isize + r_offset).clamp(0, ny_i - 1)) as usize;
                            let src_row = &is[sy * nx..(sy + 1) * nx]; // contiguous
                            let dst_row = &mut os[iy * nx..(iy + 1) * nx]; // contiguous
                            for (d, &s) in dst_row.iter_mut().zip(src_row.iter()) {
                                *d += s * w;
                            }
                        }
                    }
                });
        }
        0 => {
            // Parallel over output Z-slices. For each output slice iz, accumulate
            // contributions from input Z-slices via SAXPY (contiguous reads).
            // Complexity: O(nz x ksz x nyx) with nz-way Rayon parallelism.
            let ksz = kernel.len();
            let r = (ksz / 2) as isize;
            let nz_i = nz as isize;
            dst.par_chunks_mut(nyx)
                .enumerate()
                .for_each(|(iz, out_slice)| {
                    out_slice.fill(0.0);
                    for (kj, &w) in kernel.iter().enumerate() {
                        let sz = ((iz as isize + kj as isize - r).clamp(0, nz_i - 1)) as usize;
                        let src_z = &src[sz * nyx..(sz + 1) * nyx]; // contiguous Z-slice
                        for (o, &s) in out_slice.iter_mut().zip(src_z.iter()) {
                            *o += s * w;
                        }
                    }
                });
        }
        _ => unreachable!(),
    }
}

fn convolve_nd_dim_serial(
    src: &[f32],
    dst: &mut [f32],
    shape: &[usize],
    dim: usize,
    kernel: &[f32],
) {
    let d = shape.len();
    let mut strides = vec![1usize; d];
    for i in (0..d.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let line_len = shape[dim];
    let line_stride = strides[dim];
    let n_total: usize = shape.iter().product();
    let n_lines = n_total / line_len.max(1);
    let ksz = kernel.len();
    let r = (ksz / 2) as isize;
    let ll_i = line_len as isize;
    let mut idx = vec![0usize; d];
    let mut ob = vec![0.0f32; line_len];
    for _line in 0..n_lines {
        let base: usize = idx.iter().zip(strides.iter()).map(|(i, s)| i * s).sum();
        for (i, ob_i) in ob.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (kj, &k) in kernel.iter().enumerate() {
                let pos = ((i as isize + kj as isize - r).clamp(0, ll_i - 1)) as usize;
                acc += src[base + pos * line_stride] * k;
            }
            *ob_i = acc;
        }
        for (i, &val) in ob.iter().enumerate() {
            dst[base + i * line_stride] = val;
        }
        let mut carry = true;
        for dd in (0..d).rev() {
            if dd == dim {
                continue;
            }
            if carry {
                idx[dd] += 1;
                if idx[dd] < shape[dd] {
                    carry = false;
                } else {
                    idx[dd] = 0;
                }
            }
        }
    }
}

fn convolve_separable<const D: usize>(
    mut data: Vec<f32>,
    shape: [usize; D],
    kernels: &[Option<Vec<f32>>; D],
) -> Vec<f32> {
    let n: usize = shape.iter().product();
    let mut buf = vec![0.0f32; n];
    for (dim, kernel_opt) in kernels.iter().enumerate() {
        let Some(kernel) = kernel_opt else {
            continue;
        };
        if D == 3 {
            convolve3d_dim(&data, &mut buf, shape[0], shape[1], shape[2], dim, kernel);
        } else {
            convolve_nd_dim_serial(&data, &mut buf, &shape, dim, kernel);
        }
        std::mem::swap(&mut data, &mut buf);
    }
    data
}

#[cfg(test)]
#[path = "tests_discrete_gaussian.rs"]
mod tests;
