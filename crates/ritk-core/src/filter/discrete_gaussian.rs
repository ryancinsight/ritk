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
        Self { variance, maximum_error: 0.01, use_image_spacing: true, _b: PhantomData }
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
        Image::new(smoothed, image.origin().clone(), image.spacing().clone(), image.direction().clone())
    }

    fn variance_for_dim(&self, d: usize) -> f64 {
        if d < self.variance.len() { self.variance[d] } else { *self.variance.last().unwrap() }
    }

    fn pixel_sigma_for_dim<const D: usize>(&self, d: usize, spacing: &crate::spatial::Spacing<D>) -> f64 {
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
        if sigma_pixel < 1e-9 { return 0; }
        let r = (-2.0 * sigma_pixel * sigma_pixel * self.maximum_error.ln()).sqrt().ceil() as usize;
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
        for val in &mut k { *val /= sum as f32; }
        k
    }
    fn apply_inner<const D: usize>(&self, data: Tensor<B, D>, spacing: &crate::spatial::Spacing<D>) -> Tensor<B, D> {
        let device = data.device();
        let dims: [usize; D] = data.shape().dims();

        // Build kernels for each dimension (None = skip, identity).
        let mut kernels: [Option<Vec<f32>>; D] = std::array::from_fn(|_| None);
        for d in 0..D {
            let sigma = self.pixel_sigma_for_dim::<D>(d, spacing);
            if sigma >= 1e-9 {
                let radius = self.kernel_radius(sigma);
                if radius > 0 {
                    kernels[d] = Some(self.build_kernel(sigma, radius));
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

#[inline]
fn conv1d_replicate(input: &[f32], kernel: &[f32], output: &mut [f32]) {
    let n = input.len(); let ksz = kernel.len();
    let radius = (ksz / 2) as isize; let n_isize = n as isize;
    for i in 0..n {
        let mut acc = 0.0f32;
        for kj in 0..ksz {
            let pos = ((i as isize + kj as isize - radius).clamp(0, n_isize - 1)) as usize;
            acc += input[pos] * kernel[kj];
        }
        output[i] = acc;
    }
}

/// Convolve flat C-order [NZ,NY,NX] array along one axis. Rayon-parallel.
fn convolve3d_dim(src: &[f32], dst: &mut [f32], nz: usize, ny: usize, nx: usize, dim: usize, kernel: &[f32]) {
    let nyx = ny * nx;
    match dim {
        2 => {
            dst.par_chunks_mut(nx).zip(src.par_chunks(nx))
                .for_each(|(o, i)| conv1d_replicate(i, kernel, o));
        }
        1 => {
            let ksz = kernel.len(); let r = (ksz/2) as isize; let ny_i = ny as isize;
            dst.par_chunks_mut(nyx).zip(src.par_chunks(nyx)).for_each(|(os, is)| {
                let mut buf = vec![0.0f32; ny];
                for ix in 0..nx {
                    for iy in 0..ny {
                        let mut acc = 0.0f32;
                        for kj in 0..ksz {
                            let sy = ((iy as isize + kj as isize - r).clamp(0, ny_i-1)) as usize;
                            acc += is[sy*nx+ix] * kernel[kj];
                        }
                        buf[iy] = acc;
                    }
                    for iy in 0..ny { os[iy*nx+ix] = buf[iy]; }
                }
            });
        }
        0 => {
            // Z-axis: nyx independent lines of length nz, stride=1 in Z-direction.
            // Serial iteration: nyx*nz*ksz MACs; for 64^3 with ksz=7 ~ 1.8M MACs < 1ms.
            let ksz = kernel.len(); let r = (ksz/2) as isize; let nz_i = nz as isize;
            for yx in 0..nyx {
                for iz in 0..nz {
                    let mut acc = 0.0f32;
                    for kj in 0..ksz {
                        let sz = ((iz as isize + kj as isize - r).clamp(0, nz_i-1)) as usize;
                        acc += src[sz*nyx+yx] * kernel[kj];
                    }
                    dst[iz*nyx+yx] = acc;
                }
            }
        }
        _ => unreachable!()
    }
}

fn convolve_nd_dim_serial(src: &[f32], dst: &mut [f32], shape: &[usize], dim: usize, kernel: &[f32]) {
    let d = shape.len();
    let mut strides = vec![1usize; d];
    for i in (0..d.saturating_sub(1)).rev() { strides[i] = strides[i+1] * shape[i+1]; }
    let line_len = shape[dim]; let line_stride = strides[dim];
    let n_total: usize = shape.iter().product(); let n_lines = n_total / line_len.max(1);
    let ksz = kernel.len(); let r = (ksz/2) as isize; let ll_i = line_len as isize;
    let mut idx = vec![0usize; d]; let mut ob = vec![0.0f32; line_len];
    for _line in 0..n_lines {
        let base: usize = idx.iter().zip(strides.iter()).map(|(i,s)| i*s).sum();
        for i in 0..line_len {
            let mut acc = 0.0f32;
            for kj in 0..ksz {
                let pos = ((i as isize + kj as isize - r).clamp(0, ll_i-1)) as usize;
                acc += src[base + pos*line_stride] * kernel[kj];
            }
            ob[i] = acc;
        }
        for i in 0..line_len { dst[base + i*line_stride] = ob[i]; }
        let mut carry = true;
        for dd in (0..d).rev() {
            if dd == dim { continue; }
            if carry { idx[dd] += 1; if idx[dd] < shape[dd] { carry = false; } else { idx[dd] = 0; } }
        }
    }
}

fn convolve_separable<const D: usize>(mut data: Vec<f32>, shape: [usize; D], kernels: &[Option<Vec<f32>>; D]) -> Vec<f32> {
    let n: usize = shape.iter().product();
    let mut buf = vec![0.0f32; n];
    for dim in 0..D {
        let Some(kernel) = &kernels[dim] else { continue };
        if D == 3 { convolve3d_dim(&data, &mut buf, shape[0], shape[1], shape[2], dim, kernel); }
        else { convolve_nd_dim_serial(&data, &mut buf, &shape, dim, kernel); }
        std::mem::swap(&mut data, &mut buf);
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    type B = burn_ndarray::NdArray<f32>;

    fn make_image(vals: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
        let dev = Default::default();
        let t = Tensor::<B,3>::from_data(TensorData::new(vals, Shape::new(shape)), &dev);
        Image::new(t, Point::new([0.0;3]), Spacing::new([1.0;3]), Direction::identity())
    }
    fn make_image_with_spacing(vals: Vec<f32>, shape: [usize;3], spacing: [f64;3]) -> Image<B,3> {
        let dev = Default::default();
        let t = Tensor::<B,3>::from_data(TensorData::new(vals, Shape::new(shape)), &dev);
        Image::new(t, Point::new([0.0;3]), Spacing::new(spacing), Direction::identity())
    }
    fn vals(img: &Image<B,3>) -> Vec<f32> {
        img.data().clone().into_data().into_vec::<f32>().unwrap()
    }

    #[test]
    fn test_uniform_image_is_unchanged_by_gaussian() {
        let img = make_image(vec![7.0_f32; 125], [5,5,5]);
        let out = DiscreteGaussianFilter::<B>::new(vec![1.0]).apply(&img);
        for &x in &vals(&out) { assert!((x-7.0).abs() < 1e-4); }
    }

    #[test]
    fn test_output_shape_matches_input() {
        let img = make_image(vec![1.0_f32; 216], [6,6,6]);
        let out = DiscreteGaussianFilter::<B>::new(vec![2.0]).apply(&img);
        assert_eq!(out.shape(), img.shape());
    }

    #[test]
    fn test_larger_variance_produces_more_smoothing_on_step_edge() {
        let mut v: Vec<f32> = vec![0.0;8]; v.extend(vec![100.0;8]);
        let img = make_image(v, [1,1,16]);
        let sv = vals(&DiscreteGaussianFilter::<B>::new(vec![0.5,0.5,0.5]).with_use_image_spacing(false).apply(&img));
        let lv = vals(&DiscreteGaussianFilter::<B>::new(vec![0.5,0.5,4.0]).with_use_image_spacing(false).apply(&img));
        assert!((50.0-lv[8]).abs() < (50.0-sv[8]).abs());
    }

    #[test]
    fn test_use_image_spacing_accounts_for_spacing() {
        let mut v: Vec<f32> = vec![0.0;8]; v.extend(vec![100.0;8]);
        let img_a = make_image_with_spacing(v.clone(),[1,1,16],[1.0,1.0,1.0]);
        let img_b = make_image_with_spacing(v.clone(),[1,1,16],[1.0,1.0,2.0]);
        let f = DiscreteGaussianFilter::<B>::new(vec![4.0]);
        let a8 = vals(&f.apply(&img_a))[8]; let b8 = vals(&f.apply(&img_b))[8];
        assert!((100.0-a8).abs() > (100.0-b8).abs());
    }

    #[test]
    fn test_zero_variance_produces_identity() {
        let v: Vec<f32> = (0..27).map(|i| i as f32).collect();
        let img = make_image(v.clone(), [3,3,3]);
        let out = DiscreteGaussianFilter::<B>::new(vec![0.0]).with_use_image_spacing(false).apply(&img);
        for (&e,&a) in v.iter().zip(vals(&out).iter()) { assert!((e-a).abs() < 1e-4); }
    }

    #[test]
    fn test_spatial_metadata_preserved() {
        let dev: <B as burn::tensor::backend::Backend>::Device = Default::default();
        let t = Tensor::<B,3>::from_data(TensorData::new(vec![1.0_f32;27],Shape::new([3,3,3])),&dev);
        let origin = Point::new([1.0,2.0,3.0]); let spacing = Spacing::new([0.5,1.0,2.0]); let dir = Direction::identity();
        let img = Image::new(t, origin, spacing, dir);
        let out = DiscreteGaussianFilter::<B>::new(vec![1.0]).apply(&img);
        assert_eq!(out.origin(), &origin); assert_eq!(out.spacing(), &spacing); assert_eq!(out.direction(), &dir);
    }

    #[test]
    fn test_maximum_error_smaller_produces_larger_kernel() {
        let mut v: Vec<f32> = vec![0.0;8]; v.extend(vec![100.0;8]);
        let img = make_image(v, [1,1,16]);
        let loose = vals(&DiscreteGaussianFilter::<B>::new(vec![0.0,0.0,4.0]).with_maximum_error(0.1).with_use_image_spacing(false).apply(&img));
        let strict = vals(&DiscreteGaussianFilter::<B>::new(vec![0.0,0.0,4.0]).with_maximum_error(0.001).with_use_image_spacing(false).apply(&img));
        assert!((50.0-strict[8]).abs() <= (50.0-loose[8]).abs()+1.0);
    }

    #[test]
    fn test_per_dimension_variance_applied_independently() {
        let mut v = vec![0.0_f32;64]; v[4*8+4] = 100.0;
        let img = make_image(v, [1,8,8]);
        let ov = vals(&DiscreteGaussianFilter::<B>::new(vec![0.0,0.0,4.0]).with_use_image_spacing(false).apply(&img));
        assert!(ov[4*8+3] > 1.0); assert!(ov[4*8+5] > 1.0);
        assert!(ov[3*8+4] < 1.0); assert!(ov[5*8+4] < 1.0);
    }

    #[test]
    fn test_impulse_response_matches_analytical_gaussian() {
        // sigma=sqrt(4)=2; impulse at 15 in 1x1x31. Tol 1e-3.
        let n=31usize; let c=15usize; let var=4.0f64;
        let mut imp = vec![0.0_f32;n]; imp[c] = 1.0;
        let img = make_image(imp, [1,1,n]);
        let ov = vals(&DiscreteGaussianFilter::<B>::new(vec![0.0,0.0,var]).with_use_image_spacing(false).apply(&img));
        let tv = 2.0*var;
        let raw: Vec<f64> = (0..n).map(|k| (-((k as f64 - c as f64).powi(2))/tv).exp()).collect();
        let z: f64 = raw.iter().sum();
        let wb: Vec<f64> = raw.iter().map(|&w| w/z).collect();
        for k in 0..n { assert!((ov[k] as f64 - wb[k]).abs() < 1e-3); }
    }

    #[test] #[should_panic]
    fn test_empty_variance_panics() { let _ = DiscreteGaussianFilter::<B>::new(vec![]); }
    #[test] #[should_panic]
    fn test_maximum_error_zero_panics() { let _ = DiscreteGaussianFilter::<B>::new(vec![1.0]).with_maximum_error(0.0); }
    #[test] #[should_panic]
    fn test_maximum_error_one_panics() { let _ = DiscreteGaussianFilter::<B>::new(vec![1.0]).with_maximum_error(1.0); }
}
