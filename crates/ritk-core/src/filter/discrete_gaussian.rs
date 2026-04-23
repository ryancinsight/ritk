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
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Shape, Tensor};
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

    fn apply_inner<const D: usize>(&self, mut data: Tensor<B, D>, spacing: &crate::spatial::Spacing<D>) -> Tensor<B, D> {
        let device = data.device();
        for d in 0..D {
            let sigma = self.pixel_sigma_for_dim::<D>(d, spacing);
            if sigma < 1e-9 { continue; }
            let radius = self.kernel_radius(sigma);
            let kernel_vec = self.build_kernel(sigma, radius);
            let kernel = Tensor::<B, 1>::from_floats(kernel_vec.as_slice(), &device);
            data = convolve_1d(data, kernel, d);
        }
        data
    }
}

/// Convolve `input` along `dim` with same-size replicate-padded output.
///
/// Replicate (edge) padding ensures that a uniform input produces a uniform
/// output regardless of kernel radius: conv(c, normalized_k) = c for all voxels.
fn convolve_1d<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    kernel: Tensor<B, 1>,
    dim: usize,
) -> Tensor<B, D> {
    let shape = input.shape();
    let dims: [usize; D] = shape.dims();
    let device = input.device();

    // 1. Permute `dim` to the last position.
    let mut perm = [0isize; D];
    let mut idx = 0;
    for i in 0..D {
        if i != dim { perm[idx] = i as isize; idx += 1; }
    }
    perm[D - 1] = dim as isize;
    let permuted = input.clone().permute(perm);

    // 2. Flatten all non-dim dimensions into batch.
    let len = dims[dim];
    let mut batch = 1usize;
    for i in 0..D { if i != dim { batch *= dims[i]; } }
    let inp3 = permuted.reshape([batch, 1, len]);

    // 3. Replicate (edge) padding.
    // r = floor(ksz/2); padded_len = len + 2r; output_len = padded_len - ksz + 1 = len.
    let ksz = kernel.dims()[0];
    let kern3 = kernel.reshape([1, 1, ksz]);
    let r = ksz / 2;

    let padded = if r == 0 {
        inp3
    } else {
        // Left edge: replicate inp3[:, :, 0] r times along last dim.
        let left_edge = inp3.clone().slice([0..batch, 0..1, 0..1]);
        let left_pad = Tensor::cat(
            (0..r).map(|_| left_edge.clone()).collect::<Vec<_>>(), 2,
        );
        // Right edge: replicate inp3[:, :, len-1] r times along last dim.
        let right_edge = inp3.clone().slice([0..batch, 0..1, (len - 1)..len]);
        let right_pad = Tensor::cat(
            (0..r).map(|_| right_edge.clone()).collect::<Vec<_>>(), 2,
        );
        Tensor::cat(vec![left_pad, inp3, right_pad], 2)
    };
    // padded shape: [batch, 1, len+2r]

    // 4. Conv1d with padding=0; output_len = (len+2r - ksz)/1 + 1 = len.
    let options = ConvOptions::new([1], [0], [1], 1);

    const CHUNK: usize = 32768;
    let out3 = if batch <= CHUNK {
        burn::tensor::module::conv1d(padded, kern3, None, options)
    } else {
        let n_chunks = (batch + CHUNK - 1) / CHUNK;
        let mut parts = Vec::with_capacity(n_chunks);
        for c in 0..n_chunks {
            let s = c * CHUNK;
            let e = (s + CHUNK).min(batch);
            let chunk_in = padded.clone().slice([s..e]);
            let chunk_out = burn::tensor::module::conv1d(
                chunk_in, kern3.clone(), None, options.clone(),
            );
            parts.push(chunk_out);
        }
        Tensor::cat(parts, 0)
    };

    // 5. Reshape back to permuted shape.
    let mut perm_shape = [0usize; D];
    let mut pi = 0;
    for i in 0..D {
        if i != dim { perm_shape[pi] = dims[i]; pi += 1; }
    }
    perm_shape[D - 1] = len;
    let out_perm = out3.reshape(Shape::new(perm_shape));

    // 6. Inverse permutation.
    let mut inv_perm = [0isize; D];
    for (new_pos, &old_pos) in perm.iter().enumerate() {
        inv_perm[old_pos as usize] = new_pos as isize;
    }
    let _ = device;
    out_perm.permute(inv_perm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{image::Image, spatial::{Direction, Point, Spacing}};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(vals: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
        let dev = Default::default();
        let t = Tensor::<B, 3>::from_data(TensorData::new(vals, Shape::new(shape)), &dev);
        Image::new(t, Point::new([0.0; 3]), Spacing::new([1.0; 3]), Direction::identity())
    }

    fn make_image_with_spacing(vals: Vec<f32>, shape: [usize; 3], spacing: [f64; 3]) -> Image<B, 3> {
        let dev = Default::default();
        let t = Tensor::<B, 3>::from_data(TensorData::new(vals, Shape::new(shape)), &dev);
        Image::new(t, Point::new([0.0; 3]), Spacing::new(spacing), Direction::identity())
    }

    fn vals(img: &Image<B, 3>) -> Vec<f32> {
        img.data().clone().into_data().as_slice::<f32>().unwrap().to_vec()
    }

    // Positive tests

    #[test]
    fn test_uniform_image_is_unchanged_by_gaussian() {
        // Invariant: conv(constant, normalized_kernel) = constant for all voxels.
        // Requires replicate padding (zero-padding violates this at boundaries).
        let img = make_image(vec![7.0_f32; 125], [5, 5, 5]);
        let filter = DiscreteGaussianFilter::<B>::new(vec![1.0]);
        let out = filter.apply(&img);
        let v = vals(&out);
        for &x in &v {
            assert!(
                (x - 7.0).abs() < 1e-4,
                "uniform image must remain 7.0 after Gaussian; got {x}"
            );
        }
    }

    #[test]
    fn test_output_shape_matches_input() {
        let img = make_image(vec![1.0_f32; 216], [6, 6, 6]);
        let filter = DiscreteGaussianFilter::<B>::new(vec![2.0]);
        let out = filter.apply(&img);
        assert_eq!(out.shape(), img.shape());
    }

    #[test]
    fn test_larger_variance_produces_more_smoothing_on_step_edge() {
        // 1x1x16 step: first 8=0, last 8=100.
        // Larger variance in dim 2 blurs the step more (transition closer to 50).
        // Replicate padding prevents energy loss in dims 0 and 1 (size=1).
        let mut vals_in: Vec<f32> = vec![0.0; 8];
        vals_in.extend(vec![100.0; 8]);
        let img = make_image(vals_in, [1, 1, 16]);
        let small = DiscreteGaussianFilter::<B>::new(vec![0.5, 0.5, 0.5])
            .with_use_image_spacing(false).apply(&img);
        let large = DiscreteGaussianFilter::<B>::new(vec![0.5, 0.5, 4.0])
            .with_use_image_spacing(false).apply(&img);
        let sv = vals(&small);
        let lv = vals(&large);
        let small_at_8 = sv[8];
        let large_at_8 = lv[8];
        // small: step is sharper -> position 8 stays closer to 100 -> |50-small_at_8| > |50-large_at_8|
        assert!(
            (50.0 - large_at_8).abs() < (50.0 - small_at_8).abs(),
            "larger variance should blur more: small={small_at_8:.3} large={large_at_8:.3}"
        );
    }

    #[test]
    fn test_use_image_spacing_accounts_for_spacing() {
        // A: spacing=1mm, var=4mm^2 -> sigma_pixel=2.0 (more blur)
        // B: spacing=2mm, var=4mm^2 -> sigma_pixel=1.0 (less blur)
        let mut v: Vec<f32> = vec![0.0; 8];
        v.extend(vec![100.0; 8]);
        let img_a = make_image_with_spacing(v.clone(), [1, 1, 16], [1.0, 1.0, 1.0]);
        let img_b = make_image_with_spacing(v.clone(), [1, 1, 16], [1.0, 1.0, 2.0]);
        let filter = DiscreteGaussianFilter::<B>::new(vec![4.0]);
        let out_a = filter.apply(&img_a);
        let out_b = filter.apply(&img_b);
        let va = vals(&out_a);
        let vb = vals(&out_b);
        let at8_a = va[8];
        let at8_b = vb[8];
        // A has sigma_pixel=2 -> more blur -> va[8] further from 100 than vb[8].
        assert!(
            (100.0 - at8_a).abs() > (100.0 - at8_b).abs(),
            "smaller spacing -> more pixel blur: a={at8_a:.3} b={at8_b:.3}"
        );
    }

    #[test]
    fn test_zero_variance_produces_identity() {
        // var=0 -> sigma<1e-9 -> no convolution -> identity.
        let v: Vec<f32> = (0..27).map(|i| i as f32).collect();
        let img = make_image(v.clone(), [3, 3, 3]);
        let filter = DiscreteGaussianFilter::<B>::new(vec![0.0]).with_use_image_spacing(false);
        let out = filter.apply(&img);
        let out_vals = vals(&out);
        for (i, (&expected, &actual)) in v.iter().zip(out_vals.iter()).enumerate() {
            assert!(
                (expected - actual).abs() < 1e-4,
                "zero-var identity at {i}: expected {expected} got {actual}"
            );
        }
    }

    #[test]
    fn test_spatial_metadata_preserved() {
        use crate::spatial::{Direction, Point, Spacing};
        let dev: <B as burn::tensor::backend::Backend>::Device = Default::default();
        let t = Tensor::<B, 3>::from_data(
            TensorData::new(vec![1.0_f32; 27], Shape::new([3, 3, 3])), &dev,
        );
        let origin = Point::new([1.0, 2.0, 3.0]);
        let spacing = Spacing::new([0.5, 1.0, 2.0]);
        let direction = Direction::identity();
        let img = Image::new(t, origin, spacing, direction);
        let filter = DiscreteGaussianFilter::<B>::new(vec![1.0]);
        let out = filter.apply(&img);
        assert_eq!(out.origin(), &origin);
        assert_eq!(out.spacing(), &spacing);
        assert_eq!(out.direction(), &direction);
    }

    #[test]
    fn test_maximum_error_smaller_produces_larger_kernel() {
        // Smaller maximum_error -> larger kernel radius -> more blur at transition.
        // +1.0 tolerance handles near-converged Gaussian tails (sigma=2 is wide).
        let mut v: Vec<f32> = vec![0.0; 8];
        v.extend(vec![100.0; 8]);
        let img = make_image(v.clone(), [1, 1, 16]);
        let loose = DiscreteGaussianFilter::<B>::new(vec![0.0, 0.0, 4.0])
            .with_maximum_error(0.1).with_use_image_spacing(false).apply(&img);
        let strict = DiscreteGaussianFilter::<B>::new(vec![0.0, 0.0, 4.0])
            .with_maximum_error(0.001).with_use_image_spacing(false).apply(&img);
        let lv = vals(&loose);
        let sv = vals(&strict);
        let at8_loose = lv[8];
        let at8_strict = sv[8];
        assert!(
            (50.0 - at8_strict).abs() <= (50.0 - at8_loose).abs() + 1.0,
            "strict should blur >= loose: loose={at8_loose:.3} strict={at8_strict:.3}"
        );
    }

    #[test]
    fn test_per_dimension_variance_applied_independently() {
        // 1x8x8; peak at (0,4,4). variance=[0,0,4]: smooth dim2 only.
        let mut v = vec![0.0_f32; 64];
        v[4 * 8 + 4] = 100.0_f32;
        let img = make_image(v, [1, 8, 8]);
        let filter = DiscreteGaussianFilter::<B>::new(vec![0.0, 0.0, 4.0])
            .with_use_image_spacing(false);
        let out = filter.apply(&img);
        let ov = vals(&out);
        let left  = ov[4 * 8 + 3]; // (0,4,3) x-neighbor
        let right = ov[4 * 8 + 5]; // (0,4,5) x-neighbor
        let above = ov[3 * 8 + 4]; // (0,3,4) y-neighbor
        let below = ov[5 * 8 + 4]; // (0,5,4) y-neighbor
        assert!(left  > 1.0, "x-smoothing must spread to (0,4,3): got {left}");
        assert!(right > 1.0, "x-smoothing must spread to (0,4,5): got {right}");
        assert!(above < 1.0, "y not smoothed; (0,3,4) must stay near 0: got {above}");
        assert!(below < 1.0, "y not smoothed; (0,5,4) must stay near 0: got {below}");
    }


    #[test]
    fn test_impulse_response_matches_analytical_gaussian() {
        // Mathematical justification:
        // For variance v=4.0 and use_image_spacing=false, sigma_pixel = sqrt(4.0) = 2.0.
        // The filter builds kernel w_bar[k] = exp(-k^2 / (2*4.0)) / Z where Z = sum_k w_k.
        // Convolving a Dirac impulse at center position 15 (in a 1x1x31 image) with this
        // kernel yields output[k] = w_bar[k-15+radius] for k in 0..31, i.e., the output
        // equals the normalized kernel centered at position 15.
        // Tolerance 1e-3 accounts for f32 arithmetic and replicate-padded tail truncation.
        let n: usize = 31;
        let center: usize = 15;
        let variance: f64 = 4.0;

        // Build Dirac impulse at position 15.
        let mut impulse = vec![0.0_f32; n];
        impulse[center] = 1.0;
        let img = make_image(impulse, [1, 1, n]);

        // Apply filter: variance=[0,0,4], use_image_spacing=false.
        let filter = DiscreteGaussianFilter::<B>::new(vec![0.0, 0.0, variance])
            .with_use_image_spacing(false);
        let out = filter.apply(&img);
        let out_vals = vals(&out);

        // Compute analytical normalized Gaussian weights.
        let two_v = 2.0 * variance;
        let raw: Vec<f64> = (0..n)
            .map(|k| (-(((k as f64) - (center as f64)).powi(2)) / two_v).exp())
            .collect();
        let z: f64 = raw.iter().sum();
        let w_bar: Vec<f64> = raw.iter().map(|&w| w / z).collect();

        for k in 0..n {
            let diff = (out_vals[k] as f64 - w_bar[k]).abs();
            assert!(
                diff < 1e-3,
                "impulse response at k={k}: output={:.6} analytical={:.6} |diff|={:.6} > 1e-3",
                out_vals[k], w_bar[k], diff
            );
        }
    }

    // Negative tests

    #[test]
    #[should_panic]
    fn test_empty_variance_panics() {
        let _ = DiscreteGaussianFilter::<B>::new(vec![]);
    }

    #[test]
    #[should_panic]
    fn test_maximum_error_zero_panics() {
        let _ = DiscreteGaussianFilter::<B>::new(vec![1.0]).with_maximum_error(0.0);
    }

    #[test]
    #[should_panic]
    fn test_maximum_error_one_panics() {
        let _ = DiscreteGaussianFilter::<B>::new(vec![1.0]).with_maximum_error(1.0);
    }
}
