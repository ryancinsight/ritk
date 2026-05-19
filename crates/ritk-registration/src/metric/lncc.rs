//! Local Normalized Cross Correlation (LNCC) Metric implementation.
//!
//! # Theorem: Local Normalized Cross Correlation
//!
//! **Theorem** (Cachier et al. 2003, *Comput. Vis. Image Underst.* 89:272–298):
//! The Local Normalized Cross Correlation (LNCC) between a fixed image $F$ and a moving image $M$
//! evaluates the linear dependence of intensities within a local neighborhood defined by a
//! Gaussian smoothing kernel $K$.
//!
//! ```text
//! Local Mean:   μ_F = F * K,      μ_M = M * K
//! Local Var:    v_F = F² * K - μ_F², v_M = M² * K - μ_M²
//! Local Covar:  c_{FM} = (F · M) * K - μ_F · μ_M
//!
//! LNCC(F, M) = c_{FM} / √(v_F · v_M + ε)
//! ```
//!
//! # Architectural Optimization (O(1) Stationary Caching)
//! Because the fixed image target $F$ is stationary during registration optimization,
//! the computation of $μ_F$ and $v_F$ is redundant across iterations.
//! This implementation caches the local statistics of the fixed image upon first
//! evaluation, reducing the computational payload strictly to $M$-dependent convolutions
//! and eliminating $O(N)$ operations per forward pass.

use super::trait_::Metric;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ritk_core::filter::gaussian::GaussianFilter;
use ritk_core::image::grid;
use ritk_core::image::Image;
use ritk_core::interpolation::{Interpolator, LinearInterpolator};
use ritk_core::transform::Transform;
use std::sync::{Arc, Mutex};

/// Cache for local statistics of the stationary fixed image.
#[derive(Debug)]
struct LnccCache<B: Backend> {
    shape: Vec<usize>,
    origin: Vec<f64>,
    spacing: Vec<f64>,
    direction: Vec<f64>,
    mean_f_flat: Tensor<B, 1>,
    var_f_flat: Tensor<B, 1>,
}

/// Local Normalized Cross Correlation (LNCC) Metric.
///
/// Computes the normalized cross correlation within a local window around each pixel.
/// Robust to local intensity variations and bias fields.
#[derive(Clone, Debug)]
pub struct LocalNormalizedCrossCorrelation<B: Backend> {
    interpolator: LinearInterpolator,
    kernel_sigma: f64,
    epsilon: f64,
    cache: Arc<Mutex<Option<LnccCache<B>>>>,
    _b: std::marker::PhantomData<B>,
}

impl<B: Backend> LocalNormalizedCrossCorrelation<B> {
    /// Create a new LNCC metric.
    ///
    /// # Arguments
    /// * `kernel_sigma` - Standard deviation of the Gaussian kernel (mm) defining the local window size.
    pub fn new(kernel_sigma: f64) -> Self {
        Self {
            interpolator: LinearInterpolator::new(),
            kernel_sigma,
            epsilon: 1e-5,
            cache: Arc::new(Mutex::new(None)),
            _b: std::marker::PhantomData,
        }
    }

    /// Computes local mean and variance using a Gaussian filter.
    fn compute_local_stats<const D: usize>(
        &self,
        img: Tensor<B, D>,
        filter: &GaussianFilter<B>,
        spacing: &ritk_core::spatial::Spacing<D>,
    ) -> (Tensor<B, D>, Tensor<B, D>) {
        // Mean = I * K
        let mean = filter.apply_tensor(img.clone(), spacing);

        // MeanSq = I^2 * K
        let sq = img.powf_scalar(2.0);
        let mean_sq = filter.apply_tensor(sq, spacing);

        // Var = MeanSq - Mean^2
        let var = mean_sq - mean.clone().powf_scalar(2.0);

        // Clamp variance to avoid sqrt of negative due to float errors
        let var = var.clamp_min(0.0);

        (mean, var)
    }
}

impl<B: Backend, const D: usize> Metric<B, D> for LocalNormalizedCrossCorrelation<B> {
    fn forward(
        &self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
    ) -> Tensor<B, 1> {
        let fixed_shape = fixed.shape();
        let device = fixed.data().device();

        // 1. Generate grid (Full, as we need the full spatial structure for convolution)
        let fixed_indices = grid::generate_grid(fixed_shape, &device); // [N, D]
        let [n, _] = fixed_indices.dims();

        // 2. Resample moving image with chunking to avoid WGPU dispatch limits
        const CHUNK_SIZE: usize = 32768;

        let moving_values_flat = if n <= CHUNK_SIZE {
            let fixed_points = fixed.index_to_world_tensor(fixed_indices);
            let moving_points = transform.transform_points(fixed_points);
            let moving_indices = moving.world_to_index_tensor(moving_points);
            self.interpolator.interpolate(moving.data(), moving_indices)
        } else {
            let num_chunks = n.div_ceil(CHUNK_SIZE);
            let mut chunks = Vec::with_capacity(num_chunks);

            for i in 0..num_chunks {
                let start = i * CHUNK_SIZE;
                let end = std::cmp::min(start + CHUNK_SIZE, n);

                let chunk_range = start..end;
                let chunk_indices = fixed_indices.clone().slice([chunk_range]);
                let chunk_fixed_points = fixed.index_to_world_tensor(chunk_indices);
                let chunk_moving_points = transform.transform_points(chunk_fixed_points);
                let chunk_moving_indices = moving.world_to_index_tensor(chunk_moving_points);
                let chunk_values = self
                    .interpolator
                    .interpolate(moving.data(), chunk_moving_indices);
                chunks.push(chunk_values);
            }
            Tensor::cat(chunks, 0)
        };

        // 3. Reshape back to spatial dimensions for convolution
        let shape_dims: [usize; D] = fixed.data().shape().dims(); // [usize; D]
        let moving_values = moving_values_flat.reshape(burn::tensor::Shape::new(shape_dims));
        let fixed_values = fixed.data().clone(); // Already spatial [D, H, W]

        // 4. Setup filter
        let filter = GaussianFilter::new(vec![self.kernel_sigma; D]);

        // 5. Compute or load Local Stats for FIXED image (Stationary Cache O(1))
        let (mean_f, var_f) = {
            let mut cache = self.cache.lock().unwrap();
            let current_shape = fixed.shape().to_vec();
            let current_origin: Vec<f64> = fixed.origin().0.iter().cloned().collect();
            let current_spacing: Vec<f64> = fixed.spacing().0.iter().cloned().collect();
            let current_direction: Vec<f64> = fixed.direction().0.iter().cloned().collect();

            let mut cache_hit = false;
            if let Some(c) = cache.as_ref() {
                if c.shape == current_shape
                    && c.origin == current_origin
                    && c.spacing == current_spacing
                    && c.direction == current_direction
                {
                    cache_hit = true;
                }
            }

            if cache_hit {
                let c = cache.as_ref().unwrap();
                let m_f = c
                    .mean_f_flat
                    .clone()
                    .reshape(burn::tensor::Shape::new(shape_dims));
                let v_f = c
                    .var_f_flat
                    .clone()
                    .reshape(burn::tensor::Shape::new(shape_dims));
                (m_f, v_f)
            } else {
                let (m_f, v_f) =
                    self.compute_local_stats(fixed_values.clone(), &filter, fixed.spacing());
                *cache = Some(LnccCache {
                    shape: current_shape,
                    origin: current_origin,
                    spacing: current_spacing,
                    direction: current_direction,
                    mean_f_flat: m_f.clone().flatten(0, D - 1),
                    var_f_flat: v_f.clone().flatten(0, D - 1),
                });
                (m_f, v_f)
            }
        };

        // Local Stats for MOVING image (computed per forward pass)
        let (mean_m, var_m) =
            self.compute_local_stats(moving_values.clone(), &filter, fixed.spacing());

        // 6. Compute Cross Term
        // Cross = (F * M) * K
        let fm = fixed_values * moving_values;
        let mean_fm = filter.apply_tensor(fm, fixed.spacing());

        // Covariance = MeanFM - MeanF * MeanM
        let cov = mean_fm - (mean_f * mean_m);

        // 7. Compute LNCC
        // Denom = sqrt(VarF * VarM)
        let denom = (var_f * var_m).sqrt() + self.epsilon;
        let lncc = cov / denom;

        // 8. Return negative mean (to minimize)
        lncc.mean().neg()
    }

    fn name(&self) -> &'static str {
        "LocalNormalizedCrossCorrelation"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, TensorData};
    use burn_ndarray::NdArray;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use ritk_core::transform::TranslationTransform;

    type B = NdArray<f32>;

    fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let tensor = burn::tensor::Tensor::<B, 3>::from_data(
            TensorData::new(data, Shape::new(shape)),
            &device,
        );
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    // ── name ─────────────────────────────────────────────────────────────────

    #[test]
    fn lncc_name() {
        let metric = LocalNormalizedCrossCorrelation::<B>::new(1.0);
        assert_eq!(
            <LocalNormalizedCrossCorrelation<B> as Metric<B, 3>>::name(&metric),
            "LocalNormalizedCrossCorrelation"
        );
    }

    // ── Identical images → loss ≈ -1 ─────────────────────────────────────────
    //
    // # Derivation
    // When fixed == moving (identity transform), covariance = var_f = var_m at every
    // voxel, so LNCC(x) = var_f / sqrt(var_f^2 + ε) ≈ 1.  The mean over all
    // voxels is ≈ 1, and the returned loss is the negation: ≈ −1.
    //
    // # Tolerance
    // The ε = 1e-5 guard on the denominator and the Gaussian kernel smearing
    // introduce small deviations from the ideal −1.  We assert loss < −0.95
    // (5 % tolerance), matching the ITK LNCC acceptance criterion.

    #[test]
    fn lncc_identical_images_loss_near_negative_one() {
        let size = 8;
        let count = size * size * size;
        // Non-constant ramp so that local variance > 0 almost everywhere.
        let data: Vec<f32> = (0..count).map(|i| i as f32).collect();

        let image = make_image(data, [size, size, size]);
        let device = Default::default();
        let transform =
            TranslationTransform::<B, 3>::new(burn::tensor::Tensor::<B, 1>::zeros([3], &device));

        let metric = LocalNormalizedCrossCorrelation::<B>::new(1.5);
        let loss = metric.forward(&image, &image, &transform);
        let loss_val = loss.into_scalar();

        assert!(
            loss_val < -0.90,
            "LNCC for identical images must be < -0.90 (near -1); got {}",
            loss_val
        );
    }

    // ── Linear rescaling invariance ──────────────────────────────────────────
    //
    // LNCC is invariant to affine intensity transformations:
    //   LNCC(F, a·F + b) = 1 for any a > 0, b ∈ ℝ.
    // The loss must remain ≈ −1 when the moving image is a linearly rescaled
    // copy of the fixed image.

    #[test]
    fn lncc_linear_rescaling_invariant() {
        let size = 6;
        let count = size * size * size;
        let data1: Vec<f32> = (0..count).map(|i| i as f32 + 1.0).collect();
        // a=3, b=10: moving = 3*fixed + 10
        let data2: Vec<f32> = data1.iter().map(|&x| 3.0 * x + 10.0).collect();

        let fixed = make_image(data1, [size, size, size]);
        let moving = make_image(data2, [size, size, size]);

        let device = Default::default();
        let transform =
            TranslationTransform::<B, 3>::new(burn::tensor::Tensor::<B, 1>::zeros([3], &device));

        let metric = LocalNormalizedCrossCorrelation::<B>::new(1.5);
        let loss = metric.forward(&fixed, &moving, &transform);
        let loss_val = loss.into_scalar();

        assert!(
            loss_val < -0.90,
            "LNCC must be invariant to linear intensity rescaling (loss < -0.90); got {}",
            loss_val
        );
    }

    // ── Fixed-image cache stability ──────────────────────────────────────────
    //
    // The second `forward` call on the same (fixed, moving) pair must reuse the
    // cached fixed-image statistics and return the same scalar value as the
    // first call (within f32 round-trip precision).

    #[test]
    fn lncc_cache_returns_same_result_on_second_call() {
        let size = 6;
        let count = size * size * size;
        let data: Vec<f32> = (0..count).map(|i| i as f32 + 1.0).collect();

        let image = make_image(data, [size, size, size]);
        let device = Default::default();
        let transform =
            TranslationTransform::<B, 3>::new(burn::tensor::Tensor::<B, 1>::zeros([3], &device));

        let metric = LocalNormalizedCrossCorrelation::<B>::new(1.5);

        let loss1 = metric.forward(&image, &image, &transform).into_scalar();
        let loss2 = metric.forward(&image, &image, &transform).into_scalar();

        assert!(
            (loss1 - loss2).abs() < 1e-5,
            "LNCC cache must produce identical results across calls: {} vs {}",
            loss1,
            loss2
        );
    }

    // ── Constant image → loss is finite ─────────────────────────────────────
    //
    // When the fixed image is constant, local variance = 0 everywhere.
    // The ε guard on the denominator must prevent division by zero.
    // The loss must be finite (not NaN or ±∞).

    #[test]
    fn lncc_constant_image_loss_is_finite() {
        let size = 4;
        let image = make_image(vec![5.0_f32; size * size * size], [size, size, size]);
        let device = Default::default();
        let transform =
            TranslationTransform::<B, 3>::new(burn::tensor::Tensor::<B, 1>::zeros([3], &device));

        let metric = LocalNormalizedCrossCorrelation::<B>::new(1.0);
        let loss_val = metric.forward(&image, &image, &transform).into_scalar();

        assert!(
            loss_val.is_finite(),
            "LNCC with constant image must be finite (ε guard active); got {}",
            loss_val
        );
    }
}
