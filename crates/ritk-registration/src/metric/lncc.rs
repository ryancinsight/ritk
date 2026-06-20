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

use super::cache_slot::CacheSlot;
use super::histogram::cache::collect_array;
use super::trait_::Metric;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ritk_filter::gaussian::GaussianFilter;
use ritk_filter::GaussianSigma;
use ritk_image::grid;
use ritk_image::Image;
use ritk_interpolation::{Interpolator, LinearInterpolator};
use ritk_transform::Transform;
use std::sync::{Arc, Mutex};

// ── FilterSlot (REG-03) ───────────────────────────────────────────────────────

/// Lazy-initialized slot for a `GaussianFilter<B>`, Arc-shared across clones.
///
/// Stores the spatial dimension `D` alongside the filter to detect dimension
/// mismatches and reinitialize when needed. Implements `Clone` and `Debug`
/// without requiring either trait on `GaussianFilter<B>`.
struct FilterSlot<B: Backend>(Arc<Mutex<Option<(usize, GaussianFilter<B>)>>>);

impl<B: Backend> FilterSlot<B> {
    fn empty() -> Self {
        Self(Arc::new(Mutex::new(None)))
    }

    /// Returns a cloned `GaussianFilter<B>`.
    ///
    /// Initializes the filter with `D` copies of `sigma` on first access or
    /// when the cached dimension differs from `D`. The Mutex lock is dropped
    /// immediately after cloning.
    fn get_or_init<const D: usize>(
        &self,
        sigma: GaussianSigma,
    ) -> GaussianFilter<B> {
        let mut guard = self
            .0
            .lock()
            .expect("invariant: FilterSlot mutex not poisoned");
        if guard.as_ref().is_none_or(|(d, _)| *d != D) {
            *guard = Some((D, GaussianFilter::new(vec![sigma; D])));
        }
        guard.as_ref().unwrap().1.clone()
    }
}

impl<B: Backend> Clone for FilterSlot<B> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<B: Backend> std::fmt::Debug for FilterSlot<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let guard = self
            .0
            .lock()
            .expect("invariant: FilterSlot mutex not poisoned");
        f.debug_struct("FilterSlot")
            .field("dim", &guard.as_ref().map(|(d, _)| *d))
            .finish()
    }
}

/// Cached local statistics for the stationary fixed image.
#[derive(Debug, Clone)]
struct LnccCacheEntry<B: Backend> {
    shape: Vec<usize>,
    origin: [f64; 3],
    spacing: [f64; 3],
    direction: [f64; 9],
    mean_f_flat: Tensor<B, 1>,
    var_f_flat: Tensor<B, 1>,
}

impl<B: Backend> LnccCacheEntry<B> {
    /// Returns `true` when this entry was computed from an image with the same
    /// spatial geometry as `fixed` (shape, origin, spacing, direction).
    fn is_valid_for<const D: usize>(&self, fixed: &Image<B, D>) -> bool {
        let fs = fixed.shape();
        self.shape.as_slice() == fs
            && self.origin.iter().eq(fixed.origin().0.iter())
            && self.spacing.iter().eq(fixed.spacing().0.iter())
            && self.direction.iter().eq(fixed.direction().0.iter())
    }
}

/// Local Normalized Cross Correlation (LNCC) Metric.
///
/// Computes the normalized cross correlation within a local window around each pixel.
/// Robust to local intensity variations and bias fields.
#[derive(Clone, Debug)]
pub struct LocalNormalizedCrossCorrelation<B: Backend> {
    interpolator: LinearInterpolator,
    kernel_sigma: GaussianSigma,
    epsilon: f64,
    // CacheSlot: lazy-initialized, validity-checked per fixed-image geometry, Arc-shared across clones.
    cache: CacheSlot<LnccCacheEntry<B>>,
    // FilterSlot: GaussianFilter built once per dimension D (REG-03).
    filter_slot: FilterSlot<B>,
}

impl<B: Backend> LocalNormalizedCrossCorrelation<B> {
    /// Create a new LNCC metric.
    ///
    /// # Arguments
    /// * `kernel_sigma` - Standard deviation of the Gaussian kernel (mm) defining the local window size.
    pub fn new(kernel_sigma: GaussianSigma) -> Self {
        Self {
            interpolator: LinearInterpolator::new(),
            kernel_sigma,
            epsilon: 1e-5,
            cache: CacheSlot::empty(),
            filter_slot: FilterSlot::empty(),
        }
    }

    /// Computes local mean and variance using a Gaussian filter.
    fn compute_local_stats<const D: usize>(
        &self,
        img: Tensor<B, D>,
        filter: &GaussianFilter<B>,
        spacing: &ritk_spatial::Spacing<D>,
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
        let moving_values_flat = if n <= ritk_wgpu_compat::WGPU_CHUNK_SIZE {
            let fixed_points = fixed.index_to_world_tensor(fixed_indices);
            let moving_points = transform.transform_points(fixed_points);
            let moving_indices = moving.world_to_index_tensor(moving_points);
            self.interpolator.interpolate(moving.data(), moving_indices)
        } else {
            let num_chunks = n.div_ceil(ritk_wgpu_compat::WGPU_CHUNK_SIZE);
            let mut chunks = Vec::with_capacity(num_chunks);

            for i in 0..num_chunks {
                let start = i * ritk_wgpu_compat::WGPU_CHUNK_SIZE;
                let end = std::cmp::min(start + ritk_wgpu_compat::WGPU_CHUNK_SIZE, n);

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

        // 4. Setup filter (REG-03: get or construct once per dimension D).
        // The filter is cloned and the Mutex lock is dropped immediately,
        // eliminating lock contention during metric evaluation.
        let filter = self.filter_slot.get_or_init::<D>(self.kernel_sigma);

        // 5. Compute or load Local Stats for FIXED image (Stationary Cache O(1))
        let entry = self.cache.get_or_reinit_if(
            |e| e.is_valid_for::<D>(fixed),
            || {
                let (m_f, v_f) =
                    self.compute_local_stats(fixed_values.clone(), &filter, fixed.spacing());
                LnccCacheEntry {
                    shape: fixed.shape().to_vec(),
                    origin: collect_array::<3>(fixed.origin().0.iter().copied()),
                    spacing: collect_array::<3>(fixed.spacing().0.iter().copied()),
                    direction: collect_array::<9>(fixed.direction().0.iter().copied()),
                    mean_f_flat: m_f.flatten(0, D - 1),
                    var_f_flat: v_f.flatten(0, D - 1),
                }
            },
        );
        let mean_f = entry
            .mean_f_flat
            .clone()
            .reshape(burn::tensor::Shape::new(shape_dims));
        let var_f = entry
            .var_f_flat
            .clone()
            .reshape(burn::tensor::Shape::new(shape_dims));

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
#[path = "tests_lncc.rs"]
mod tests;
