use ritk_image::tensor::Backend;
use ritk_image::tensor::Tensor;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::metric::cache_slot::CacheSlot;

#[cfg(feature = "direct-parzen")]
use self::direct::HistogramPool;
use super::cache::{HistogramCache, MaskedHistogramCache};

pub(crate) mod compute;
pub(crate) mod compute_image;
pub(crate) mod direct;
pub(crate) mod dispatch;
pub(crate) mod image_cache_helpers;
pub(crate) mod oob;
pub(crate) mod sparse;

#[cfg(test)]
mod tests;

pub(super) use oob::compute_oob_mask;

/// Joint Histogram Calculator using Parzen windowing.
#[derive(Debug)]
pub struct ParzenJointHistogram<B: Backend> {
    /// Number of histogram bins
    pub num_bins: usize,
    /// Minimum intensity value (fixed-image axis)
    pub min_intensity: f32,
    /// Maximum intensity value (fixed-image axis)
    pub max_intensity: f32,
    /// Parzen window sigma for histogram smoothing (fixed-image axis)
    pub parzen_sigma: f32,
    /// Optional separate minimum intensity for the moving image.
    /// When `None`, falls back to `min_intensity` (shared-range behaviour).
    pub moving_min_intensity: Option<f32>,
    /// Optional separate maximum intensity for the moving image.
    /// When `None`, falls back to `max_intensity` (shared-range behaviour).
    pub moving_max_intensity: Option<f32>,
    /// Optional separate Parzen sigma for the moving image.
    /// When `None`, falls back to `parzen_sigma`.
    pub moving_parzen_sigma: Option<f32>,
    /// Pre-computed bin center tensor `[1, num_bins]` — eagerly constructed in
    /// `new()` and reused across all Parzen weight computations. Eliminates 2
    /// GPU kernel dispatches (arange + int→float cast) per
    /// `compute_joint_histogram*` call.
    ///
    /// Wrapped in `Option<Tensor<B, 2>>` so that struct literals can set this
    /// field to `None` in test or deserialization contexts. In practice always
    /// `Some` after [`new()`](Self::new) initialises it.
    bins_exp: Option<Tensor<B, 2>>,
    /// Lazily populated cache of fixed-image Parzen weights and grid data for
    /// the image-grid histogram path.
    ///
    /// Uses [`CacheSlot<HistogramCache<B>>`] — a shared `Arc<Mutex<Option<T>>>`
    /// wrapper — so that `Clone` shares the cache across multi-resolution clones
    /// (all levels observe the same lazily-built data) and interior mutability
    /// allows population/invalidation without `&mut self`.
    pub(super) cache: CacheSlot<HistogramCache<B>>,
    /// Lazily populated cache for the masked histogram path (Strategy 2:
    /// caller-supplied cache key).
    ///
    /// Uses [`CacheSlot<MaskedHistogramCache<B>>`] for the same shared-ownership
    /// and interior-mutability reasons as [`Self::cache`].
    pub(super) masked_cache: CacheSlot<MaskedHistogramCache<B>>,
    /// Phantom data
    _phantom: PhantomData<fn() -> B>,
    /// Reusable histogram buffer pool, allocated once in `new()` and reused
    /// across CMA-ES iterations to avoid repeated O(num_bins²) allocations.
    /// Shared directly via `Arc` across clones.
    #[cfg(feature = "direct-parzen")]
    pub(super) histogram_pool: Arc<HistogramPool>,
}

/// Cloning a [`ParzenJointHistogram`] creates a new handle that **shares** the caches with
/// the original — both the original and the clone observe each other's cache updates and
/// invalidations via the shared `Arc` inside each `CacheSlot`. This is the intended behavior
/// in multi-resolution pipelines, where one metric handle per resolution level is created via
/// `.clone()` and all levels share a single lazily-built fixed-image cache.
impl<B: Backend> Clone for ParzenJointHistogram<B> {
    fn clone(&self) -> Self {
        Self {
            num_bins: self.num_bins,
            min_intensity: self.min_intensity,
            max_intensity: self.max_intensity,
            parzen_sigma: self.parzen_sigma,
            moving_min_intensity: self.moving_min_intensity,
            moving_max_intensity: self.moving_max_intensity,
            moving_parzen_sigma: self.moving_parzen_sigma,
            bins_exp: self.bins_exp.clone(),
            cache: self.cache.clone(),
            masked_cache: self.masked_cache.clone(),
            _phantom: PhantomData,
            #[cfg(feature = "direct-parzen")]
            histogram_pool: Arc::clone(&self.histogram_pool),
        }
    }
}

impl<B: Backend> ParzenJointHistogram<B> {
    /// Create a new Parzen Joint Histogram calculator.
    ///
    /// `device` is used to eagerly allocate the pre-computed bin-center tensor
    /// `[1, num_bins]`, eliminating 2 GPU kernel dispatches (arange + int→float
    /// cast) per weight-computation call on the hot path.
    pub fn new(
        num_bins: usize,
        min_intensity: f32,
        max_intensity: f32,
        parzen_sigma: f32,
        device: &B::Device,
    ) -> Self {
        Self {
            num_bins,
            min_intensity,
            max_intensity,
            parzen_sigma,
            moving_min_intensity: None,
            moving_max_intensity: None,
            moving_parzen_sigma: None,
            bins_exp: Some(compute::arange_bins(num_bins, device)),
            cache: CacheSlot::empty(),
            masked_cache: CacheSlot::empty(),
            _phantom: PhantomData,
            #[cfg(feature = "direct-parzen")]
            histogram_pool: Arc::new({
                let buffer_count = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1);
                HistogramPool::new_with_capacity(num_bins * num_bins, buffer_count)
            }),
        }
    }

    /// Configure a separate intensity range for the moving image (elastix-style independent binning).
    ///
    /// When set, each axis of the joint histogram uses its own normalization:
    /// the fixed axis spans `[min_intensity, max_intensity]` and the moving axis
    /// spans `[moving_min, moving_max]`, giving each image the full bin resolution.
    ///
    /// `moving_parzen_sigma` is set to `(moving_max - moving_min).max(1e-6) / num_bins`
    /// (Mattes parameterization: sigma = bin_width).
    pub fn with_separate_moving_range(mut self, moving_min: f32, moving_max: f32) -> Self {
        let sigma = (moving_max - moving_min).max(1e-6) / self.num_bins as f32;
        self.moving_min_intensity = Some(moving_min);
        self.moving_max_intensity = Some(moving_max);
        self.moving_parzen_sigma = Some(sigma);
        self
    }

    /// Compute the [`direct::ParzenConfig`] for the fixed-image axis (DRY-320-01).
    ///
    /// Encapsulates the repeated `ParzenConfig::from_intensity_sigma(
    /// self.parzen_sigma, self.min_intensity, self.max_intensity, self.num_bins)`
    /// pattern that appeared at 8 call sites across `compute.rs`,
    /// `compute_image/mod.rs`, and `masked/mod.rs`.
    pub(super) fn fixed_sigma_cfg(&self) -> direct::ParzenConfig {
        direct::ParzenConfig::from_intensity_sigma(
            self.parzen_sigma,
            self.min_intensity,
            self.max_intensity,
            self.num_bins,
        )
    }

    /// Compute the [`direct::ParzenConfig`] for the moving-image axis (DRY-320-01).
    ///
    /// Uses `moving_parzen_sigma`, `moving_min_intensity`, and
    /// `moving_max_intensity` when set; falls back to the fixed-image
    /// range (backward-compatible).
    pub(super) fn moving_sigma_cfg(&self) -> direct::ParzenConfig {
        let mov_min = self.moving_min_intensity.unwrap_or(self.min_intensity);
        let mov_max = self.moving_max_intensity.unwrap_or(self.max_intensity);
        let mov_sigma = self.moving_parzen_sigma.unwrap_or(self.parzen_sigma);
        direct::ParzenConfig::from_intensity_sigma(mov_sigma, mov_min, mov_max, self.num_bins)
    }

    /// Compute Entropy of a distribution P.
    pub fn compute_entropy(&self, p: Tensor<B, 1>) -> Tensor<B, 1> {
        let eps = 1e-10;
        let log_p = (p.clone() + eps).log();
        p.mul(log_p).sum().neg()
    }

    /// Invalidate the image-grid cache.
    ///
    /// Clears the cached fixed-image weights, points, and sparse data that
    /// are reused across [`compute_image_joint_histogram`](Self::compute_image_joint_histogram)
    /// calls for the same spatial metadata.  Call this:
    ///
    /// - when switching to a different fixed image mid-registration,
    /// - between multi-resolution levels to free GPU/CPU memory, or
    /// - any time you want to ensure the next histogram computation starts
    ///   from scratch.
    ///
    /// The cache will be rebuilt on the next call to
    /// [`compute_image_joint_histogram`](Self::compute_image_joint_histogram).
    ///
    /// Invalidation is idempotent: clearing an already-`None` cache is a
    /// no-op.
    pub fn invalidate_cache(&self) {
        self.cache.invalidate();
    }

    /// Invalidate the masked-path cache.
    ///
    /// Clears the cached `W_fixed^T`, sparse weights, and associated
    /// metadata that are reused across
    /// [`compute_masked_joint_histogram`](Self::compute_masked_joint_histogram)
    /// calls with the same `cache_key`.  Call this:
    ///
    /// - when switching to a different fixed image or mask between
    ///   registration stages,
    /// - to release the `sparse_w_fixed` tensor (~2 MB for 32K samples) and
    ///   other cached data between independent registration runs, or
    /// - any time you want to ensure the next masked histogram computation
    ///   starts from scratch.
    ///
    /// The cache will be rebuilt on the next call to
    /// [`compute_masked_joint_histogram`](Self::compute_masked_joint_histogram)
    /// with a `cache_key`.
    ///
    /// Invalidation is idempotent: clearing an already-`None` cache is a
    /// no-op.
    pub fn invalidate_masked_cache(&self) {
        self.masked_cache.invalidate();
    }

    /// Invalidate both the image-grid cache and the masked-path cache.
    ///
    /// Convenience method that calls [`invalidate_cache`](Self::invalidate_cache)
    /// and [`invalidate_masked_cache`](Self::invalidate_masked_cache) together.
    /// Useful when fully resetting between registration stages or when
    /// switching fixed images.
    ///
    /// Both caches will be rebuilt on their respective next computation
    /// calls.
    ///
    /// Invalidation is idempotent: clearing already-`None` caches is a
    /// no-op.
    pub fn invalidate_all_caches(&self) {
        self.invalidate_cache();
        self.invalidate_masked_cache();
    }

    /// Validate that the masked cache still corresponds to the current data.
    ///
    /// Checks whether the stored `data_fingerprint` in the masked cache
    /// matches the fingerprint computed from the provided normalized
    /// fixed-image values. If they don't match (or the cache is empty),
    /// the masked cache is invalidated and `false` is returned.
    ///
    /// This guards against **partial key collisions**: the scenario where
    /// two different mask/point-sets share the same `cache_key` and point
    /// count `n`, which would otherwise cause incorrect cache reuse.
    ///
    /// # Arguments
    /// * `fixed_norm` — Normalized fixed-image values `[N]` in
    ///   `[0, num_bins - 1]`, as returned by `normalize_and_extract`.
    ///
    /// # Returns
    /// * `true` if the cache is valid (fingerprint matches, or no
    ///   fingerprint was stored).
    /// * `false` if the fingerprint mismatched (cache was invalidated).
    ///
    /// # Example
    /// ```ignore
    /// // Before reusing the cache with the same key but potentially different data:
    /// if !hist.validate_masked_cache_fingerprint(&fixed_norm) {
    ///     // Cache was invalidated — next compute call will rebuild it
    /// }
    /// ```
    #[cfg(feature = "direct-parzen")]
    pub fn validate_masked_cache_fingerprint(&self, fixed_norm: &[f32]) -> bool {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for &v in fixed_norm {
            v.to_bits().hash(&mut hasher);
        }
        let current_fp = hasher.finish();

        self.masked_cache.with_mut(|cache| {
            if let Some(ref masked) = *cache {
                if let Some(stored_fp) = masked.data_fingerprint {
                    if stored_fp != current_fp {
                        // Fingerprint mismatch — invalidate
                        *cache = None;
                        return false;
                    }
                }
            }
            true
        })
    }
}
