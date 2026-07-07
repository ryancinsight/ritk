//! Cache lookup and normalization helpers for Parzen image-level histogram computation.
//!
//! Extracted from `compute_image/mod.rs` (Sprint 356 cycle 12-13) to keep each file
//! under the 500-line structural limit. All utilities are used exclusively by
//! [`super::compute_image`].

use super::super::cache::{HistogramCache, SparseWFixedCache};
use crate::metric::cache_slot::CacheSlot;
use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_image::tensor::Tensor;

/// Check whether a cached histogram entry matches the given image's spatial metadata.
pub(crate) fn cache_matches_image<B: Backend, const D: usize>(
    cache: &HistogramCache<B>,
    fixed: &Image<B, D>,
) -> bool {
    let fs = fixed.shape();
    cache.shape.as_slice() == fs
        && cache.origin.iter().eq(fixed.origin().0.iter())
        && cache.spacing.iter().eq(fixed.spacing().as_slice().iter())
        && cache.direction.0.iter().eq(fixed.direction().0.iter())
}

/// Helper: read the dense W_fixed^T from the cache if it matches the fixed image.
pub(crate) fn get_cached_w_fixed_t<B: Backend, const D: usize>(
    cache_guard: &Option<HistogramCache<B>>,
    fixed: &Image<B, D>,
) -> Option<Tensor<B, 2>> {
    cache_guard.as_ref().and_then(|c| {
        cache_matches_image(c, fixed)
            .then(|| c.w_fixed_transposed.clone())
            .flatten()
    })
}

/// Helper: read or lazily build the sparse W_fixed^T from the cache.
///
/// If the sparse cache already exists, returns a clone. Otherwise, if the
/// cache contains `fixed_norm` (the normalized fixed-image values), builds
/// the sparse cache from it, stores it in the cache for future use, and
/// returns it. This lazy construction reduces peak memory: on the first
/// cache-miss only the dense `w_fixed_transposed` tensor and the small
/// `fixed_norm` Vec are allocated; the ~2 MB sparse cache is deferred until
/// the sparse dispatch path is first requested.
#[cfg(feature = "direct-parzen")]
pub(crate) fn get_cached_sparse_w_fixed<B: Backend, const D: usize>(
    cache_guard: &mut Option<HistogramCache<B>>,
    fixed: &Image<B, D>,
    num_bins: usize,
    sigma_sq_fix: f32,
) -> Option<super::direct::SparseWFixedT> {
    if B::ad_enabled() {
        return None;
    }
    let cache = cache_guard.as_mut()?;
    if !cache_matches_image(cache, fixed) {
        return None;
    }
    cache.get_or_build_sparse_w_fixed(num_bins, sigma_sq_fix)
}

/// Normalize fixed-image values for lazy sparse cache construction.
///
/// Returns the normalized `Vec<f32>` in `[0, num_bins - 1]` so it can be
/// stored in the cache and later used by `get_cached_sparse_w_fixed` to
/// build the sparse W_fixed^T on first access. This avoids eagerly
/// constructing the sparse cache (~2 MB) on every cache-miss; only the
/// ~128 KB `fixed_norm` Vec is stored up front.
#[cfg(feature = "direct-parzen")]
pub(crate) fn normalize_fixed_values<B: Backend>(
    fixed_values: &Tensor<B, 1>,
    min_intensity: f32,
    max_intensity: f32,
    num_bins: usize,
) -> Vec<f32> {
    // `normalize_and_extract` returns `Cow<[f32]>` for zero-copy on the
    // borrow-friendly paths; the cache stores `fixed_norm` for the lifetime of
    // the level, so it must own the data.
    super::dispatch::normalize_and_extract(fixed_values, min_intensity, max_intensity, num_bins)
        .into_owned()
}

/// Extract cached `fixed_points [N, D]` from the internal `HistogramCache`.
///
/// Returns `None` when the cache is absent or does not match `fixed`'s spatial
/// metadata. Moved here from `compute_image` (Sprint SRP-split) to keep
/// `compute_image/mod.rs` under 500 lines.
pub(crate) fn extract_cached_points<B: Backend, const D: usize>(
    fixed: &Image<B, D>,
    cache: &CacheSlot<HistogramCache<B>>,
) -> Option<Tensor<B, 2>> {
    cache.with_ref(|guard| {
        guard
            .as_ref()
            .filter(|c| cache_matches_image(c, fixed))
            .map(|c| c.points.clone())
    })
}
