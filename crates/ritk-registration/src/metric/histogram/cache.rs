use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::types::DirectionFingerprint;

/// Collect up to `N` elements from `iter` into `[f64; N]`; remaining slots are zero-filled.
pub(crate) fn collect_array<const N: usize>(iter: impl Iterator<Item = f64>) -> [f64; N] {
    let mut arr = [0.0f64; N];
    for (i, v) in iter.take(N).enumerate() {
        arr[i] = v;
    }
    arr
}

#[cfg(feature = "direct-parzen")]
use super::parzen::direct::SparseWFixedT;

/// Cache entry for the per-`MutualInformation` W_fixed^T reuse (350-P1-03).
///
/// Stores the precomputed Parzen weight matrix `W_fixed^T [num_bins, N]` keyed
/// on the fixed image's spatial fingerprint (shape, origin, spacing, direction)
/// and the point count `n`. Subsequent `forward_with_cache` calls with the same
/// (fingerprint, n) pair reuse the cached `W_fixed^T` instead of recomputing
/// the O(N × num_bins) Parzen weight matrix every iteration.
///
/// This is the public cache exposed via
/// `ParzenJointHistogram::compute_image_joint_histogram_with_w_fixed`. The
/// internal `HistogramCache` (private to `parzen::compute_image`) is the build
/// site; `WFixedCache` is the lightweight struct `MutualInformation` owns for
/// cross-call state.
#[derive(Clone, Debug)]
pub(crate) struct WFixedCache<B: Backend> {
    /// Fixed image shape (cloned from `fixed.shape()` at first build).
    pub shape: Vec<usize>,
    /// Fixed image origin `[3]` (stack-allocated to avoid heap for 24 bytes).
    pub origin: [f64; 3],
    /// Fixed image spacing `[3]`.
    pub spacing: [f64; 3],
    /// Fixed image direction (column-major 3×3 matrix, flattened).
    pub direction: DirectionFingerprint,
    /// Number of points `N` in the W_fixed^T matrix.
    pub n: usize,
    /// Cached W_fixed^T `[num_bins, N]`.
    pub w_fixed_t: Tensor<B, 2>,
}

impl<B: Backend> WFixedCache<B> {
    /// Build a `WFixedCache` from a fixed image and its computed W_fixed^T.
    pub fn from_image<const D: usize>(
        fixed: &ritk_core::image::Image<B, D>,
        n: usize,
        w_fixed_t: Tensor<B, 2>,
    ) -> Self {
        Self {
            shape: fixed.shape().to_vec(),
            origin: collect_array::<3>(fixed.origin().0.iter().copied()),
            spacing: collect_array::<3>(fixed.spacing().0.iter().copied()),
            direction: DirectionFingerprint(collect_array::<9>(
                fixed.direction().0.iter().copied(),
            )),
            n,
            w_fixed_t,
        }
    }

    /// Return `true` iff this cache entry matches the given fixed image and
    /// point count. Used by `MutualInformation` to detect cache hits without
    /// re-running the full `cache_matches_image` flow.
    pub fn matches<const D: usize>(&self, fixed: &ritk_core::image::Image<B, D>, n: usize) -> bool {
        self.n == n
            && self.shape.as_slice() == fixed.shape()
            && self.origin.iter().eq(fixed.origin().0.iter())
            && self.spacing.iter().eq(fixed.spacing().0.iter())
            && self.direction.0.iter().eq(fixed.direction().0.iter())
    }
}

/// Trait for cache entries that can lazily build a sparse W_fixed^T representation.
///
/// Both `HistogramCache` and `MaskedHistogramCache` share identical lazy-build
/// logic for `sparse_w_fixed` from `fixed_norm`. This trait eliminates the
/// duplicated "check sparse_w_fixed → take fixed_norm → build → store → return clone"
/// pattern that was previously inlined in both `impl` blocks.
///
/// Only available when the `direct-parzen` feature is enabled, since the
/// `sparse_w_fixed` and `fixed_norm` fields only exist under that cfg.
#[cfg(feature = "direct-parzen")]
pub(crate) trait SparseWFixedCache {
    /// Read the current sparse W_fixed^T, if already built.
    fn sparse_w_fixed(&self) -> &Option<SparseWFixedT>;

    /// Mutably access the sparse W_fixed^T storage.
    fn sparse_w_fixed_mut(&mut self) -> &mut Option<SparseWFixedT>;

    /// Take the normalized fixed-image values, consuming them.
    ///
    /// Returns `None` if the values have already been consumed by a prior
    /// lazy build.
    fn take_fixed_norm(&mut self) -> Option<Vec<f32>>;

    /// Read or lazily build the sparse W_fixed^T from this cache entry.
    ///
    /// If the sparse cache already exists, returns a clone. Otherwise, if
    /// `fixed_norm` is present, builds the sparse cache from it, stores it
    /// for future use, consumes `fixed_norm`, and returns the sparse cache.
    /// Returns `None` if `fixed_norm` has already been consumed and the
    /// sparse cache hasn't been built yet (shouldn't happen in practice).
    fn get_or_build_sparse_w_fixed(
        &mut self,
        num_bins: usize,
        sigma_sq_fix: f32,
    ) -> Option<SparseWFixedT> {
        if self.sparse_w_fixed().is_some() {
            return self.sparse_w_fixed().clone();
        }
        let fixed_norm = self.take_fixed_norm()?;
        let sparse = super::parzen::direct::build_sparse_w_fixed_transposed(
            &fixed_norm,
            num_bins,
            sigma_sq_fix,
            None,
        );
        *self.sparse_w_fixed_mut() = Some(sparse);
        self.sparse_w_fixed().clone()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct HistogramCache<B: Backend> {
    /// World-space coordinates of all fixed-image voxels [N, D].
    pub points: Tensor<B, 2>,

    /// Precomputed Parzen weight matrix for the fixed image, transposed: [num_bins, N].
    /// Constant across all registration iterations because the fixed image never changes.
    /// Reusing this avoids O(N × num_bins) kernel computation and removes the fixed-image
    /// Parzen path from the autodiff graph on every iteration after the first.
    pub w_fixed_transposed: Option<Tensor<B, 2>>,

    /// Sparse representation of W_fixed^T for the direct CPU path.
    /// Each `sparse_w_fixed[i]` contains only the ~7 non-zero `SparseWFixedEntry`
    /// values for sample `i`, eliminating the full `0..num_bins` scan and
    /// `if w_f > 0.0` branch in the hot loop. Built lazily from `fixed_norm`
    /// the first time the sparse dispatch path is taken, then reused every
    /// iteration.
    ///
    /// Only populated when the `direct-parzen` feature is enabled.
    ///
    /// Because this field is gated by `#[cfg]`, `HistogramCache` cannot be
    /// constructed by a single function under both feature configurations —
    /// the struct literal would be missing the field in one cfg or reference a
    /// non-existent type in the other. See `compute_image/mod.rs::make_cache` for
    /// the two cfg-specific constructors that work around this.
    #[cfg(feature = "direct-parzen")]
    pub sparse_w_fixed: Option<SparseWFixedT>,

    /// Normalized fixed-image values `[N]` in `[0, num_bins - 1]`.
    ///
    /// Stored so the sparse W_fixed^T cache can be built lazily from it on
    /// first access, rather than eagerly at cache construction time. This
    /// reduces peak memory during the initial cache-miss: instead of holding
    /// both the dense tensor (~4 MB) *and* the sparse cache (~2 MB)
    /// simultaneously, we keep only the dense tensor plus this ~128 KB Vec.
    ///
    /// Only needed when the `direct-parzen` feature is enabled.
    #[cfg(feature = "direct-parzen")]
    pub fixed_norm: Option<Vec<f32>>,

    pub shape: Vec<usize>,
    pub origin: [f64; 3],
    pub spacing: [f64; 3],
    pub direction: DirectionFingerprint,
}

/// Cache for the masked joint histogram path.
///
/// Unlike `HistogramCache` (which matches on image spatial metadata), this cache
/// uses a caller-supplied `cache_key: u64` to identify a particular mask/point-set.
/// The caller (e.g., CMA-ES optimizer) provides a generation counter or hash that
/// changes only when the mask changes, so the cached fixed-image Parzen weights
/// are reused across iterations for the same mask.
///
/// # Cache key collision guard
///
/// A known risk with caller-supplied `cache_key` is partial key collision: two
/// different masks with the same `cache_key` and point count `n` would incorrectly
/// reuse cached weights. To mitigate this, an optional `data_fingerprint` can be
/// stored — a SipHash-1-3 of the normalized fixed-image data. On cache hit, if a
/// fingerprint is present and doesn't match the current data, the cache is
/// invalidated. This provides deterministic collision detection.
#[derive(Debug, Clone)]
pub(crate) struct MaskedHistogramCache<B: Backend> {
    /// Caller-supplied key that identifies this particular mask/point-set.
    pub cache_key: u64,

    /// Pre-computed Parzen weight matrix for the fixed image, transposed: [num_bins, N].
    pub w_fixed_transposed: Option<Tensor<B, 2>>,

    /// Sparse representation of W_fixed^T for the direct CPU path.
    ///
    /// Built lazily from `fixed_norm` the first time the sparse dispatch path
    /// is taken, then reused every iteration. Only populated when the
    /// `direct-parzen` feature is enabled.
    #[cfg(feature = "direct-parzen")]
    pub sparse_w_fixed: Option<SparseWFixedT>,

    /// Normalized fixed-image values `[N]` in `[0, num_bins - 1]`.
    ///
    /// Stored so the sparse W_fixed^T cache can be built lazily from it on
    /// first access, rather than eagerly at cache construction time.
    ///
    /// Only needed when the `direct-parzen` feature is enabled.
    #[cfg(feature = "direct-parzen")]
    pub fixed_norm: Option<Vec<f32>>,

    /// Number of points in the mask (to detect mismatch).
    pub n: usize,

    /// SipHash-1-3 fingerprint of the full normalized fixed-image data.
    ///
    /// When provided, `get_masked_cached_w_fixed_t` and
    /// `get_masked_cached_sparse_w_fixed` check this fingerprint against the
    /// current data on cache hit. A mismatch indicates that the same
    /// `cache_key` was reused for different data (partial key collision),
    /// and the cache is treated as a miss.
    ///
    /// Only read when the `direct-parzen` feature is enabled (via
    /// `validate_masked_cache_fingerprint`).
    #[allow(dead_code)]
    // read only via validate_masked_cache_fingerprint (direct-parzen feature)
    pub data_fingerprint: Option<u64>,
}

#[cfg(feature = "direct-parzen")]
impl<B: Backend> SparseWFixedCache for HistogramCache<B> {
    fn sparse_w_fixed(&self) -> &Option<SparseWFixedT> {
        &self.sparse_w_fixed
    }

    fn sparse_w_fixed_mut(&mut self) -> &mut Option<SparseWFixedT> {
        &mut self.sparse_w_fixed
    }

    fn take_fixed_norm(&mut self) -> Option<Vec<f32>> {
        self.fixed_norm.take()
    }
}

#[cfg(feature = "direct-parzen")]
impl<B: Backend> SparseWFixedCache for MaskedHistogramCache<B> {
    fn sparse_w_fixed(&self) -> &Option<SparseWFixedT> {
        &self.sparse_w_fixed
    }

    fn sparse_w_fixed_mut(&mut self) -> &mut Option<SparseWFixedT> {
        &mut self.sparse_w_fixed
    }

    fn take_fixed_norm(&mut self) -> Option<Vec<f32>> {
        self.fixed_norm.take()
    }
}

// ============================================================================
// Cache constructors (Sprint 354, DRY-354-03)
// ============================================================================
//
// The `HistogramCache` and `MaskedHistogramCache` structs have fields gated by
// `#[cfg(feature = "direct-parzen")]` (`sparse_w_fixed`, `fixed_norm`,
// `data_fingerprint`). A single function cannot construct both variants because
// the struct literal would fail to compile in one cfg or the other. We provide
// two cfg-gated overloads of `make_cache` and `make_masked_cache` that share
// the same call signature, so the caller in `compute_image/mod.rs` /
// `masked/mod.rs` does not need to duplicate surrounding logic under both cfgs.
//
// The sparse cache (`sparse_w_fixed`) is **not** built here — it is constructed
// lazily by `get_cached_sparse_w_fixed` / `get_masked_cached_sparse_w_fixed` on
// first access from `fixed_norm`. This reduces peak memory on the initial
// cache-miss.

/// Construct a `HistogramCache` with dense representation and normalized fixed values.
///
/// `fixed_norm` carries the normalized `[0, num_bins - 1]` values used to
/// lazily build the sparse W_fixed^T on first access. When the
/// `direct-parzen` feature is off, this parameter must be `None::<()>`.
#[cfg(feature = "direct-parzen")]
pub(crate) fn make_cache<B: Backend, const D: usize>(
    points: Tensor<B, 2>,
    w_fixed_transposed: Tensor<B, 2>,
    fixed: &ritk_core::image::Image<B, D>,
    fixed_norm: Option<Vec<f32>>,
) -> HistogramCache<B> {
    HistogramCache {
        points,
        w_fixed_transposed: Some(w_fixed_transposed),
        sparse_w_fixed: None,
        fixed_norm,
        shape: fixed.shape().to_vec(),
        origin: collect_array::<3>(fixed.origin().0.iter().copied()),
        spacing: collect_array::<3>(fixed.spacing().0.iter().copied()),
        direction: DirectionFingerprint(collect_array::<9>(fixed.direction().0.iter().copied())),
    }
}

/// Construct a `HistogramCache` with only the dense representation
/// (non-`direct-parzen` overload).
#[cfg(not(feature = "direct-parzen"))]
pub(crate) fn make_cache<B: Backend, const D: usize>(
    points: Tensor<B, 2>,
    w_fixed_transposed: Tensor<B, 2>,
    fixed: &ritk_core::image::Image<B, D>,
    _fixed_norm: Option<()>,
) -> HistogramCache<B> {
    HistogramCache {
        points,
        w_fixed_transposed: Some(w_fixed_transposed),
        shape: fixed.shape().to_vec(),
        origin: collect_array::<3>(fixed.origin().0.iter().copied()),
        spacing: collect_array::<3>(fixed.spacing().0.iter().copied()),
        direction: DirectionFingerprint(collect_array::<9>(fixed.direction().0.iter().copied())),
    }
}

/// Construct a `MaskedHistogramCache` with dense representation and normalized
/// fixed values. When `fixed_norm` is `Some`, a `data_fingerprint` is computed
/// from the values and stored for collision detection.
#[cfg(feature = "direct-parzen")]
pub(crate) fn make_masked_cache<B: Backend>(
    cache_key: u64,
    w_fixed_transposed: Tensor<B, 2>,
    n: usize,
    fixed_norm: Option<Vec<f32>>,
) -> MaskedHistogramCache<B> {
    let data_fingerprint = fixed_norm.as_ref().map(|v| compute_fingerprint(v));
    MaskedHistogramCache {
        cache_key,
        w_fixed_transposed: Some(w_fixed_transposed),
        sparse_w_fixed: None,
        fixed_norm,
        n,
        data_fingerprint,
    }
}

/// Construct a `MaskedHistogramCache` with only the dense representation
/// (non-`direct-parzen` overload).
#[cfg(not(feature = "direct-parzen"))]
pub(crate) fn make_masked_cache<B: Backend>(
    cache_key: u64,
    w_fixed_transposed: Tensor<B, 2>,
    n: usize,
    _fixed_norm: Option<()>,
) -> MaskedHistogramCache<B> {
    MaskedHistogramCache {
        cache_key,
        w_fixed_transposed: Some(w_fixed_transposed),
        n,
        data_fingerprint: None,
    }
}

/// Compute a SipHash-1-3 fingerprint from normalized fixed-image values.
///
/// Uses `to_bits()` to convert each `f32` to a `u32` before hashing,
/// providing deterministic hashing (NaN is not expected in normalized data).
#[cfg(feature = "direct-parzen")]
fn compute_fingerprint(fixed_norm: &[f32]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    for &v in fixed_norm {
        v.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}
