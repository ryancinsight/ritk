use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[cfg(feature = "direct-parzen")]
use super::parzen::direct::SparseWFixedT;

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

#[derive(Debug)]
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
    /// non-existent type in the other. See `compute_image.rs::make_cache` for
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
    pub origin: Vec<f64>,
    pub spacing: Vec<f64>,
    pub direction: Vec<f64>,
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
#[derive(Debug)]
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
