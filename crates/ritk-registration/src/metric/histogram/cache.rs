use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[cfg(feature = "direct-parzen")]
use super::parzen::direct::SparseWFixedT;

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
    /// Each `sparse_w_fixed[i]` contains only the ~7 non-zero (bin_index, weight)
    /// pairs for sample `i`, eliminating the full `0..num_bins` scan and
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
}
