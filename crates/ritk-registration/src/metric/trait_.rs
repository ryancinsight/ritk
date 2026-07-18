//! Metric trait for image similarity measurement.
//!
//! This module defines the core Metric trait that all similarity metrics
//! must implement for image registration.

use ritk_image::tensor::Backend;
use ritk_image::tensor::Tensor;
use ritk_image::Image;
use ritk_transform::Transform;

/// Metric trait for measuring similarity between images.
///
/// Metrics compute a loss value that represents the dissimilarity between
/// a fixed (reference) image and a moving (aligned) image. Lower values
/// indicate better alignment.
///
/// # Type Parameters
/// * `B` - The tensor backend
/// * `D` - The spatial dimensionality (2 or 3)
pub trait Metric<B: Backend, const D: usize> {
    /// Calculate the loss (dissimilarity) between fixed and moving images.
    ///
    /// # Arguments
    /// * `fixed` - The fixed (reference) image
    /// * `moving` - The moving (aligned) image
    /// * `transform` - The spatial transform applied to map from fixed to moving image space
    ///
    /// # Returns
    /// Scalar tensor representing the loss value
    fn forward(
        &self,
        fixed: &Image<f32, B, D>,
        moving: &Image<f32, B, D>,
        transform: &impl Transform<B, D>,
    ) -> Tensor<f32, B>;

    /// Calculate the loss with explicit fixed-image cache reuse (Phase 1-P2).
    ///
    /// This method exposes the same path as [`forward`](Self::forward) but
    /// documents the contract that **constant terms** (those depending only on
    /// `fixed` and not on `transform` or `moving`) may be cached and reused
    /// across calls â€” e.g. the Parzen weight matrix `W_fixed^T [num_bins, N]`
    /// for histogram-based metrics like `MutualInformation`.
    ///
    /// The default implementation simply delegates to [`forward`](Self::forward)
    /// and recomputes constant terms every call. Metrics whose forward path
    /// already caches constant terms internally (e.g. `MutualInformation` via
    /// `ParzenJointHistogram::compute_image_joint_histogram`) do not need to
    /// override this method; the cache contract is already satisfied.
    ///
    /// # Arguments
    /// * `fixed` - The fixed (reference) image (cache key)
    /// * `moving` - The moving (aligned) image
    /// * `transform` - The spatial transform applied to map from fixed to moving image space
    ///
    /// # Returns
    /// Scalar tensor representing the loss value
    ///
    /// # Performance note
    /// For a 256Â³ volume with Mattes MI (50 bins), the full-grid `W_fixed^T`
    /// matrix is `[50, 16M]` = ~3.2 GB. Recomputing it every iteration costs
    /// ~400 ms on a 16-core CPU; caching saves ~99 % of that on iteration 2+.
    /// See `docs/audit_optimization_sprint_350.md` Â§2.3 for the breakdown.
    fn forward_with_cache(
        &self,
        fixed: &Image<f32, B, D>,
        moving: &Image<f32, B, D>,
        transform: &impl Transform<B, D>,
    ) -> Tensor<f32, B> {
        // Default: cache contract is met by recomputing each call. Metrics with
        // built-in caching (e.g. MI via ParzenJointHistogram) inherit this
        // default but the cache hit is handled inside compute_image_joint_histogram.
        let _ = fixed; // suppress unused warning in default impl
        self.forward(fixed, moving, transform)
    }

    /// Get the name of this metric.
    ///
    /// # Returns
    /// String identifier for the metric
    fn name(&self) -> &'static str;
}
