//! Interpolator trait for sampling values at continuous coordinates.
//!
//! This module defines the core Interpolator trait that all interpolation methods must implement.

use coeus_core::{ComputeBackend, Scalar};
use coeus_tensor::Tensor as CoeusTensor;
use ritk_image::tensor::{Backend, Tensor};

/// **Soft-deprecated (Atlas Batch #3 sub-batch #2, 2026-07-06; docstring-only)** —
/// prefer the Atlas-typed parallel [`InterpolatorAtlas`] (uses
/// `coeus_core::ComputeBackend` instead of Burn's `Backend`).
/// No `#[deprecated]` attribute on this item to avoid cascading
/// `#[warn(deprecated)]` warnings across 671 burner consumer files in
/// `xtask/burn_surface.allowlist`. Migration plan:
/// `atlas/docs/adr/0012-ritk-burn-trait-rebind.md` §Sub-batch #2.
///
/// Interpolator trait for sampling values at continuous coordinates.
///
/// Interpolators are used to sample image values at non-integer coordinates,
/// which is essential for image registration and resampling.
///
/// # Type Parameters
/// * `B` - The Burn backend
///
/// # Dimension restriction
/// Only `D ∈ {1, 2, 3, 4}` is supported. The dispatch layer enforces
/// this at runtime (panicking for unsupported dimensions), while the
/// compiler fully monomorphizes each supported dimension.
pub trait Interpolator<B: Backend> {
    /// Interpolate values from a tensor at given continuous indices.
    ///
    /// # Arguments
    /// * `data` - The source tensor (e.g., 3D volume `[D, H, W]` or 2D image `[H, W]`)
    /// * `indices` - The indices at which to interpolate `[Batch, Rank]`
    ///   Rank must match `data` dimensionality
    ///
    /// # Returns
    /// Tensor of sampled values `[Batch]`
    fn interpolate<const D: usize>(
        &self,
        data: &Tensor<B, D>,
        indices: Tensor<B, 2>,
    ) -> Tensor<B, 1>;
}

// Atlas-side parallel (Additive, day-1 surface; concrete impls land in
// sub-batch #3+). Cross-walked at
// `atlas/docs/adr/0012-ritk-burn-trait-rebind.md` §Decision §Sub-batch #1.

/// Atlas-typed parallel to [`Interpolator`]. Mirrors the Burn surface shape
/// verbatim: data is the source tensor and indices is the batched continuous-
/// coord tensor at rank 2 (matching the legacy `Tensor<B, 2>` shape).
#[allow(dead_code)]
pub trait InterpolatorAtlas<T, B>
where
    T: Scalar,
    B: ComputeBackend,
{
    /// Interpolate values from an Atlas tensor at given continuous indices.
    fn interpolate(
        &self,
        data: &CoeusTensor<T, B>,
        indices: &CoeusTensor<T, B>,
    ) -> CoeusTensor<T, B>;
}
