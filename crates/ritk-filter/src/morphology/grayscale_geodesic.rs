//! Geodesic grayscale morphological filters.
//!
//! # Mathematical Specification
//!
//! **Geodesic dilation** (Vincent 1993, Dilation Reconstruction):
//! Given marker `M` and mask `I` with `M â‰¤ I`:
//!
//! `M* = lim_{kâ†’âˆž} min(D_B(M_k), I)`
//!
//! where `D_B` is one-step grayscale dilation with the unit-radius cubic element `B`.
//! The fixed point `M*` is the morphological reconstruction of `I` from `M` by dilation.
//!
//! **Geodesic erosion** (Erosion Reconstruction):
//! Given marker `M` and mask `I` with `M â‰¥ I`:
//!
//! `M* = lim_{kâ†’âˆž} max(E_B(M_k), I)`
//!
//! # ITK / SimpleITK Parity
//!
//! | Filter                               | ITK class                                    |
//! |--------------------------------------|----------------------------------------------|
//! | `GrayscaleGeodesicDilationFilter`    | `GrayscaleGeodesicDilationImageFilter`       |
//! | `GrayscaleGeodesicErosionFilter`     | `GrayscaleGeodesicErosionImageFilter`        |
//!
//! Both filters delegate to [`crate::morphology::label_morphology::MorphologicalReconstruction`]
//! with the appropriate [`crate::morphology::label_morphology::ReconstructionMode`].
//!
//! # References
//! - Vincent, L. (1993). Morphological grayscale reconstruction in image analysis.
//!   *IEEE Trans. Image Process.* 2(2):176â€“201.

use crate::morphology::label_morphology::{MorphologicalReconstruction, ReconstructionMode};
use crate::morphology::Connectivity;
use ritk_image::tensor::Backend;
use ritk_image::Image;

// â”€â”€ GrayscaleGeodesicDilationFilter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Geodesic dilation: reconstruct `I` from marker `M` by iterative constrained dilation.
///
/// # Precondition
///
/// `M(x) â‰¤ I(x)` for all x. If the precondition is violated, values are clamped
/// to `min(M(x), I(x))` before reconstruction, matching ITK's behaviour.
///
/// # Usage
///
/// ```rust,ignore
/// let out = GrayscaleGeodesicDilationFilter::new().apply(&marker, &mask)?;
/// ```
#[derive(Debug, Clone)]
pub struct GrayscaleGeodesicDilationFilter {
    inner: MorphologicalReconstruction,
}

impl Default for GrayscaleGeodesicDilationFilter {
    fn default() -> Self {
        Self {
            inner: MorphologicalReconstruction::new(ReconstructionMode::Dilation),
        }
    }
}

impl GrayscaleGeodesicDilationFilter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the structuring-element adjacency (face vs full connectivity).
    /// Defaults to [`Connectivity::Face6`], matching ITK's `FullyConnectedOff`.
    pub fn with_connectivity(mut self, connectivity: Connectivity) -> Self {
        self.inner = self.inner.with_connectivity(connectivity);
        self
    }

    /// Apply geodesic dilation: reconstruct `mask` from `marker` by dilation.
    ///
    /// - `marker`: the seed image (must have `marker â‰¤ mask` pointwise)
    /// - `mask`: the constraint image
    pub fn apply<B: Backend>(
        &self,
        marker: &Image<f32, B, 3>,
        mask: &Image<f32, B, 3>,
    ) -> anyhow::Result<Image<f32, B, 3>> {
        self.inner.apply(marker, mask)
    }

    /// Coeus-native sister of [`GrayscaleGeodesicDilationFilter::apply`].
    ///
    /// Delegates to the native [`MorphologicalReconstruction::apply_native`],
    /// bitwise-identical to the Burn path. No Burn tensor is constructed.
    ///
    /// # Errors
    /// Returns an error on shape mismatch or non-contiguous buffers.
    pub fn apply_native<B>(
        &self,
        marker: &ritk_image::native::Image<f32, B, 3>,
        mask: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        self.inner.apply_native(marker, mask, backend)
    }
}

// â”€â”€ GrayscaleGeodesicErosionFilter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Geodesic erosion: reconstruct `I` from marker `M` by iterative constrained erosion.
///
/// # Precondition
///
/// `M(x) â‰¥ I(x)` for all x. If violated, values are clamped to `max(M(x), I(x))`
/// before reconstruction.
///
/// # Usage
///
/// ```rust,ignore
/// let out = GrayscaleGeodesicErosionFilter::new().apply(&marker, &mask)?;
/// ```
#[derive(Debug, Clone)]
pub struct GrayscaleGeodesicErosionFilter {
    inner: MorphologicalReconstruction,
}

impl Default for GrayscaleGeodesicErosionFilter {
    fn default() -> Self {
        Self {
            inner: MorphologicalReconstruction::new(ReconstructionMode::Erosion),
        }
    }
}

impl GrayscaleGeodesicErosionFilter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the structuring-element adjacency (face vs full connectivity).
    /// Defaults to [`Connectivity::Face6`], matching ITK's `FullyConnectedOff`.
    pub fn with_connectivity(mut self, connectivity: Connectivity) -> Self {
        self.inner = self.inner.with_connectivity(connectivity);
        self
    }

    /// Apply geodesic erosion: reconstruct `mask` from `marker` by erosion.
    ///
    /// - `marker`: the seed image (must have `marker â‰¥ mask` pointwise)
    /// - `mask`: the constraint image
    pub fn apply<B: Backend>(
        &self,
        marker: &Image<f32, B, 3>,
        mask: &Image<f32, B, 3>,
    ) -> anyhow::Result<Image<f32, B, 3>> {
        self.inner.apply(marker, mask)
    }

    /// Coeus-native sister of [`GrayscaleGeodesicErosionFilter::apply`].
    ///
    /// Delegates to the native [`MorphologicalReconstruction::apply_native`],
    /// bitwise-identical to the Burn path. No Burn tensor is constructed.
    ///
    /// # Errors
    /// Returns an error on shape mismatch or non-contiguous buffers.
    pub fn apply_native<B>(
        &self,
        marker: &ritk_image::native::Image<f32, B, 3>,
        mask: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        self.inner.apply_native(marker, mask, backend)
    }
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
#[path = "tests_grayscale_geodesic.rs"]
mod tests_grayscale_geodesic;
