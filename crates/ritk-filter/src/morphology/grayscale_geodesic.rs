//! Geodesic grayscale morphological filters.
//!
//! # Mathematical Specification
//!
//! **Geodesic dilation** (Vincent 1993, Dilation Reconstruction):
//! Given marker `M` and mask `I` with `M ≤ I`:
//!
//! `M* = lim_{k→∞} min(D_B(M_k), I)`
//!
//! where `D_B` is one-step grayscale dilation with the unit-radius cubic element `B`.
//! The fixed point `M*` is the morphological reconstruction of `I` from `M` by dilation.
//!
//! **Geodesic erosion** (Erosion Reconstruction):
//! Given marker `M` and mask `I` with `M ≥ I`:
//!
//! `M* = lim_{k→∞} max(E_B(M_k), I)`
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
//!   *IEEE Trans. Image Process.* 2(2):176–201.

use crate::morphology::label_morphology::{MorphologicalReconstruction, ReconstructionMode};
use crate::morphology::Connectivity;
use burn::tensor::backend::Backend;
use ritk_image::Image;

// ── GrayscaleGeodesicDilationFilter ──────────────────────────────────────────

/// Geodesic dilation: reconstruct `I` from marker `M` by iterative constrained dilation.
///
/// # Precondition
///
/// `M(x) ≤ I(x)` for all x. If the precondition is violated, values are clamped
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
    /// - `marker`: the seed image (must have `marker ≤ mask` pointwise)
    /// - `mask`: the constraint image
    pub fn apply<B: Backend>(
        &self,
        marker: &Image<B, 3>,
        mask: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        self.inner.apply(marker, mask)
    }
}

// ── GrayscaleGeodesicErosionFilter ───────────────────────────────────────────

/// Geodesic erosion: reconstruct `I` from marker `M` by iterative constrained erosion.
///
/// # Precondition
///
/// `M(x) ≥ I(x)` for all x. If violated, values are clamped to `max(M(x), I(x))`
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
    /// - `marker`: the seed image (must have `marker ≥ mask` pointwise)
    /// - `mask`: the constraint image
    pub fn apply<B: Backend>(
        &self,
        marker: &Image<B, 3>,
        mask: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        self.inner.apply(marker, mask)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_grayscale_geodesic.rs"]
mod tests_grayscale_geodesic;
