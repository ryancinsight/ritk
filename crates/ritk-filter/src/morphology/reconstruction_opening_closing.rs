//! Opening- and closing-by-reconstruction grayscale morphological filters.
//!
//! # Mathematical Specification
//!
//! Unlike a plain morphological opening/closing (erosion∘dilation), the
//! *by-reconstruction* variants restore the exact contours of the features that
//! survive the marker step:
//!
//! - **Opening by reconstruction**: `OBR_B(f) = R^δ_f(ε_B(f))` — erode with the
//!   structuring element `B`, then reconstruct the eroded marker under `f` by
//!   dilation. Removes bright structures the SE cannot contain while leaving the
//!   surviving structures geometrically intact (no corner rounding).
//! - **Closing by reconstruction**: `CBR_B(f) = R^ε_f(δ_B(f))` — dilate, then
//!   reconstruct under `f` by erosion. The dual, for dark structures.
//!
//! The structuring element is a flat cubic (box) element of half-width
//! `radius`, matching `ritk`'s grayscale erosion/dilation. With a box SE these
//! are bit-exact to `sitk.OpeningByReconstruction` / `sitk.ClosingByReconstruction`
//! (`kernelType = sitkBox`).
//!
//! # ITK / SimpleITK Parity
//!
//! | Filter                            | ITK class                                   | SimpleITK                |
//! |-----------------------------------|---------------------------------------------|--------------------------|
//! | `OpeningByReconstructionFilter`   | `OpeningByReconstructionImageFilter`        | `OpeningByReconstruction`|
//! | `ClosingByReconstructionFilter`   | `ClosingByReconstructionImageFilter`        | `ClosingByReconstruction`|
//!
//! # References
//! - Vincent, L. (1993). Morphological grayscale reconstruction in image
//!   analysis. *IEEE Trans. Image Process.* 2(2):176–201.

use crate::morphology::label_morphology::{MorphologicalReconstruction, ReconstructionMode};
use crate::morphology::{Connectivity, GrayscaleDilation, GrayscaleErosion};
use burn::tensor::backend::Backend;
use ritk_image::Image;

/// Opening by reconstruction: `R^δ_f(ε_B(f))`.
#[derive(Debug, Clone)]
pub struct OpeningByReconstructionFilter {
    radius: usize,
    connectivity: Connectivity,
}

impl OpeningByReconstructionFilter {
    /// Create an opening-by-reconstruction filter with a box SE of half-width
    /// `radius`. The reconstruction step defaults to [`Connectivity::Face6`]
    /// (ITK `FullyConnectedOff`).
    pub fn new(radius: usize) -> Self {
        Self {
            radius,
            connectivity: Connectivity::Face6,
        }
    }

    /// Set the reconstruction-step structuring-element adjacency.
    pub fn with_connectivity(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// Apply opening by reconstruction.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let marker = GrayscaleErosion::new(self.radius).apply(image)?;
        MorphologicalReconstruction::new(ReconstructionMode::Dilation)
            .with_connectivity(self.connectivity)
            .apply(&marker, image)
    }
}

/// Closing by reconstruction: `R^ε_f(δ_B(f))`.
#[derive(Debug, Clone)]
pub struct ClosingByReconstructionFilter {
    radius: usize,
    connectivity: Connectivity,
}

impl ClosingByReconstructionFilter {
    /// Create a closing-by-reconstruction filter with a box SE of half-width
    /// `radius`. The reconstruction step defaults to [`Connectivity::Face6`].
    pub fn new(radius: usize) -> Self {
        Self {
            radius,
            connectivity: Connectivity::Face6,
        }
    }

    /// Set the reconstruction-step structuring-element adjacency.
    pub fn with_connectivity(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// Apply closing by reconstruction.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let marker = GrayscaleDilation::new(self.radius).apply(image)?;
        MorphologicalReconstruction::new(ReconstructionMode::Erosion)
            .with_connectivity(self.connectivity)
            .apply(&marker, image)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_reconstruction_opening_closing.rs"]
mod tests_reconstruction_opening_closing;
