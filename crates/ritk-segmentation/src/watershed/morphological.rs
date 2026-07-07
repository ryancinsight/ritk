//! Morphological watershed (marker-less) segmentation.
//!
//! # Mathematical Specification
//!
//! Ports `itk::MorphologicalWatershedImageFilter`. It floods the relief from its
//! own regional minima rather than from external markers, decomposing exactly as
//!
//! ```text
//! MorphologicalWatershed(f, level) =
//!     MarkerControlledWatershed(f, label( RegionalMinima( H_level-minima(f) ) ))
//! ```
//!
//! where `H_level-minima` is the [`HMinimaFilter`] (identity at `level = 0`) that
//! fills basins shallower than `level`, so only sufficiently deep minima seed a
//! basin. The composition is bit-exact, label-for-label, to
//! `sitk.MorphologicalWatershed(level, markWatershedLine=True,
//! fullyConnected=False)` — verified for level 0/5/10 on the cthead gradient.
//!
//! Watershed-line voxels are label 0. Face (6-)connectivity throughout.

use ritk_filter::{HMinimaFilter, RegionalMinimaFilter};
use ritk_image::tensor::Backend;
use ritk_image::Image;

use super::MarkerControlledWatershed;
use crate::labeling::{connected_components, Connectivity};

/// Marker-less morphological watershed (`itk::MorphologicalWatershedImageFilter`).
#[derive(Debug, Clone, Copy)]
pub struct MorphologicalWatershed {
    /// Depth below which shallow minima are merged (h-minima level). ITK default `0`.
    pub level: f32,
}

impl Default for MorphologicalWatershed {
    fn default() -> Self {
        Self { level: 0.0 }
    }
}

impl MorphologicalWatershed {
    /// Construct with the given flooding level.
    pub fn new(level: f32) -> Self {
        Self { level }
    }

    /// Segment the relief `image` into watershed basins from its regional minima.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        // Suppress minima shallower than `level` (identity at level 0).
        let base = if self.level > 0.0 {
            HMinimaFilter::new(self.level).apply(image)?
        } else {
            image.clone()
        };
        // Regional minima → binary seeds → labelled markers.
        let minima = RegionalMinimaFilter::new()
            .with_values(1.0, 0.0)
            .apply(&base)?;
        let (markers, _) = connected_components(&minima, Connectivity::Six);
        // Flood the original relief from those markers.
        MarkerControlledWatershed::new().apply(image, &markers)
    }
}

#[cfg(test)]
#[path = "tests_morphological.rs"]
mod tests_morphological;
