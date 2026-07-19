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
//! fullyConnected=False)` â€” verified for level 0/5/10 on the cthead gradient.
//!
//! Watershed-line voxels are label 0. Face (6-)connectivity throughout.

use ritk_filter::{HMinimaFilter, RegionalMinimaFilter};
use ritk_image::tensor::Backend;
use ritk_image::Image;

use super::MarkerControlledWatershed;
use crate::labeling::{connected_components, ConnectedComponentsFilter, Connectivity};

/// Marker-less morphological watershed (`itk::MorphologicalWatershedImageFilter`).
#[derive(Debug, Clone, Copy)]
pub struct MorphologicalWatershed {
    level: f32,
}

impl Default for MorphologicalWatershed {
    fn default() -> Self {
        Self { level: 0.0 }
    }
}

impl MorphologicalWatershed {
    /// Construct with the given flooding level.
    ///
    /// # Errors
    ///
    /// Returns an error unless `level` is finite and nonnegative.
    pub fn new(level: f32) -> anyhow::Result<Self> {
        anyhow::ensure!(
            level.is_finite() && level >= 0.0,
            "morphological watershed level must be finite and nonnegative, got {level}"
        );
        Ok(Self { level })
    }

    /// Return the h-minima suppression level.
    pub fn level(&self) -> f32 {
        self.level
    }

    /// Segment the relief `image` into watershed basins from its regional minima.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        // Suppress minima shallower than `level` (identity at level 0).
        let base = if self.level > 0.0 {
            HMinimaFilter::new(self.level).apply(image)?
        } else {
            image.clone()
        };
        // Regional minima â†’ binary seeds â†’ labelled markers.
        let minima = RegionalMinimaFilter::new()
            .with_values(1.0, 0.0)
            .apply(&base)?;
        let (markers, _) = connected_components(&minima, Connectivity::Six);
        // Flood the original relief from those markers.
        MarkerControlledWatershed::new().apply(image, &markers)
    }

    /// Segment a Coeus-native relief image from its regional minima.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid relief/storage, a failed native
    /// intermediate, or marker-controlled flooding failure.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let suppressed = if self.level > 0.0 {
            Some(HMinimaFilter::new(self.level).apply_native(image, backend)?)
        } else {
            None
        };
        let base = match &suppressed {
            Some(filtered) => filtered,
            None => image,
        };
        let minima = RegionalMinimaFilter::new()
            .with_values(1.0, 0.0)
            .apply_native(base, backend)?;
        let (markers, _) = ConnectedComponentsFilter::with_connectivity(Connectivity::Six)
            .apply_native(&minima, backend)?;
        MarkerControlledWatershed::new().apply_native(image, &markers, backend)
    }
}

#[cfg(test)]
#[path = "tests_morphological.rs"]
mod tests_morphological;
