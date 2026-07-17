//! Grayscale grind-peak filter for 3-D images (dual of grayscale fill-hole).
//!
//! # Mathematical Specification
//!
//! A "peak" is a bright regional maximum enclosed by a darker surround that is
//! not connected to the image boundary. Grind-peak grinds each such peak down to
//! the level of the highest "saddle" connecting it to the border — the dual of
//! [`super::grayscale_fillhole`], which raises enclosed dark pits.
//!
//! Computed as a morphological reconstruction by dilation:
//! `GrindPeak(f) = R^δ_f(J)` where the marker `J` equals `f` on the image border
//! and the global minimum of `f` in the interior. Reconstruction by dilation
//! lifts the interior marker back up to `f`, but bright peaks not reachable from
//! the border (where the marker stayed at the global minimum) are ground down to
//! the highest connecting saddle.
//!
//! # ITK / SimpleITK Parity
//!
//! Matches `itk::GrayscaleGrindPeakImageFilter` / `sitk.GrayscaleGrindPeak`
//! (`FullyConnectedOff` → 6-connected in 3-D). Output satisfies `g[x] ≤ f[x]`.
//!
//! # References
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer, §6.3.

use crate::morphology::label_morphology::{MorphologicalReconstruction, ReconstructionMode};
use crate::morphology::{on_image_border, Connectivity};
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Grayscale grind-peak filter: grinds down bright peaks not connected to the
/// image border (dual of grayscale fill-hole).
#[derive(Debug, Clone)]
pub struct GrayscaleGrindPeakFilter {
    connectivity: Connectivity,
}

impl Default for GrayscaleGrindPeakFilter {
    fn default() -> Self {
        Self {
            connectivity: Connectivity::Face6,
        }
    }
}

impl GrayscaleGrindPeakFilter {
    /// Create a new grayscale grind-peak filter (6-connected, ITK default).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the reconstruction-step structuring-element adjacency.
    pub fn with_connectivity(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// Apply the grind-peak filter to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let [nz, ny, nx] = dims;
        let gmin = vals.iter().copied().fold(f32::INFINITY, f32::min);

        // Marker: `f` on the border, global minimum in the interior.
        let mut marker = vec![gmin; vals.len()];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    if on_image_border(iz, iy, ix, dims) {
                        let flat = iz * ny * nx + iy * nx + ix;
                        marker[flat] = vals[flat];
                    }
                }
            }
        }
        let marker_img = rebuild(marker, dims, image);
        MorphologicalReconstruction::new(ReconstructionMode::Dilation)
            .with_connectivity(self.connectivity)
            .apply(&marker_img, image)
    }

    /// Coeus-native counterpart to the legacy application method.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (vals, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let [nz, ny, nx] = dims;
        let gmin = vals.iter().copied().fold(f32::INFINITY, f32::min);

        let mut marker = vec![gmin; vals.len()];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    if on_image_border(iz, iy, ix, dims) {
                        let flat = iz * ny * nx + iy * nx + ix;
                        marker[flat] = vals[flat];
                    }
                }
            }
        }
        let marker_img = crate::native_support::rebuild_image(marker, dims, image, backend)?;
        MorphologicalReconstruction::new(ReconstructionMode::Dilation)
            .with_connectivity(self.connectivity)
            .apply_native(&marker_img, image, backend)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_grayscale_grind_peak.rs"]
mod tests_grayscale_grind_peak;
