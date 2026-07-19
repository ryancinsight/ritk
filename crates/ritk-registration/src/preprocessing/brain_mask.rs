//! Brain-mask generation for cross-modal rigid registration.
//!
//! Cross-modal CTâ†”MRI mutual-information registration is prone to converging on
//! a geometrically wrong pose: evaluated over the whole field of view, the MI
//! objective is dominated by the high-contrast skull/air boundary (and any
//! scanner bed/headrest in the CT), whose global maximum need not coincide with
//! brain alignment. Restricting the metric to a brain mask removes that bias and
//! lets MI peak at the true brain alignment.
//!
//! [`ct_brain_mask`] derives such a mask from a CT volume by threshold +
//! morphology, with no atlas or learned model:
//!
//! 1. Threshold to the soft-tissue Hounsfield window (default `[0, 100]` HU).
//! 2. Erode â€” break thin connections (skull, meninges, neck muscle).
//! 3. Keep the largest 26-connected component (the brain).
//! 4. Dilate â€” restore the eroded brain boundary.
//! 5. Fill internal holes.
//!
//! The pipeline previously lived only in the RIRE brain-mask integration test;
//! it is promoted here so registration consumers can mask without reimplementing
//! it.

use ritk_core::Image;
use ritk_filter::{BinaryDilateFilter, BinaryErodeFilter, BinaryFillholeFilter};
use ritk_image::tensor::Backend;
use ritk_segmentation::{binary_threshold, ConnectedComponentsFilter, Connectivity};

/// Parameters for [`ct_brain_mask`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CtBrainMaskConfig {
    /// Lower soft-tissue Hounsfield bound (inclusive).
    pub hu_low: f32,
    /// Upper soft-tissue Hounsfield bound (inclusive).
    pub hu_high: f32,
    /// Erosion radius \[voxels\] â€” severs thin skull/meninges/muscle bridges.
    pub erode_radius: usize,
    /// Dilation radius \[voxels\] â€” restores the brain boundary after erosion.
    pub dilate_radius: usize,
}

impl Default for CtBrainMaskConfig {
    /// Soft-tissue window `[0, 100]` HU with radius-2 erode/dilate â€” the values
    /// validated on RIRE Patient-001.
    fn default() -> Self {
        Self {
            hu_low: 0.0,
            hu_high: 100.0,
            erode_radius: 2,
            dilate_radius: 2,
        }
    }
}

/// Generate a binary brain mask (`1.0` inside, `0.0` outside) from a CT volume by
/// threshold + morphology. See the [module docs](self) for the pipeline.
///
/// # Panics
/// Panics if a morphology/threshold stage fails or the eroded soft-tissue mask
/// has no connected component (e.g. an empty or non-CT input).
#[must_use]
pub fn ct_brain_mask<B: Backend>(
    ct: &Image<f32, B, 3>,
    config: &CtBrainMaskConfig,
) -> Image<f32, B, 3> {
    let mask = binary_threshold(ct, config.hu_low, config.hu_high, 1.0, 0.0);

    let eroded = BinaryErodeFilter::new(config.erode_radius)
        .apply(&mask)
        .expect("brain-mask erosion failed");

    let cc = ConnectedComponentsFilter::with_connectivity(Connectivity::TwentySix);
    let (label_img, stats) = cc.apply(&eroded);
    let largest = stats
        .iter()
        .max_by_key(|s| s.voxel_count)
        .expect("brain mask: eroded soft-tissue mask has no connected component");
    let lv = largest.label as f32;
    let largest_only = binary_threshold(&label_img, lv, lv, 1.0, 0.0);

    let dilated = BinaryDilateFilter::new(config.dilate_radius)
        .apply(&largest_only)
        .expect("brain-mask dilation failed");

    BinaryFillholeFilter::new()
        .apply(&dilated)
        .expect("brain-mask hole-fill failed")
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use ritk_core::{Direction, Point, Spacing};
    use ritk_image::tensor::Tensor;

    type B = SequentialBackend;

    /// A solid soft-tissue cube (50 HU) embedded in air (âˆ’1000 HU), wrapped in a
    /// one-voxel bone shell (1000 HU): the mask must recover the cube interior.
    #[test]
    fn recovers_soft_tissue_core() {
        let n = 16usize;
        let device = Default::default();
        let mut v = vec![-1000.0f32; n * n * n];
        let at = |z: usize, y: usize, x: usize| z * n * n + y * n + x;
        for z in 2..14 {
            for y in 2..14 {
                for x in 2..14 {
                    // Bone shell at the outer ring, soft tissue inside.
                    let edge = z == 2 || z == 13 || y == 2 || y == 13 || x == 2 || x == 13;
                    v[at(z, y, x)] = if edge { 1000.0 } else { 50.0 };
                }
            }
        }
        let img = Image::<f32, B, 3>::new(
            Tensor::<f32, B>::from_slice_on([n, n, n], &v, &device),
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
        .expect("invariant: fixture tensor preserves the declared image rank");

        let mask = ct_brain_mask(&img, &CtBrainMaskConfig::default());
        let m = mask.data_slice().expect("fixture image is CPU-addressable");
        let fg = m.iter().filter(|&&x| x > 0.5).count();
        // Non-empty, and well inside the total volume (not the whole cube/air).
        assert!(fg > 0, "brain mask is empty");
        assert!(fg < n * n * n, "brain mask filled the whole volume");
        // The geometric centre (soft tissue) is masked in.
        assert!(m[at(8, 8, 8)] > 0.5, "core voxel not in mask");
        // A corner (air) is excluded.
        assert!(m[at(0, 0, 0)] < 0.5, "air corner wrongly in mask");
    }
}
