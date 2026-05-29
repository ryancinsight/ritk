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
//! Both filters delegate to [`crate::filter::morphology::label_morphology::MorphologicalReconstruction`]
//! with the appropriate [`crate::filter::morphology::label_morphology::ReconstructionMode`].
//!
//! # References
//! - Vincent, L. (1993). Morphological grayscale reconstruction in image analysis.
//!   *IEEE Trans. Image Process.* 2(2):176–201.

use crate::filter::morphology::label_morphology::{
    MorphologicalReconstruction, ReconstructionMode,
};
use crate::image::Image;
use burn::tensor::backend::Backend;

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
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let td = TensorData::new(vals, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0_f64, 0.0, 0.0]),
            Spacing::new([1.0_f64, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn voxels(img: &Image<B, 3>) -> Vec<f32> {
        img.data_vec()
    }

    /// When marker equals mask, reconstruction by dilation returns marker unchanged.
    #[test]
    fn geodesic_dilation_marker_equals_mask_is_identity() {
        let vals = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let marker = make_image(vals.clone(), [2, 2, 2]);
        let mask = make_image(vals.clone(), [2, 2, 2]);
        let out = GrayscaleGeodesicDilationFilter::new()
            .apply(&marker, &mask)
            .unwrap();
        let v = voxels(&out);
        for (i, (&a, &b)) in v.iter().zip(vals.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "voxel {}: expected {}, got {}",
                i,
                b,
                a
            );
        }
    }

    /// Marker ≤ mask: reconstruction expands marker but never exceeds mask.
    #[test]
    fn geodesic_dilation_result_bounded_by_mask() {
        // 1×1×5: marker=[0,0,5,0,0], mask=[3,3,5,3,3]
        let marker = make_image(vec![0.0f32, 0.0, 5.0, 0.0, 0.0], [1, 1, 5]);
        let mask = make_image(vec![3.0f32, 3.0, 5.0, 3.0, 3.0], [1, 1, 5]);
        let out = GrayscaleGeodesicDilationFilter::new()
            .apply(&marker, &mask)
            .unwrap();
        let v = voxels(&out);
        let mv = [3.0f32, 3.0, 5.0, 3.0, 3.0];
        for (i, (&a, &b)) in v.iter().zip(mv.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "voxel {}: expected {}, got {}",
                i,
                b,
                a
            );
        }
    }

    /// Spatial metadata is preserved identically.
    #[test]
    fn geodesic_dilation_preserves_spatial_metadata() {
        let vals = vec![1.0f32; 8];
        let marker = make_image(vals.clone(), [2, 2, 2]);
        let mask = make_image(vals, [2, 2, 2]);
        let out = GrayscaleGeodesicDilationFilter::new()
            .apply(&marker, &mask)
            .unwrap();
        assert_eq!(out.shape(), marker.shape());
        assert_eq!(out.spacing(), marker.spacing());
        assert_eq!(out.origin(), marker.origin());
    }

    /// When marker equals mask, reconstruction by erosion returns marker unchanged.
    #[test]
    fn geodesic_erosion_marker_equals_mask_is_identity() {
        let vals = vec![7.0f32, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0];
        let marker = make_image(vals.clone(), [2, 2, 2]);
        let mask = make_image(vals.clone(), [2, 2, 2]);
        let out = GrayscaleGeodesicErosionFilter::new()
            .apply(&marker, &mask)
            .unwrap();
        let v = voxels(&out);
        for (i, (&a, &b)) in v.iter().zip(vals.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "voxel {}: expected {}, got {}",
                i,
                b,
                a
            );
        }
    }

    /// Marker ≥ mask: reconstruction contracts marker but never goes below mask.
    #[test]
    fn geodesic_erosion_result_bounded_below_by_mask() {
        // 1×1×5: marker=[5,5,0,5,5], mask=[3,3,0,3,3]; result must ≥ mask
        let marker = make_image(vec![5.0f32, 5.0, 0.0, 5.0, 5.0], [1, 1, 5]);
        let mask = make_image(vec![3.0f32, 3.0, 0.0, 3.0, 3.0], [1, 1, 5]);
        let out = GrayscaleGeodesicErosionFilter::new()
            .apply(&marker, &mask)
            .unwrap();
        let v = voxels(&out);
        let mask_v = [3.0f32, 3.0, 0.0, 3.0, 3.0];
        for (i, (&a, &b)) in v.iter().zip(mask_v.iter()).enumerate() {
            assert!(a >= b - 1e-4, "voxel {}: result {} below mask {}", i, a, b);
        }
    }
}
