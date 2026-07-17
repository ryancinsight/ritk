//! Double-threshold (hysteresis) filter.
//!
//! Matches ITK `DoubleThresholdImageFilter` / `sitk.DoubleThreshold`: a voxel is
//! foreground if it lies in the **inner** band `[t2, t3]`, or in the **outer**
//! band `[t1, t4]` and is connected (through the outer band) to an inner-band
//! voxel. This is exactly a binary morphological reconstruction of the inner
//! band (marker) under the outer band (mask).

use crate::morphology::{Connectivity, MorphologicalReconstruction, ReconstructionMode};
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Double-threshold (hysteresis) filter with four ordered thresholds
/// `t1 ≤ t2 ≤ t3 ≤ t4`.
#[derive(Debug, Clone)]
pub struct DoubleThresholdImageFilter {
    /// Lower bound of the outer band.
    pub threshold1: f32,
    /// Lower bound of the inner band.
    pub threshold2: f32,
    /// Upper bound of the inner band.
    pub threshold3: f32,
    /// Upper bound of the outer band.
    pub threshold4: f32,
    /// Output value for foreground voxels.
    pub inside_value: f32,
    /// Output value for background voxels.
    pub outside_value: f32,
    /// Reconstruction adjacency (ITK `FullyConnectedOff` → [`Connectivity::Face6`]).
    pub connectivity: Connectivity,
}

impl DoubleThresholdImageFilter {
    /// Construct with the four thresholds and inside/outside values
    /// (defaults: face-connected).
    pub fn new(
        threshold1: f32,
        threshold2: f32,
        threshold3: f32,
        threshold4: f32,
        inside_value: f32,
        outside_value: f32,
    ) -> Self {
        Self {
            threshold1,
            threshold2,
            threshold3,
            threshold4,
            inside_value,
            outside_value,
            connectivity: Connectivity::Face6,
        }
    }

    /// Set the reconstruction adjacency.
    pub fn with_connectivity(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// Apply the double-threshold transform.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let out = self.double_threshold_flat(&vals, dims);
        Ok(rebuild(out, dims, image))
    }

    /// Coeus-native sister of [`DoubleThresholdImageFilter::apply`].
    ///
    /// Runs the identical band-indicator + binary morphological reconstruction
    /// via the shared `double_threshold_flat` host core (which delegates to
    /// `MorphologicalReconstruction::reconstruct_flat`)
    /// on the image's contiguous host buffer, so the result is bitwise-identical
    /// to the Burn path. No Burn tensor is constructed. Spatial metadata is
    /// preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        crate::native_support::map_flat_image(image, backend, |vals, dims| {
            self.double_threshold_flat(vals, dims)
        })
    }

    /// Substrate-agnostic host core: the inner-band marker `[t2, t3]` is
    /// binary-morphologically reconstructed under the outer-band mask `[t1, t4]`,
    /// then mapped to inside/outside values. Single source of truth for the Burn
    /// [`apply`](Self::apply) and Coeus-native [`apply_native`](Self::apply_native)
    /// paths.
    fn double_threshold_flat(&self, vals: &[f32], dims: [usize; 3]) -> Vec<f32> {
        // Inner band [t2, t3] → marker; outer band [t1, t4] → mask.
        let inner: Vec<f32> = vals
            .iter()
            .map(|&v| (v >= self.threshold2 && v <= self.threshold3) as u8 as f32)
            .collect();
        let outer: Vec<f32> = vals
            .iter()
            .map(|&v| (v >= self.threshold1 && v <= self.threshold4) as u8 as f32)
            .collect();
        let recon = MorphologicalReconstruction::new(ReconstructionMode::Dilation)
            .with_connectivity(self.connectivity)
            .reconstruct_flat(&inner, &outer, dims);

        recon
            .iter()
            .map(|&v| {
                if v > 0.5 {
                    self.inside_value
                } else {
                    self.outside_value
                }
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "tests_double_threshold.rs"]
mod tests_double_threshold;

#[cfg(test)]
mod tests_native {
    use super::DoubleThresholdImageFilter;
    use crate::native_support::{assert_native_matches_burn, make_native_image, native_vals};
    use coeus_core::SequentialBackend;

    #[test]
    fn matches_burn() {
        // Mixed intensities so inner/outer bands and connectivity all exercise.
        let vals: Vec<f32> = (0..60).map(|i| ((i * 7) % 20) as f32).collect();
        assert_native_matches_burn(
            vals,
            [3, 4, 5],
            |img| {
                DoubleThresholdImageFilter::new(2.0, 8.0, 12.0, 18.0, 1.0, 0.0)
                    .apply(img)
                    .expect("burn double threshold")
            },
            |img, backend| {
                DoubleThresholdImageFilter::new(2.0, 8.0, 12.0, 18.0, 1.0, 0.0)
                    .apply_native(img, backend)
            },
        );
    }

    #[test]
    fn oracle_all_inner_band_is_all_inside() {
        // Every voxel lies in the inner band [t2, t3] → the marker is full → the
        // reconstruction fills the whole (also-full) mask → all inside_value.
        let img = make_native_image(vec![10.0f32; 27], [3, 3, 3]);
        let out = DoubleThresholdImageFilter::new(0.0, 5.0, 15.0, 20.0, 1.0, 0.0)
            .apply_native(&img, &SequentialBackend)
            .expect("native double threshold");
        for &v in &native_vals(&out) {
            assert_eq!(v, 1.0, "inner-band voxel must map to inside_value");
        }
    }

    #[test]
    fn oracle_empty_marker_is_all_outside() {
        // No voxel lies in the inner band [t2, t3] → empty marker → empty
        // reconstruction → all outside_value, regardless of the outer band.
        let img = make_native_image(vec![10.0f32; 27], [3, 3, 3]);
        let out = DoubleThresholdImageFilter::new(0.0, 50.0, 60.0, 100.0, 1.0, 0.0)
            .apply_native(&img, &SequentialBackend)
            .expect("native double threshold");
        for &v in &native_vals(&out) {
            assert_eq!(v, 0.0, "no inner-band marker must map to outside_value");
        }
    }
}
