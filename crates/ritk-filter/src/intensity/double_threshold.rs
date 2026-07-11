//! Double-threshold (hysteresis) filter.
//!
//! Matches ITK `DoubleThresholdImageFilter` / `sitk.DoubleThreshold`: a voxel is
//! foreground if it lies in the **inner** band `[t2, t3]`, or in the **outer**
//! band `[t1, t4]` and is connected (through the outer band) to an inner-band
//! voxel. This is exactly a binary morphological reconstruction of the inner
//! band (marker) under the outer band (mask).

use crate::morphology::{Connectivity, MorphologicalReconstruction, ReconstructionMode};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::tensor::Backend;
use ritk_image::{native::Image as NativeImage, Image};
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
        let (inner, outer) = self.marker_and_mask(&vals);
        let marker = rebuild(inner, dims, image);
        let mask = rebuild(outer, dims, image);
        let recon = MorphologicalReconstruction::new(ReconstructionMode::Dilation)
            .with_connectivity(self.connectivity)
            .apply(&marker, &mask)?;

        let (rvals, _) = extract_vec(&recon)?;
        let out: Vec<f32> = rvals
            .iter()
            .map(|&v| {
                if v > 0.5 {
                    self.inside_value
                } else {
                    self.outside_value
                }
            })
            .collect();
        Ok(rebuild(out, dims, image))
    }

    /// Apply double-threshold hysteresis to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &NativeImage<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<NativeImage<f32, B, 3>>
    where
        B: ComputeBackend,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let (inner, outer) = self.marker_and_mask(image.data_slice()?);
        let marker = NativeImage::from_flat_on(
            inner,
            image.shape(),
            *image.origin(),
            *image.spacing(),
            *image.direction(),
            backend,
        )?;
        let mask = NativeImage::from_flat_on(
            outer,
            image.shape(),
            *image.origin(),
            *image.spacing(),
            *image.direction(),
            backend,
        )?;
        let reconstruction = MorphologicalReconstruction::new(ReconstructionMode::Dilation)
            .with_connectivity(self.connectivity)
            .apply_native(&marker, &mask, backend)?;
        let values = reconstruction
            .data_slice()?
            .iter()
            .map(|&value| {
                if value > 0.5 {
                    self.inside_value
                } else {
                    self.outside_value
                }
            })
            .collect();
        NativeImage::from_flat_on(
            values,
            image.shape(),
            *image.origin(),
            *image.spacing(),
            *image.direction(),
            backend,
        )
    }

    fn marker_and_mask(&self, values: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let marker = values
            .iter()
            .map(|&value| (value >= self.threshold2 && value <= self.threshold3) as u8 as f32)
            .collect();
        let mask = values
            .iter()
            .map(|&value| (value >= self.threshold1 && value <= self.threshold4) as u8 as f32)
            .collect();
        (marker, mask)
    }
}

#[cfg(test)]
#[path = "tests_double_threshold.rs"]
mod tests_double_threshold;
