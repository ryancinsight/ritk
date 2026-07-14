//! Unsigned Euclidean distance transform filter.

use super::super::types::{BinarizationThreshold, DistanceMeasure};
use super::core::euclidean_dt_with_measure;
use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Unsigned Euclidean distance transform.
///
/// For each voxel, computes the physical distance (mm) to the nearest voxel
/// with intensity strictly greater than `threshold` (default 0.5, appropriate
/// for binary images).
///
/// Foreground voxels receive distance 0.
///
/// # Mathematical Specification
///
/// `out(x) = min_{y ∈ S} ||x − y||₂` where `S = { y : in(y) > threshold }`
///
/// # ITK Parity
///
/// `DanielssonDistanceMapImageFilter` with `UseImageSpacing = true`.
///
/// # Complexity
///
/// O(N) time via Meijster 2000; O(N) additional space.
#[derive(Debug, Clone)]
pub struct DistanceTransformImageFilter {
    /// Intensity threshold separating background (≤ threshold) from foreground (> threshold).
    threshold: BinarizationThreshold,
    measure: DistanceMeasure,
}

impl Default for DistanceTransformImageFilter {
    fn default() -> Self {
        Self {
            threshold: BinarizationThreshold::DEFAULT,
            measure: DistanceMeasure::Euclidean,
        }
    }
}

impl DistanceTransformImageFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threshold(mut self, threshold: BinarizationThreshold) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn with_measure(mut self, measure: DistanceMeasure) -> Self {
        self.measure = measure;
        self
    }

    pub fn threshold(&self) -> BinarizationThreshold {
        self.threshold
    }

    pub fn measure(&self) -> DistanceMeasure {
        self.measure
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        let [nz, ny, nx] = dims;
        let (vals, _shape) = extract_vec_infallible(image);
        let fg: Vec<bool> = vals
            .iter()
            .map(|&v| v > f32::from(self.threshold))
            .collect();
        let sp = image.spacing();
        let spacing = [sp[0], sp[1], sp[2]];
        validate_input(&vals, dims, spacing, self.threshold)?;
        let result = distance_values(&fg, dims, spacing, self.measure);

        Ok(rebuild(result, [nz, ny, nx], image))
    }

    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let values = image.data_slice()?;
        let dims = image.shape();
        let spacing = [image.spacing()[0], image.spacing()[1], image.spacing()[2]];
        validate_input(values, dims, spacing, self.threshold)?;
        let foreground: Vec<bool> = values
            .iter()
            .map(|&value| value > f32::from(self.threshold))
            .collect();
        crate::native_support::map_flat_image(image, backend, |_, _| {
            distance_values(&foreground, dims, spacing, self.measure)
        })
    }
}

fn distance_values(
    foreground: &[bool],
    dims: [usize; 3],
    spacing: [f64; 3],
    measure: DistanceMeasure,
) -> Vec<f32> {
    if !foreground.iter().any(|&value| value) {
        return vec![0.0; foreground.len()];
    }
    euclidean_dt_with_measure(foreground, dims, spacing, measure)
}

pub(super) fn validate_input(
    values: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
    threshold: BinarizationThreshold,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        dims.iter().all(|&dimension| dimension > 0),
        "distance-transform dimensions must be non-zero, got {dims:?}"
    );
    anyhow::ensure!(
        spacing
            .iter()
            .all(|value| value.is_finite() && *value > 0.0),
        "distance-transform spacing must be finite and positive, got {spacing:?}"
    );
    let threshold = f32::from(threshold);
    anyhow::ensure!(
        threshold.is_finite() && threshold >= 0.0,
        "distance-transform threshold must be finite and non-negative, got {threshold}"
    );
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        anyhow::bail!(
            "distance-transform sample at flat index {index} must be finite, got {value}"
        );
    }
    Ok(())
}
