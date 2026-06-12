//! Unsigned Euclidean distance transform filter.

use super::super::types::BinarizationThreshold;
use super::core::euclidean_dt;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_core::image::Image;
use ritk_tensor_ops::extract_vec_infallible;

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
    pub threshold: BinarizationThreshold,
}

impl Default for DistanceTransformImageFilter {
    fn default() -> Self {
        Self {
            threshold: BinarizationThreshold::DEFAULT,
        }
    }
}

impl DistanceTransformImageFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threshold(mut self, t: impl Into<BinarizationThreshold>) -> Self {
        self.threshold = t.into();
        self
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
        let result = euclidean_dt(&fg, dims, spacing);

        let device = image.data().device();
        let td_out = TensorData::new(result, Shape::new([nz, ny, nx]));
        let tensor = Tensor::<B, 3>::from_data(td_out, &device);
        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}
