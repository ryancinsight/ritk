//! Signed Euclidean distance transform filter.

use super::super::types::BinarizationThreshold;
use super::core::euclidean_dt;
use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Signed Euclidean distance transform (voxel-centre convention).
///
/// Convention:
/// - Background voxels: `+dist` (positive = outside the object)
/// - Foreground voxels: `−dist` (negative = inside the object, distance to nearest background)
///
/// # Mathematical Specification
///
/// `SEDT(x) = EDT_bg(x)` if `in(x) ≤ threshold` (outside object)
/// `SEDT(x) = −EDT_fg(x)` if `in(x) > threshold` (inside object)
///
/// where `EDT_bg` = distance to nearest background voxel **centre**, `EDT_fg` =
/// distance to nearest foreground voxel **centre**.
///
/// # Parity
///
/// Float-exact to `scipy.ndimage.distance_transform_edt` applied to each region
/// (signed: `−edt(mask)` inside, `+edt(¬mask)` outside), with `UseImageSpacing`.
/// This is **distance to the nearest opposite-class voxel centre** — it does NOT
/// match ITK `SignedMaurerDistanceMapImageFilter`, which measures distance to the
/// object *boundary/interface* (the two differ by up to √2 voxel; an earlier doc
/// claimed Maurer parity in error).
#[derive(Debug, Clone)]
pub struct SignedDistanceTransformImageFilter {
    /// Intensity threshold separating background from foreground.
    threshold: BinarizationThreshold,
}

impl Default for SignedDistanceTransformImageFilter {
    fn default() -> Self {
        Self {
            threshold: BinarizationThreshold::DEFAULT,
        }
    }
}

impl SignedDistanceTransformImageFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threshold(mut self, threshold: BinarizationThreshold) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn threshold(&self) -> BinarizationThreshold {
        self.threshold
    }

    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let dims = image.shape();
        let [nz, ny, nx] = dims;
        let (vals, _shape) = extract_vec_infallible(image);
        let fg: Vec<bool> = vals
            .iter()
            .map(|&v| v > f32::from(self.threshold))
            .collect();
        let bg: Vec<bool> = fg.iter().map(|&b| !b).collect();
        let sp = image.spacing();
        let spacing = [sp[0], sp[1], sp[2]];
        super::unsigned::validate_input(&vals, dims, spacing, self.threshold)?;
        let result = signed_values(&fg, &bg, dims, spacing);

        Ok(rebuild(result, [nz, ny, nx], image))
    }

    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let values = image.data_slice()?;
        let dims = image.shape();
        let spacing = [image.spacing()[0], image.spacing()[1], image.spacing()[2]];
        super::unsigned::validate_input(values, dims, spacing, self.threshold)?;
        let foreground: Vec<bool> = values
            .iter()
            .map(|&value| value > f32::from(self.threshold))
            .collect();
        let background: Vec<bool> = foreground.iter().map(|&value| !value).collect();
        crate::native_support::map_flat_image(image, backend, |_, _| {
            signed_values(&foreground, &background, dims, spacing)
        })
    }
}

fn signed_values(
    foreground: &[bool],
    background: &[bool],
    dims: [usize; 3],
    spacing: [f64; 3],
) -> Vec<f32> {
    let to_foreground = if foreground.iter().any(|&value| value) {
        euclidean_dt(foreground, dims, spacing)
    } else {
        vec![0.0; foreground.len()]
    };
    let to_background = if background.iter().any(|&value| value) {
        euclidean_dt(background, dims, spacing)
    } else {
        vec![0.0; background.len()]
    };
    foreground
        .iter()
        .zip(to_foreground)
        .zip(to_background)
        .map(|((&is_foreground, to_foreground), to_background)| {
            if is_foreground {
                -to_background
            } else {
                to_foreground
            }
        })
        .collect()
}
