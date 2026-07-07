//! Signed Euclidean distance transform filter.

use super::super::types::BinarizationThreshold;
use super::core::euclidean_dt;
use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_tensor_ops::extract_vec_infallible;

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
    pub threshold: BinarizationThreshold,
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
        let bg: Vec<bool> = fg.iter().map(|&b| !b).collect();
        let sp = image.spacing();
        let spacing = [sp[0], sp[1], sp[2]];

        // EDT from each voxel to nearest foreground (background voxels get positive value)
        let edt_to_fg = euclidean_dt(&fg, dims, spacing);
        // EDT from each voxel to nearest background (foreground voxels get positive value)
        let edt_to_bg = euclidean_dt(&bg, dims, spacing);

        // Signed: outside (+) = edt_to_fg, inside (−) = −edt_to_bg
        let result: Vec<f32> = fg
            .iter()
            .zip(edt_to_fg.iter())
            .zip(edt_to_bg.iter())
            .map(|((&is_fg, &d_fg), &d_bg)| if is_fg { -d_bg } else { d_fg })
            .collect();

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
