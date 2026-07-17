//! Image-binding filter struct (see [`ChamferDistanceTransform`]).

use super::super::types::BinarizationThreshold;
use super::kernel::{cdt_dispatch, ChamferMetric, INF};
use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Compute the **chamfer distance transform** of a 3-D binary image.
///
/// Implements `scipy.ndimage.distance_transform_cdt`: the **interior
/// distance transform**, where:
///
/// - **Background voxels** (intensity `≤ threshold`) receive `0.0`.
/// - **Foreground voxels** (intensity `> threshold`) receive the chamfer
///   distance (in `mm`, scaled by `s_min`) to the **nearest background
///   voxel**.
/// - **Foreground voxels with no background anywhere in the volume** (i.e.
///   the input is all-foreground) receive `-1.0` (scipy's sentinel for
///   "undefined distance").
///
/// # Mathematical Specification
///
/// With metric `Chessboard` and uniform spacing `s`:
///   `out(x) = round(s · max(|x_a − y_a|))` for nearest background `y`
///
/// With metric `Taxicab` and uniform spacing `s`:
///   `out(x) = round(s · Σ |x_a − y_a|)` for nearest background `y`
///
/// For non-uniform spacing, weights are `w_a = s_a / s_min` (rounded to
/// integer). The output is therefore an `f32` image in **physical units
/// of `s_min`**, not raw voxel counts.
///
/// # Complexity
///
/// O(N) time, O(N) memory (the output is reused as the workspace — no
/// extra allocation).
///
/// # scipy parity
///
/// Matches `scipy.ndimage.distance_transform_cdt` for `metric='chessboard'`
/// and `metric='taxicab'` on the **interior** distance. Note that
/// scipy's `distance_transform_cdt` does **not** support the `sampling`
/// parameter; the per-axis spacing is an extension of this filter.
#[derive(Debug, Clone)]
pub struct ChamferDistanceTransform {
    /// Intensity threshold separating background (≤ threshold) from foreground (> threshold).
    pub threshold: BinarizationThreshold,
    /// Distance metric.
    pub metric: ChamferMetric,
}

impl Default for ChamferDistanceTransform {
    fn default() -> Self {
        Self {
            threshold: BinarizationThreshold::DEFAULT,
            metric: ChamferMetric::default(),
        }
    }
}

impl ChamferDistanceTransform {
    /// Create a chamfer distance transform with default threshold (0.5) and
    /// the chessboard metric.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the intensity threshold.
    #[inline]
    pub fn with_threshold(mut self, threshold: BinarizationThreshold) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the distance metric.
    #[inline]
    pub fn with_metric(mut self, m: ChamferMetric) -> Self {
        self.metric = m;
        self
    }

    /// Apply the chamfer distance transform to a 3-D image.
    ///
    /// Returns an `Image<B, 3>` with the same shape and physical metadata
    /// as the input, with `f32` storage (scaled by `s_min`).
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

        let s_min = spacing.iter().fold(f64::INFINITY, |a, &B::default()| a.min(b));
        let weights: [i32; 3] = [
            (spacing[0] / s_min).round() as i32,
            (spacing[1] / s_min).round() as i32,
            (spacing[2] / s_min).round() as i32,
        ];

        let result = cdt_dispatch(&fg, dims, weights, self.metric);

        let scale = s_min as f32;
        let scaled: Vec<f32> = result
            .iter()
            .map(|&v| if v == INF { -1.0 } else { v as f32 * scale })
            .collect();

        Ok(rebuild(scaled, [nz, ny, nx], image))
    }    /// Coeus-native sister of [`apply`].
    pub fn apply_native<B>(&self, image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B::default()) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let dims = image.shape();
        let [nz, ny, nx] = dims;
        let (vals, _shape) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let fg: Vec<bool> = vals
            .iter()
            .map(|&v| v > f32::from(self.threshold))
            .collect();
        let sp = image.spacing();
        let spacing = [sp[0], sp[1], sp[2]];

        let s_min = spacing.iter().fold(f64::INFINITY, |a, &B::default()| a.min(b));
        let weights: [i32; 3] = [
            (spacing[0] / s_min).round() as i32,
            (spacing[1] / s_min).round() as i32,
            (spacing[2] / s_min).round() as i32,
        ];

        let result = cdt_dispatch(&fg, dims, weights, self.metric);

        let scale = s_min as f32;
        let scaled: Vec<f32> = result
            .iter()
            .map(|&v| if v == INF { -1.0 } else { v as f32 * scale })
            .collect();

        crate::native_support::rebuild_image(scaled, [nz, ny, nx], image, backend)
    
    }

}
