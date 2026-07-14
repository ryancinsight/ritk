//! Linear intensity rescaling filter.
//!
//! # Mathematical Specification
//!
//! Let I_min = min_{x} I(x), I_max = max_{x} I(x).
//! If I_min == I_max: output(x) = out_min for all x.
//! Else: output(x) = (I(x) - I_min) / (I_max - I_min) x (out_max - out_min) + out_min
//!
//! This is the unique affine bijection mapping [I_min, I_max] to [out_min, out_max].

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

use crate::native_support::map_flat_image;

/// Linear rescale of image intensity to [out_min, out_max].
///
/// Computes the global minimum and maximum of the input image and maps the
/// intensity range [I_min, I_max] linearly to [out_min, out_max].
#[derive(Debug, Clone)]
pub struct RescaleIntensityFilter {
    /// Minimum output intensity value.
    pub out_min: f32,
    /// Maximum output intensity value.
    pub out_max: f32,
}

impl RescaleIntensityFilter {
    /// Construct with explicit output range.
    pub fn new(out_min: f32, out_max: f32) -> Self {
        Self { out_min, out_max }
    }

    /// Construct with unit output range [0.0, 1.0].
    pub fn unit() -> Self {
        Self {
            out_min: 0.0,
            out_max: 1.0,
        }
    }

    /// Apply the rescaling to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let out = rescale_vec(&vals, self.out_min, self.out_max);
        Ok(rebuild(out, dims, image))
    }

    /// Coeus-native sister of [`RescaleIntensityFilter::apply`].
    ///
    /// Runs the identical global-min/max affine remap via the shared
    /// [`rescale_vec`] host core on the image's contiguous host buffer, so the
    /// result is bitwise-identical to the Burn path. No Burn tensor is
    /// constructed. Spatial metadata (origin, spacing, direction) is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        crate::native_support::map_flat_image(image, backend, |vals, _dims| {
            rescale_vec(vals, self.out_min, self.out_max)
        })
    }
}

/// Substrate-agnostic host core for [`RescaleIntensityFilter`].
///
/// Computes the global `[I_min, I_max]` extrema in one fused parallel pass,
/// then affinely remaps every voxel to `[out_min, out_max]`. When the input is
/// constant (`I_max == I_min` within `f32::EPSILON`), every output voxel is
/// `out_min`, matching the mathematical specification.
pub(crate) fn rescale_vec(vals: &[f32], out_min: f32, out_max: f32) -> Vec<f32> {
    let n = vals.len();

    // Fused parallel min/max reduction (one pass over the data instead of
    // two sequential folds). NaN compares `false` in `min`/`max`, leaving
    // the running extremum unchanged — matching the prior `f32::min`/`max`.
    let (i_min, i_max) = moirai::fold_reduce_with::<moirai::Adaptive, _, _, _, _>(
        n,
        || (f32::INFINITY, f32::NEG_INFINITY),
        |(mn, mx), i| {
            let v = vals[i];
            (mn.min(v), mx.max(v))
        },
        |(a_mn, a_mx), (b_mn, b_mx)| (a_mn.min(b_mn), a_mx.max(b_mx)),
    );

    if (i_max - i_min).abs() < f32::EPSILON {
        vec![out_min; n]
    } else {
        // Affine remap, parallelized element-wise (independent per voxel).
        let scale = (out_max - out_min) / (i_max - i_min);
        moirai::map_collect_index_with::<moirai::Adaptive, _, _>(n, |i| {
            (vals[i] - i_min) * scale + out_min
        })
    }
}

#[cfg(test)]
#[path = "tests_rescale.rs"]
mod tests;
