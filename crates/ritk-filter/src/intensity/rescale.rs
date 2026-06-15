//! Linear intensity rescaling filter.
//!
//! # Mathematical Specification
//!
//! Let I_min = min_{x} I(x), I_max = max_{x} I(x).
//! If I_min == I_max: output(x) = out_min for all x.
//! Else: output(x) = (I(x) - I_min) / (I_max - I_min) x (out_max - out_min) + out_min
//!
//! This is the unique affine bijection mapping [I_min, I_max] to [out_min, out_max].

use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

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
        let n = vals.len();

        // Fused parallel min/max reduction (one pass over the data instead of
        // two sequential folds). NaN compares `false` in `min`/`max`, leaving
        // the running extremum unchanged â€” matching the prior `f32::min`/`max`.
        let (i_min, i_max) = moirai::fold_reduce_with::<moirai::Adaptive, _, _, _, _>(
            n,
            || (f32::INFINITY, f32::NEG_INFINITY),
            |(mn, mx), i| {
                let v = vals[i];
                (mn.min(v), mx.max(v))
            },
            |(a_mn, a_mx), (b_mn, b_mx)| (a_mn.min(b_mn), a_mx.max(b_mx)),
        );

        let out: Vec<f32> = if (i_max - i_min).abs() < f32::EPSILON {
            vec![self.out_min; n]
        } else {
            // Affine remap, parallelized element-wise (independent per voxel).
            let scale = (self.out_max - self.out_min) / (i_max - i_min);
            let out_min = self.out_min;
            moirai::map_collect_index_with::<moirai::Adaptive, _, _>(n, |i| {
                (vals[i] - i_min) * scale + out_min
            })
        };

        Ok(rebuild(out, dims, image))
    }
}

#[cfg(test)]
#[path = "tests_rescale.rs"]
mod tests;
