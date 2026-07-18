use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec_infallible as extract_vec, rebuild};

use crate::native_support::map_flat_image;

/// Zero-mean, unit-variance intensity normalization filter.
///
/// # Mathematical Specification
///
/// Let `N = n_z Â· n_y Â· n_x` be the total voxel count.
/// Define (matching ITK, whose `NormalizeImageFilter` divides by the
/// `StatisticsImageFilter` *sample* sigma â€” Bessel-corrected, `Ã· (Nâˆ’1)`):
///
///   `Î¼  = Î£_{x} in(x) / N`
///   `ÏƒÂ² = Î£_{x} (in(x) âˆ’ Î¼)Â² / (N âˆ’ 1)`
///   `Ïƒ  = âˆšÏƒÂ²`
///
/// Then:
///
///   `out(x) = (in(x) âˆ’ Î¼) / Ïƒ`      if N > 1 and Ïƒ > 0
///   `out(x) = 0`                      otherwise (N â‰¤ 1, or constant image)
///
/// # Properties
/// - `Î£ out(x) / N = 0` (zero mean, exactly by construction).
/// - `Î£ (out(x))Â² / (N âˆ’ 1) = 1` (unit *sample* variance, by construction); the
///   output population variance is `(N âˆ’ 1) / N`.
/// - Constant image â†’ all-zero output (undefined normalisation â†’ zero by convention).
///
/// # References
/// - ITK `itk::NormalizeImageFilter<TInputImage, TOutputImage>` (float-exact).
#[derive(Debug, Clone, Copy, Default)]
pub struct NormalizeImageFilter;

impl NormalizeImageFilter {
    /// Construct a new `NormalizeImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply zero-mean unit-variance normalization to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> Image<f32, B, 3> {
        let (vals, dims) = extract_vec(image);
        let n = vals.len() as f64;
        // PRECISION: f64 accumulation required â€” f32 sum of n>10^7 elements loses
        // precision; see numerical_discipline in AGENTS.md.
        let mean = vals.iter().map(|&v| v as f64).sum::<f64>() / n;
        // Sample (Bessel-corrected) variance to match ITK NormalizeImageFilter.
        // n â‰¤ 1 has no defined sample variance â†’ fall through to the zero output.
        let variance = if n > 1.0 {
            vals.iter()
                .map(|&v| {
                    let d = v as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / (n - 1.0)
        } else {
            0.0
        };
        let std = variance.sqrt() as f32;
        let mean_f = mean as f32;
        let out: Vec<f32> = if std < f32::EPSILON {
            vec![0.0_f32; vals.len()]
        } else {
            vals.into_iter().map(|v| (v - mean_f) / std).collect()
        };
        rebuild(out, dims, image)
    }

    /// Apply sample-standard-deviation normalization to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        map_flat_image(image, backend, |values, _| {
            let count = values.len() as f64;
            let mean = values.iter().map(|&value| f64::from(value)).sum::<f64>() / count;
            let variance = if count > 1.0 {
                values
                    .iter()
                    .map(|&value| {
                        let delta = f64::from(value) - mean;
                        delta * delta
                    })
                    .sum::<f64>()
                    / (count - 1.0)
            } else {
                0.0
            };
            let standard_deviation = variance.sqrt() as f32;
            if standard_deviation < f32::EPSILON {
                vec![0.0; values.len()]
            } else {
                let mean = mean as f32;
                values
                    .iter()
                    .map(|&value| (value - mean) / standard_deviation)
                    .collect()
            }
        })
    }
}

/// Scale an image so that the sum of all voxels equals a target `constant`.
///
/// `out(x) = in(x) Â· constant / Î£ in`. Matches ITK `NormalizeToConstantImageFilter`
/// (`sitk.NormalizeToConstant`). A zero-sum image yields all zeros.
#[derive(Debug, Clone, Copy)]
pub struct NormalizeToConstantImageFilter {
    /// Target sum of the output image.
    pub constant: f64,
}

impl NormalizeToConstantImageFilter {
    /// Construct with the given target sum.
    pub fn new(constant: f64) -> Self {
        Self { constant }
    }

    /// Apply the normalisation. f64 accumulation for the sum (precision).
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> Image<f32, B, 3> {
        let (vals, dims) = extract_vec(image);
        let sum = vals.iter().map(|&v| v as f64).sum::<f64>();
        let factor = if sum != 0.0 {
            (self.constant / sum) as f32
        } else {
            0.0
        };
        let out: Vec<f32> = vals.into_iter().map(|v| v * factor).collect();
        rebuild(out, dims, image)
    }

    /// Apply constant-sum normalization to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        map_flat_image(image, backend, |values, _| {
            let sum = values.iter().map(|&value| f64::from(value)).sum::<f64>();
            let factor = if sum != 0.0 {
                (self.constant / sum) as f32
            } else {
                0.0
            };
            values.iter().map(|&value| value * factor).collect()
        })
    }
}

#[cfg(test)]
#[path = "tests_normalize.rs"]
mod tests;
