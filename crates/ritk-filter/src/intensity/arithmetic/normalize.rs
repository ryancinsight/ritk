use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use ritk_tensor_ops::{extract_vec_infallible as extract_vec, rebuild};

/// Zero-mean, unit-variance intensity normalization filter.
///
/// # Mathematical Specification
///
/// Let `N = n_z · n_y · n_x` be the total voxel count.
/// Define:
///
///   `μ  = Σ_{x} in(x) / N`
///   `σ² = Σ_{x} (in(x) − μ)² / N`
///   `σ  = √σ²`
///
/// Then:
///
///   `out(x) = (in(x) − μ) / σ`      if σ > 0
///   `out(x) = 0`                      if σ = 0  (constant image)
///
/// # Properties
/// - `Σ out(x) / N = 0` (zero mean, exactly by construction).
/// - `Σ (out(x))² / N = 1` (unit variance, exactly by construction).
/// - Constant image → all-zero output (undefined normalisation → zero by convention).
///
/// # References
/// - ITK `itk::NormalizeImageFilter<TInputImage, TOutputImage>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct NormalizeImageFilter;

impl NormalizeImageFilter {
    /// Construct a new `NormalizeImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply zero-mean unit-variance normalization to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let n = vals.len() as f64;
        // PRECISION: f64 accumulation required — f32 sum of n>10^7 elements loses
        // precision; see numerical_discipline in AGENTS.md.
        let mean = vals.iter().map(|&v| v as f64).sum::<f64>() / n;
        let variance = vals
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        let std = variance.sqrt() as f32;
        let mean_f = mean as f32;
        let out: Vec<f32> = if std < f32::EPSILON {
            vec![0.0_f32; vals.len()]
        } else {
            vals.into_iter().map(|v| (v - mean_f) / std).collect()
        };
        rebuild(out, dims, image)
    }
}

#[cfg(test)]
#[path = "tests_normalize.rs"]
mod tests;
