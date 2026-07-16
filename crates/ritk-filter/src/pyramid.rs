use super::downsample::DownsampleFilter;
use super::gaussian::GaussianFilter;
use crate::edge::GaussianSigma;
use ritk_image::native::Image;

/// Multi-resolution image pyramid.
///
/// Generates a sequence of images at different resolution/scales.
/// Typically used for coarse-to-fine registration strategies.
pub struct MultiResolutionPyramid {
    images: Vec<Image<f32, coeus_core::SequentialBackend, 3>>,
}

impl MultiResolutionPyramid {
    /// Create a pyramid from an input image and schedules.
    ///
    /// # Arguments
    /// * `input` - The original high-resolution image.
    /// * `shrink_factors` - Shrink factors for each level `[level][dim]`, as
    ///   stack-allocated `[usize; 3]` arrays (one allocation per level, no
    ///   inner `Vec<usize>` indirection).
    /// * `smoothing_sigmas` - Smoothing sigmas for each level `[level][dim]`,
    ///   as stack-allocated `[f64; 3]` arrays.
    ///
    /// # Panics
    /// Panics if schedules have different lengths.
    ///
    /// P1-01 (Sprint 350): API consumes `[T; 3]` arrays directly, removing
    /// the `Vec<Vec<T>>` outer + inner heap allocations that the legacy shape
    /// imposed on every pyramid build.
    pub fn new(
        input: &Image<f32, coeus_core::SequentialBackend, 3>,
        shrink_factors: &[[usize; 3]],
        smoothing_sigmas: &[[f64; 3]],
    ) -> anyhow::Result<Self> {
        assert_eq!(
            shrink_factors.len(),
            smoothing_sigmas.len(),
            "Schedule lengths must match"
        );

        let backend = coeus_core::SequentialBackend;
        let mut images = Vec::with_capacity(shrink_factors.len());

        for (factors, sigmas) in shrink_factors.iter().zip(smoothing_sigmas.iter()) {
            // Optimization: if identity transform, just clone
            let is_identity_shrink = factors.iter().all(|&f| f == 1);
            let is_identity_smooth = sigmas.iter().all(|&s| s <= 1e-6);

            if is_identity_shrink && is_identity_smooth {
                images.push(input.clone());
                continue;
            }

            // 1. Smooth
            // Only smooth if sigmas are significant. GaussianFilter::new takes
            // a `Vec<GaussianSigma>`, so we materialise the per-axis sigmas into
            // a small D-entry Vec here — one allocation per pyramid level, not a hot
            // path.
            let smoothed = if !is_identity_smooth {
                let sigmas_val: Vec<GaussianSigma> = sigmas
                    .iter()
                    .map(|&s| {
                        GaussianSigma::new(s).unwrap_or_else(|| GaussianSigma::new_unchecked(1e-9))
                    })
                    .collect();
                let smoother = GaussianFilter::new(sigmas_val);
                smoother.apply(input, &backend)?
            } else {
                input.clone()
            };

            // 2. Downsample
            let result = if !is_identity_shrink {
                let downsampler = DownsampleFilter::new(factors.to_vec());
                downsampler.apply(&smoothed, &backend)?
            } else {
                smoothed
            };

            images.push(result);
        }

        Ok(Self { images })
    }

    /// Get image at specific level.
    pub fn get_level(&self, level: usize) -> &Image<f32, coeus_core::SequentialBackend, 3> {
        &self.images[level]
    }

    /// Get number of levels.
    pub fn levels(&self) -> usize {
        self.images.len()
    }

    /// Create a default schedule for N levels with power-of-2 shrinking.
    ///
    /// Returns `(shrink_factors, smoothing_sigmas)` as `Vec<[usize; 3]>` and
    /// `Vec<[f64; 3]>` respectively (stack arrays per level, no inner Vec).
    /// Levels are ordered from coarsest to finest.
    /// E.g. `levels=3` -> factors `[4, 2, 1]`, sigmas `[2.0, 1.0, 0.0]`
    pub fn default_schedule(levels: usize) -> (Vec<[usize; 3]>, Vec<[f64; 3]>) {
        let mut shrink_factors = Vec::with_capacity(levels);
        let mut smoothing_sigmas = Vec::with_capacity(levels);

        for i in 0..levels {
            let exponent = (levels - 1 - i) as u32;
            let factor = 2usize.pow(exponent);
            let sigma = if factor > 1 {
                0.5 * (factor as f64)
            } else {
                0.0
            };

            shrink_factors.push([factor; 3]);
            smoothing_sigmas.push([sigma; 3]);
        }

        (shrink_factors, smoothing_sigmas)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_pyramid.rs"]
mod tests_pyramid;
