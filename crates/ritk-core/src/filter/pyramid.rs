use super::downsample::DownsampleFilter;
use super::gaussian::GaussianFilter;
use crate::image::Image;
use burn::tensor::backend::Backend;

/// Multi-resolution image pyramid.
///
/// Generates a sequence of images at different resolutions/scales.
/// Typically used for coarse-to-fine registration strategies.
pub struct MultiResolutionPyramid<B: Backend, const D: usize> {
    images: Vec<Image<B, D>>,
}

impl<B: Backend, const D: usize> MultiResolutionPyramid<B, D> {
    /// Create a pyramid from an input image and schedules.
    ///
    /// # Arguments
    /// * `input` - The original high-resolution image.
    /// * `shrink_factors` - Shrink factors for each level `[level][dim]`, as
    ///   stack-allocated `[usize; D]` arrays (one allocation per level, no
    ///   inner `Vec<usize>` indirection).
    /// * `smoothing_sigmas` - Smoothing sigmas for each level `[level][dim]`,
    ///   as stack-allocated `[f64; D]` arrays.
    ///
    /// # Panics
    /// Panics if schedules have different lengths.
    ///
    /// P1-01 (Sprint 350): API consumes `[T; D]` arrays directly, removing
    /// the `Vec<Vec<T>>` outer + inner heap allocations that the legacy shape
    /// imposed on every pyramid build.
    pub fn new(
        input: &Image<B, D>,
        shrink_factors: &[[usize; D]],
        smoothing_sigmas: &[[f64; D]],
    ) -> Self {
        assert_eq!(
            shrink_factors.len(),
            smoothing_sigmas.len(),
            "Schedule lengths must match"
        );

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
            // a `Vec<f64>`, so we materialise the per-axis sigmas into a small
            // D-entry Vec here — one allocation per pyramid level, not a hot
            // path. Future Sprint could plumb `[f64; D]` through the filter API.
            let smoothed = if !is_identity_smooth {
                let smoother = GaussianFilter::new(sigmas.to_vec());
                smoother.apply(input)
            } else {
                input.clone()
            };

            // 2. Downsample
            let result = if !is_identity_shrink {
                let downsampler = DownsampleFilter::new(factors.to_vec());
                downsampler.apply(&smoothed)
            } else {
                smoothed
            };

            images.push(result);
        }

        Self { images }
    }

    /// Get image at specific level.
    pub fn get_level(&self, level: usize) -> &Image<B, D> {
        &self.images[level]
    }

    /// Get number of levels.
    pub fn levels(&self) -> usize {
        self.images.len()
    }

    /// Create a default schedule for N levels with power-of-2 shrinking.
    ///
    /// Returns `(shrink_factors, smoothing_sigmas)` as `Vec<[usize; D]>` and
    /// `Vec<[f64; D]>` respectively (stack arrays per level, no inner Vec).
    /// Levels are ordered from coarsest to finest.
    /// E.g. `levels=3` -> factors `[4, 2, 1]`, sigmas `[2.0, 1.0, 0.0]`
    pub fn default_schedule(levels: usize) -> (Vec<[usize; D]>, Vec<[f64; D]>) {
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

            shrink_factors.push([factor; D]);
            smoothing_sigmas.push([sigma; D]);
        }

        (shrink_factors, smoothing_sigmas)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(shape: [usize; 3]) -> Image<B, 3> {
        let n = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let device = Default::default();
        let t = Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(shape)), &device);
        Image::new(
            t,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    // ── MultiResolutionPyramid ────────────────────────────────────────────────────

    /// Level count matches the schedule length.
    #[test]
    fn pyramid_level_count_matches_schedule() {
        let img = make_image([8, 8, 8]);
        let shrink: Vec<[usize; 3]> = vec![[4, 4, 4], [2, 2, 2], [1, 1, 1]];
        let sigmas: Vec<[f64; 3]> = vec![[2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]];
        let pyr = MultiResolutionPyramid::<B, 3>::new(&img, &shrink, &sigmas);
        assert_eq!(pyr.levels(), 3, "pyramid must have 3 levels");
    }

    /// Identity schedule (factor=1, sigma=0): every level is a clone of the input.
    #[test]
    fn pyramid_identity_schedule_clones_image() {
        let img = make_image([6, 6, 6]);
        let shrink: Vec<[usize; 3]> = vec![[1, 1, 1]];
        let sigmas: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]];
        let pyr = MultiResolutionPyramid::<B, 3>::new(&img, &shrink, &sigmas);
        assert_eq!(pyr.levels(), 1);
        assert_eq!(
            pyr.get_level(0).shape(),
            img.shape(),
            "identity schedule must preserve shape"
        );
    }

    /// Coarser levels produce shapes smaller than or equal to finer levels.
    ///
    /// # Derivation
    /// Schedule [4,4,4] -> [2,2,2] -> [1,1,1]: shapes are \u2248 [2,2,2] < \u2248 [4,4,4] < [8,8,8].
    #[test]
    fn pyramid_coarser_levels_have_smaller_shape() {
        let img = make_image([8, 8, 8]);
        let (shrink, sigmas) = MultiResolutionPyramid::<B, 3>::default_schedule(3);
        let pyr = MultiResolutionPyramid::<B, 3>::new(&img, &shrink, &sigmas);
        // Levels ordered coarsest-to-finest (default_schedule convention).
        let n0: usize = pyr.get_level(0).shape().iter().product();
        let n1: usize = pyr.get_level(1).shape().iter().product();
        let n2: usize = pyr.get_level(2).shape().iter().product();
        assert!(
            n0 <= n1,
            "level 0 (coarsest) must have <= voxels than level 1: {n0} vs {n1}"
        );
        assert!(
            n1 <= n2,
            "level 1 must have <= voxels than level 2 (finest): {n1} vs {n2}"
        );
    }

    // ── default_schedule ──────────────────────────────────────────────────────

    /// default_schedule(3) produces 3 levels with factors [4,2,1] and sigmas [2.0,1.0,0.0].
    ///
    /// # Derivation
    /// For `levels=3`, `i=0..2`, `exponent = 2-i = [2,1,0]`.
    /// factor = 2^exponent = [4, 2, 1].
    /// sigma = 0.5 * factor if factor>1 else 0.0 = [2.0, 1.0, 0.0].
    #[test]
    fn default_schedule_3_levels_correct_factors_and_sigmas() {
        let (shrink, sigmas) = MultiResolutionPyramid::<B, 3>::default_schedule(3);
        assert_eq!(shrink.len(), 3);
        assert_eq!(shrink[0], [4, 4, 4], "level 0 factor must be 4");
        assert_eq!(shrink[1], [2, 2, 2], "level 1 factor must be 2");
        assert_eq!(shrink[2], [1, 1, 1], "level 2 factor must be 1");
        assert!(
            (sigmas[0][0] - 2.0).abs() < 1e-9,
            "level 0 sigma must be 2.0"
        );
        assert!(
            (sigmas[1][0] - 1.0).abs() < 1e-9,
            "level 1 sigma must be 1.0"
        );
        assert!(
            (sigmas[2][0] - 0.0).abs() < 1e-9,
            "level 2 sigma must be 0.0"
        );
    }

    /// default_schedule(1) produces one level with factor=1 and sigma=0.0.
    #[test]
    fn default_schedule_single_level_is_identity() {
        let (shrink, sigmas) = MultiResolutionPyramid::<B, 3>::default_schedule(1);
        assert_eq!(shrink[0], [1, 1, 1]);
        assert!((sigmas[0][0] - 0.0).abs() < 1e-9);
    }
}
