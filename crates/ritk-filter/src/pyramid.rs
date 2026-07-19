use super::downsample::DownsampleFilter;
use super::gaussian::GaussianFilter;
use crate::edge::GaussianSigma;
use anyhow::bail;
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_core::image::Image;
use ritk_image::tensor::Backend;

/// Multi-resolution image pyramid.
///
/// Generates a sequence of images at different resolutions/scales.
/// Typically used for coarse-to-fine registration strategies.
pub struct MultiResolutionPyramid<B: Backend, const D: usize> {
    images: Vec<Image<f32, B, D>>,
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
        input: &Image<f32, B, D>,
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
            // a `Vec<GaussianSigma>`, so we materialise the per-axis sigmas into
            // a small D-entry Vec here â€” one allocation per pyramid level, not a hot
            // path.
            let smoothed = if !is_identity_smooth {
                let sigmas_val: Vec<GaussianSigma> = sigmas
                    .iter()
                    .map(|&s| {
                        GaussianSigma::new(s).unwrap_or_else(|| GaussianSigma::new_unchecked(1e-9))
                    })
                    .collect();
                let smoother = GaussianFilter::new(sigmas_val);
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
    pub fn get_level(&self, level: usize) -> &Image<f32, B, D> {
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
        default_pyramid_schedule(levels)
    }
}

fn default_pyramid_schedule<const D: usize>(levels: usize) -> (Vec<[usize; D]>, Vec<[f64; D]>) {
    let mut shrink_factors = Vec::with_capacity(levels);
    let mut smoothing_sigmas = Vec::with_capacity(levels);

    for level in 0..levels {
        let factor = 2usize.pow((levels - 1 - level) as u32);
        shrink_factors.push([factor; D]);
        smoothing_sigmas.push(if factor > 1 {
            [0.5 * factor as f64; D]
        } else {
            [0.0; D]
        });
    }

    (shrink_factors, smoothing_sigmas)
}

/// Coeus-native 3-D multi-resolution image pyramid.
///
/// Each level is smoothed in physical space and then sampled at integer index
/// strides. Levels are ordered by the supplied schedule, conventionally from
/// coarsest to finest. The image origin and direction are preserved; spacing is
/// multiplied by the stride along each axis.
pub struct NativeMultiResolutionPyramid<B>
where
    B: ComputeBackend,
{
    images: Vec<ritk_image::Image<f32, B, 3>>,
}

impl<B> NativeMultiResolutionPyramid<B>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    /// Build a native pyramid from a 3-D image and per-level schedules.
    ///
    /// # Errors
    ///
    /// Returns an error when schedule lengths differ, a shrink factor is zero,
    /// a negative smoothing sigma is supplied, or a native image boundary fails.
    pub fn new(
        input: &ritk_image::Image<f32, B, 3>,
        shrink_factors: &[[usize; 3]],
        smoothing_sigmas: &[[f64; 3]],
        backend: &B,
    ) -> anyhow::Result<Self> {
        if shrink_factors.len() != smoothing_sigmas.len() {
            bail!(
                "pyramid schedule lengths differ: shrink={} smoothing={}",
                shrink_factors.len(),
                smoothing_sigmas.len()
            );
        }

        let mut images = Vec::with_capacity(shrink_factors.len());
        for (&factors, &sigmas) in shrink_factors.iter().zip(smoothing_sigmas) {
            if factors.contains(&0) {
                bail!("pyramid shrink factors must be positive, got {factors:?}");
            }

            if let Some(&sigma) = sigmas.iter().find(|&&sigma| sigma < 0.0) {
                bail!("pyramid smoothing sigma must be non-negative, got {sigma}");
            }
            let smoothed = if sigmas.iter().any(|&sigma| sigma > 0.0) {
                let sigmas = sigmas.map(|sigma| match GaussianSigma::new(sigma) {
                    Some(sigma) => sigma,
                    None => GaussianSigma::new_unchecked(1e-9),
                });
                GaussianFilter::<B>::new(sigmas.to_vec()).apply_native(input, backend)?
            } else {
                input.clone()
            };

            images.push(downsample_native(&smoothed, factors, backend)?);
        }

        Ok(Self { images })
    }

    /// Return the image at a scheduled level.
    #[must_use]
    pub fn get_level(&self, level: usize) -> &ritk_image::Image<f32, B, 3> {
        &self.images[level]
    }

    /// Return the number of scheduled levels.
    #[must_use]
    pub fn levels(&self) -> usize {
        self.images.len()
    }

    /// Create the default coarse-to-fine 3-D schedule.
    #[must_use]
    pub fn default_schedule(levels: usize) -> (Vec<[usize; 3]>, Vec<[f64; 3]>) {
        default_pyramid_schedule(levels)
    }
}

fn downsample_native<B>(
    image: &ritk_image::Image<f32, B, 3>,
    factors: [usize; 3],
    backend: &B,
) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
where
    B: ComputeBackend,
{
    let input_shape = image.shape();
    let output_shape = std::array::from_fn(|axis| input_shape[axis].div_ceil(factors[axis]));
    let values = image.try_data_vec_on(backend)?;
    let mut output = Vec::with_capacity(output_shape.iter().product());

    for z in 0..output_shape[0] {
        for y in 0..output_shape[1] {
            for x in 0..output_shape[2] {
                let input_index = ((z * factors[0]) * input_shape[1] + y * factors[1])
                    * input_shape[2]
                    + x * factors[2];
                output.push(values[input_index]);
            }
        }
    }

    let mut spacing = *image.spacing();
    for axis in 0..3 {
        spacing[axis] *= factors[axis] as f64;
    }
    ritk_image::Image::from_flat_on(
        output,
        output_shape,
        *image.origin(),
        spacing,
        *image.direction(),
        backend,
    )
}
