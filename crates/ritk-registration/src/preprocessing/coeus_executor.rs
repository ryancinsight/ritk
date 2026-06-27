//! Coeus-backed preprocessing executor for scalar image steps.

use anyhow::{Context, Result};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::coeus::Image;

use super::pipeline::PreprocessingPipeline;
use super::step::PreprocessingStep;
use super::value_ops::{apply_mask_values, clamp_values, normalize_values, validate_mask};

impl PreprocessingPipeline {
    /// Execute scalar preprocessing steps on a Coeus-backed image.
    ///
    /// The method supports steps whose semantics are pure pointwise or
    /// whole-buffer scalar transforms. Filter-backed steps still depend on the
    /// legacy Burn filter implementations and return an explicit error here
    /// rather than silently downgrading the Coeus path.
    ///
    /// # Errors
    /// Returns an error when tensor extraction/rebuild validation fails, mask
    /// dimensions do not match the image, or a filter-backed step is requested
    /// before its Coeus implementation exists.
    pub fn execute_coeus<B>(
        &self,
        mut image: Image<f32, B, 3>,
        backend: &B,
    ) -> Result<Image<f32, B, 3>>
    where
        B: ComputeBackend,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        for step in &self.steps {
            image = match step {
                PreprocessingStep::IntensityNormalization { mode } => {
                    let (vals, dims) = ritk_tensor_ops::coeus::extract_image_vec(&image)
                        .context("coeus preprocessing intensity normalization requires contiguous f32 image data")?;
                    let result = normalize_values(&vals, mode);
                    ritk_tensor_ops::coeus::rebuild_image(result, dims, &image, backend)?
                }
                PreprocessingStep::Clamp { lower, upper } => {
                    let (vals, dims) = ritk_tensor_ops::coeus::extract_image_vec(&image)
                        .context("coeus preprocessing clamp requires contiguous f32 image data")?;
                    let result = clamp_values(&vals, *lower, *upper);
                    ritk_tensor_ops::coeus::rebuild_image(result, dims, &image, backend)?
                }
                PreprocessingStep::Masking {
                    mask,
                    dims: mask_dims,
                } => {
                    validate_mask(mask, *mask_dims, image.shape())?;
                    let (vals, dims) = ritk_tensor_ops::coeus::extract_image_vec(&image).context(
                        "coeus preprocessing masking requires contiguous f32 image data",
                    )?;
                    let result = apply_mask_values(&vals, mask)?;
                    ritk_tensor_ops::coeus::rebuild_image(result, dims, &image, backend)?
                }
                PreprocessingStep::N4BiasCorrection { .. } => {
                    anyhow::bail!(
                        "coeus preprocessing does not support N4BiasCorrection; use the legacy Burn executor until N4 is migrated"
                    );
                }
                PreprocessingStep::Smoothing { .. } => {
                    anyhow::bail!(
                        "coeus preprocessing does not support Smoothing; use the legacy Burn executor until Gaussian smoothing is migrated"
                    );
                }
            };
        }
        Ok(image)
    }
}

#[cfg(test)]
mod tests {
    use coeus_core::SequentialBackend;
    use coeus_tensor::Tensor;
    use ritk_spatial::{Direction, Point, Spacing};

    use crate::preprocessing::{IntensityRescaleMode, PreprocessingPipeline, PreprocessingStep};

    use super::*;

    type B = SequentialBackend;

    fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
        let data = Tensor::<f32, B>::from_slice(dims, &vals);
        Image::new(
            data,
            Point::new([10.0, 20.0, 30.0]),
            Spacing::new([0.5, 1.5, 2.5]),
            Direction::identity(),
        )
        .unwrap()
    }

    #[test]
    fn execute_coeus_clamp_preserves_metadata() {
        let backend = B::new();
        let image = make_image(vec![-1.0, 0.25, 0.75, 2.0], [1, 2, 2]);
        let origin = *image.origin();
        let spacing = *image.spacing();
        let direction = *image.direction();
        let pipeline = PreprocessingPipeline::new().add_step(PreprocessingStep::Clamp {
            lower: 0.0,
            upper: 1.0,
        });

        let out = pipeline.execute_coeus(image, &backend).unwrap();

        assert_eq!(out.data_slice().unwrap(), &[0.0, 0.25, 0.75, 1.0]);
        assert_eq!(out.origin(), &origin);
        assert_eq!(out.spacing(), &spacing);
        assert_eq!(out.direction(), &direction);
    }

    #[test]
    fn execute_coeus_masking_matches_mask_values() {
        let backend = B::new();
        let image = make_image(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 2]);
        let pipeline = PreprocessingPipeline::new().add_step(PreprocessingStep::Masking {
            mask: vec![1, 0, 1, 0],
            dims: [1, 2, 2],
        });

        let out = pipeline.execute_coeus(image, &backend).unwrap();

        assert_eq!(out.data_slice().unwrap(), &[1.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn execute_coeus_minmax_matches_bounded_reference() {
        let backend = B::new();
        let image = make_image(vec![2.0, 4.0, 6.0], [1, 1, 3]);
        let pipeline =
            PreprocessingPipeline::new().add_step(PreprocessingStep::IntensityNormalization {
                mode: IntensityRescaleMode::MinMax {
                    out_min: 0.0,
                    out_max: 1.0,
                },
            });

        let out = pipeline.execute_coeus(image, &backend).unwrap();

        assert_eq!(out.data_slice().unwrap(), &[0.0, 0.5, 1.0]);
    }

    #[test]
    fn execute_coeus_rejects_filter_backed_steps_explicitly() {
        let backend = B::new();
        let image = make_image(vec![1.0; 4], [1, 2, 2]);
        let pipeline =
            PreprocessingPipeline::new().add_step(PreprocessingStep::Smoothing { sigma: 1.0 });

        let err = pipeline.execute_coeus(image, &backend).unwrap_err();

        assert_eq!(
            err.to_string(),
            "coeus preprocessing does not support Smoothing; use the legacy Burn executor until Gaussian smoothing is migrated"
        );
    }
}
