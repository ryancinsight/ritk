//! Coeus-backed preprocessing executor for scalar image steps.

use anyhow::{Context, Result};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::native::Image;
use ritk_spatial::{Spacing, VolumeDims};

use crate::deformable_field_ops::gaussian_smooth_with_scratch_per_axis;

use super::pipeline::PreprocessingPipeline;
use super::step::PreprocessingStep;
use super::value_ops::{
    apply_mask_values, clamp_values, normalize_values, validate_mask, validate_value_count,
};

impl PreprocessingPipeline {
    /// Execute preprocessing steps on a Coeus-backed image.
    ///
    /// The method supports pointwise scalar transforms, masking, and Gaussian
    /// smoothing. N4 still depends on the legacy Burn filter implementation and
    /// returns an explicit error rather than silently downgrading the Coeus path.
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
        let mut smoothing_scratch = Vec::new();
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
                PreprocessingStep::Smoothing { sigma } => {
                    let (mut vals, dims) = ritk_tensor_ops::coeus::extract_image_vec(&image)
                        .context("coeus preprocessing smoothing requires contiguous f32 image data")?;
                    smooth_values(
                        &mut vals,
                        dims,
                        image.spacing(),
                        *sigma,
                        &mut smoothing_scratch,
                    )?;
                    ritk_tensor_ops::coeus::rebuild_image(vals, dims, &image, backend)?
                }
                PreprocessingStep::N4BiasCorrection { .. } => anyhow::bail!(
                    "coeus preprocessing does not support N4BiasCorrection; use the legacy Burn executor until N4 is migrated"
                ),
            };
        }
        Ok(image)
    }
}

fn smooth_values(
    vals: &mut [f32],
    dims: [usize; 3],
    spacing: &Spacing<3>,
    sigma: f32,
    scratch: &mut Vec<f32>,
) -> Result<()> {
    validate_value_count(vals.len(), dims, "coeus smoothing")?;

    if !sigma.is_finite() {
        anyhow::bail!("coeus smoothing sigma must be finite, got {sigma}");
    }
    if sigma <= 0.0 {
        return Ok(());
    }

    let physical_sigma = f64::from(sigma);
    let voxel_sigmas = [
        physical_sigma / spacing[0],
        physical_sigma / spacing[1],
        physical_sigma / spacing[2],
    ];
    scratch.resize(vals.len(), 0.0);
    gaussian_smooth_with_scratch_per_axis(vals, VolumeDims::new(dims), voxel_sigmas, scratch);
    Ok(())
}

#[cfg(test)]
mod tests {
    use coeus_core::SequentialBackend;
    use ritk_spatial::{Direction, Point, Spacing};

    use crate::preprocessing::{IntensityRescaleMode, PreprocessingPipeline, PreprocessingStep};

    use super::*;

    type B = SequentialBackend;

    fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
        Image::from_flat(
            vals,
            dims,
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
    fn execute_coeus_smoothing_preserves_constant_image() {
        let backend = B::new();
        let image = make_image(vec![3.25; 27], [3, 3, 3]);
        let pipeline =
            PreprocessingPipeline::new().add_step(PreprocessingStep::Smoothing { sigma: 1.0 });

        let out = pipeline.execute_coeus(image, &backend).unwrap();

        for &value in out.data_slice().unwrap() {
            // Three f64-normalized convolution passes round once per pass into
            // f32 output; 1e-6 is a bounded multi-ulp tolerance at this magnitude.
            assert!(
                (value - 3.25).abs() <= 1.0e-6,
                "constant smoothing changed value to {value}"
            );
        }
    }

    #[test]
    fn execute_coeus_smoothing_reduces_impulse_peak() {
        let backend = B::new();
        let mut values = vec![0.0; 27];
        values[13] = 1.0;
        let image = make_image(values, [3, 3, 3]);
        let pipeline =
            PreprocessingPipeline::new().add_step(PreprocessingStep::Smoothing { sigma: 1.0 });

        let out = pipeline.execute_coeus(image, &backend).unwrap();
        let data = out.data_slice().unwrap();

        // Physical sigma 1.0 maps through the fixture spacing to per-axis voxel
        // sigmas [2.0, 2.0/3.0, 0.4]. Representative voxels are products of the
        // corresponding normalized kernel weights; the tolerance covers f64
        // accumulation rounded to f32 after each separable pass.
        let assert_close = |actual: f32, expected: f32, label: &str| {
            assert!(
                (actual - expected).abs() <= 1.0e-6,
                "{label} expected {expected}, got {actual}"
            );
        };
        assert!(
            data[13] > 0.0 && data[13] < 1.0,
            "smoothing must reduce the impulse peak, got {}",
            data[13]
        );
        assert_close(data[13], 0.109_807_3, "center voxel");
        assert_close(data[14], 0.004_824_596, "axis neighbor voxel");
        assert_close(data[17], 0.001_566_317, "face diagonal voxel");
        assert_close(data[26], 0.001_382_269_9, "corner voxel");
    }

    #[test]
    fn execute_coeus_rejects_nonfinite_smoothing_sigma() {
        let backend = B::new();
        let image = make_image(vec![1.0; 4], [1, 2, 2]);
        let pipeline =
            PreprocessingPipeline::new().add_step(PreprocessingStep::Smoothing { sigma: f32::NAN });

        let err = pipeline.execute_coeus(image, &backend).unwrap_err();

        assert_eq!(
            err.to_string(),
            "coeus smoothing sigma must be finite, got NaN"
        );
    }

    #[test]
    fn execute_coeus_rejects_n4_explicitly() {
        let backend = B::new();
        let image = make_image(vec![1.0; 4], [1, 2, 2]);
        let pipeline = PreprocessingPipeline::new().add_step(PreprocessingStep::N4BiasCorrection {
            n_iterations: 1,
            n_fitting_levels: 1,
        });

        let err = pipeline.execute_coeus(image, &backend).unwrap_err();

        assert_eq!(
            err.to_string(),
            "coeus preprocessing does not support N4BiasCorrection; use the legacy Burn executor until N4 is migrated"
        );
    }
}
