//! `execute()` implementation and per-step dispatch logic.

use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_filter::bias::N4Config;
use ritk_filter::{GaussianFilter, GaussianSigma, N4BiasFieldCorrectionFilter};

use super::pipeline::PreprocessingPipeline;
use super::step::{IntensityRescaleMode, PreprocessingStep};

impl PreprocessingPipeline {
    /// Execute all steps sequentially.
    ///
    /// Invariant: each step receives the output of the previous step.
    pub fn execute<B: Backend>(&self, mut image: Image<B, 3>) -> Result<Image<B, 3>> {
        for step in &self.steps {
            image = match step {
                PreprocessingStep::N4BiasCorrection {
                    n_iterations,
                    n_fitting_levels,
                } => {
                    let cfg = N4Config {
                        num_fitting_levels: *n_fitting_levels as usize,
                        num_iterations: *n_iterations as usize,
                        ..N4Config::default()
                    };
                    N4BiasFieldCorrectionFilter::new(cfg).apply(&image)?
                }

                PreprocessingStep::IntensityNormalization { mode } => {
                    let vals = image
                        .try_data_vec()
                        .context("IntensityNormalization requires f32 image data")?;
                    let n = vals.len();
                    let result = match mode {
                        IntensityRescaleMode::ZScore => {
                            let mean = vals.iter().sum::<f32>() / n as f32;
                            let variance =
                                vals.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n as f32;
                            let std = variance.sqrt();
                            if std < 1e-8 {
                                vec![0.0f32; n]
                            } else {
                                vals.iter().map(|&v| (v - mean) / std).collect()
                            }
                        }
                        IntensityRescaleMode::MinMax { out_min, out_max } => {
                            let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
                            let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let range = (max - min).max(1e-8);
                            vals.iter()
                                .map(|&v| {
                                    let n01 = (v - min) / range;
                                    n01 * (out_max - out_min) + out_min
                                })
                                .collect()
                        }
                    };
                    rebuild_image(&image, result)?
                }

                PreprocessingStep::Clamp { lower, upper } => {
                    let vals = image
                        .try_data_vec()
                        .context("Clamp requires f32 image data")?;
                    let result: Vec<f32> = vals.iter().map(|&v| v.clamp(*lower, *upper)).collect();
                    rebuild_image(&image, result)?
                }

                PreprocessingStep::Masking {
                    mask,
                    dims: mask_dims,
                } => {
                    let shape = image.shape();
                    let [nz, ny, nx] = shape;
                    if mask_dims != &[nz, ny, nx] {
                        return Err(anyhow::anyhow!(
                            "mask dims {:?} do not match image shape {:?}",
                            mask_dims,
                            shape
                        ));
                    }
                    let expected = nz * ny * nx;
                    if mask.len() != expected {
                        return Err(anyhow::anyhow!(
                            "mask length {} != voxel count {}",
                            mask.len(),
                            expected
                        ));
                    }
                    let vals = image
                        .try_data_vec()
                        .context("Masking requires f32 image data")?;
                    let result: Vec<f32> = vals
                        .iter()
                        .zip(mask.iter())
                        .map(|(&v, &m)| if m == 0 { 0.0 } else { v })
                        .collect();
                    rebuild_image(&image, result)?
                }

                PreprocessingStep::Smoothing { sigma } => {
                    let s_val = *sigma as f64;
                    let g_sigma = GaussianSigma::new(s_val)
                        .unwrap_or_else(|| GaussianSigma::new_unchecked(1e-9));
                    GaussianFilter::new(vec![g_sigma; 3]).apply(&image)
                }
            };
        }
        Ok(image)
    }
}

/// Reconstruct a 3-D image from a flat `Vec<f32>`, preserving spatial metadata.
fn rebuild_image<B: Backend>(src: &Image<B, 3>, vals: Vec<f32>) -> Result<Image<B, 3>> {
    let shape = src.shape();
    let device = src.data().device();
    let tensor = Tensor::<B, 3>::from_data(TensorData::new(vals, Shape::new(shape)), &device);
    Ok(Image::new(
        tensor,
        *src.origin(),
        *src.spacing(),
        *src.direction(),
    ))
}

#[cfg(test)]
mod tests {
    use super::super::pipeline::PreprocessingPipeline;
    use crate::preprocessing::{IntensityRescaleMode, PreprocessingStep};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;
    use ritk_image::Image;
    use ritk_spatial::{Direction, Point, Spacing};

    type B = NdArray<f32>;

    fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let t = Tensor::<B, 3>::from_data(TensorData::new(vals, Shape::new(dims)), &device);
        Image::new(
            t,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn extract(img: &Image<B, 3>) -> Vec<f32> {
        img.data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    // Test: Clamp [0,1] clamps all out-of-range values
    #[test]
    fn test_clamp_clips_values() {
        let vals = vec![-1.0f32, 0.0, 0.5, 1.0, 1.5, 2.0, -0.5, 0.3];
        let img = make_image(vals, [2, 2, 2]);
        let pipeline = PreprocessingPipeline::new().add_step(PreprocessingStep::Clamp {
            lower: 0.0,
            upper: 1.0,
        });
        let out = extract(&pipeline.execute(img).unwrap());
        for &v in &out {
            assert!((0.0..=1.0).contains(&v), "value {} outside [0,1]", v);
        }
        assert_eq!(out[0], 0.0f32);
        assert_eq!(out[4], 1.0f32);
        assert!((out[2] - 0.5f32).abs() < 1e-6);
    }

    // Test: Masking zeros masked voxels, preserves unmasked
    #[test]
    fn test_masking_zeros_masked_voxels() {
        let vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mask = vec![1u8, 0, 1, 0, 1, 1, 0, 1];
        let img = make_image(vals.clone(), [2, 2, 2]);
        let pipeline = PreprocessingPipeline::new().add_step(PreprocessingStep::Masking {
            mask: mask.clone(),
            dims: [2, 2, 2],
        });
        let out = extract(&pipeline.execute(img).unwrap());
        assert_eq!(out[1], 0.0f32, "voxel 1 (mask=0) must be 0");
        assert_eq!(out[3], 0.0f32, "voxel 3 (mask=0) must be 0");
        assert_eq!(out[6], 0.0f32, "voxel 6 (mask=0) must be 0");
        assert_eq!(out[0], vals[0], "voxel 0 (mask=1) must be original");
        assert_eq!(out[2], vals[2], "voxel 2 (mask=1) must be original");
    }

    // Test: Masking with wrong dims returns Err
    #[test]
    fn test_masking_wrong_dims_returns_err() {
        let img = make_image(vec![1.0f32; 8], [2, 2, 2]);
        let pipeline = PreprocessingPipeline::new().add_step(PreprocessingStep::Masking {
            mask: vec![1u8; 8],
            dims: [3, 2, 2], // wrong: 3*2*2=12 != 8
        });
        assert!(
            pipeline.execute(img).is_err(),
            "wrong dims must produce Err"
        );
    }

    // Test: ZScore on constant image -> all zeros
    #[test]
    fn test_zscore_constant_image_all_zeros() {
        let img = make_image(vec![5.0f32; 8], [2, 2, 2]);
        let pipeline =
            PreprocessingPipeline::new().add_step(PreprocessingStep::IntensityNormalization {
                mode: IntensityRescaleMode::ZScore,
            });
        let out = extract(&pipeline.execute(img).unwrap());
        for &v in &out {
            assert!(
                v.abs() < 1e-6,
                "constant image z-score must be 0, got {}",
                v
            );
        }
    }

    // Test: MinMax [0,1] -> actual min becomes 0.0, max becomes 1.0
    #[test]
    fn test_minmax_min_is_zero_max_is_one() {
        // Values [2, 5, 3, 8, 1, 6, 4, 7]: min=1, max=8
        let vals = vec![2.0f32, 5.0, 3.0, 8.0, 1.0, 6.0, 4.0, 7.0];
        let img = make_image(vals, [2, 2, 2]);
        let pipeline =
            PreprocessingPipeline::new().add_step(PreprocessingStep::IntensityNormalization {
                mode: IntensityRescaleMode::MinMax {
                    out_min: 0.0,
                    out_max: 1.0,
                },
            });
        let out = extract(&pipeline.execute(img).unwrap());
        let out_min = out.iter().cloned().fold(f32::INFINITY, f32::min);
        let out_max = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            out_min.abs() < 1e-5,
            "min after MinMax must be 0, got {}",
            out_min
        );
        assert!(
            (out_max - 1.0).abs() < 1e-5,
            "max after MinMax must be 1, got {}",
            out_max
        );
    }
}
