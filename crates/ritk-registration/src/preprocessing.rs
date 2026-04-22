//! ANTs-style preprocessing pipeline for volumetric images.
//!
//! Mathematical specification:
//!   P = (steps: Vec<PreprocessingStep>) applied sequentially.
//!   execute(P, I_0) = fold(steps, I_0, apply_step)
//!
//! Each step is a deterministic, pure transform Image<B,3> -> Result<Image<B,3>>.
//!
//! Steps:
//!   N4BiasCorrection  : I' = exp(ln(I) - B_spline_estimate)
//!   IntensityNorm ZScore : I'[i] = (I[i] - mu) / sigma  (sigma=0 -> 0)
//!   IntensityNorm MinMax : I'[i] = (I[i]-min)/(max-min) * (hi-lo) + lo
//!   Clamp             : I'[i] = clamp(I[i], lower, upper)
//!   Masking           : I'[i] = if mask[i]==0 { 0 } else { I[i] }
//!   Smoothing         : I' = Gaussian_sigma(I)

use anyhow::Result;
use ritk_core::filter::bias::N4Config;
use ritk_core::filter::{GaussianFilter, N4BiasFieldCorrectionFilter};
use ritk_core::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

/// Intensity normalization mode.
#[derive(Debug, Clone)]
pub enum NormalizationMode {
    /// z-score: (x - mu) / sigma.  Constant images produce all-zero output.
    ZScore,
    /// Min-max rescale to [out_min, out_max].
    MinMax { out_min: f32, out_max: f32 },
}

/// A single preprocessing step.
#[derive(Debug, Clone)]
pub enum PreprocessingStep {
    N4BiasCorrection { n_iterations: u32, n_fitting_levels: u32 },
    IntensityNormalization { mode: NormalizationMode },
    Clamp { lower: f32, upper: f32 },
    /// Zero out voxels where mask[i] == 0.  mask.len() must equal voxel count.
    Masking { mask: Vec<u8>, dims: [usize; 3] },
    Smoothing { sigma: f32 },
}

/// Sequential preprocessing pipeline.
#[derive(Debug, Clone, Default)]
pub struct PreprocessingPipeline {
    steps: Vec<PreprocessingStep>,
}

impl PreprocessingPipeline {
    pub fn new() -> Self { Self::default() }

    /// Builder: append step and return self.
    pub fn add_step(mut self, step: PreprocessingStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Mutable push (does not consume self).
    pub fn push_step(&mut self, step: PreprocessingStep) {
        self.steps.push(step);
    }

    pub fn step_count(&self) -> usize { self.steps.len() }

    /// Execute all steps sequentially.
    ///
    /// Invariant: each step receives the output of the previous step.
    pub fn execute<B: Backend>(&self, mut image: Image<B, 3>) -> Result<Image<B, 3>> {
        for step in &self.steps {
            image = match step {
                PreprocessingStep::N4BiasCorrection { n_iterations, n_fitting_levels } => {
                    let cfg = N4Config {
                        num_fitting_levels: *n_fitting_levels as usize,
                        num_iterations: *n_iterations as usize,
                        ..N4Config::default()
                    };
                    N4BiasFieldCorrectionFilter::new(cfg).apply(&image)?
                }

                PreprocessingStep::IntensityNormalization { mode } => {
                    let td = image.data().clone().into_data();
                    let vals: Vec<f32> = td
                        .as_slice::<f32>()
                        .map_err(|e| anyhow::anyhow!("f32 required: {:?}", e))?
                        .to_vec();
                    let n = vals.len();
                    let result = match mode {
                        NormalizationMode::ZScore => {
                            let mean = vals.iter().sum::<f32>() / n as f32;
                            let variance = vals.iter()
                                .map(|&v| (v - mean).powi(2))
                                .sum::<f32>() / n as f32;
                            let std = variance.sqrt();
                            if std < 1e-8 {
                                vec![0.0f32; n]
                            } else {
                                vals.iter().map(|&v| (v - mean) / std).collect()
                            }
                        }
                        NormalizationMode::MinMax { out_min, out_max } => {
                            let min = vals.iter().cloned()
                                .fold(f32::INFINITY, f32::min);
                            let max = vals.iter().cloned()
                                .fold(f32::NEG_INFINITY, f32::max);
                            let range = (max - min).max(1e-8);
                            vals.iter().map(|&v| {
                                let n01 = (v - min) / range;
                                n01 * (out_max - out_min) + out_min
                            }).collect()
                        }
                    };
                    rebuild_image_3d(&image, result)?
                }

                PreprocessingStep::Clamp { lower, upper } => {
                    let td = image.data().clone().into_data();
                    let vals: Vec<f32> = td
                        .as_slice::<f32>()
                        .map_err(|e| anyhow::anyhow!("f32 required: {:?}", e))?
                        .to_vec();
                    let result: Vec<f32> = vals.iter().map(|&v| v.clamp(*lower, *upper)).collect();
                    rebuild_image_3d(&image, result)?
                }

                PreprocessingStep::Masking { mask, dims: mask_dims } => {
                    let shape = image.shape();
                    let [nz, ny, nx] = shape;
                    if mask_dims != &[nz, ny, nx] {
                        return Err(anyhow::anyhow!(
                            "mask dims {:?} do not match image shape {:?}",
                            mask_dims, shape
                        ));
                    }
                    let expected = nz * ny * nx;
                    if mask.len() != expected {
                        return Err(anyhow::anyhow!(
                            "mask length {} != voxel count {}",
                            mask.len(), expected
                        ));
                    }
                    let td = image.data().clone().into_data();
                    let vals: Vec<f32> = td
                        .as_slice::<f32>()
                        .map_err(|e| anyhow::anyhow!("f32 required: {:?}", e))?
                        .to_vec();
                    let result: Vec<f32> = vals.iter().zip(mask.iter())
                        .map(|(&v, &m)| if m == 0 { 0.0 } else { v })
                        .collect();
                    rebuild_image_3d(&image, result)?
                }

                PreprocessingStep::Smoothing { sigma } => {
                    GaussianFilter::new(vec![*sigma as f64; 3]).apply(&image)
                }
            };
        }
        Ok(image)
    }
}

/// Reconstruct a 3-D image from a flat `Vec<f32>`, preserving spatial metadata.
fn rebuild_image_3d<B: Backend>(src: &Image<B, 3>, vals: Vec<f32>) -> Result<Image<B, 3>> {
    let shape = src.shape();
    let device = src.data().device();
    let tensor = Tensor::<B, 3>::from_data(TensorData::new(vals, Shape::new(shape)), &device);
    Ok(Image::new(
        tensor,
        src.origin().clone(),
        src.spacing().clone(),
        src.direction().clone(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let t = Tensor::<B, 3>::from_data(
            TensorData::new(vals, Shape::new(dims)), &device);
        Image::new(t,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity())
    }

    fn extract(img: &Image<B, 3>) -> Vec<f32> {
        img.data().clone().into_data()
            .as_slice::<f32>().unwrap().to_vec()
    }

    // Test 1: empty pipeline is identity
    #[test]
    fn test_empty_pipeline_identity() {
        let vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let img = make_image(vals.clone(), [2, 2, 2]);
        let result = PreprocessingPipeline::new().execute(img).unwrap();
        let out = extract(&result);
        assert_eq!(out, vals);
    }

    // Test 2: Clamp [0,1] clamps all out-of-range values
    #[test]
    fn test_clamp_clips_values() {
        let vals = vec![-1.0f32, 0.0, 0.5, 1.0, 1.5, 2.0, -0.5, 0.3];
        let img = make_image(vals, [2, 2, 2]);
        let pipeline = PreprocessingPipeline::new()
            .add_step(PreprocessingStep::Clamp { lower: 0.0, upper: 1.0 });
        let out = extract(&pipeline.execute(img).unwrap());
        for &v in &out {
            assert!(v >= 0.0 && v <= 1.0, "value {} outside [0,1]", v);
        }
        assert_eq!(out[0], 0.0f32);
        assert_eq!(out[4], 1.0f32);
        assert!((out[2] - 0.5f32).abs() < 1e-6);
    }

    // Test 3: Masking zeros masked voxels, preserves unmasked
    #[test]
    fn test_masking_zeros_masked_voxels() {
        let vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mask = vec![1u8, 0, 1, 0, 1, 1, 0, 1];
        let img = make_image(vals.clone(), [2, 2, 2]);
        let pipeline = PreprocessingPipeline::new()
            .add_step(PreprocessingStep::Masking { mask: mask.clone(), dims: [2, 2, 2] });
        let out = extract(&pipeline.execute(img).unwrap());
        // Index 1: mask=0 -> 0
        assert_eq!(out[1], 0.0f32, "voxel 1 (mask=0) must be 0");
        // Index 3: mask=0 -> 0
        assert_eq!(out[3], 0.0f32, "voxel 3 (mask=0) must be 0");
        // Index 6: mask=0 -> 0
        assert_eq!(out[6], 0.0f32, "voxel 6 (mask=0) must be 0");
        // Index 0: mask=1 -> original
        assert_eq!(out[0], vals[0], "voxel 0 (mask=1) must be original");
        // Index 2: mask=1 -> original
        assert_eq!(out[2], vals[2], "voxel 2 (mask=1) must be original");
    }

    // Test 4: Masking with wrong dims returns Err
    #[test]
    fn test_masking_wrong_dims_returns_err() {
        let img = make_image(vec![1.0f32; 8], [2, 2, 2]);
        let pipeline = PreprocessingPipeline::new()
            .add_step(PreprocessingStep::Masking {
                mask: vec![1u8; 8],
                dims: [3, 2, 2], // wrong: 3*2*2=12 != 8
            });
        let result = pipeline.execute(img);
        assert!(result.is_err(), "wrong dims must produce Err");
    }

    // Test 5: ZScore on constant image -> all zeros
    #[test]
    fn test_zscore_constant_image_all_zeros() {
        let img = make_image(vec![5.0f32; 8], [2, 2, 2]);
        let pipeline = PreprocessingPipeline::new()
            .add_step(PreprocessingStep::IntensityNormalization {
                mode: NormalizationMode::ZScore });
        let out = extract(&pipeline.execute(img).unwrap());
        for &v in &out {
            assert!(v.abs() < 1e-6, "constant image z-score must be 0, got {}", v);
        }
    }

    // Test 6: MinMax [0,1] -> actual min becomes 0.0, max becomes 1.0
    #[test]
    fn test_minmax_min_is_zero_max_is_one() {
        // Values [2, 5, 3, 8, 1, 6, 4, 7]: min=1, max=8
        let vals = vec![2.0f32, 5.0, 3.0, 8.0, 1.0, 6.0, 4.0, 7.0];
        let img = make_image(vals, [2, 2, 2]);
        let pipeline = PreprocessingPipeline::new()
            .add_step(PreprocessingStep::IntensityNormalization {
                mode: NormalizationMode::MinMax { out_min: 0.0, out_max: 1.0 } });
        let out = extract(&pipeline.execute(img).unwrap());
        let out_min = out.iter().cloned().fold(f32::INFINITY, f32::min);
        let out_max = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(out_min.abs() < 1e-5, "min after MinMax must be 0, got {}", out_min);
        assert!((out_max - 1.0).abs() < 1e-5, "max after MinMax must be 1, got {}", out_max);
    }

    // Test 7: Clamp + MinMax composition applies in order
    #[test]
    fn test_clamp_then_minmax_composition() {
        // Values [-5, 2, 8, 15] clamped to [0,10] -> [0, 2, 8, 10]
        // Then MinMax [0,1]: min=0, max=10
        //   0->0, 2->0.2, 8->0.8, 10->1.0
        let vals = vec![-5.0f32, 2.0, 8.0, 15.0];
        let img = make_image(vals, [1, 1, 4]);
        let pipeline = PreprocessingPipeline::new()
            .add_step(PreprocessingStep::Clamp { lower: 0.0, upper: 10.0 })
            .add_step(PreprocessingStep::IntensityNormalization {
                mode: NormalizationMode::MinMax { out_min: 0.0, out_max: 1.0 } });
        let out = extract(&pipeline.execute(img).unwrap());
        assert!(out[0].abs() < 1e-5, "clamped -5 -> 0, minmax -> 0.0; got {}", out[0]);
        assert!((out[1] - 0.2).abs() < 1e-4, "clamped 2, minmax -> 0.2; got {}", out[1]);
        assert!((out[2] - 0.8).abs() < 1e-4, "clamped 8, minmax -> 0.8; got {}", out[2]);
        assert!((out[3] - 1.0).abs() < 1e-4, "clamped 10, minmax -> 1.0; got {}", out[3]);
    }

    // Test 8: add_step builder increases step_count
    #[test]
    fn test_add_step_increases_step_count() {
        let p0 = PreprocessingPipeline::new();
        assert_eq!(p0.step_count(), 0);
        let p1 = p0.add_step(PreprocessingStep::Clamp { lower: 0.0, upper: 1.0 });
        assert_eq!(p1.step_count(), 1);
        let p2 = p1.add_step(PreprocessingStep::IntensityNormalization {
            mode: NormalizationMode::ZScore });
        assert_eq!(p2.step_count(), 2);
    }

    // Test 9: N4 pipeline step executes without error on a simple test image
    #[test]
    fn test_n4_pipeline_step_executes() {
        // Small 4x4x4 image filled with a smooth ramp to give N4 something realistic.
        let vals: Vec<f32> = (0..64).map(|i| 1.0 + i as f32 * 0.1).collect();
        let img = make_image(vals, [4, 4, 4]);
        let pipeline = PreprocessingPipeline::new()
            .add_step(PreprocessingStep::N4BiasCorrection {
                n_iterations: 5,
                n_fitting_levels: 1,
            });
        let result = pipeline.execute(img);
        assert!(result.is_ok(), "N4 pipeline step must not return Err: {:?}", result.err());
        // Verify the output has the same shape and all positive values.
        let out = extract(&result.unwrap());
        assert_eq!(out.len(), 64);
        for &v in &out {
            assert!(v > 0.0, "N4 output must be positive, got {}", v);
        }
    }
}
