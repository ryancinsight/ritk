//! `PreprocessingPipeline` struct and builder methods.

use super::step::PreprocessingStep;

/// Sequential preprocessing pipeline.
#[derive(Debug, Clone, Default)]
pub struct PreprocessingPipeline {
    /// Steps visible to sibling `executor` module for dispatch.
    pub(crate) steps: Vec<PreprocessingStep>,
}

impl PreprocessingPipeline {
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: append step and return self.
    pub fn add_step(mut self, step: PreprocessingStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Mutable push (does not consume self).
    pub fn push_step(&mut self, step: PreprocessingStep) {
        self.steps.push(step);
    }

    pub fn step_count(&self) -> usize {
        self.steps.len()
    }
}

#[cfg(test)]
mod tests {
    use super::PreprocessingPipeline;
    use crate::preprocessing::{IntensityRescaleMode, PreprocessingStep};
    use coeus_core::SequentialBackend;
    use ritk_image::tensor::Tensor;
    use ritk_image::Image;
    use ritk_spatial::{Direction, Point, Spacing};

    type B = SequentialBackend;

    fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
        let device = Default::default();
        let t = Tensor::<f32, B>::from_slice_on(dims, &vals, &device);
        Image::new(
            t,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
        .expect("invariant: fixture tensor preserves the declared image rank")
    }

    fn extract(img: &Image<f32, B, 3>) -> Vec<f32> {
        img.data_slice()
            .expect("fixture image is CPU-addressable")
            .to_vec()
    }

    // Test: empty pipeline is identity
    #[test]
    fn test_empty_pipeline_identity() {
        let vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let img = make_image(vals.clone(), [2, 2, 2]);
        let result = PreprocessingPipeline::new().execute(img).unwrap();
        let out = extract(&result);
        assert_eq!(out, vals);
    }

    // Test: add_step builder increases step_count
    #[test]
    fn test_add_step_increases_step_count() {
        let p0 = PreprocessingPipeline::new();
        assert_eq!(p0.step_count(), 0);
        let p1 = p0.add_step(PreprocessingStep::Clamp {
            lower: 0.0,
            upper: 1.0,
        });
        assert_eq!(p1.step_count(), 1);
        let p2 = p1.add_step(PreprocessingStep::IntensityNormalization {
            mode: IntensityRescaleMode::ZScore,
        });
        assert_eq!(p2.step_count(), 2);
    }

    // Test: Clamp + MinMax composition applies in order
    #[test]
    fn test_clamp_then_minmax_composition() {
        // Values [-5, 2, 8, 15] clamped to [0,10] -> [0, 2, 8, 10]
        // Then MinMax [0,1]: min=0, max=10 => 0->0, 2->0.2, 8->0.8, 10->1.0
        let vals = vec![-5.0f32, 2.0, 8.0, 15.0];
        let img = make_image(vals, [1, 1, 4]);
        let pipeline = PreprocessingPipeline::new()
            .add_step(PreprocessingStep::Clamp {
                lower: 0.0,
                upper: 10.0,
            })
            .add_step(PreprocessingStep::IntensityNormalization {
                mode: IntensityRescaleMode::MinMax {
                    out_min: 0.0,
                    out_max: 1.0,
                },
            });
        let out = extract(&pipeline.execute(img).unwrap());
        assert!(
            out[0].abs() < 1e-5,
            "clamped -5 -> 0, minmax -> 0.0; got {}",
            out[0]
        );
        assert!(
            (out[1] - 0.2).abs() < 1e-4,
            "clamped 2, minmax -> 0.2; got {}",
            out[1]
        );
        assert!(
            (out[2] - 0.8).abs() < 1e-4,
            "clamped 8, minmax -> 0.8; got {}",
            out[2]
        );
        assert!(
            (out[3] - 1.0).abs() < 1e-4,
            "clamped 10, minmax -> 1.0; got {}",
            out[3]
        );
    }

    // Test: N4 pipeline step executes without error on a simple test image
    #[test]
    fn test_n4_pipeline_step_executes() {
        // Small 4x4x4 image filled with a smooth ramp to give N4 something realistic.
        let vals: Vec<f32> = (0..64).map(|i| 1.0 + i as f32 * 0.1).collect();
        let img = make_image(vals, [4, 4, 4]);
        let pipeline = PreprocessingPipeline::new().add_step(PreprocessingStep::N4BiasCorrection {
            n_iterations: 5,
            n_fitting_levels: 1,
        });
        let result = pipeline.execute(img);
        assert!(
            result.is_ok(),
            "N4 pipeline step must not return Err: {:?}",
            result.err()
        );
        let out = extract(&result.unwrap());
        assert_eq!(out.len(), 64);
        for &v in &out {
            assert!(v > 0.0, "N4 output must be positive, got {}", v);
        }
    }
}
