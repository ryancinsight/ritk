//! The Anti-Alias step: `CurvatureFlowFunction::ComputeUpdate` as an engine
//! policy. The band machinery it runs on lives in [`crate::sparse_field`].

use super::{curvature::curvature, AntiAliasBinaryImageFilter, CGV, DT};
use crate::sparse_field::{evolve, GridTopology, SparseFieldConfig, SparseFieldStep};

/// Mean-curvature flow with the level-set sign locked to the input binary.
struct CurvatureFlowStep<'a> {
    binary: &'a [f32],
    foreground: f32,
    topo: GridTopology,
}

impl SparseFieldStep<f32> for CurvatureFlowStep<'_> {
    fn stage(&self, phi: &[f32], active: &[usize]) -> (Vec<f32>, f32) {
        (
            active
                .iter()
                .map(|&f| curvature(phi, &self.topo, f))
                .collect(),
            DT,
        )
    }

    fn clamp(&self, f: usize, value: f32) -> f32 {
        if self.binary[f] == self.foreground {
            value.max(0.0)
        } else {
            value.min(0.0)
        }
    }
}

impl AntiAliasBinaryImageFilter {
    pub(super) fn run(&self, binary: &[f32], dims: [usize; 3]) -> Vec<f32> {
        let topo = GridTopology::new(dims);
        if topo.len() == 0 {
            return Vec::new();
        }

        // iso = (max+min)/2 (MinimumMaximumImageCalculator).
        let (mn, mx) = binary
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        let iso = mx - (mx - mn) / 2.0;
        let shifted: Vec<f32> = binary.iter().map(|&v| v - iso).collect();

        let step = CurvatureFlowStep {
            binary,
            foreground: mx,
            topo,
        };
        evolve(
            &shifted,
            &SparseFieldConfig {
                dims,
                // CurvatureFlow overrides NumberOfLayers with the image dimension.
                number_of_layers: topo.ndim(),
                constant_gradient: CGV,
                iterations: self.number_of_iterations,
                max_rms_error: self.max_rms_error as f64,
            },
            &step,
        )
    }
}
