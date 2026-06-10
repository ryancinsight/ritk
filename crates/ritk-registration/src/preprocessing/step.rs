//! `NormalizationMode` and `PreprocessingStep` — step type definitions.

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
    N4BiasCorrection {
        n_iterations: u32,
        n_fitting_levels: u32,
    },
    IntensityNormalization {
        mode: NormalizationMode,
    },
    Clamp {
        lower: f32,
        upper: f32,
    },
    /// Zero out voxels where `mask[i] == 0`.  `mask.len()` must equal voxel count.
    Masking {
        mask: Vec<u8>,
        dims: [usize; 3],
    },
    Smoothing {
        sigma: f32,
    },
}
