//! ANTs-style preprocessing pipeline for volumetric images.
//!
//! Mathematical specification:
//!   P = (`steps: Vec<PreprocessingStep>`) applied sequentially.
//!   `execute(P, I_0) = fold(steps, I_0, apply_step)`
//!
//! Each step is a deterministic, pure transform `Image<B,3> -> Result<Image<B,3>>`.
//!
//! Steps:
//!   N4BiasCorrection  : `I' = exp(ln(I) - B_spline_estimate)`
//!   IntensityNorm ZScore : `I'[i] = (I[i] - mu) / sigma`  (sigma=0 -> 0)
//!   IntensityNorm MinMax : `I'[i] = (I[i]-min)/(max-min) * (hi-lo) + lo`
//!   Clamp             : `I'[i] = clamp(I[i], lower, upper)`
//!   Masking           : `I'[i] = if mask[i]==0 { 0 } else { I[i] }`
//!   Smoothing         : `I' = Gaussian_sigma(I)`

pub mod brain_mask;
pub(crate) mod native_executor;
pub(crate) mod executor;
pub(crate) mod pipeline;
pub(crate) mod step;
pub(crate) mod value_ops;

pub use brain_mask::{ct_brain_mask, CtBrainMaskConfig};
pub use pipeline::PreprocessingPipeline;
pub use step::{IntensityRescaleMode, PreprocessingStep};
