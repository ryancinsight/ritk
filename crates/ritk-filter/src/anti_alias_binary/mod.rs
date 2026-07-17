//! Anti-alias binary image filter via the ITK SparseField level-set solver.
//!
//! # Mathematical Specification
//!
//! Ports `sitk.AntiAliasBinaryImageFilter` / `itk::AntiAliasBinaryImageFilter`,
//! which derives from `itk::SparseFieldLevelSetImageFilter` with a
//! `CurvatureFlowFunction` difference function. The boundary of a binary object
//! is smoothed by evolving a narrow-band signed level-set under mean curvature
//! flow, constrained so the zero crossing never leaves the original boundary
//! band (the per-pixel sign is locked to the input binary).

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

mod curvature;
mod solver;

// ── Constants ─────────────────────────────────────────────────────────────────

/// ITK `CurvatureFlowImageFilter` default explicit-Euler time step.
const DT: f32 = 0.05;
/// `m_ConstantGradientValue` (unit spacing).
const CGV: f32 = 1.0;
/// Gradient-magnitude-squared floor preventing 0/0 in flat regions.
const MSQ_EPS: f32 = 1e-9;

// Status sentinels (non-layer states are negative; layer indices are 0..num).
const ST_NULL: i32 = -1;
const ST_CHG: i32 = -2;
const ST_CUP: i32 = -3;
const ST_CDN: i32 = -4;

// ── Filter ────────────────────────────────────────────────────────────────────

/// Anti-alias binary image filter (faithful ITK SparseField solver).
///
/// Smooths the boundary of a binary object, returning the signed level-set φ
/// (negative inside the smoothed object, positive outside; the zero crossing is
/// the anti-aliased sub-voxel boundary). Bit-exact to `sitk.AntiAliasBinary`.
///
/// # Defaults
/// - `max_rms_error = 0.07` (ITK default)
/// - `number_of_iterations = 1000` (ITK default)
#[derive(Debug, Clone)]
pub struct AntiAliasBinaryImageFilter {
    /// Per-voxel RMS change threshold for early termination (ITK default 0.07).
    pub max_rms_error: f32,
    /// Maximum number of level-set evolution iterations (ITK default 1000).
    pub number_of_iterations: usize,
}

impl Default for AntiAliasBinaryImageFilter {
    fn default() -> Self {
        Self {
            max_rms_error: 0.07,
            number_of_iterations: 1000,
        }
    }
}

impl AntiAliasBinaryImageFilter {
    /// Evolve the binary boundary under the SparseField mean-curvature solver.
    ///
    /// `image`: binary float32 (foreground == max value, background == min),
    /// shape `[nz, ny, nx]` (`nz == 1` is treated as a 2-D image, matching sitk).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (binary, dims) = extract_vec_infallible(image);
        let out = self.run(&binary, dims);
        rebuild(out, dims, image)
    }
    /// Coeus-native sister of [`apply`].
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (binary, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let out = self.run(&binary, dims);
        crate::native_support::rebuild_image(out, dims, image, backend)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "../tests_anti_alias_binary.rs"]
mod tests_anti_alias_binary;
