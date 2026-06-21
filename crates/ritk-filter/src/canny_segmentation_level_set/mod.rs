//! Canny-edge-guided segmentation level set via the ITK SparseField solver.
//!
//! # Mathematical Specification
//!
//! Ports `sitk.CannySegmentationLevelSet` / `itk::CannySegmentationLevelSetImageFilter`,
//! which is a `SegmentationLevelSetImageFilter` (a `SparseFieldLevelSetImageFilter`)
//! driven by a `CannySegmentationLevelSetFunction`.
//!
//! ## Speed / advection construction (computed once)
//!
//! 1. **Canny edges** of the feature image via
//!    [`CannyEdgeDetectionImageFilter`] (`variance`, `maximum_error = 0.01`,
//!    `lower_threshold = 0`, `upper_threshold = threshold`).
//! 2. **Speed image** `P = DanielssonDistanceMap(cannyEdges)` (unsigned Euclidean
//!    distance to the nearest edge voxel) via [`DistanceTransformImageFilter`].
//! 3. **Advection field** `A = P · ∇P` (central interior differences, one-sided
//!    boundaries — `numpy.gradient` convention).
//!
//! ## SparseField evolution (`itk::LevelSetFunction::ComputeUpdate`)
//!
//! Per active-layer voxel the update is `κ·c_w − P·√(godunov)·p_w − (A·∇φ)·a_w`,
//! where
//! - `κ = (Σ_{i≠j} φ_jj·φ_i² − φ_i·φ_j·φ_ij) / (1e-6 + |∇φ|²)` is ITK's
//!   `ComputeCurvatureTerm` (mean-curvature numerator over squared gradient),
//! - the propagation gradient uses the Godunov upwind scheme in the sign of `P`,
//! - the advection term uses simple upwinding in the sign of each `A` component.
//!
//! **InterpolateSurfaceLocation** (ON): `P` and `A` are sampled not at the pixel
//! centre but at the sub-voxel surface location `idx − offset`, where
//! `offset[i] = d[i]·φ(x) / (Σ d² + MIN_NORM)` and `d[i]` is the larger-magnitude
//! one-sided φ-derivative along axis `i` (or the zero-surface direction when the
//! axis neighbours straddle the surface) — `itkSparseFieldLevelSetImageFilter`
//! `CalculateChange`. Sampling is multilinear (`LinearInterpolateImageFunction`).
//!
//! The global time step is `Δt = min(waveDT/(maxAdv+maxProp), DT/maxCurv)` with
//! `waveDT = DT = 1/(2·dim)` (`ComputeGlobalTimeStep`), recomputed each iteration
//! from the per-voxel maxima of the (weighted, offset-sampled) terms.
//!
//! Narrow-band bookkeeping (status lists, layer construction, value propagation,
//! the `ProcessStatusList` cascade and orphan node-deletion) is identical to
//! [`crate::AntiAliasBinaryImageFilter`]; `NumberOfLayers = 2` (the SparseField
//! default — only AntiAlias overrides it to the image dimension).
//!
//! Output is the evolved level set φ (band values plus a `±(NumberOfLayers+1)`
//! far field), **not** a binary mask — threshold at φ < 0 for the region.
//!
//! Validated bit-exact (max-err 0.0 across iterations 1–5) against
//! `sitk.CannySegmentationLevelSet` on a square feature with a circular init.
//!
//! ## References
//! - Whitaker, R.T. (1998). "A Level-Set Approach to 3D Reconstruction from Range
//!   Data." *IJCV*, 29(3), 203–231.
//! - ITK `itkCannySegmentationLevelSetFunction.hxx`,
//!   `itkSegmentationLevelSetFunction.hxx`, `itkLevelSetFunction.hxx`,
//!   `itkSparseFieldLevelSetImageFilter.hxx`.

mod advection;
mod solver;

use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

use crate::{CannyEdgeDetectionImageFilter, DistanceTransformImageFilter};
use advection::advection_field;

// ── Constants ─────────────────────────────────────────────────────────────────

/// `m_ConstantGradientValue` (unit spacing).
const CGV: f64 = 1.0;
/// `MIN_NORM` floor for the surface-location offset (unit spacing).
const MIN_NORM: f64 = 1.0e-6;
/// `m_GradMagSqr` seed / curvature denominator floor (`itkLevelSetFunction`).
const GRAD_EPS: f64 = 1.0e-6;
/// Internal Canny `MaximumError` fixed by `CannySegmentationLevelSetFunction`.
const CANNY_MAX_ERROR: f64 = 0.01;

// Status sentinels (non-layer states are negative; layer indices are 0..num).
const ST_NULL: i32 = -1;
const ST_CHG: i32 = -2;
const ST_CUP: i32 = -3;
const ST_CDN: i32 = -4;

// ── Public API ─────────────────────────────────────────────────────────────────

/// Canny-edge-guided segmentation level set (faithful ITK SparseField solver).
///
/// Returns the evolved level set φ (negative inside the segmented region). The
/// zero crossing is the segmentation boundary. Bit-exact to
/// `sitk.CannySegmentationLevelSet`.
///
/// # Defaults (match `sitk.CannySegmentationLevelSet`)
/// - `canny_threshold = 0.0`, `canny_variance = 0.0`
/// - `propagation_scaling = 1.0`, `curvature_scaling = 1.0`, `advection_scaling = 1.0`
/// - `iso_surface_value = 0.0`
/// - `max_rms_error = 0.02`, `number_of_iterations = 1000`
#[derive(Debug, Clone)]
pub struct CannySegmentationLevelSet {
    /// Upper hysteresis threshold for the internal Canny edge detector.
    pub canny_threshold: f32,
    /// Gaussian variance for the internal Canny edge detector.
    pub canny_variance: f32,
    /// Maximum number of PDE iterations.
    pub number_of_iterations: usize,
    /// RMS convergence criterion: stop when the active-layer RMS change < this.
    pub max_rms_error: f32,
    /// Propagation (balloon) force scaling.
    pub propagation_scaling: f32,
    /// Curvature regularisation weight.
    pub curvature_scaling: f32,
    /// Advection (edge-attraction) weight.
    pub advection_scaling: f32,
    /// Iso-surface value of the initial level set treated as the zero crossing.
    pub iso_surface_value: f32,
}

impl Default for CannySegmentationLevelSet {
    fn default() -> Self {
        Self {
            canny_threshold: 0.0,
            canny_variance: 0.0,
            number_of_iterations: 1000,
            max_rms_error: 0.02,
            propagation_scaling: 1.0,
            curvature_scaling: 1.0,
            advection_scaling: 1.0,
            iso_surface_value: 0.0,
        }
    }
}

impl CannySegmentationLevelSet {
    /// Evolve a level set toward Canny edges in the feature image.
    ///
    /// # Arguments
    /// - `initial_level_set`: φ₀ with **φ < `iso_surface_value` inside** the ROI.
    /// - `feature_image`: the image from which Canny edges are derived. Must have
    ///   the same shape as `initial_level_set`.
    ///
    /// # Errors
    /// Returns `Err` if tensor extraction fails or the shapes differ.
    pub fn apply<B: Backend>(
        &self,
        initial_level_set: &Image<B, 3>,
        feature_image: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        let dims = initial_level_set.shape();
        if dims != feature_image.shape() {
            anyhow::bail!(
                "initial_level_set shape {:?} and feature_image shape {:?} must match",
                dims,
                feature_image.shape()
            );
        }
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        if n == 0 {
            return Ok(initial_level_set.clone());
        }

        // ── Speed image P = DanielssonDistanceMap(CannyEdges(feature)) ───────
        let edges = CannyEdgeDetectionImageFilter {
            variance: self.canny_variance as f64,
            maximum_error: CANNY_MAX_ERROR,
            lower_threshold: 0.0,
            upper_threshold: self.canny_threshold,
        }
        .apply(feature_image);
        let p_img = DistanceTransformImageFilter::new().apply(&edges)?;
        let (p_f32, _) = extract_vec(&p_img)?;
        let p: Vec<f64> = p_f32.iter().map(|&v| v as f64).collect();

        // ── Advection field A[axis] = P · ∂P/∂axis (numpy.gradient convention) ──
        // axis 0 = x (innermost), 1 = y, 2 = z, matching ITK index ordering.
        let adv = advection_field(&p, dims);

        // ── Initial level set, shifted so iso_surface_value maps to 0 ────────
        let (sh_f32, _) = extract_vec(initial_level_set)?;
        let iso = self.iso_surface_value as f64;
        let shifted: Vec<f64> = sh_f32.iter().map(|&v| v as f64 - iso).collect();

        let phi = self.run(&shifted, &p, &adv, dims);
        let result: Vec<f32> = phi.iter().map(|&v| v as f32).collect();
        Ok(rebuild(result, dims, initial_level_set))
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "../tests_canny_segmentation_level_set.rs"]
mod tests;
