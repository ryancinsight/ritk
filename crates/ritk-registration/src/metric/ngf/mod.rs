//! Normalized Gradient Fields (NGF) metric for multi-modal registration.
//!
//! # Theorem: edge-orientation similarity (Haber & Modersitzki 2006)
//!
//! Cross-modal pairs (CTâ†”MRI) lack a functional intensity relationship, so
//! intensity metrics (MI, NCC) can be weak â€” e.g. a near-uniform CT brain
//! interior gives almost no mutual-information signal, and a rotation about the
//! centroid barely perturbs the joint histogram. NGF instead aligns the
//! **orientation** of image gradients, which co-locate across modalities even
//! where intensities do not (a skull/ventricle boundary is an edge in *both* CT
//! and MRI). For fixed `F` and moving `M` resampled onto the fixed grid,
//!
//! ```text
//! NGF(F, M) = (1/N) Â· Î£_x  (âˆ‡FÂ·âˆ‡M)Â² / ((|âˆ‡F|Â² + Î·_FÂ²)(|âˆ‡M|Â² + Î·_MÂ²))
//! ```
//!
//! Each term is `1` when the gradients are parallel **or anti-parallel** (so a
//! bright-CT / dark-MR edge still scores `1` â€” the squared dot product is
//! sign-invariant) and `0` where either side is flat. `Î·` is the edge-noise
//! scale (the mean masked gradient magnitude, per Haber & Modersitzki), which
//! suppresses flat-region noise. `NGF âˆˆ [0, 1]`; higher is better aligned, so the
//! metric returns `âˆ’NGF` as a minimization loss.
//!
//! This is a **gradient-free** metric (the gradients are spatial image gradients,
//! not autodiff gradients of the transform): it returns a scalar for the
//! derivative-free optimizers (CMA-ES, coordinate descent) that cross-modal rigid
//! registration uses, where intensity-MI hill-climbing from identity is unreliable.
//! Pre-masking the images (e.g. to a brain mask) focuses NGF on the shared rigid
//! structure, since flat masked-out regions contribute ~0.

pub mod fixed_prep;
pub mod native;
pub mod scalar;

pub(crate) use fixed_prep::NgfFixedPrep;
pub use scalar::center_gaussian_weight_field;

use super::trait_::Metric;
use ritk_image::tensor::Backend;
use ritk_image::tensor::{Tensor };
use ritk_image::Image;
use ritk_transform::Transform;

/// Normalized Gradient Fields metric (Haber & Modersitzki 2006).
///
/// Returns `âˆ’NGF âˆˆ [âˆ’1, 0]` as a loss to be minimized. Robust for cross-modal
/// (CTâ†”MRI) alignment where intensity MI/NCC are weak. See the [module docs](self).
pub struct NormalizedGradientField;

impl NormalizedGradientField {
    /// Create a new NGF metric (linear interpolation of the moving image, held by
    /// the per-registration `NgfFixedPrep`).
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for NormalizedGradientField {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend, const D: usize> Metric<B, D> for NormalizedGradientField {
    fn forward(
        &self,
        fixed: &Image<f32, B, D>,
        moving: &Image<f32, B, D>,
        transform: &impl Transform<B, D>,
    ) -> Tensor<f32, B> {
        let device = fixed.data().device();
        let ngf = self.ngf_value(fixed, moving, transform, None);
        // âˆ’NGF as a minimization loss.
        Tensor::<f32, B>::from_slice_on([1], &[-ngf], &device)
    }

    fn name(&self) -> &'static str {
        "NormalizedGradientField"
    }
}

impl NormalizedGradientField {
    /// Resample `moving` onto the `fixed` grid through `transform`, then return
    /// `NGF âˆˆ [0, 1]` over the `true` voxels of `mask` (or all if `None`). The
    /// masked path is used by the cross-modal rigid registration; the unmasked
    /// path backs [`Metric::forward`].
    pub(crate) fn ngf_value<B: Backend, const D: usize>(
        &self,
        fixed: &Image<f32, B, D>,
        moving: &Image<f32, B, D>,
        transform: &impl Transform<B, D>,
        mask: Option<&[bool]>,
    ) -> f32 {
        self.ngf_value_weighted(fixed, moving, transform, mask, None)
    }

    /// As [`ngf_value`](Self::ngf_value), but each masked voxel's contribution is
    /// scaled by `weights[flat]` (row-major, same length as the fixed image).
    /// A brain-centroid Gaussian weight (see [`center_gaussian_weight_field`])
    /// down-weights the high-gradient skull/scalp rim â€” which otherwise dominates
    /// the uniform NGF average and lets the optimiser ignore deep structures
    /// (ventricles, deep gray) â€” so the metric becomes sensitive to the central
    /// anatomy where rigid alignment is anatomically defined.
    pub(crate) fn ngf_value_weighted<B: Backend, const D: usize>(
        &self,
        fixed: &Image<f32, B, D>,
        moving: &Image<f32, B, D>,
        transform: &impl Transform<B, D>,
        mask: Option<&[bool]>,
        weights: Option<&[f32]>,
    ) -> f32 {
        // One-shot path (trait `forward`): build the fixed context and evaluate
        // once. The registration hot loop instead builds [`NgfFixedPrep`] ONCE and
        // calls [`NgfFixedPrep::eval`] per transform, reusing the fixed work.
        NgfFixedPrep::new(fixed, mask, weights).eval(moving, transform)
    }
}

#[cfg(test)]
mod tests;
