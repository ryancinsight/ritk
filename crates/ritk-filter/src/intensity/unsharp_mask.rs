//! Unsharp mask filter (ITK `UnsharpMaskingImageFilter` / ImageJ "Unsharp Mask" parity).
//!
//! # Mathematical Specification
//!
//! Given input I and blurred image B = G_Ïƒ âˆ— I (Gaussian convolution):
//!
//! ```text
//! mask(p)  = I(p) âˆ’ B(p)
//! output(p) = I(p)                                          if |mask(p)| < Ï„
//! output(p) = I(p) + a Â· (|mask(p)| âˆ’ Ï„) Â· sign(mask(p))  if |mask(p)| â‰¥ Ï„
//! ```
//!
//! where:
//! - Ïƒ is the spatial Gaussian standard deviation (per dimension, physical units)
//! - a (`amount`) is the sharpening strength in [0, âˆž)
//! - Ï„ (`threshold`) is the minimum absolute mask value required to trigger sharpening
//! - Output is optionally clamped to `[min(I), max(I)]`
//!
//! # Equivalence to ITK
//!
//! ITK `UnsharpMaskingImageFilter` uses the same formula with identical parameter
//! semantics:
//! - `Sigmas` â†’ `sigmas` (per-dimension Ïƒ; single value is broadcast)
//! - `Amount` â†’ `amount` (default 0.5)
//! - `Threshold` â†’ `threshold` (default 0.0)
//! - `Clamp` â†’ `clamp` (default true)
//!
//! # ImageJ "Unsharp Mask"
//!
//! ImageJ uses the special case Ï„ = 0, a âˆˆ (0, 1], with Gaussian radius in pixels.
//! Equivalent: set `threshold = 0.0`, `sigmas` to the ImageJ radius (1 pixel = 1mm if
//! `use_image_spacing = false`), and `amount` to the ImageJ mask weight p where
//! `amount = p / (1 âˆ’ p)` (ImageJ sharpens as `I + pÂ·mask / (1âˆ’p)` but the conventions
//! differ; for direct equivalence set `amount = p` and disable clamping).
//!
//! # Precision
//!
//! All arithmetic is performed in `f32` matching the `Image<B,3>` storage type.
//! The intermediate Gaussian blur uses the recursive (Deriche) Gaussian
//! (`smoothing_recursive_gaussian`) â€” the smoother ITK/SimpleITK `UnsharpMask`
//! uses (`SmoothingRecursiveGaussian`), float-exact to it.
//!
//! # Complexity
//!
//! O(N Â· r) per dimension where r = âŒˆ3Ïƒ/hâŒ‰ (kernel radius in pixels, h = voxel size).
//!
//! # References
//!
//! - Crane, R. (1997). A simplified approach to image processing. Ch. 6: Unsharp Masking.
//! - Maini, R., & Aggarwal, H. (2010). A comprehensive review of image enhancement techniques.
//!   *Journal of Computing, 2*(3), 8â€“13.
//! - ITK Software Guide 4th Ed., Â§6.5.2 UnsharpMaskingImageFilter.

use crate::edge::GaussianSigma;
use crate::recursive_gaussian::smoothing_recursive_gaussian_vals;
use anyhow::Result;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};
use serde::{Deserialize, Serialize};

/// Whether to clamp the unsharp mask output to the input intensity range.
///
/// - `NoClamp`: output may exceed `[min(I), max(I)]`.
/// - `ClampToInputRange`: output is restricted to `[min(I), max(I)]` (ITK default).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ClampPolicy {
    /// Do not clamp; sharpening can produce values outside the input range.
    #[default]
    NoClamp,
    /// Clamp output to `[min(I), max(I)]`.
    ClampToInputRange,
}

/// Unsharp mask sharpening filter for 3-D volumes.
///
/// Sharpens edges and fine detail by adding back a scaled, thresholded version
/// of the high-frequency component `I âˆ’ G_Ïƒâˆ—I` to the original image.
///
/// # Default parameters
/// - `sigmas = [1.0]` â€” isotropic Ïƒ = 1.0 physical unit (mm)
/// - `amount = 0.5` â€” ITK default sharpening strength
/// - `threshold = 0.0` â€” no minimum edge contrast requirement
/// - `clamp = true` â€” ITK default: clamp output to input intensity range
///
/// # Invariants
/// - `amount = 0` â†’ output identical to input (no sharpening applied)
/// - uniform input â†’ output identical to input (mask = 0 everywhere)
/// - `threshold > max(|mask|)` â†’ output identical to input
/// - `clamp = true` â†’ `output(p) âˆˆ [min(I), max(I)]` for all p
pub struct UnsharpMaskFilter {
    /// Per-dimension Gaussian sigma in physical units (mm).
    /// A single-element `Vec` is broadcast to all dimensions.
    pub sigmas: Vec<GaussianSigma>,
    /// Sharpening strength. Typical range [0.0, 5.0]. ITK default: 0.5.
    pub amount: f64,
    /// Minimum absolute mask value to trigger sharpening. ITK default: 0.0.
    pub threshold: f64,
    /// Clamp policy for the output. ITK default: `ClampToInputRange`.
    pub clamp: ClampPolicy,
}

impl Default for UnsharpMaskFilter {
    fn default() -> Self {
        Self {
            sigmas: vec![GaussianSigma::new_unchecked(1.0)],
            amount: 0.5,
            threshold: 0.0,
            clamp: ClampPolicy::ClampToInputRange,
        }
    }
}

impl UnsharpMaskFilter {
    /// Construct an unsharp mask filter.
    ///
    /// # Arguments
    /// * `sigmas`    â€” per-dimension Gaussian sigma in physical units; broadcast if length 1.
    /// * `amount`    â€” sharpening strength (ITK default 0.5).
    /// * `threshold` â€” minimum |mask| to trigger sharpening (ITK default 0.0).
    /// * `clamp` â€” clamp output to input range (ITK default `ClampToInputRange`).
    pub fn new(
        sigmas: Vec<GaussianSigma>,
        amount: f64,
        threshold: f64,
        clamp: ClampPolicy,
    ) -> Self {
        Self {
            sigmas,
            amount,
            threshold,
            clamp,
        }
    }

    /// Builder-style setter for the clamp policy.
    pub fn with_clamp(mut self, clamp: ClampPolicy) -> Self {
        self.clamp = clamp;
        self
    }

    /// Apply the unsharp mask filter to a 3-D image.
    ///
    /// # Algorithm
    /// 1. Compute `B = DiscreteGaussianFilter(variance = ÏƒÂ²) Â· I`.
    /// 2. Compute per-voxel mask `m = I âˆ’ B`.
    /// 3. Apply `output(p) = I(p) + amount Â· max(0, |m(p)| âˆ’ threshold) Â· sign(m(p))`.
    /// 4. If `clamp`, restrict each output voxel to `[min(I), max(I)]`.
    ///
    /// # Errors
    /// Returns `Err` if tensor data cannot be extracted as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        let (input, dims) = extract_vec_infallible(image);
        let sp = image.spacing();
        let output = self.unsharp_mask_flat(&input, dims, [sp[0], sp[1], sp[2]]);
        Ok(rebuild(output, dims, image))
    }

    /// Coeus-native sister of [`UnsharpMaskFilter::apply`].
    ///
    /// Runs the identical recursive-Gaussian unsharp-mask formula via the shared
    /// `unsharp_mask_flat` host core on the image's
    /// contiguous host buffer, so the result is bitwise-identical to the Burn
    /// path. No Burn tensor is constructed. Spatial metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let sp = image.spacing();
        let spacing = [sp[0], sp[1], sp[2]];
        crate::native_support::map_flat_image(image, backend, |vals, dims| {
            self.unsharp_mask_flat(vals, dims, spacing)
        })
    }

    /// Substrate-agnostic host core: recursive-Gaussian blur, thresholded
    /// mask-add, and optional input-range clamp on a flat z-major buffer. Single
    /// source of truth for the Burn [`apply`](Self::apply) and Coeus-native
    /// [`apply_native`](Self::apply_native) paths.
    fn unsharp_mask_flat(&self, input: &[f32], dims: [usize; 3], spacing: [f64; 3]) -> Vec<f32> {
        let n = input.len();

        // Blur via the recursive (Deriche) Gaussian — the smoother ITK/SimpleITK
        // `UnsharpMask` uses (`SmoothingRecursiveGaussian`), not the discrete
        // Gaussian. Per-dimension physical sigma, broadcast from the last entry.
        let sigmas: Vec<f64> = self.sigmas.iter().map(|s| s.get()).collect();
        let blurred = smoothing_recursive_gaussian_vals(input.to_vec(), dims, spacing, &sigmas);

        let amount = self.amount as f32;
        let threshold = self.threshold as f32;

        // Derive input intensity range for optional clamping.
        let (v_min, v_max) = match self.clamp {
            ClampPolicy::ClampToInputRange => {
                let (mn, mx) = moirai::fold_reduce_with::<moirai::Adaptive, _, _, _, _>(
                    n,
                    || (f32::INFINITY, f32::NEG_INFINITY),
                    |(mn, mx), i| {
                        let v = input[i];
                        if v.is_finite() {
                            (mn.min(v), mx.max(v))
                        } else {
                            (mn, mx)
                        }
                    },
                    |(a_mn, a_mx), (b_mn, b_mx)| (a_mn.min(b_mn), a_mx.max(b_mx)),
                );
                (mn, mx)
            }
            ClampPolicy::NoClamp => (f32::NEG_INFINITY, f32::INFINITY),
        };

        // Apply unsharp mask formula.
        //
        // For each voxel p:
        //   mask = I(p) − B(p)
        //   |mask| < threshold  → output = I(p)
        //   |mask| ≥ threshold  → output = I(p) + amount · (|mask| − threshold) · sign(mask)
        //
        // Then clamp to [v_min, v_max] if requested.
        let blurred_ref = &blurred;
        moirai::map_collect_index_with::<moirai::Adaptive, _, _>(n, |i| {
            let inp = input[i];
            let mask = inp - blurred_ref[i];
            let abs_mask = mask.abs();
            let sharpened = if abs_mask < threshold {
                inp
            } else {
                inp + amount * (abs_mask - threshold) * mask.signum()
            };
            sharpened.clamp(v_min, v_max)
        })
    }
}

#[cfg(test)]
#[path = "tests_unsharp_mask.rs"]
mod tests;

#[cfg(test)]
mod tests_native {
    use super::{ClampPolicy, GaussianSigma, UnsharpMaskFilter};
    use crate::native_support::{assert_native_matches_burn, make_native_image, native_vals};
    use coeus_core::SequentialBackend;

    fn filter() -> UnsharpMaskFilter {
        UnsharpMaskFilter::new(
            vec![GaussianSigma::new_unchecked(1.0)],
            1.5,
            0.0,
            ClampPolicy::NoClamp,
        )
    }

    #[test]
    fn matches_burn() {
        let vals: Vec<f32> = (0..60).map(|i| ((i * 3) % 11) as f32).collect();
        assert_native_matches_burn(
            vals,
            [3, 4, 5],
            |img| filter().apply(img).expect("burn unsharp mask"),
            |img, backend| filter().apply_native(img, backend),
        );
    }

    #[test]
    fn oracle_uniform_input_unchanged() {
        // Blur of a uniform field is the field itself → mask = 0 everywhere →
        // output == input (no sharpening), independent of `amount`.
        let img = make_native_image(vec![9.0f32; 27], [3, 3, 3]);
        let out = filter()
            .apply_native(&img, &SequentialBackend)
            .expect("native unsharp mask");
        for &v in &native_vals(&out) {
            assert!(
                (v - 9.0).abs() < 1e-4,
                "uniform input must be preserved, got {v}"
            );
        }
    }
}
