//! Unsharp mask filter (ITK `UnsharpMaskingImageFilter` / ImageJ "Unsharp Mask" parity).
//!
//! # Mathematical Specification
//!
//! Given input I and blurred image B = G_ÃÆ’ Ã¢Ë†â€” I (Gaussian convolution):
//!
//! ```text
//! mask(p)  = I(p) Ã¢Ë†â€™ B(p)
//! output(p) = I(p)                                          if |mask(p)| < Ãâ€ž
//! output(p) = I(p) + a Ã‚Â· (|mask(p)| Ã¢Ë†â€™ Ãâ€ž) Ã‚Â· sign(mask(p))  if |mask(p)| Ã¢â€°Â¥ Ãâ€ž
//! ```
//!
//! where:
//! - ÃÆ’ is the spatial Gaussian standard deviation (per dimension, physical units)
//! - a (`amount`) is the sharpening strength in [0, Ã¢Ë†Å¾)
//! - Ãâ€ž (`threshold`) is the minimum absolute mask value required to trigger sharpening
//! - Output is optionally clamped to `[min(I), max(I)]`
//!
//! # Equivalence to ITK
//!
//! ITK `UnsharpMaskingImageFilter` uses the same formula with identical parameter
//! semantics:
//! - `Sigmas` Ã¢â€ â€™ `sigmas` (per-dimension ÃÆ’; single value is broadcast)
//! - `Amount` Ã¢â€ â€™ `amount` (default 0.5)
//! - `Threshold` Ã¢â€ â€™ `threshold` (default 0.0)
//! - `Clamp` Ã¢â€ â€™ `clamp` (default true)
//!
//! # ImageJ "Unsharp Mask"
//!
//! ImageJ uses the special case Ãâ€ž = 0, a Ã¢Ë†Ë† (0, 1], with Gaussian radius in pixels.
//! Equivalent: set `threshold = 0.0`, `sigmas` to the ImageJ radius (1 pixel = 1mm if
//! `use_image_spacing = false`), and `amount` to the ImageJ mask weight p where
//! `amount = p / (1 Ã¢Ë†â€™ p)` (ImageJ sharpens as `I + pÃ‚Â·mask / (1Ã¢Ë†â€™p)` but the conventions
//! differ; for direct equivalence set `amount = p` and disable clamping).
//!
//! # Precision
//!
//! All arithmetic is performed in `f32` matching the `Image<f32, B,3>` storage type.
//! The intermediate Gaussian blur uses the recursive (Deriche) Gaussian
//! (`smoothing_recursive_gaussian`) Ã¢â‚¬â€ the smoother ITK/SimpleITK `UnsharpMask`
//! uses (`SmoothingRecursiveGaussian`), float-exact to it.
//!
//! # Complexity
//!
//! O(N Ã‚Â· r) per dimension where r = Ã¢Å’Ë†3ÃÆ’/hÃ¢Å’â€° (kernel radius in pixels, h = voxel size).
//!
//! # References
//!
//! - Crane, R. (1997). A simplified approach to image processing. Ch. 6: Unsharp Masking.
//! - Maini, R., & Aggarwal, H. (2010). A comprehensive review of image enhancement techniques.
//!   *Journal of Computing, 2*(3), 8Ã¢â‚¬â€œ13.
//! - ITK Software Guide 4th Ed., Ã‚Â§6.5.2 UnsharpMaskingImageFilter.

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
/// of the high-frequency component `I Ã¢Ë†â€™ G_ÃÆ’Ã¢Ë†â€”I` to the original image.
///
/// # Default parameters
/// - `sigmas = [1.0]` Ã¢â‚¬â€ isotropic ÃÆ’ = 1.0 physical unit (mm)
/// - `amount = 0.5` Ã¢â‚¬â€ ITK default sharpening strength
/// - `threshold = 0.0` Ã¢â‚¬â€ no minimum edge contrast requirement
/// - `clamp = true` Ã¢â‚¬â€ ITK default: clamp output to input intensity range
///
/// # Invariants
/// - `amount = 0` Ã¢â€ â€™ output identical to input (no sharpening applied)
/// - uniform input Ã¢â€ â€™ output identical to input (mask = 0 everywhere)
/// - `threshold > max(|mask|)` Ã¢â€ â€™ output identical to input
/// - `clamp = true` Ã¢â€ â€™ `output(p) Ã¢Ë†Ë† [min(I), max(I)]` for all p
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
    /// * `sigmas`    Ã¢â‚¬â€ per-dimension Gaussian sigma in physical units; broadcast if length 1.
    /// * `amount`    Ã¢â‚¬â€ sharpening strength (ITK default 0.5).
    /// * `threshold` Ã¢â‚¬â€ minimum |mask| to trigger sharpening (ITK default 0.0).
    /// * `clamp` Ã¢â‚¬â€ clamp output to input range (ITK default `ClampToInputRange`).
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
    /// 1. Compute `B = DiscreteGaussianFilter(variance = ÃÆ’Ã‚Â²) Ã‚Â· I`.
    /// 2. Compute per-voxel mask `m = I Ã¢Ë†â€™ B`.
    /// 3. Apply `output(p) = I(p) + amount Ã‚Â· max(0, |m(p)| Ã¢Ë†â€™ threshold) Ã‚Â· sign(m(p))`.
    /// 4. If `clamp`, restrict each output voxel to `[min(I), max(I)]`.
    ///
    /// # Errors
    /// Returns `Err` if tensor data cannot be extracted as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> Result<Image<f32, B, 3>> {
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

        // Blur via the recursive (Deriche) Gaussian â€” the smoother ITK/SimpleITK
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
        //   mask = I(p) âˆ’ B(p)
        //   |mask| < threshold  â†’ output = I(p)
        //   |mask| â‰¥ threshold  â†’ output = I(p) + amount Â· (|mask| âˆ’ threshold) Â· sign(mask)
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
    use crate::native_support::{assert_coeus_matches_coeus, make_native_image, native_vals};
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
        assert_coeus_matches_coeus(
            vals,
            [3, 4, 5],
            |img| filter().apply(img).expect("burn unsharp mask"),
            |img, backend| filter().apply_native(img, backend),
        );
    }

    #[test]
    fn oracle_uniform_input_unchanged() {
        // Blur of a uniform field is the field itself â†’ mask = 0 everywhere â†’
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
