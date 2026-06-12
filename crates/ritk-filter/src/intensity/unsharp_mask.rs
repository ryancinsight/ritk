//! Unsharp mask filter (ITK `UnsharpMaskingImageFilter` / ImageJ "Unsharp Mask" parity).
//!
//! # Mathematical Specification
//!
//! Given input I and blurred image B = G_σ ∗ I (Gaussian convolution):
//!
//! ```text
//! mask(p)  = I(p) − B(p)
//! output(p) = I(p)                                          if |mask(p)| < τ
//! output(p) = I(p) + a · (|mask(p)| − τ) · sign(mask(p))  if |mask(p)| ≥ τ
//! ```
//!
//! where:
//! - σ is the spatial Gaussian standard deviation (per dimension, physical units)
//! - a (`amount`) is the sharpening strength in [0, ∞)
//! - τ (`threshold`) is the minimum absolute mask value required to trigger sharpening
//! - Output is optionally clamped to `[min(I), max(I)]`
//!
//! # Equivalence to ITK
//!
//! ITK `UnsharpMaskingImageFilter` uses the same formula with identical parameter
//! semantics:
//! - `Sigmas` → `sigmas` (per-dimension σ; single value is broadcast)
//! - `Amount` → `amount` (default 0.5)
//! - `Threshold` → `threshold` (default 0.0)
//! - `Clamp` → `clamp` (default true)
//!
//! # ImageJ "Unsharp Mask"
//!
//! ImageJ uses the special case τ = 0, a ∈ (0, 1], with Gaussian radius in pixels.
//! Equivalent: set `threshold = 0.0`, `sigmas` to the ImageJ radius (1 pixel = 1mm if
//! `use_image_spacing = false`), and `amount` to the ImageJ mask weight p where
//! `amount = p / (1 − p)` (ImageJ sharpens as `I + p·mask / (1−p)` but the conventions
//! differ; for direct equivalence set `amount = p` and disable clamping).
//!
//! # Precision
//!
//! All arithmetic is performed in `f32` matching the `Image<B,3>` storage type.
//! The intermediate Gaussian blur uses `DiscreteGaussianFilter` (variance parameterised)
//! with replicate boundary conditions — the same as ITK's discrete Gaussian.
//!
//! # Complexity
//!
//! O(N · r) per dimension where r = ⌈3σ/h⌉ (kernel radius in pixels, h = voxel size).
//!
//! # References
//!
//! - Crane, R. (1997). A simplified approach to image processing. Ch. 6: Unsharp Masking.
//! - Maini, R., & Aggarwal, H. (2010). A comprehensive review of image enhancement techniques.
//!   *Journal of Computing, 2*(3), 8–13.
//! - ITK Software Guide 4th Ed., §6.5.2 UnsharpMaskingImageFilter.

use crate::discrete_gaussian::DiscreteGaussianFilter;
use crate::edge::GaussianSigma;
use anyhow::Result;
use burn::tensor::backend::Backend;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};
use ritk_image::Image;
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
/// of the high-frequency component `I − G_σ∗I` to the original image.
///
/// # Default parameters
/// - `sigmas = [1.0]` — isotropic σ = 1.0 physical unit (mm)
/// - `amount = 0.5` — ITK default sharpening strength
/// - `threshold = 0.0` — no minimum edge contrast requirement
/// - `clamp = true` — ITK default: clamp output to input intensity range
///
/// # Invariants
/// - `amount = 0` → output identical to input (no sharpening applied)
/// - uniform input → output identical to input (mask = 0 everywhere)
/// - `threshold > max(|mask|)` → output identical to input
/// - `clamp = true` → `output(p) ∈ [min(I), max(I)]` for all p
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
    /// * `sigmas`    — per-dimension Gaussian sigma in physical units; broadcast if length 1.
    /// * `amount`    — sharpening strength (ITK default 0.5).
    /// * `threshold` — minimum |mask| to trigger sharpening (ITK default 0.0).
    /// * `clamp` — clamp output to input range (ITK default `ClampToInputRange`).
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
    /// 1. Compute `B = DiscreteGaussianFilter(variance = σ²) · I`.
    /// 2. Compute per-voxel mask `m = I − B`.
    /// 3. Apply `output(p) = I(p) + amount · max(0, |m(p)| − threshold) · sign(m(p))`.
    /// 4. If `clamp`, restrict each output voxel to `[min(I), max(I)]`.
    ///
    /// # Errors
    /// Returns `Err` if tensor data cannot be extracted as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        let (input, dims) = extract_vec_infallible(image);
        let n = input.len();

        // Compute blurred image via DiscreteGaussianFilter (variance = sigma², computed internally).
        let blur = DiscreteGaussianFilter::<B>::new(self.sigmas.clone()).apply(image);
        let (blurred, _) = extract_vec_infallible(&blur);

        let amount = self.amount as f32;
        let threshold = self.threshold as f32;

        // Derive input intensity range for optional clamping.
        let (v_min, v_max) = match self.clamp {
            ClampPolicy::ClampToInputRange => {
                let mut mn = f32::INFINITY;
                let mut mx = f32::NEG_INFINITY;
                for &v in &input {
                    if v.is_finite() {
                        mn = mn.min(v);
                        mx = mx.max(v);
                    }
                }
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
        let mut output: Vec<f32> = Vec::with_capacity(n);
        for i in 0..n {
            let inp = input[i];
            let mask = inp - blurred[i];
            let abs_mask = mask.abs();
            let sharpened = if abs_mask < threshold {
                inp
            } else {
                inp + amount * (abs_mask - threshold) * mask.signum()
            };
            output.push(sharpened.clamp(v_min, v_max));
        }

        Ok(rebuild(output, dims, image))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;
    use ritk_image::Image;
    use ritk_spatial::{Direction, Point, Spacing};

    type B = NdArray<f32>;

    fn make_image(vals: Vec<f32>, depth: usize, rows: usize, cols: usize) -> Image<B, 3> {
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let td = TensorData::new(vals, Shape::new([depth, rows, cols]));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn image_vals(img: &Image<B, 3>) -> Vec<f32> {
        img.data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .expect("f32 extraction failed")
            .to_vec()
    }

    /// Invariant: uniform input → output = input.
    ///
    /// Proof: G_σ * constant = constant (Gaussian kernel sums to 1),
    /// so mask = 0 everywhere, |mask| = 0 < threshold for any threshold ≥ 0,
    /// output = input.
    #[test]
    fn uniform_input_is_identity() {
        let img = make_image(vec![3.0_f32; 2 * 4 * 4], 2, 4, 4);
        let filter = UnsharpMaskFilter::new(
            vec![GaussianSigma::new_unchecked(1.0)],
            2.0,
            0.0,
            ClampPolicy::ClampToInputRange,
        );
        let out = filter.apply::<B>(&img).expect("apply failed");
        let vals = image_vals(&out);
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                (v - 3.0_f32).abs() < 1e-4,
                "voxel {i}: expected 3.0, got {v} (uniform input violated identity)"
            );
        }
    }

    /// Invariant: amount = 0 → output = input identically.
    ///
    /// Proof: output = I + 0 · (...) = I for all p.
    #[test]
    fn amount_zero_is_exact_identity() {
        // Non-trivial image with gradient values.
        let vals: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let img = make_image(vals.clone(), 2, 4, 4);
        let filter = UnsharpMaskFilter::new(
            vec![GaussianSigma::new_unchecked(1.0)],
            0.0,
            0.0,
            ClampPolicy::NoClamp,
        );
        let out = filter.apply::<B>(&img).expect("apply failed");
        let out_vals = image_vals(&out);
        for (i, (&expected, &got)) in vals.iter().zip(out_vals.iter()).enumerate() {
            assert!(
                (expected - got).abs() < 1e-5,
                "voxel {i}: expected {expected}, got {got} (amount=0 identity violated)"
            );
        }
    }

    /// Invariant: threshold > all |mask| values → output = input.
    ///
    /// Construction: use a constant image (mask = 0 everywhere) with threshold = 100.0;
    /// since |mask| = 0 < 100.0 for all voxels, sharpening is never triggered.
    #[test]
    fn threshold_suppresses_all_sharpening() {
        // Constant image → mask = 0 < threshold = 100.0 everywhere.
        let img = make_image(vec![42.0_f32; 2 * 3 * 3], 2, 3, 3);
        let filter = UnsharpMaskFilter::new(
            vec![GaussianSigma::new_unchecked(1.0)],
            5.0,
            100.0,
            ClampPolicy::NoClamp,
        );
        let out = filter.apply::<B>(&img).expect("apply failed");
        let out_vals = image_vals(&out);
        for (i, &v) in out_vals.iter().enumerate() {
            assert!(
                (v - 42.0_f32).abs() < 1e-4,
                "voxel {i}: expected 42.0, got {v} (threshold suppression violated)"
            );
        }
    }

    /// Invariant: clamp=true → output(p) ≤ max(input) for all p.
    ///
    /// Construction: step edge [0, 0, ..., 1, 1, ...] with large amount (5.0);
    /// without clamping, edge voxels would exceed 1.0. Clamping enforces ≤ 1.0.
    #[test]
    fn clamp_enforces_upper_bound() {
        // 1×1×8 step edge: [0,0,0,0,1,1,1,1]
        let vals: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let img = make_image(vals.clone(), 1, 1, 8);
        // Large amount to ensure edge overshoot without clamping.
        let filter = UnsharpMaskFilter::new(
            vec![GaussianSigma::new_unchecked(0.5)],
            5.0,
            0.0,
            ClampPolicy::ClampToInputRange,
        );
        let out = filter.apply::<B>(&img).expect("apply failed");
        let out_vals = image_vals(&out);
        let input_max = 1.0_f32;
        let input_min = 0.0_f32;
        for (i, &v) in out_vals.iter().enumerate() {
            assert!(
                v <= input_max + 1e-5,
                "voxel {i}: clamp=true violated upper bound: {v} > {input_max}"
            );
            assert!(
                v >= input_min - 1e-5,
                "voxel {i}: clamp=true violated lower bound: {v} < {input_min}"
            );
        }
    }

    /// Invariant: clamp=false → sharpening can produce values outside [min(I), max(I)].
    ///
    /// Construction: same step edge with large amount; at least one edge voxel must
    /// exceed 1.0 (the input maximum), proving the unsharp mask is genuinely applied.
    #[test]
    fn no_clamp_allows_overshoot() {
        // 1×1×8 step edge: [0,0,0,0,1,1,1,1]
        let vals: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let img = make_image(vals.clone(), 1, 1, 8);
        let filter = UnsharpMaskFilter::new(
            vec![GaussianSigma::new_unchecked(0.5)],
            5.0,
            0.0,
            ClampPolicy::NoClamp,
        );
        let out = filter.apply::<B>(&img).expect("apply failed");
        let out_vals = image_vals(&out);
        // At the step boundary (positions 4–5), the sharpened output must exceed 1.0.
        let any_above_max = out_vals.iter().any(|&v| v > 1.0 + 1e-5);
        assert!(
            any_above_max,
            "no_clamp_allows_overshoot: expected at least one voxel > 1.0 at step edge, \
             got max = {:.6}. The unsharp mask formula is not applying sharpening.",
            out_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );
    }

    /// Invariant: spatial metadata (origin, spacing, direction) is preserved.
    #[test]
    fn spatial_metadata_preserved() {
        use nalgebra::SMatrix;
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let td = TensorData::new(vec![1.0_f32; 2 * 3 * 3], Shape::new([2, 3, 3]));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        let origin = Point::new([10.0, 20.0, 30.0]);
        let spacing = Spacing::new([1.5, 2.0, 0.75]);
        let dir = Direction::<3>(SMatrix::<f64, 3, 3>::identity());
        let img = Image::new(tensor, origin, spacing, dir);
        let filter = UnsharpMaskFilter::default();
        let out = filter.apply::<B>(&img).expect("apply failed");
        assert_eq!(out.origin(), img.origin(), "origin changed");
        assert_eq!(out.spacing(), img.spacing(), "spacing changed");
        assert_eq!(out.direction(), img.direction(), "direction changed");
    }

    /// Invariant: sharpening genuinely increases contrast near step edges.
    ///
    /// For a step edge [0.0, 1.0] embedded in a 1×1×4 image, after sharpening with
    /// threshold=0 and amount>0, the difference between the edge voxels must be
    /// strictly greater than in the input.
    #[test]
    fn sharpening_increases_edge_contrast() {
        // 1×1×4 step: [0, 0, 1, 1]
        let vals: Vec<f32> = vec![0.0, 0.0, 1.0, 1.0];
        let img = make_image(vals, 1, 1, 4);
        // amount=2.0, no clamping so we can observe the actual sharpened values.
        let filter = UnsharpMaskFilter::new(
            vec![GaussianSigma::new_unchecked(0.5)],
            2.0,
            0.0,
            ClampPolicy::NoClamp,
        );
        let out = filter.apply::<B>(&img).expect("apply failed");
        let out_vals = image_vals(&out);
        // The output step contrast (max − min) must be > input contrast (1.0 − 0.0 = 1.0).
        let out_min = out_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let out_max = out_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let out_contrast = out_max - out_min;
        assert!(
            out_contrast > 1.0,
            "sharpening did not increase edge contrast: \
             output contrast = {out_contrast:.4} (expected > 1.0, input contrast = 1.0)"
        );
    }
}
