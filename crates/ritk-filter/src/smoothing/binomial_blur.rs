//! Binomial blur smoothing filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Applies the separable 1-D binomial kernel `[¼, ½, ¼]` independently along
//! each axis, repeated `repetitions` times:
//!
//! ```text
//! O = (B_x ∘ B_y ∘ B_z)^repetitions (I)
//! B_d(I)(…, i, …) = ¼·I(…, i−1, …) + ½·I(…, i, …) + ¼·I(…, i+1, …)
//! ```
//!
//! The kernel approximates a Gaussian with variance `repetitions/2` voxels per
//! axis (central-limit). Because each pass is a linear separable convolution,
//! the per-axis and per-repetition order is irrelevant to the result.
//!
//! # Boundary
//!
//! ITK `BinomialBlurImageFilter` uses an **asymmetric** boundary (a documented
//! quirk of the filter): the low end reflects (`I[−1] = I[1]`) while the high
//! end clamps (`I[N] = I[N−1]`). Verified against `sitk.BinomialBlur`: a
//! constant image is preserved; a low-edge impulse `[4,0,0,0,0]` blurs to
//! `[2,1,0,0,0]` (`out[0]` = ½·I₀+½·I₁, reflect) but a high-edge impulse
//! `[0,0,0,0,4]` blurs to `[0,0,0,1,3]` (out[N−1] = ¼·I_{N−2}+¾·I_{N−1}, clamp).
//! Zero-flux would give 3 at the low edge, and symmetric reflect would give 2 at
//! the high edge — only this asymmetric rule reproduces both.
//!
//! # ITK parity
//!
//! Corresponds to `itk::BinomialBlurImageFilter`. ITK default
//! `repetitions = 1`. `repetitions = 0` is the identity.
//!
//! # Complexity
//!
//! O(repetitions · D · N) — three linear passes per repetition.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Binomial blur smoothing filter (`itk::BinomialBlurImageFilter`).
#[derive(Debug, Clone, Copy)]
pub struct BinomialBlurImageFilter {
    /// Number of times the `[¼, ½, ¼]` kernel is applied along each axis.
    /// `0` is the identity. ITK default `1`.
    pub repetitions: usize,
}

impl BinomialBlurImageFilter {
    /// Construct with the given number of repetitions.
    pub fn new(repetitions: usize) -> Self {
        Self { repetitions }
    }

    /// Apply the binomial blur to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (mut vals, dims) = extract_vec_infallible(image);
        for _ in 0..self.repetitions {
            for axis in 0..3 {
                vals = blur_axis(&vals, dims, axis);
            }
        }
        rebuild(vals, dims, image)
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
        let (mut vals, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        for _ in 0..self.repetitions {
            for axis in 0..3 {
                vals = blur_axis(&vals, dims, axis);
            }
        }
        crate::native_support::rebuild_image(vals, dims, image, backend)
    }
}

/// One `[¼, ½, ¼]` pass along `axis` with reflect (mirror) boundary.
fn blur_axis(vals: &[f32], dims: [usize; 3], axis: usize) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let (len, stride) = match axis {
        0 => (nz, ny * nx),
        1 => (ny, nx),
        _ => (nx, 1),
    };
    if len < 2 {
        // A size-1 axis has no neighbours; reflect makes the pass the identity.
        return vals.to_vec();
    }

    // ITK BinomialBlur boundary (asymmetric): left of index 0 reflects to index
    // 1; right of the last index clamps to itself.
    let left_idx = |p: usize| -> usize {
        if p == 0 {
            1
        } else {
            p - 1
        }
    };
    let right_idx = |p: usize| -> usize {
        if p == len - 1 {
            len - 1
        } else {
            p + 1
        }
    };

    let mut out = vec![0.0f32; vals.len()];
    let (n_outer, n_mid) = match axis {
        0 => (ny, nx),
        1 => (nz, nx),
        _ => (nz, ny),
    };
    for a in 0..n_outer {
        for b in 0..n_mid {
            let base = match axis {
                0 => a * nx + b,
                1 => a * ny * nx + b,
                _ => a * ny * nx + b * nx,
            };
            for p in 0..len {
                let l = vals[base + left_idx(p) * stride];
                let c = vals[base + p * stride];
                let r = vals[base + right_idx(p) * stride];
                out[base + p * stride] = 0.25 * l + 0.5 * c + 0.25 * r;
            }
        }
    }
    out
}

#[cfg(test)]
#[path = "tests_binomial_blur.rs"]
mod tests_binomial_blur;
