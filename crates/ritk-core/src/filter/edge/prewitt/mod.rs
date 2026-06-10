//! 3-D Prewitt gradient filter via separable convolution.
//!
//! # Mathematical Specification
//!
//! The 3-D Prewitt operator estimates spatial derivatives using separable
//! convolution kernels that combine a derivative operator with uniform
//! smoothing. For each axis direction, the Prewitt kernel is the outer
//! product of three 1-D kernels:
//!
//! d = [-1, 0, 1] (derivative)
//! s = [ 1, 1, 1] (uniform smoothing)
//!
//! For the x-derivative: K_x = s ⊗ s ⊗ d (smooth z, smooth y, derivative x)
//! For the y-derivative: K_y = s ⊗ d ⊗ s (smooth z, derivative y, smooth x)
//! For the z-derivative: K_z = d ⊗ s ⊗ s (derivative z, smooth y, smooth x)
//!
//! ## Normalization
//!
//! factor = 2 · h · 3 · 3 = 18 · h
//!
//! # Architecture
//!
//! `PrewittFilter` is a type alias for [`SeparableGradientFilter<PrewittKernel>`],
//! which is the canonical monomorphized implementation. All convolution logic
//! lives in [`super::separable_gradient`].
//!
//! ## Sobel vs. Prewitt
//!
//! - Sobel uses binomial smoothing [1, 2, 1] (weights 1-2-1).
//! - Prewitt uses uniform smoothing [1, 1, 1] (weights 1-1-1).
//! - Prewitt is computationally cheaper (integer arithmetic with sum=3 vs. 4).
//! - Sobel provides slight additional smoothing perpendicular to the
//!   derivative axis, suppressing diagonal noise more aggressively.
//!
//! # Reference
//!
//! Prewitt, J. M. S. (1970). "Object enhancement and extraction."
//! In *Picture Processing and Psychopictorics*, Academic Press.

use super::separable_gradient::{PrewittKernel, SeparableGradientFilter};

/// 3-D Prewitt gradient filter.
///
/// Computes spatial derivatives using the 3-D Prewitt operator, which combines
/// central-difference derivative estimation with uniform smoothing along the
/// two orthogonal axes. The output is normalized to physical gradient units
/// (intensity per unit spacing).
///
/// ## Kernel structure
///
/// For derivative axis `a` with orthogonal axes `b`, `c`:
///
/// ```text
/// K_a[db][dc][da] = s[db] · s[dc] · d[da]
/// where d = [-1, 0, 1], s = [1, 1, 1]
/// ```
///
/// ## Normalization factor derivation
///
/// | Component | Factor | Source |
/// |------------------|--------|----------------------------------------------|
/// | Central diff | 2·h | [-1,0,1] spans 2 voxels of spacing h |
/// | Smoothing axis 1 | 3 | sum(\[1,1,1\]) |
/// | Smoothing axis 2 | 3 | sum(\[1,1,1\]) |
/// | **Total** | 18·h | |
pub type PrewittFilter = SeparableGradientFilter<PrewittKernel>;

#[cfg(test)]
mod tests;
