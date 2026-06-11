//! 3-D Sobel gradient filter via separable convolution.
//!
//! # Mathematical Specification
//!
//! The 3-D Sobel operator estimates spatial derivatives using separable
//! convolution kernels that combine a derivative operator with smoothing.
//! For each axis direction, the Sobel kernel is the outer product of three
//! 1-D kernels:
//!
//! d = [-1, 0, 1] (derivative)
//! s = [ 1, 2, 1] (smoothing)
//!
//! For the x-derivative: K_x = s ⊗ s ⊗ d (smooth z, smooth y, derivative x)
//! For the y-derivative: K_y = s ⊗ d ⊗ s (smooth z, derivative y, smooth x)
//! For the z-derivative: K_z = d ⊗ s ⊗ s (derivative z, smooth y, smooth x)
//!
//! Each 3×3×3 kernel is applied via three sequential 1-D convolutions
//! with replicate (clamp) boundary padding.
//!
//! ## Normalization
//!
//! The raw convolution output is normalized to approximate the true spatial
//! gradient in physical units. The normalization factor for each component is:
//!
//! factor = 2 · h · 4 · 4 = 32 · h
//!
//! where h is the physical spacing along the derivative axis. The factor of 2·h
//! accounts for the central-difference step size (the derivative kernel
//! [-1, 0, 1] computes f(i+1) − f(i−1), spanning 2 voxels), and each factor
//! of 4 is the sum of one smoothing kernel [1, 2, 1].
//!
//! # Architecture
//!
//! `SobelFilter` is a type alias for [`SeparableGradientFilter<SobelKernel>`],
//! which is the canonical monomorphized implementation. All convolution logic
//! lives in [`super::separable_gradient`].
//!
//! # Reference
//!
//! Zucker, S. W. & Hummel, R. A. (1981). "A three-dimensional edge operator."
//! *IEEE Trans. Pattern Analysis and Machine Intelligence*, 3(3), 324–331.

use super::separable_gradient::{SeparableGradientFilter, SobelKernel};

/// 3-D Sobel gradient filter.
///
/// Computes spatial derivatives using the 3-D Sobel operator, which combines
/// central-difference derivative estimation with binomial smoothing along
/// the two orthogonal axes. The output is normalized to physical gradient
/// units (intensity per unit spacing).
///
/// ## Kernel structure
///
/// For derivative axis `a` with orthogonal axes `b`, `c`:
///
/// ```text
/// K_a[db][dc][da] = s[db] · s[dc] · d[da]
/// where d = [-1, 0, 1], s = [1, 2, 1]
/// ```
///
/// ## Normalization factor derivation
///
/// | Component | Factor | Source |
/// |------------------|--------|----------------------------------------------|
/// | Central diff | 2·h | [-1,0,1] spans 2 voxels of spacing h |
/// | Smoothing axis 1 | 4 | sum(\[1,2,1\]) |
/// | Smoothing axis 2 | 4 | sum(\[1,2,1\]) |
/// | **Total** | 32·h | |
pub type SobelFilter = SeparableGradientFilter<SobelKernel>;

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[path = "tests_sobel.rs"]
mod tests_sobel;
