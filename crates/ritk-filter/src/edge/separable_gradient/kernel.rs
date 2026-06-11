//! ZST trait and kernel types for separable gradient filters.
//!
//! [`GradientKernel`] encodes the smoothing kernel and normalization factor
//! at the type level. Each implementor is a zero-sized type so that
//! [`SeparableGradientFilter<K>`](super::SeparableGradientFilter) monomorphizes
//! to a specialized, branch-free implementation identical to a hand-written
//! version.

// ── Trait ─────────────────────────────────────────────────────────────────────

/// ZST trait encoding a separable gradient filter's smoothing kernel and
/// normalization factor.
///
/// Each implementor is a zero-sized type. The compiler monomorphizes
/// [`SeparableGradientFilter<K>`](super::SeparableGradientFilter) into a
/// specialized, branch-free implementation identical to a hand-written version.
pub trait GradientKernel: Default {
    /// Smoothing kernel: 3-tap `[k_{-1}, k_0, k_{+1}]`.
    ///
    /// Sobel: `[1, 2, 1]` (binomial). Prewitt: `[1, 1, 1]` (uniform).
    const SMOOTH: [f32; 3];

    /// Normalization divisor per axis, excluding the spacing factor `h`.
    ///
    /// Derivation: `norm = 2 · sum(SMOOTH)²`. The factor of 2 accounts for the
    /// central-difference kernel `[-1, 0, 1]` spanning 2 voxels. Each
    /// `sum(SMOOTH)` is the gain of one orthogonal smoothing pass.
    ///
    /// Sobel: `2 · 4 · 4 = 32`. Prewitt: `2 · 3 · 3 = 18`.
    const NORM_BASE: f32;
}

// ── ZST kernel types ──────────────────────────────────────────────────────────

/// Sobel binomial smoothing kernel `[1, 2, 1]` with normalization factor `32·h`.
///
/// The Sobel kernel weights the center pixel twice as much as its neighbors,
/// providing slightly more smoothing perpendicular to the derivative axis than
/// Prewitt, which suppresses diagonal noise more aggressively.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SobelKernel;

/// Prewitt uniform smoothing kernel `[1, 1, 1]` with normalization factor `18·h`.
///
/// The Prewitt kernel applies equal weights to all three neighbors, yielding
/// a computationally simpler filter with integer-friendly arithmetic.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrewittKernel;

impl GradientKernel for SobelKernel {
    const SMOOTH: [f32; 3] = [1.0, 2.0, 1.0];
    const NORM_BASE: f32 = 32.0;
}

impl GradientKernel for PrewittKernel {
    const SMOOTH: [f32; 3] = [1.0, 1.0, 1.0];
    const NORM_BASE: f32 = 18.0;
}
