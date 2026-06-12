//! Image deconvolution / restoration filters (GAP-262-FLT-02).
//!
//! Deconvolution reverses degradation caused by a known point-spread function
//! (PSF / kernel), restoring the original image. All methods operate in the
//! frequency domain via FFT for O(N log N) efficiency.
//!
//! # Methods
//!
//! | Filter | Type | Equation |
//! |---|---|---|
//! | [`WienerDeconvolution`] | Non-iterative | `G = H* / (|H|² + K)` |
//! | [`TikhonovDeconvolution`] | Non-iterative | `G = H* / (|H|² + λ|L|²)` |
//! | [`RichardsonLucyDeconvolution`] | Iterative | `uₖ₊₁ = uₖ · H* ⋆ (f / (H ⋆ uₖ))` |
//! | [`LandweberDeconvolution`] | Iterative | `uₖ₊₁ = uₖ + α · H* ⋆ (f − H ⋆ uₖ)` |
//!
//! All filters expose a single generic `apply<B: Backend, const D: usize>` that
//! accepts images of any supported dimensionality (currently `D = 2` and `D = 3`).
//!
//! # Theory
//!
//! Given a degraded image `g = h ∗ u + n` where `h` is the PSF kernel
//! and `n` is additive noise, deconvolution estimates the original image `u`.
//!
//! In the frequency domain (capital letters denote FFT):
//!
//! ```text
//! G(ω) = H(ω) · U(ω) + N(ω)
//! ```
//!
//! Direct inverse filtering `U = G / H` amplifies noise where |H| is small.
//! Regularization suppresses this amplification.

mod helpers;
mod landweber;
mod regularization;
mod rl;
mod tikhonov;
mod wiener;

pub use landweber::LandweberDeconvolution;
pub use rl::RichardsonLucyDeconvolution;
pub use tikhonov::TikhonovDeconvolution;
pub use wiener::WienerDeconvolution;

#[cfg(test)]
#[path = "tests_2d.rs"]
mod tests_2d;

#[cfg(test)]
#[path = "tests_3d.rs"]
mod tests_3d;
