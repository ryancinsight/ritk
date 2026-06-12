//! Noise simulation filters for medical image robustness testing (GAP-262-FLT-05).
//!
//! These filters add synthetic noise to images, useful for:
//! - Evaluating segmentation/registration robustness under varying noise levels
//! - Generating training data with realistic noise profiles
//! - Testing filter and denoising algorithm quality
//!
//! # Filters
//!
//! | Filter | Noise model | Parameters |
//! |---|---|---|
//! | `AdditiveGaussianNoiseFilter` | `I'(x) = I(x) + N(mean, std)` | `mean`, `std` |
//! | `SaltAndPepperNoiseFilter` | Random pixel replacement with min/max | `probability` |
//! | `ShotNoiseFilter` | Poisson process: `I'(x) ~ Poisson(λ·I(x)) / λ` | `scale` |
//! | `SpeckleNoiseFilter` | Multiplicative: `I'(x) = I(x) · (1 + N(0, std))` | `std` |

pub mod gaussian;
pub mod salt_pepper;
pub mod shot;
pub mod speckle;

pub use gaussian::AdditiveGaussianNoiseFilter;
pub use salt_pepper::SaltAndPepperNoiseFilter;
pub use shot::ShotNoiseFilter;
pub use speckle::SpeckleNoiseFilter;

/// Default random seed used by all noise filters for reproducible output.
pub(crate) const DEFAULT_NOISE_SEED: u64 = 42;

/// Box-Muller transform: given two uniform random variates u1 ∈ (0, 1] and u2 ∈ [0, 1),
/// returns a standard normal variate Z ~ N(0, 1).
#[inline]
pub(crate) fn box_muller(u1: f64, u2: f64) -> f64 {
    (-2.0_f64 * u1.max(f64::MIN_POSITIVE).ln()).sqrt() * (2.0 * std::f64::consts::TAU * u2).cos()
}

#[cfg(test)]
#[path = "tests_noise.rs"]
mod tests_noise;
