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

#[cfg(test)]
#[path = "tests_noise.rs"]
mod tests_noise;
