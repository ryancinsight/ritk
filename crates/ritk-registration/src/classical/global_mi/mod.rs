//! Global Mutual Information registration pipeline.
//!
//! Multi-resolution Mattes MI + RegularStepGradientDescent (RSGD) registration,
//! following ITK's `ImageRegistrationMethod` architecture. Closes GAP-R08a:
//! "Global MI optimizer (Mattes MI + RSGD with sparse sampling)".
//!
//! # Algorithm
//!
//! ```text
//! Multi-Resolution Pyramid (coarse → fine)
//! └── Per Level:
//!     1. Resample fixed/moving to current resolution
//!     2. Estimate intensity range for MI binning
//!     3. Compute Mattes MI loss: L(θ) = -MI(A, B∘T; θ)
//!     4. Backward pass: ∇θ L
//!     5. RSGD step: θₖ₊₁ = θₖ − Δₖ · ĝ
//!     6. Accept/reject step, shrink Δ if rejected
//!     7. Continue until convergence or max iterations
//! ```
//!
//! # Theorem: Mattes Mutual Information
//!
//! Given N randomly sampled voxels xᵢ from fixed image A and moving image B∘T:
//!
//! ```text
//! I_Mattes(A, B; T) = Σ_{a,b} p̂(a,b) · log( p̂(a,b) / (p̂_A(a) · p̂_B(b)) )
//! ```
//!
//! where the joint density is estimated by cubic B-spline Parzen windows.
//!
//! # References
//!
//! - Mattes, D., et al. (2003). PET-CT image registration in the chest using
//!   free-form deformations. *IEEE Trans. Med. Imaging* 22(1):120–128.
//! - ITK `ImageRegistrationMethod`:
//!   <https://itk.org/Doxygen/html/classitk_1_1ImageRegistrationMethod.html>

pub mod center_of_mass;
pub mod cma_mi_registration;
pub mod config;
pub mod multistart;
pub mod registration;
pub mod result;
pub(crate) mod transforms;

#[cfg(test)]
mod tests;

pub use center_of_mass::{compute_center_of_mass, translation_from_centers_of_mass};
pub use cma_mi_registration::{CmaMiConfig, CmaMiRegistration, CmaMiResult};
pub use config::{GlobalMiConfig, GlobalMiTransformType};
pub use multistart::{MultiStartConfig, MultiStartMiRegistration, MultiStartResult};
pub use registration::GlobalMiRegistration;
pub use result::GlobalMiResult;
