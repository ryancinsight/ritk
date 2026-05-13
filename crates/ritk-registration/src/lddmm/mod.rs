//! LDDMM — Large Deformation Diffeomorphic Metric Mapping.
//!
//! # Mathematical Specification
//!
//! LDDMM (Beg et al. 2005) generates geodesic paths in the group of
//! diffeomorphisms by optimising the initial velocity v₀ of a time-dependent
//! velocity field v(t), t ∈ \[0, 1\].
//!
//! ## Energy functional
//!
//! E(v₀) = λ ‖v₀‖²\_V + MSE(I ∘ φ₁, J)
//!
//! where:
//! - ‖·‖\_V is the Sobolev norm induced by Gaussian kernel K\_σ,
//! - φ₁ is the diffeomorphism at t = 1 obtained by integrating v(t),
//! - I is the moving image, J is the fixed image,
//! - λ is the regularisation weight (`regularization_weight`).
//!
//! ## EPDiff shooting (geodesic integration)
//!
//! The velocity field evolves according to the EPDiff equation:
//!
//!   ∂v/∂t = −K\_σ ∗ ad\*\_v(m)
//!
//! where m = K\_σ ∗ v is the momentum and ad\*\_v(m) is the coadjoint
//! operator of the Lie-algebra adjoint:
//!
//!   (ad\*\_v m)\_i = Σ\_j \[v\_j · ∂m\_i/∂x\_j + m\_j · ∂v\_i/∂x\_j\] + m\_i · div(v)
//!
//! Integration proceeds via forward Euler over N\_t steps with dt = 1/N\_t.
//! At each step the displacement field is composed with the incremental map
//! id + v(t)·dt to accumulate the full diffeomorphism φ₁.
//!
//! ## Gradient descent update
//!
//!   ∂E/∂v₀ = 2λ v₀ + K\_σ ∗ \[2(I∘φ₁ − J) · ∇(I∘φ₁)\]
//!
//!   v₀ ← v₀ − lr · ∂E/∂v₀
//!
//! # References
//!
//! - Beg, M. F., Miller, M. I., Trouvé, A. & Younes, L. (2005).
//!   Computing large deformation metric mappings via geodesic flows of
//!   diffeomorphisms. *Int. J. Comput. Vis.* 61(2):139–157.
//! - Vialard, F.-X., Risser, L., Rueckert, D. & Cotter, C. J. (2012).
//!   Diffeomorphic 3D image registration via geodesic shooting using an
//!   efficient adjoint calculation. *Int. J. Comput. Vis.* 97(2):153–174.

mod adjoint;
mod config;
mod geodesic;
mod registration;

#[cfg(test)]
mod tests;

pub use config::{LddmmConfig, LddmmResult};
pub use registration::LddmmRegistration;
