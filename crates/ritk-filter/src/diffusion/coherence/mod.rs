//! Coherence-Enhancing Diffusion (CED) filter (Weickert 1999).
//!
//! # Mathematical Specification
//!
//! Coherence-Enhancing Diffusion is an anisotropic diffusion filter that
//! smooths images **along** coherent structures (edges, ridges, fibres) while
//! preserving them **across** their orientation. It uses the structure tensor
//! to steer the diffusion tensor.
//!
//! ## Structure Tensor
//!
//! Given image I(x), the structure tensor J_ρ at integration scale ρ is:
//!
//! J_ρ = G_ρ * (∇I · ∇Iᵀ)
//!
//! where G_ρ is a Gaussian of standard deviation ρ and * denotes convolution.
//!
//! In 3-D the 6 independent components are:
//!
//! J_11 = G_ρ * (I_z²), J_22 = G_ρ * (I_y²), J_33 = G_ρ * (I_x²)
//! J_12 = G_ρ * (I_z·I_y), J_13 = G_ρ * (I_z·I_x), J_23 = G_ρ * (I_y·I_x)
//!
//! ## Diffusion Tensor
//!
//! Let λ₁ ≤ λ₂ ≤ λ₃ be the eigenvalues of J_ρ with eigenvectors e₁, e₂, e₃.
//! The diffusion tensor D shares the eigenvectors but replaces eigenvalues:
//!
//! α₁ = α + (1 − α) · (1 − exp(−C · (λ₃ − λ₁)² / (λ₃² + ε))) (coherence dir)
//! α₂ = α + (1 − α) · (1 − exp(−C · (λ₂ − λ₁)² / (λ₃² + ε))) (intermediate)
//! α₃ = α (edge dir)
//!
//! α is the flat-region smoothing parameter, C the contrast parameter.
//!
//! Along e₁ (smallest eigenvalue = coherence direction) diffusion is maximal;
//! along e₃ (largest eigenvalue = edge direction) diffusion is minimal.
//!
//! ## PDE
//!
//! ∂I/∂t = div(D · ∇I)
//!
//! Discretised with explicit Euler on the 3-D grid. Stability requires
//! Δt ≤ 1 / (2·D·max(α_i)).
//!
//! # Complexity
//!
//! Per iteration: O(N·k) where N is the number of voxels and k is the
//! Gaussian kernel size (radius ⌈3ρ⌉ along each axis). The eigendecomposition
//! is O(1) per voxel (analytical closed-form).
//!
//! # Invariants
//!
//! - Constant image: ∇I = 0 → J_ρ = 0 → D = α·I → div(D·∇I) = 0 → unchanged.
//! - Linear image: ∇I = const → J_ρ rank-1 → λ₂ = λ₁ = 0 → α₁ = α₂ = α
//!   (no excess diffusion), div(D·∇I) = 0 → unchanged.
//!
//! # References
//!
//! - Weickert, J. (1999). *Coherence-Enhancing Diffusion Filtering*.
//!   Int. J. Comput. Vis. 31(2/3):111–127.
//! - Weickert, J. (1998). *Anisotropic Diffusion in Image Processing*.
//!   Teubner, Stuttgart.

mod filter;
mod pde;
mod scratch;
mod tensor;

pub use filter::{CoherenceConfig, CoherenceEnhancingDiffusionFilter};
pub use scratch::{CedScratch, StructureTensorProducts};
pub use tensor::EigenDecomp;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
