//! CMA-ES → RSGD cascade registration using Mattes Mutual Information.
//!
//! # Motivation: The Local-Maxima Problem in MI-Based Registration
//!
//! Multi-modal image registration by mutual information maximisation is
//! fundamentally a non-convex optimisation problem. The MI landscape
//! `I(θ) : ℝ⁶ → ℝ` over rigid-body parameters exhibits numerous local
//! maxima caused by:
//!
//! - **Aliasing**: when voxel grids of the two modalities are commensurate
//!   along a given axis, a false MI peak appears at every integer-voxel offset.
//! - **Symmetry**: anatomically symmetric structures (e.g. bilateral brain
//!   hemispheres) produce mirror-image MI peaks.
//! - **Parzen window width**: wide bins smooth the landscape and help escape
//!   local optima, but the optimal landscape is still non-convex at coarse
//!   resolution.
//!
//! Gradient-based methods such as RSGD / Adam are susceptible to local-maxima
//! trapping and are highly sensitive to initialisation. The [`GlobalMiRegistration`]
//! pipeline (RSGD) therefore benefits from a good starting point.
//!
//! # Solution: CMA-ES Global Search + RSGD Local Refinement
//!
//! This module implements a two-phase cascade:
//!
//! ```text
//! Phase 0: Center-of-mass initialisation (zeroth-order translation estimate)
//! Phase 1: CMA-ES global search on a coarse pyramid level
//!           ├─ Derivative-free evolution strategy — can cross MI valleys
//!           ├─ Normalised parameter space [−1, 1]⁶ for uniform scaling
//!           └─ Minimises −MI(A, B∘T; θ) over 6-DOF rigid parameters
//! Phase 2 (optional): RSGD fine refinement on the full-resolution image
//!           using the CMA-ES solution as warm start
//! ```
//!
//! # Why CMA-ES?
//!
//! The (μ/μ_w, λ)-CMA-ES (Hansen & Ostermeier 2001) is an evolutionary
//! strategy that adapts a full covariance matrix to the curvature of the
//! objective. Key properties for registration:
//!
//! - **No gradients required**: suitable for metrics whose autodiff graph is
//!   expensive to compute at full resolution.
//! - **Invariant to rotation of parameter space**: affine parameter rescaling
//!   does not affect convergence speed.
//! - **Population-based**: evaluates `λ` candidates per generation, naturally
//!   exploring multiple modes of the MI landscape simultaneously.
//! - **Self-adaptive step size**: the step-size control (CSA / cumulative
//!   path-length control) prevents premature convergence.
//!
//! IPOP-CMA-ES (Auger & Hansen 2005) with increasing population offers
//! stronger global guarantees but is not implemented here; the coarse-pyramid
//! strategy (large `coarse_shrink`) achieves a similar smoothing effect at
//! lower computational cost.
//!
//! # References
//!
//! - Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation
//!   in evolution strategies. *Evol. Comput.* 9(2):159–195.
//!   DOI: 10.1162/106365601750190398
//! - Auger, A., & Hansen, N. (2005). A restart CMA evolution strategy with
//!   increasing population size. *CEC 2005*, vol. 2, pp. 1769–1776.
//!   DOI: 10.1109/CEC.2005.1554902
//! - Klein, S., et al. (2007). Evaluation of optimization methods for
//!   nonrigid medical image registration using mutual information and
//!   B-splines. *IEEE Trans. Image Process.* 16(12):2879–2890.
//!   DOI: 10.1109/TIP.2007.909412

mod config;
mod helpers;
mod registration;

pub use config::{CmaMiConfig, CmaMiLevelConfig, CmaMiResult};
pub use registration::CmaMiRegistration;
