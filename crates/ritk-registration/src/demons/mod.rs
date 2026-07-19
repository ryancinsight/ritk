//! Demons-family deformable image registration algorithms.
//!
//! This module provides three variants of the Demons registration algorithm,
//! all operating on flat `Vec<f32>` buffers with shape `[nz, ny, nx]` (Z-major).
//!
//! # Variants
//!
//! | Type | Reference | Key property |
//! |---|---|---|
//! | [`ThirionDemonsRegistration`] | Thirion (1998) | Classic optical-flow forces |
//! | [`DiffeomorphicDemonsRegistration`] | Vercauteren et al. (2009) | Invertible via exp-map |
//! | [`SymmetricDemonsRegistration`] | Pennec et al. (1999) | Symmetric fixed/moving forces |
//!
//! # Quick Start
//!
//! ```no_run
//! use ritk_registration::demons::{ThirionDemonsRegistration, DemonsConfig};
//!
//! let dims = [32usize, 32, 32];
//! let n = dims[0] * dims[1] * dims[2];
//! let fixed = vec![0.0_f32; n];
//! let moving = vec![0.0_f32; n];
//!
//! let reg = ThirionDemonsRegistration::new(DemonsConfig::default());
//! let result = reg.register(&fixed, &moving, dims, [1.0, 1.0, 1.0]).unwrap();
//! println!("Final MSE: {}", result.final_mse);
//! ```
//!
//! # Module Layout
/// ```text
/// demons/
/// â”œâ”€â”€ mod.rs â† this file (re-exports)
/// â”œâ”€â”€ config.rs â† DemonsConfig, DemonsResult (SSOT)
/// â”œâ”€â”€ level_set_motion.rs â† ITK LevelSetMotionRegistrationFilter variant
/// â”œâ”€â”€ thirion/ â† Thirion 1998 classic Demons
/// â”œâ”€â”€ diffeomorphic/ â† Vercauteren 2009 diffeomorphic variant
/// â”œâ”€â”€ symmetric/  â† Pennec 1999 symmetric-force variant
/// â”‚   â”œâ”€â”€ mod.rs
/// â”‚   â””â”€â”€ tests.rs
/// â”œâ”€â”€ inverse/ â† Exact SVF inverse + iterative displacement inverse
/// â”œâ”€â”€ exact_inverse_diffeomorphic/ â† Inverse-consistent diffeomorphic Demons
/// â””â”€â”€ multires/ â† Multi-resolution coarse-to-fine pyramid
///     â”œâ”€â”€ mod.rs
///     â”œâ”€â”€ resample.rs
///     â””â”€â”€ tests_multires.rs
/// ```
///
/// Shared CPU primitives (indexing, interpolation, gradient, smoothing,
/// field composition, scaling-and-squaring) live in
/// `crate::deformable_field_ops` (crate-level SSOT).
pub mod config;
pub mod diffeomorphic;
pub mod exact_inverse_diffeomorphic;
pub mod inverse;
pub mod level_set_motion;
pub mod multires;
pub mod symmetric;
pub mod thirion;

pub use config::{DemonsConfig, DemonsResult, DemonsVariant};
pub use diffeomorphic::DiffeomorphicDemonsRegistration;
pub use exact_inverse_diffeomorphic::{
    InverseConsistentDemonsConfig, InverseConsistentDemonsResult,
    InverseConsistentDiffeomorphicDemonsRegistration,
};
pub use inverse::{invert_displacement_field, invert_velocity_field, InverseFieldConfig};
pub use level_set_motion::LevelSetMotionRegistration;
pub use multires::{MultiResDemonsConfig, MultiResDemonsRegistration};
pub use symmetric::SymmetricDemonsRegistration;
pub use thirion::ThirionDemonsRegistration;
