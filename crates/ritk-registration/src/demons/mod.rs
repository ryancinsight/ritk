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
//! let result = reg.register(&fixed, &moving, dims, [1.0, 1.0, 1.0]).expect("infallible: validated precondition");
//! println!("Final MSE: {}", result.final_mse);
//! ```
//!
//! # Module Layout
/// ```text
/// demons/
/// ├── mod.rs ← this file (re-exports)
/// ├── config.rs ← DemonsConfig, DemonsResult (SSOT)
/// ├── level_set_motion.rs ← ITK LevelSetMotionRegistrationFilter variant
/// ├── thirion/ ← Thirion 1998 classic Demons
/// ├── diffeomorphic/ ← Vercauteren 2009 diffeomorphic variant
/// ├── symmetric/  ← Pennec 1999 symmetric-force variant
/// │   ├── mod.rs
/// │   └── tests.rs
/// ├── inverse/ ← Exact SVF inverse + iterative displacement inverse
/// ├── exact_inverse_diffeomorphic/ ← Inverse-consistent diffeomorphic Demons
/// └── multires/ ← Multi-resolution coarse-to-fine pyramid
///     ├── mod.rs
///     ├── resample.rs
///     └── tests_multires.rs
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
