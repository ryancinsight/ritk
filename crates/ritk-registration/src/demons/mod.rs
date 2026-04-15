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
/// ├── mod.rs           ← this file (re-exports)
/// ├── thirion.rs       ← Thirion 1998 classic Demons
/// ├── diffeomorphic.rs ← Vercauteren 2009 diffeomorphic variant
/// ├── symmetric.rs     ← Pennec 1999 symmetric-force variant
/// └── inverse.rs       ← Exact SVF inverse + iterative displacement inverse
/// ```
///
/// Shared CPU primitives (indexing, interpolation, gradient, smoothing,
/// field composition, scaling-and-squaring) live in
/// [`crate::deformable_field_ops`] (crate-level SSOT).
pub mod diffeomorphic;
pub mod inverse;
pub mod symmetric;
pub mod thirion;

pub use diffeomorphic::DiffeomorphicDemonsRegistration;
pub use inverse::{invert_displacement_field, invert_velocity_field, InverseFieldConfig};
pub use symmetric::SymmetricDemonsRegistration;
pub use thirion::{DemonsConfig, DemonsResult, ThirionDemonsRegistration};
