//! Diffusion MRI signal and orientation models.
//!
//! This crate owns diffusion-model fitting under
//! ADR 0017 (`diffusion-mri-pipeline`) defines the crate boundary. The current
//! implemented model is regularized analytical Q-ball imaging. Unimplemented
//! model families are not advertised as available APIs.
//!
//! # Module map
//!
//! | Module | Model | Solver |
//! |--------|-------|--------|
//! | [`odf`] | Analytical Q-ball ODF via real spherical harmonics | `leto_ops::solve_least_squares` |

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod odf;
