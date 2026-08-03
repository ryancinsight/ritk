//! Diffusion MRI signal and orientation models.
//!
//! This crate owns diffusion-model fitting under
//! [ADR 0017](../../../docs/adr/0017-diffusion-mri-pipeline.md). The current
//! implemented models are regularized analytical Q-ball imaging, constrained
//! spherical deconvolution (CSD), diffusion tensor imaging (DTI), and
//! diffusion kurtosis imaging (DKI).
//! Unimplemented model families are not advertised as available APIs.
//!
//! # Module map
//!
//! | Module | Model | Solver |
//! |--------|-------|--------|
//! | [`dti`] | Log-linear diffusion tensor (FA / MD / PEV) | `leto_ops::solve_least_squares` |
//! | [`dki`] | Nonlinear kurtosis tensor (MK / AK / RK) | `coeus_optim::levenberg_marquardt` |
//! | [`noddi`] | NODDI ball-and-stick (NDI / f_ISO) | `coeus_optim::levenberg_marquardt` |
//! | [`odf`] | Analytical Q-ball ODF via real spherical harmonics | `leto_ops::solve_least_squares` |
//! | [`csd`] | Constrained spherical deconvolution (non-negative fODF) | `leto_ops::nnls` |

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod csd;
pub mod dki;
pub mod dti;
pub mod noddi;
pub mod odf;
