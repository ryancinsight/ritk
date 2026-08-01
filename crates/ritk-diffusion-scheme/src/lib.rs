//! Validated diffusion-MRI acquisition schemes.
//!
//! A diffusion series is meaningful only when each volume has a diffusion
//! weighting, a gradient direction, and a declared coordinate frame. This
//! crate owns that format-neutral contract. Format crates convert DICOM,
//! NRRD, or companion-file metadata at their trust boundaries.
//!
//! Diffusion weighting has dimension time per area. Values are stored in
//! canonical SI seconds per square meter through Aequitas and are converted
//! explicitly to the MRI convention seconds per square millimeter.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod error;
mod fsl;
mod gradient;
mod scheme;
mod weighting;

pub use error::GradientSchemeError;
pub use fsl::{parse_fsl_bval, parse_fsl_bvec, read_fsl_scheme};
pub use gradient::{GradientDirection, GradientFrame};
pub use scheme::GradientScheme;
pub use weighting::DiffusionWeighting;

#[cfg(test)]
mod tests;
