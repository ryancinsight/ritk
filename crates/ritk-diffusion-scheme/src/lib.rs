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
mod mrtrix;
mod scheme;
mod weighting;

pub use error::GradientSchemeError;
pub use fsl::{parse_fsl_bval, parse_fsl_bvec, read_fsl_scheme, write_fsl_scheme};
pub use gradient::{GradientDirection, GradientFrame};
pub use mrtrix::{read_mrtrix_scheme, write_mrtrix_scheme};
pub use scheme::GradientScheme;
pub use weighting::DiffusionWeighting;

/// Default scanner-facing threshold separating baseline and weighted volumes,
/// in seconds per square millimeter.
///
/// External values at or below this threshold are canonicalized to exact b0
/// entries because small nonzero values commonly encode scanner baseline
/// acquisitions whose gradient orientation is absent or not meaningful.
pub const DEFAULT_B0_THRESHOLD_SECONDS_PER_SQUARE_MILLIMETER: f64 = 50.0;

#[cfg(test)]
mod tests;
