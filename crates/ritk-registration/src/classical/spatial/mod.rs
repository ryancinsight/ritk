//! Classical spatial transformation: centroid, Kabsch SVD, rigid/affine perturbations.
//!
//! # Module structure
//! - `error`     — `SpatialError`
//! - `transform` — `SpatialTransform`, homogeneous-matrix helpers, `transform_point`
//! - `centroid`  — `compute_centroid`, `center_points`
//! - `kabsch`    — Kabsch SVD algorithm, `compute_fre`
//! - `rigid`     — 6-DOF perturbation generation and application
//! - `affine`    — 9-DOF perturbation generation and application
//! - `volume`    — volume warping via 4×4 homogeneous transforms

mod affine;
mod centroid;
mod error;
mod kabsch;
mod rigid;
mod transform;
mod volume;
#[cfg(test)]
mod tests;

/// Shared step-size constants used by both rigid and affine perturbation modules.
pub(super) const EULER_STEP: f64 = 0.01;
pub(super) const TRANSLATION_STEP: f64 = 1.0;
pub(super) const SCALE_STEP: f64 = 0.02;

pub use error::SpatialError;
pub use transform::SpatialTransform;

pub(crate) use centroid::{center_points, compute_centroid};
pub(crate) use kabsch::{compute_fre, kabsch_algorithm};
pub(crate) use rigid::{apply_transform_perturbation, generate_transform_perturbations};
pub(crate) use affine::{apply_affine_perturbation, generate_affine_perturbations};
pub(crate) use transform::{build_homogeneous_matrix, extract_spatial_transform};
pub use volume::apply_transform;
