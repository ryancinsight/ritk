//! Deterministic streamline tractography producing Gaia polyline geometry.
//!
//! RITK owns integration and termination policy; Gaia owns curve geometry.
//! The current strategy is explicit Euler stepping with direction continuity,
//! field-boundary, turning-angle, and step-count termination. It is intended
//! for deterministic examples and baseline algorithms, not as a clinical
//! tractography validation claim.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod direction_fields;
mod dti;
mod export;
mod tracking;
mod types;

pub use direction_fields::{
    dti_pev_direction_field, dti_volume_direction_field, fod_peak_direction_field,
    fod_volume_direction_field, noddi_direction_field,
};
pub use dti::{
    dti_volume_seed_points, dti_volume_seed_points_with_mask, dti_volume_tractography,
    dti_volume_tractography_with_mask,
};
pub use tracking::euler_tractography;
pub use types::{
    DtiTractographyConfig, Streamline, TerminationReason, TrackingDirection, TractographyConfig,
    TractographyError, TractographyResult,
};

#[cfg(test)]
mod tests;
