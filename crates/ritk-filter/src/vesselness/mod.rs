//! Vesselness filters for curvilinear structure detection in 3-D medical images.

pub mod frangi;
pub mod hessian;
pub mod sato;

use serde::{Deserialize, Serialize};

/// Vessel polarity â€” whether the target structures are bright or dark relative
/// to the background.
///
/// - `Dark`: detect dark structures on a bright background.
/// - `Bright`: detect bright structures on a dark background (e.g. vessels in MRA).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum VesselPolarity {
    /// Detect dark structures on a bright background.
    #[default]
    Dark,
    /// Detect bright structures on a dark background.
    Bright,
}

pub use frangi::{FrangiConfig, FrangiVesselnessFilter};
pub use hessian::compute_hessian;
pub use sato::{SatoConfig, SatoLineFilter};
