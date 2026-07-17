//! Native Coeus boundaries for classical registration.
//!
//! Classical mutual-information optimisation operates on Leto volumes in
//! index coordinates. This module owns the explicit conversions to and from
//! RITK's Coeus-native image and physical-coordinate contracts.

mod error;
mod transform;
mod volume;

pub use error::NativeConversionError;
pub use transform::index_affine_to_physical;
pub use volume::{image_to_leto_volume, leto_volume_to_image};
