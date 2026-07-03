//! Image types and operations — Image, RgbVolume, ColorVolume, grid generation, metadata.
//!
//! Depends on `ritk-spatial` for spatial types, `burn` for the legacy root
//! image backend, and an optional Atlas-native tensor image (`native` module, `coeus` feature).

pub mod native;
pub mod color;
pub mod grid;
pub mod host_extract;
pub mod metadata;
#[cfg(any(test, feature = "test-helpers"))]
pub mod test_support;
pub mod transform;
pub mod types;

pub use color::{ColorVolume, RgbVolume};
pub use grid::generate_grid;
pub use host_extract::HostExtract;
pub use metadata::ImageMetadata;
pub use types::Image;
