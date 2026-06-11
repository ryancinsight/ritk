//! Image types and operations — Image, RgbVolume, ColorVolume, grid generation, metadata.
//!
//! Depends on `ritk-spatial` for spatial types and `burn` for tensor backend.

pub mod color;
pub mod grid;
pub mod metadata;
pub mod transform;
pub mod types;

pub use color::{ColorVolume, RgbVolume};
pub use grid::generate_grid;
pub use metadata::ImageMetadata;
pub use types::Image;
