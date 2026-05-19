//! Image types and operations.
//!
//! This module provides the Image type and related functionality
//! for representing medical images with physical metadata.

pub mod color;
pub mod grid;
pub mod metadata;
pub mod transform;
pub mod types;

pub use color::{ColorVolume, RgbVolume};
pub use grid::generate_grid;
pub use metadata::ImageMetadata;
pub use types::Image;
