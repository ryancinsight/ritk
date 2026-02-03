//! Image types and operations.
//!
//! This module provides the Image type and related functionality
//! for representing medical images with physical metadata.

pub mod image;
pub mod metadata;
pub mod grid;

pub use image::Image;
pub use metadata::ImageMetadata;
pub use grid::generate_grid;
