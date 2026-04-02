//! Image types and operations.
//!
//! This module provides the Image type and related functionality
//! for representing medical images with physical metadata.

pub mod grid;
pub mod image;
pub mod metadata;

pub use grid::generate_grid;
pub use image::Image;
pub use metadata::ImageMetadata;
