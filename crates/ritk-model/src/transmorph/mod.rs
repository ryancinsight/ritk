//! TransMorph Model Implementation
//!
//! This module implements the TransMorph architecture for medical image registration,
//! as described in:
//!
//! Chen, J., Frey, E. C., He, Y., Segars, W. P., Li, Y., & Du, Y. (2022).
//! TransMorph: Transformer for unsupervised medical image registration.
//! Medical Image Analysis, 82, 102615.
//!
//! # Architecture
//!
//! The model consists of:
//! 1.  **Swin Transformer Encoder**: Hierarchical feature extraction with shifted window attention.
//! 2.  **Decoder**: Conv3D-based decoder with skip connections from the encoder.
//! 3.  **Integration**: Diffeomorphic integration (scaling and squaring) for topology preservation.
//! 4.  **Transformation**: Spatial Transformer Network (STN) for warping the moving image.
//!
//! # Mathematical Invariants
//!
//! - **Diffeomorphism**: The output displacement field is guaranteed to be diffeomorphic
//!   (smooth, invertible) if `integrate` is enabled, using the Scaling and Squaring method.
//! - **Topology Preservation**: The Jacobian determinant of the transformation field
//!   should be positive everywhere.
//! - **Coordinate System**: Uses index coordinates (voxel space) for internal processing.
//!   Conversion to physical space (world coordinates) happens at the I/O boundary.
//!
//! # Usage
//!
//! ```rust
//! use ritk_model::transmorph::{TransMorphConfig, TransMorph};
//! use burn::tensor::backend::Backend;
//!
//! fn create_model<B: Backend>(device: &B::Device) -> TransMorph<B> {
//!     TransMorphConfig::new(1, 12, 3)
//!         .with_window_size(4)
//!         .init(device)
//! }
//! ```

pub mod config;
pub mod integration;
pub mod model;
pub mod spatial_transform;
pub mod swin;

pub use config::TransMorphConfig;
pub use model::{TransMorph, TransMorphOutput};

#[cfg(test)]
mod tests;
