//! TransMorph Model Implementation, Coeus-native.
//!
//! This module implements the TransMorph architecture for medical image
//! registration, as described in:
//!
//! Chen, J., Frey, E. C., He, Y., Segars, W. P., Li, Y., & Du, Y. (2022).
//! TransMorph: Transformer for unsupervised medical image registration.
//! Medical Image Analysis, 82, 102615.
//!
//! # Architecture
//!
//! 1.  **Swin Transformer Encoder**: Hierarchical feature extraction with
//!     shifted-window attention.
//! 2.  **Decoder**: Conv3d-based decoder with skip connections from the encoder.
//! 3.  **Integration**: Diffeomorphic integration (scaling and squaring) for
//!     topology preservation.
//! 4.  **Transformation**: Spatial-transformer warp of the moving image.
//!
//! # Mathematical Invariants
//!
//! - **Diffeomorphism**: With integration enabled, the displacement field is
//!   diffeomorphic (smooth, invertible) via scaling and squaring.
//! - **Topology Preservation**: The Jacobian determinant of the transformation
//!   field should be positive everywhere.
//! - **Coordinate System**: Index coordinates (voxel space) internally;
//!   conversion to physical space happens at the I/O boundary.
//!
//! # Usage
//!
//! ```rust
//! use ritk_model::transmorph::{TransMorph, TransMorphConfig};
//! use coeus_core::SequentialBackend;
//!
//! let model: TransMorph<SequentialBackend> = TransMorphConfig::new(1, 12, 3)
//!     .with_window_size(4)
//!     .init();
//! let _ = model.parameters();
//! ```

pub mod config;
pub mod integration;
pub mod model;
pub mod spatial_transform;
pub mod swin;

pub use config::{TransMorphConfig, TransformIntegration};
pub use integration::VecInt;
pub use model::{TransMorph, TransMorphOutput};
pub use spatial_transform::SpatialTransformer;
