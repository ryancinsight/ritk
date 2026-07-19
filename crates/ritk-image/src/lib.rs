//! Image types and operations — Image, RgbVolume, ColorVolume, grid generation, metadata.
//!
//! Depends on `ritk-spatial` for spatial types and `coeus` for the Atlas-native
//! tensor backend.

pub mod color;
pub mod grid;
pub mod metadata;
#[cfg(any(test, feature = "test-helpers"))]
pub mod test_support;
pub mod transform;
pub mod types;

pub use color::{ColorVolume, RgbVolume};
pub use grid::{generate_grid, generate_random_points};
pub use metadata::ImageMetadata;
pub use types::Image;

/// Coeus-backed tensor and module surface re-exported for downstream crates.
pub mod coeus {
    pub use coeus_autograd;
    pub use coeus_core;
    pub use coeus_nn;
    pub use coeus_ops;
    pub use coeus_optim;
    pub use coeus_tensor;
}

/// Coeus tensor aliases and backend re-exports.
pub mod tensor {
    pub use coeus_core::{
        Backend, ComputeBackend, Float, MoiraiBackend, Scalar, SequentialBackend,
    };
    pub use coeus_tensor::Tensor;

    /// Shape alias — coeus uses `Vec<usize>` / `&[usize]`.
    pub type Shape = Vec<usize>;

    /// Construct a coeus shape from an iterator.
    pub fn shape(dims: impl IntoIterator<Item = usize>) -> Vec<usize> {
        dims.into_iter().collect()
    }
}

/// Backend re-exports (always available via coeus).
pub mod backend {
    pub use coeus_core::{Backend, ComputeBackend, MoiraiBackend, SequentialBackend};
}
