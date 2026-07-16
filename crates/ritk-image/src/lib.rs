//! Image types and operations — Image, RgbVolume, ColorVolume, grid generation, metadata.
//!
//! Depends on `ritk-spatial` for spatial types and `coeus` for the Atlas-native
//! tensor backend. The legacy `burn` compatibility surface has been replaced
//! by direct `coeus` re-exports.

pub mod color;
pub mod grid;
pub mod metadata;
pub mod native;
#[cfg(any(test, feature = "test-helpers"))]
pub mod test_support;
pub mod transform;
pub mod types;

pub use color::{ColorVolume, RgbVolume};
pub use grid::generate_grid;
pub use metadata::ImageMetadata;
pub use types::Image;

/// Coeus-backed tensor and module surface re-exported for downstream crates.
///
/// This replaces the former `burn` compatibility shim. Downstream crates that
/// previously used `ritk_image::burn::*` should migrate to `ritk_image::coeus::*`.
pub mod coeus {
    pub use coeus_autograd;
    pub use coeus_core;
    pub use coeus_nn;
    pub use coeus_ops;
    pub use coeus_optim;
    pub use coeus_tensor;
}

/// Coeus tensor aliases replacing the former burn tensor module.
pub mod tensor {
    /// Backend trait re-exports (replaces the former `burn::tensor::backend`).
    pub mod backend {
        pub use coeus_core::{Backend, ComputeBackend, MoiraiBackend, SequentialBackend};
    }

    pub use coeus_core::{Backend, ComputeBackend, Float, Scalar};
    pub use coeus_tensor::Tensor;

    /// Shape alias — coeus uses `Vec<usize>` / `&[usize]` rather than a
    /// dedicated `Shape` type. This newtype preserves call-site ergonomics.
    pub type Shape = Vec<usize>;

    /// Construct a coeus shape from an array (replaces `burn::tensor::Shape::new`).
    pub fn shape(dims: impl IntoIterator<Item = usize>) -> Shape {
        dims.into_iter().collect()
    }
}

/// Backwards-compatible re-export of the coeus `SequentialBackend` for tests.
pub mod backend {
    pub use coeus_core::{Backend, ComputeBackend, MoiraiBackend, SequentialBackend};
}
