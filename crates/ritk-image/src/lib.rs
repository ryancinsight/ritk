//! Image types and operations — Image, RgbVolume, ColorVolume, grid generation, metadata.
//!
//! Depends on `ritk-spatial` for spatial types and `coeus` for the Atlas-native
//! tensor backend. The legacy `burn` compatibility surface is available behind the
//! `burn-compat` feature flag (enabled automatically by `test-helpers`).

#[cfg(feature = "burn-compat")]
mod burn_compat_types;

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

/// `Image` — coeus-backed `Image<T, B, D>` without `burn-compat`;
/// legacy burn-backed `Image<B, D>` with `burn-compat` (backward compat alias).
#[cfg(not(feature = "burn-compat"))]
pub use types::Image;
#[cfg(feature = "burn-compat")]
pub use burn_compat_types::Image;

/// Coeus-backed tensor and module surface re-exported for downstream crates.
pub mod coeus {
    pub use coeus_autograd;
    pub use coeus_core;
    pub use coeus_nn;
    pub use coeus_ops;
    pub use coeus_optim;
    pub use coeus_tensor;
}

/// Coeus tensor aliases + legacy burn compatibility (controlled by feature flags).
pub mod tensor {
    pub mod backend {
        pub use coeus_core::{ComputeBackend, MoiraiBackend, SequentialBackend};
        /// `Backend` — coeus ComputeBackend (always) or burn Backend (burn-compat)
        #[cfg(not(feature = "burn-compat"))]
        pub use coeus_core::Backend;
        #[cfg(feature = "burn-compat")]
        pub use ::burn::tensor::backend::{AutodiffBackend, Backend};
    }

    // ── Primary scalar / backend types ────────────────────────────────────────
    pub use coeus_core::{ComputeBackend, Float, Scalar};

    /// `Backend` type alias — coeus ComputeBackend without burn-compat, burn's Backend with it.
    #[cfg(not(feature = "burn-compat"))]
    pub use coeus_core::Backend;
    #[cfg(feature = "burn-compat")]
    pub use ::burn::tensor::backend::Backend;

    /// `Tensor` type alias — coeus Tensor without burn-compat, burn Tensor with it.
    #[cfg(not(feature = "burn-compat"))]
    pub use coeus_tensor::Tensor;
    #[cfg(feature = "burn-compat")]
    pub use ::burn::tensor::Tensor;

    pub type Int = i32;

    /// Shape alias — coeus uses `Vec<usize>` / `&[usize]`.
    #[cfg(not(feature = "burn-compat"))]
    pub type Shape = Vec<usize>;
    #[cfg(feature = "burn-compat")]
    pub use ::burn::tensor::Shape;

    /// Construct a coeus shape from an iterator (burn-compat: delegates to burn Shape::new).
    pub fn shape(dims: impl IntoIterator<Item = usize>) -> Vec<usize> {
        dims.into_iter().collect()
    }

    // ── Burn-only exports (burn-compat feature only) ─────────────────────────
    #[cfg(feature = "burn-compat")]
    pub use ::burn::tensor::{
        activation, cast, Distribution, ElementConversion, TensorData, TensorPrimitive,
    };
    #[cfg(feature = "burn-compat")]
    pub use ::burn::tensor::backend::AutodiffBackend;
    #[cfg(feature = "burn-compat")]
    pub use ::burn::tensor::{module, ops};

    /// Legacy `burn::module` / `burn::record` shim.
    #[cfg(feature = "burn-compat")]
    pub mod burn {
        pub mod module {
            pub use ::burn::module::{
                AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault, Param,
            };
        }
        pub mod record {
            pub use ::burn::record::{PrecisionSettings, Record};
        }
    }
}

/// Legacy Burn compatibility surface (burn-compat feature).
#[cfg(feature = "burn-compat")]
pub mod burn {
    pub use ::burn::{backend, module, nn, optim, prelude, record, tensor};
}

/// Backend re-exports (always available via coeus).
pub mod backend {
    pub use coeus_core::{Backend, ComputeBackend, MoiraiBackend, SequentialBackend};
}
