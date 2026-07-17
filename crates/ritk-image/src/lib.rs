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

/// Legacy burn-backed `generate_grid<B, D>` for burn-compat callers.
#[cfg(feature = "burn-compat")]
pub mod burn_compat_grid {
    use crate::tensor::{backend::Backend, Distribution, Shape, Tensor, TensorData};

    /// Legacy burn-backed grid generator — `generate_grid::<B, D>(shape, device)`.
    pub fn generate_grid<B: Backend, const D: usize>(
        shape: [usize; D],
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let count: usize = shape.iter().product();
        let mut values = Vec::with_capacity(count * D);
        let mut index = [0usize; D];

        for _ in 0..count {
            for axis in (0..D).rev() {
                values.push(index[axis] as f32);
            }
            for axis in (0..D).rev() {
                index[axis] += 1;
                if index[axis] < shape[axis] {
                    break;
                }
                index[axis] = 0;
            }
        }

        Tensor::<B, 1>::from_data(TensorData::new(values, Shape::new([count * D])), device)
            .reshape([count, D])
    }

    /// Generate uniformly distributed continuous voxel indices.
    ///
    /// The coordinate columns follow [`generate_grid`]'s innermost-first
    /// convention, so each column is bounded by its corresponding reversed axis.
    pub fn generate_random_points<B: Backend, const D: usize>(
        shape: [usize; D],
        count: usize,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let maxima: Vec<f32> = (0..D)
            .map(|column| shape[D - 1 - column].saturating_sub(1) as f32)
            .collect();
        let scale = Tensor::<B, 1>::from_data(TensorData::new(maxima, Shape::new([D])), device)
            .reshape([D, 1]);

        Tensor::<B, 2>::random(
            Shape::new([D, count]),
            Distribution::Uniform(0.0, 1.0),
            device,
        )
        .mul(scale)
        .transpose()
    }
}

/// Legacy generate_grid re-export for burn-compat callers (burn-backed, 2-arg form).
#[cfg(feature = "burn-compat")]
pub use burn_compat_grid::generate_grid as generate_grid_burn;

#[cfg(feature = "burn-compat")]
pub use burn_compat_grid::generate_random_points as generate_random_points_burn;

#[cfg(feature = "burn-compat")]
pub use burn_compat_types::Image;
/// `Image` — coeus-backed `Image<T, B, D>` without `burn-compat`;
/// legacy burn-backed `Image<B, D>` with `burn-compat` (backward compat alias).
#[cfg(not(feature = "burn-compat"))]
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

/// Coeus tensor aliases + legacy burn compatibility (controlled by feature flags).
pub mod tensor {
    pub mod backend {
        #[cfg(feature = "burn-compat")]
        pub use ::burn::tensor::backend::{AutodiffBackend, Backend};
        /// `Backend` — coeus ComputeBackend (always) or burn Backend (burn-compat)
        #[cfg(not(feature = "burn-compat"))]
        pub use coeus_core::Backend;
        pub use coeus_core::{ComputeBackend, MoiraiBackend, SequentialBackend};
    }

    // ── Primary scalar / backend types ────────────────────────────────────────
    pub use coeus_core::{ComputeBackend, Float, Scalar};

    #[cfg(feature = "burn-compat")]
    pub use ::burn::tensor::backend::Backend;
    /// `Backend` type alias — coeus ComputeBackend without burn-compat, burn's Backend with it.
    #[cfg(not(feature = "burn-compat"))]
    pub use coeus_core::Backend;

    #[cfg(feature = "burn-compat")]
    pub use ::burn::tensor::Tensor;
    /// `Tensor` type alias — coeus Tensor without burn-compat, burn Tensor with it.
    #[cfg(not(feature = "burn-compat"))]
    pub use coeus_tensor::Tensor;

    /// `Int` marker type — `i32` without burn-compat, burn's Int marker with it.
    #[cfg(not(feature = "burn-compat"))]
    pub type Int = i32;
    #[cfg(feature = "burn-compat")]
    pub use ::burn::tensor::Int;

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
    pub use ::burn::tensor::backend::AutodiffBackend;
    #[cfg(feature = "burn-compat")]
    pub use ::burn::tensor::{
        activation, cast, Distribution, ElementConversion, TensorData, TensorPrimitive,
    };
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
