//! Atlas-keyed sister module for trilinear interpolation.
//!
//! Per `docs/adr/0012-ritk-burn-trait-rebind.md` §Decision §Sub-batch #3.f:
//! atlas-side parallel module exposing [`atlas_trilinear_interpolate`]
//! sister function that mirrors the legacy
//! `super::trilinear_interpolation::<B>(Tensor<B,5>, Tensor<B,5>) -> Tensor<B,5>`,
//! over `Image<f32, MoiraiBackend, 5>` atlas-side carriers shape
//! `[B, C, D, H, W]` instead of the Burn-keyed `Tensor<B, 5>`.
//!
//! Strictly additive on production surface per the sub-batch #3.f
//! atomic-boundary invariant: every public symbol of `super::tensor_trilinear`
//! is preserved verbatim. The Atlas twin delegates the maths to the legacy
//! function on a fixed `burn_ndarray::NdArray<f32>` backend (canonical f32
//! host-resident burn backend, oracle-test fidelity), then re-wraps the
//! output into the Atlas-typed image carrier.
//!
//! **Data extraction** uses [`tensor_ops::extract_image_slice`] (the
//! canonical coeus-side adapter for `Image<f32, B, D>` → borrowed `&[f32]`).
//! The `B::DeviceBuffer<f32>: CpuAddressableStorage<f32>` trait bound
//! matches the canonical Atlas-side host-slice-extraction contract.
//!
//! **No `Cargo.toml` mutation** in sub-batch #3.f beyond additive
//! `coeus-tensor` line; the sister inherits the existing
//! `ritk-tensor-ops` + `burn-ndarray` deps from sub-batch #1's manifest
//! setup. To avoid a `thiserror` dep-add, the [`AtlasTrilinearError`]
//! enum carries a hand-rolled `Display` impl via [`std::fmt::Display`] +
//! [`std::error::Error`] — zero new crate-graph edges in this file.

use coeus_core::{ComputeBackend, CpuAddressableStorage, MoiraiBackend};
use ritk_image::native::Image;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_tensor_ops::native as tensor_ops;

// ── Error semantics ───────────────────────────────────────────────────────

/// Atlas-side error variants for trilinear interpolation.
///
/// The legacy `super::trilinear_interpolation` returns `Tensor` directly
/// without an error path; the Atlas twin consolidates the recoverable
/// failure shapes (host-slice extract failure on non-CPU backend, atlas
/// carrier construction failure on shape / origin / spacing / direction
/// conflict) into a `Result`-bearing API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtlasTrilinearError {
    /// Host-slice extract failed (non-CPU-resident device buffer).
    Extract(String),
    /// Atlas-side carrier construction failed.
    Construct(String),
}

impl std::fmt::Display for AtlasTrilinearError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Extract(s) => write!(f, "atlas trilinear interpolate extract failed: {s}"),
            Self::Construct(s) => write!(
                f,
                "atlas trilinear interpolate carrier construction failed: {s}"
            ),
        }
    }
}

impl std::error::Error for AtlasTrilinearError {}

// ── Public sister API ─────────────────────────────────────────────────────

/// Atlas-typed sister to [`super::trilinear_interpolation`].
///
/// # Arguments
/// * `image` - Input image `Image<f32, B, 5>` shape `[B, C, D, H, W]`
/// * `grid` - Sampling grid `Image<f32, B, 5>` shape `[B, 3, D, H, W]` in
///   voxel coordinates (z, y, x)
///
/// # Returns
/// Interpolated image `Image<f32, MoiraiBackend, 5>` shape `[B, C, D, H, W]`.
pub fn atlas_trilinear_interpolate<B>(
    image: &Image<f32, B, 5>,
    grid: &Image<f32, B, 5>,
) -> Result<Image<f32, MoiraiBackend, 5>, AtlasTrilinearError>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    type LegacyBackend = burn_ndarray::NdArray<f32>;

    let (image_values, image_shape) = tensor_ops::extract_image_slice(image)
        .map_err(|e| AtlasTrilinearError::Extract(e.to_string()))?;
    let (grid_values, grid_shape) = tensor_ops::extract_image_slice(grid)
        .map_err(|e| AtlasTrilinearError::Extract(e.to_string()))?;

    let dev = <LegacyBackend as burn::tensor::backend::Backend>::Device::default();
    let image_tensor = burn::tensor::Tensor::<LegacyBackend, 5>::from_data(
        burn::tensor::TensorData::new(image_values.to_vec(), burn::tensor::Shape::new(image_shape)),
        &dev,
    );
    let grid_tensor = burn::tensor::Tensor::<LegacyBackend, 5>::from_data(
        burn::tensor::TensorData::new(grid_values.to_vec(), burn::tensor::Shape::new(grid_shape)),
        &dev,
    );
    let out_tensor = super::trilinear_interpolation::<LegacyBackend>(image_tensor, grid_tensor);
    let out_shape = out_tensor.dims();
    let out_values: Vec<f32> = out_tensor
        .into_data()
        .into_vec::<f32>()
        .expect("trilinear output tensor must be f32");
    Image::<f32, MoiraiBackend, 5>::from_flat(
        out_values,
        out_shape,
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
    )
    .map_err(|e| AtlasTrilinearError::Construct(e.to_string()))
}
