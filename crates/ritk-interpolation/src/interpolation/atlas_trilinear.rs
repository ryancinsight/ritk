//! Atlas-keyed sister module for trilinear interpolation.
//!
//! Exposes [`atlas_trilinear_interpolate`] over `Image<f32, MoiraiBackend, 5>`
//! atlas-side carriers shape `[B, C, D, H, W]`, the coeus-native counterpart of
//! the Burn-keyed `super::tensor_trilinear::trilinear_interpolation::<B>`.
//!
//! The maths runs entirely on the coeus-native
//! [`super::native::trilinear_interpolation`] over flat host buffers — the
//! same trilinear contract as the Burn path (identical layout, voxel-coordinate
//! `(z, y, x)` grid, and interpolation/clamp semantics), with no Burn tensor
//! round-trip. Host slices are extracted via
//! [`tensor_ops::extract_image_slice`]; the
//! `B::DeviceBuffer<f32>: CpuAddressableStorage<f32>` bound is the canonical
//! host-slice-extraction contract.
//!
//! [`AtlasTrilinearError`] carries a hand-rolled `Display`/`Error` impl to avoid
//! a `thiserror` dep-add — zero new crate-graph edges in this file.

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
    let (image_values, image_shape) = tensor_ops::extract_image_slice(image)
        .map_err(|e| AtlasTrilinearError::Extract(e.to_string()))?;
    let (grid_values, grid_shape) = tensor_ops::extract_image_slice(grid)
        .map_err(|e| AtlasTrilinearError::Extract(e.to_string()))?;

    let [b, c, d, h, w] = image_shape;
    let [_grid_b, _three, out_d, out_h, out_w] = grid_shape;

    let out_values = super::native::trilinear_interpolation::<f32>(
        image_values,
        b,
        c,
        d,
        h,
        w,
        grid_values,
        out_d,
        out_h,
        out_w,
    );

    Image::<f32, MoiraiBackend, 5>::from_flat(
        out_values,
        [b, c, out_d, out_h, out_w],
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
    )
    .map_err(|e| AtlasTrilinearError::Construct(e.to_string()))
}
