//! In-bounds mask computation for out-of-bounds zero-padding.
//!
//! The pattern `floor_c.clone().equal(floor_c.clamp(0.0, max)).float()` is
//! duplicated across every interpolation dimension. This helper factors it
//! into a single function, reducing clone overhead and centralising the
//! contract: a sample is in-bounds along one axis iff its floor coordinate
//! equals its clamped floor coordinate.
//!
//! # Contract
//!
//! Returns a tensor of `1.0` (in-bounds) or `0.0` (out-of-bounds) for each
//! sample along a single axis. The product across all axes gives the
//! multi-dimensional in-bounds mask.

use burn::tensor::{backend::Backend, Tensor};

/// Out-of-bounds handling policy for interpolation kernels.
///
/// - `ZeroPad`: coordinates outside the image extent produce output 0.
/// - `Clamp`: coordinates outside the image extent are clamped to the
///   nearest valid boundary value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutOfBoundsMode {
    /// Return 0 for out-of-bounds coordinates.
    ZeroPad,
    /// Clamp out-of-bounds coordinates to the nearest valid boundary.
    Clamp,
}

/// Compute a per-sample in-bounds mask for a single axis.
///
/// `floor_c` is the floor of the continuous coordinate along this axis.
/// `max` is the maximum valid index (`d_k - 1`).
///
/// A sample is in-bounds iff `floor_c == clamp(floor_c, 0, max)`, i.e.
/// the coordinate was already within `[0, max]` before clamping.
///
/// # Implementation
///
/// `floor_c.clamp(0.0, max)` consumes `floor_c` (Burn's ownership model).
/// The caller must `.clone()` `floor_c` before calling if the original
/// value is needed later. However, by accepting `floor_c` by value and
/// by accepting `floor_c` by value and
/// cloning internally **only** when `mode` is `ZeroPad`, the `Clamp`
/// path pays zero cost — the compiler dead-code eliminates the entire
/// body when `mode` is `Clamp` at monomorphization time.
///
/// When `mode` is `Clamp`, returns `None` (caller skips mask logic).
#[inline]
pub fn in_bounds_mask<B: Backend>(
    floor_c: Tensor<B, 1>,
    max: f64,
    mode: OutOfBoundsMode,
) -> Option<Tensor<B, 1>> {
    if mode == OutOfBoundsMode::ZeroPad {
        let clamped = floor_c.clone().clamp(0.0, max);
        Some(floor_c.equal(clamped).float())
    } else {
        None
    }
}

/// Multiply a list of per-axis in-bounds masks into a single joint mask.
///
/// `None` entries are treated as all-ones (axis is unconditionally in-bounds,
/// which happens when `mode` is `Clamp`). If every entry is `None`,
/// returns `None` (no masking needed).
#[inline]
pub fn joint_in_bounds_mask<B: Backend>(masks: &[Option<Tensor<B, 1>>]) -> Option<Tensor<B, 1>> {
    let mut product: Option<Tensor<B, 1>> = None;
    for m in masks.iter().flatten() {
        product = Some(match product {
            None => m.clone(),
            Some(p) => p * m.clone(),
        });
    }
    product
}
