//! Optimized flat-slice B-Spline interpolation functions.
//!
//! These functions operate on pre-extracted flat f32 slices rather than
//! tensor objects, eliminating per-point tensor allocations.

use super::cubic_bspline;
use crate::interpolation::shared::OutOfBoundsMode;

/// 3D B-Spline interpolation for a single point — flat-slice variant.
///
/// `volume_slice` is the pre-flattened data buffer with row-major layout
/// `[dim0 × dim1 × dim2]` (fastest axis = dim2).
/// `coords` are `[coord0, coord1, coord2]` indexing the respective dimensions.
///
/// When `mode` is [`OutOfBoundsMode::ZeroPad`] and `floor(coord_d)` lies outside `[0, dim_d − 1]`
/// for any dimension `d`, the function returns `0.0` immediately.
///
/// This function performs **no tensor allocations** — all work is pure Rust
/// scalar arithmetic on the pre-extracted `volume_slice`.
#[inline]
pub(super) fn interpolate_point_3d_flat(
    volume_slice: &[f32],
    coords: &[f32],
    dims: &[usize],
    mode: OutOfBoundsMode,
) -> f32 {
    let x = coords[0];
    let y = coords[1];
    let z = coords[2];

    // Zero-pad early exit: if the query coordinate itself is outside the volume,
    // return 0.0 immediately.
    if mode == OutOfBoundsMode::ZeroPad {
        let xf = x.floor() as isize;
        let yf = y.floor() as isize;
        let zf = z.floor() as isize;
        if xf < 0
            || xf >= dims[0] as isize
            || yf < 0
            || yf >= dims[1] as isize
            || zf < 0
            || zf >= dims[2] as isize
        {
            return 0.0;
        }
    }

    // Strides for row-major [dim0, dim1, dim2] layout:
    // flat_index = xi * stride0 + yi * stride1 + zi
    let stride0 = dims[1] * dims[2];
    let stride1 = dims[2];

    // Upper-left corner of the 4×4×4 neighbourhood (B-spline requires floor − 1).
    let x0 = x.floor() as isize - 1;
    let y0 = y.floor() as isize - 1;
    let z0 = z.floor() as isize - 1;

    let dim0 = dims[0] as isize;
    let dim1 = dims[1] as isize;
    let dim2 = dims[2] as isize;

    let mut result = 0.0f32;
    let mut weight_sum = 0.0f32;

    // Sample 4×4×4 neighbourhood with direct slice indexing — no allocations.
    for dx in 0..4isize {
        let xi = x0 + dx;
        if xi < 0 || xi >= dim0 {
            continue;
        }
        let wx = cubic_bspline(x - xi as f32);
        let base0 = xi as usize * stride0;
        for dy in 0..4isize {
            let yi = y0 + dy;
            if yi < 0 || yi >= dim1 {
                continue;
            }
            let wy = cubic_bspline(y - yi as f32);
            let base01 = base0 + yi as usize * stride1;
            for dz in 0..4isize {
                let zi = z0 + dz;
                if zi < 0 || zi >= dim2 {
                    continue;
                }
                let wz = cubic_bspline(z - zi as f32);
                let weight = wx * wy * wz;
                let idx = base01 + zi as usize;
                // SAFETY: bounds checked above (xi, yi, zi all in [0, dim_k)).
                result += unsafe { *volume_slice.get_unchecked(idx) } * weight;
                weight_sum += weight;
            }
        }
    }

    // Renormalize by the accumulated weight (handles boundary renormalization when
    // some neighbourhood samples lie outside the volume).
    if weight_sum > 0.0 {
        result / weight_sum
    } else {
        0.0
    }
}

/// 2D B-Spline interpolation for a single point — flat-slice variant.
///
/// `volume_slice` is the pre-flattened data buffer with row-major layout
/// `[dim0 × dim1]` (fastest axis = dim1).
/// `coords` are `[coord0, coord1]` indexing the respective dimensions.
///
/// When `mode` is [`OutOfBoundsMode::ZeroPad`] and `floor(coord_d)` lies outside `[0, dim_d − 1]`
/// for any dimension `d`, the function returns `0.0` immediately.
#[inline]
pub(super) fn interpolate_point_2d_flat(
    volume_slice: &[f32],
    coords: &[f32],
    dims: &[usize],
    mode: OutOfBoundsMode,
) -> f32 {
    let x = coords[0];
    let y = coords[1];

    // Zero-pad early exit: if the query coordinate itself is outside the image.
    if mode == OutOfBoundsMode::ZeroPad {
        let xf = x.floor() as isize;
        let yf = y.floor() as isize;
        if xf < 0 || xf >= dims[0] as isize || yf < 0 || yf >= dims[1] as isize {
            return 0.0;
        }
    }

    // Stride for row-major [dim0, dim1] layout: flat_index = xi * stride0 + yi
    let stride0 = dims[1];

    // Upper-left corner of the 4×4 neighbourhood.
    let x0 = x.floor() as isize - 1;
    let y0 = y.floor() as isize - 1;

    let dim0 = dims[0] as isize;
    let dim1 = dims[1] as isize;

    let mut result = 0.0f32;
    let mut weight_sum = 0.0f32;

    // Sample 4×4 neighbourhood with direct slice indexing — no allocations.
    for dx in 0..4isize {
        let xi = x0 + dx;
        if xi < 0 || xi >= dim0 {
            continue;
        }
        let wx = cubic_bspline(x - xi as f32);
        let base0 = xi as usize * stride0;
        for dy in 0..4isize {
            let yi = y0 + dy;
            if yi < 0 || yi >= dim1 {
                continue;
            }
            let wy = cubic_bspline(y - yi as f32);
            let weight = wx * wy;
            let idx = base0 + yi as usize;
            // SAFETY: bounds checked above (xi, yi both in [0, dim_k)).
            result += unsafe { *volume_slice.get_unchecked(idx) } * weight;
            weight_sum += weight;
        }
    }

    // Renormalize by the accumulated weight.
    if weight_sum > 0.0 {
        result / weight_sum
    } else {
        0.0
    }
}
