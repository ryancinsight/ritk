//! Optimized flat-slice B-Spline interpolation functions.
//!
//! These functions operate on the pre-extracted, **pre-filtered** B-spline
//! coefficient buffer (see [`super::prefilter`]) rather than the raw samples, so
//! the reconstruction `Σ_k c_k · β³(x − k)` interpolates the data exactly at the
//! grid points instead of smoothing them.
//!
//! # Axis convention
//! The coefficient buffer is row-major `[d0, d1, d2]` (fastest axis = `d2`), and
//! `coords` are innermost-first `[x, y, z]` — i.e. `coords[0] = x` indexes the
//! fastest axis `d2`, `coords[2] = z` the slowest axis `d0` — matching
//! `Image::world_to_index_tensor` and the linear/nearest kernels. (Pairing
//! `coords[0]` with `d0` instead silently transposed x↔z, which was invisible on
//! cubes but collapsed degenerate `z = 1` grids to zero.)

use super::cubic_bspline;
use crate::interpolation::shared::OutOfBoundsMode;

/// Whole-sample mirror reflection of index `i` into `[0, n)` (period `2n − 2`).
/// A degenerate axis (`n == 1`) always maps to `0`.
#[inline]
fn mirror(i: isize, n: isize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut m = i % period;
    if m < 0 {
        m += period;
    }
    if m >= n {
        m = period - m;
    }
    m as usize
}

/// 3-D B-spline interpolation for a single point — flat coefficient buffer.
///
/// `coeff_slice` is the pre-filtered coefficient buffer, row-major `[d0, d1, d2]`.
/// `coords = [x, y, z]` index axes `d2`, `d1`, `d0` respectively.
///
/// `Extend` uses whole-sample mirror boundary (matching ITK's B-spline
/// interpolator); `ZeroPad` returns `0.0` when the query coordinate is outside
/// the volume and treats out-of-bounds support taps as zero coefficients.
#[inline]
pub(super) fn interpolate_point_3d_flat(
    coeff_slice: &[f32],
    coords: &[f32],
    dims: &[usize],
    mode: OutOfBoundsMode,
) -> f32 {
    let x = coords[0]; // fastest axis, d2
    let y = coords[1]; // d1
    let z = coords[2]; // slowest axis, d0

    let d0 = dims[0] as isize;
    let d1 = dims[1] as isize;
    let d2 = dims[2] as isize;

    let zero_pad = mode == OutOfBoundsMode::ZeroPad;
    if zero_pad {
        let xf = x.floor() as isize;
        let yf = y.floor() as isize;
        let zf = z.floor() as isize;
        if xf < 0 || xf >= d2 || yf < 0 || yf >= d1 || zf < 0 || zf >= d0 {
            return 0.0;
        }
    }

    let stride0 = (d1 * d2) as usize; // z axis (d0)
    let stride1 = d2 as usize; // y axis (d1)

    // Upper-left corner of the 4×4×4 support (B-spline needs floor − 1).
    let x0 = x.floor() as isize - 1;
    let y0 = y.floor() as isize - 1;
    let z0 = z.floor() as isize - 1;

    let mut result = 0.0f32;
    for dz in 0..4isize {
        let zt = z0 + dz;
        let zi = if zero_pad {
            if zt < 0 || zt >= d0 {
                continue;
            }
            zt as usize
        } else {
            mirror(zt, d0)
        };
        let wz = cubic_bspline(z - zt as f32);
        let bz = zi * stride0;
        for dy in 0..4isize {
            let yt = y0 + dy;
            let yi = if zero_pad {
                if yt < 0 || yt >= d1 {
                    continue;
                }
                yt as usize
            } else {
                mirror(yt, d1)
            };
            let wy = cubic_bspline(y - yt as f32);
            let byz = bz + yi * stride1;
            for dx in 0..4isize {
                let xt = x0 + dx;
                let xi = if zero_pad {
                    if xt < 0 || xt >= d2 {
                        continue;
                    }
                    xt as usize
                } else {
                    mirror(xt, d2)
                };
                let wx = cubic_bspline(x - xt as f32);
                // SAFETY: mirror() returns an index in [0, d_k); the zero-pad
                // branch `continue`s on out-of-bounds taps, so byz + xi < len.
                result += unsafe { *coeff_slice.get_unchecked(byz + xi) } * wx * wy * wz;
            }
        }
    }
    result
}

/// 2-D B-spline interpolation for a single point — flat coefficient buffer.
///
/// `coeff_slice` is row-major `[d0, d1]`; `coords = [x, y]` index axes `d1`, `d0`.
#[inline]
pub(super) fn interpolate_point_2d_flat(
    coeff_slice: &[f32],
    coords: &[f32],
    dims: &[usize],
    mode: OutOfBoundsMode,
) -> f32 {
    let x = coords[0]; // fastest axis, d1
    let y = coords[1]; // slowest axis, d0

    let d0 = dims[0] as isize;
    let d1 = dims[1] as isize;

    let zero_pad = mode == OutOfBoundsMode::ZeroPad;
    if zero_pad {
        let xf = x.floor() as isize;
        let yf = y.floor() as isize;
        if xf < 0 || xf >= d1 || yf < 0 || yf >= d0 {
            return 0.0;
        }
    }

    let stride0 = d1 as usize; // y axis (d0)

    let x0 = x.floor() as isize - 1;
    let y0 = y.floor() as isize - 1;

    let mut result = 0.0f32;
    for dy in 0..4isize {
        let yt = y0 + dy;
        let yi = if zero_pad {
            if yt < 0 || yt >= d0 {
                continue;
            }
            yt as usize
        } else {
            mirror(yt, d0)
        };
        let wy = cubic_bspline(y - yt as f32);
        let by = yi * stride0;
        for dx in 0..4isize {
            let xt = x0 + dx;
            let xi = if zero_pad {
                if xt < 0 || xt >= d1 {
                    continue;
                }
                xt as usize
            } else {
                mirror(xt, d1)
            };
            let wx = cubic_bspline(x - xt as f32);
            // SAFETY: see the 3-D variant.
            result += unsafe { *coeff_slice.get_unchecked(by + xi) } * wx * wy;
        }
    }
    result
}
