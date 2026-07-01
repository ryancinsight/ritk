//! Coeus-native trilinear interpolation.
//!
//! Atlas migration target: the same contract as
//! [`super::tensor_trilinear::trilinear_interpolation`] (Burn-generic), but
//! operating directly on flat row-major buffers through
//! [`coeus_core::Scalar`] instead of `burn::tensor::Tensor`. Both paths
//! implement the identical trilinear-interpolation contract; this module does
//! not change the math, only the substrate. `read_jpeg_coeus`/
//! `trilinear_interpolation` (`ritk-jpeg`) established the parallel-path
//! template this module follows: production Burn API stays, a Coeus-native
//! equivalent is added and verified against it, and the Burn path is only
//! removed once every caller has migrated (tracked under the
//! MIG-433-06/437-04/439-03 backend-migration family).
//!
//! # Layout
//! `image` is `[B, C, D, H, W]` flattened row-major; `grid` is `[B, 3, D', H',
//! W']` flattened row-major with channel `0 = z, 1 = y, 2 = x` sample
//! coordinates in voxel space. Output is `[B, C, D', H', W']` flattened
//! row-major.

use coeus_core::Scalar;

/// Trilinear interpolation for 3D flat buffers (Coeus-native).
///
/// See the module documentation for the buffer layout contract. Panics if
/// `image.len() != b * c * d * h * w` or `grid.len() != b * 3 * out_d *
/// out_h * out_w` (a caller invariant violation, not an input-data error —
/// matches the Burn path, which panics identically via tensor shape
/// mismatches).
#[allow(clippy::too_many_arguments)]
pub fn trilinear_interpolation_coeus<T: Scalar>(
    image: &[T],
    b: usize,
    c: usize,
    d: usize,
    h: usize,
    w: usize,
    grid: &[T],
    out_d: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<T> {
    assert_eq!(
        image.len(),
        b * c * d * h * w,
        "trilinear_interpolation_coeus: image buffer length mismatch"
    );
    assert_eq!(
        grid.len(),
        b * 3 * out_d * out_h * out_w,
        "trilinear_interpolation_coeus: grid buffer length mismatch"
    );

    let out_elements = out_d * out_h * out_w;
    let grid_channel_stride = out_elements;
    let grid_batch_stride = 3 * out_elements;
    let image_channel_stride = d * h * w;
    let image_batch_stride = c * image_channel_stride;
    let stride_d = h * w;
    let stride_h = w;

    let mut out = vec![T::zero(); b * c * out_elements];

    for bi in 0..b {
        let grid_base = bi * grid_batch_stride;
        let z_base = grid_base;
        let y_base = grid_base + grid_channel_stride;
        let x_base = grid_base + 2 * grid_channel_stride;

        for p in 0..out_elements {
            let z = grid[z_base + p];
            let y = grid[y_base + p];
            let x = grid[x_base + p];

            let (z0_idx, z1_idx, wz0, wz1) = floor_weights(z, d);
            let (y0_idx, y1_idx, wy0, wy1) = floor_weights(y, h);
            let (x0_idx, x1_idx, wx0, wx1) = floor_weights(x, w);

            let idx00 = z0_idx * stride_d + y0_idx * stride_h;
            let idx01 = z0_idx * stride_d + y1_idx * stride_h;
            let idx10 = z1_idx * stride_d + y0_idx * stride_h;
            let idx11 = z1_idx * stride_d + y1_idx * stride_h;

            for ci in 0..c {
                let channel_base = bi * image_batch_stride + ci * image_channel_stride;
                let v000 = image[channel_base + idx00 + x0_idx];
                let v001 = image[channel_base + idx00 + x1_idx];
                let v010 = image[channel_base + idx01 + x0_idx];
                let v011 = image[channel_base + idx01 + x1_idx];
                let v100 = image[channel_base + idx10 + x0_idx];
                let v101 = image[channel_base + idx10 + x1_idx];
                let v110 = image[channel_base + idx11 + x0_idx];
                let v111 = image[channel_base + idx11 + x1_idx];

                let w00 = v000 * wx0 + v001 * wx1;
                let w01 = v010 * wx0 + v011 * wx1;
                let w10 = v100 * wx0 + v101 * wx1;
                let w11 = v110 * wx0 + v111 * wx1;

                let w0 = w00 * wy0 + w01 * wy1;
                let w1 = w10 * wy0 + w11 * wy1;

                let value = w0 * wz0 + w1 * wz1;

                let out_base = bi * c * out_elements + ci * out_elements;
                out[out_base + p] = value;
            }
        }
    }

    out
}

/// Floor the coordinate into its two neighboring voxel indices (each
/// independently clamped to `[0, extent-1]`, matching the Burn path's
/// `z0.clamp(...)` / `z1.clamp(...)` pair exactly — including the case where
/// a negative coordinate clamps both neighbors to index 0) and the
/// interpolation weight pair `(w0, w1)` derived from the *unclamped* floor
/// (also matching the Burn path, which computes weights before clamping).
/// The index round-trips through `f64`, exact for the small integer
/// magnitudes voxel coordinates take.
#[inline]
fn floor_weights<T: Scalar>(coord: T, extent: usize) -> (usize, usize, T, T) {
    let coord_f64 = coord.to_f64();
    let floor_f64 = coord_f64.floor();
    let w1 = coord - T::from_f64(floor_f64);
    let w0 = T::one() - w1;
    let max_idx = extent.saturating_sub(1);
    let clamp = |v: f64| -> usize {
        if v < 0.0 {
            0
        } else if v >= max_idx as f64 {
            max_idx
        } else {
            v as usize
        }
    };
    (clamp(floor_f64), clamp(floor_f64 + 1.0), w0, w1)
}

#[cfg(test)]
#[path = "tests_coeus_trilinear.rs"]
mod tests;
