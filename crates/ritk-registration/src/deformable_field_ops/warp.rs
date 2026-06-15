//! Image warping and streaming MSE under displacement fields.

use super::trilinear_interpolate;
use crate::parallel::CellSlice;
use ritk_spatial::VolumeDims;

/// Interpolation mode for [`warp_image`].
///
/// Intensity images use [`Trilinear`](WarpInterpolation::Trilinear) (smooth,
/// the default for registration). **Integer label maps must use
/// [`Nearest`](WarpInterpolation::Nearest)** — trilinear interpolation blends
/// distinct label ids into meaningless fractional values, corrupting the
/// segmentation. Nearest-neighbour preserves exact label ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WarpInterpolation {
    /// Trilinear interpolation (smooth) — for continuous intensity images.
    #[default]
    Trilinear,
    /// Nearest-neighbour — for integer label maps / categorical data.
    Nearest,
}

/// Nearest-neighbour sample of `data` at continuous index `(z, y, x)` with
/// clamp-to-border BC (matching [`trilinear_interpolate`]'s boundary handling).
#[inline]
fn nearest_sample(data: &[f32], dims: VolumeDims, z: f32, y: f32, x: f32) -> f32 {
    let [nz, ny, nx] = dims.0;
    let iz = (z.round() as isize).clamp(0, nz as isize - 1) as usize;
    let iy = (y.round() as isize).clamp(0, ny as isize - 1) as usize;
    let ix = (x.round() as isize).clamp(0, nx as isize - 1) as usize;
    data[(iz * ny + iy) * nx + ix]
}

/// Warp `moving` by the displacement field into a caller-provided buffer.
///
/// For each voxel `p = (iz, iy, ix)`:
///   `output[p] = moving(iz + dz[p], iy + dy[p], ix + dx[p])`
/// sampled with trilinear interpolation and clamp-to-border BC.
/// `output` must have length `dims[0] * dims[1] * dims[2]`.
pub(crate) fn warp_image_into(
    moving: &[f32],
    dims: VolumeDims,
    dz: &[f32],
    dy: &[f32],
    dx: &[f32],
    output: &mut [f32],
) {
    let [nz, ny, nx] = dims.0;
    // Parallelize over z-slices: each slice writes to a disjoint contiguous
    // range in `output`; all reads are from immutable inputs.
    let slice_len = ny * nx;
    let output = CellSlice::from_mut(output);
    moirai::for_each_index_with::<moirai::Adaptive, _>(nz, |iz| {
        let base = iz * slice_len;
        // SAFETY: `output` has length nz*ny*nx; each thread writes only to
        // its own disjoint [base, base + slice_len) range.
        let out_s = unsafe { output.slice_mut(base, slice_len) };
        for iy in 0..ny {
            for ix in 0..nx {
                let local = iy * nx + ix;
                let fi = base + local;
                let wz = iz as f32 + dz[fi];
                let wy = iy as f32 + dy[fi];
                let wx = ix as f32 + dx[fi];
                out_s[local] = trilinear_interpolate(moving, dims, wz, wy, wx);
            }
        }
    });
}

/// Warp `moving` by the displacement field `(dz, dy, dx)` into a caller-provided
/// buffer using **nearest-neighbour** sampling (clamp-to-border BC).
///
/// For label maps: preserves exact integer ids where trilinear would blend them.
pub(crate) fn warp_image_nearest_into(
    moving: &[f32],
    dims: VolumeDims,
    dz: &[f32],
    dy: &[f32],
    dx: &[f32],
    output: &mut [f32],
) {
    let [nz, ny, nx] = dims.0;
    let slice_len = ny * nx;
    let output = CellSlice::from_mut(output);
    moirai::for_each_index_with::<moirai::Adaptive, _>(nz, |iz| {
        let base = iz * slice_len;
        // SAFETY: `output` has length nz*ny*nx; each thread writes only to its
        // own disjoint [base, base + slice_len) range.
        let out_s = unsafe { output.slice_mut(base, slice_len) };
        for iy in 0..ny {
            for ix in 0..nx {
                let local = iy * nx + ix;
                let fi = base + local;
                let wz = iz as f32 + dz[fi];
                let wy = iy as f32 + dy[fi];
                let wx = ix as f32 + dx[fi];
                out_s[local] = nearest_sample(moving, dims, wz, wy, wx);
            }
        }
    });
}

/// Warp `moving` by the displacement field `(dz, dy, dx)` with the chosen
/// interpolation [`mode`](WarpInterpolation).
///
/// For each voxel `p = (iz, iy, ix)`:
///   `warped(p) = moving(iz + dz[p], iy + dy[p], ix + dx[p])`
/// sampled per `mode` (trilinear for intensities, nearest for label maps) with
/// clamp-to-border BC.
pub fn warp_image(
    moving: &[f32],
    dims: VolumeDims,
    dz: &[f32],
    dy: &[f32],
    dx: &[f32],
    mode: WarpInterpolation,
) -> Vec<f32> {
    let n = dims.total_voxels();
    let mut warped = vec![0.0_f32; n];
    match mode {
        WarpInterpolation::Trilinear => warp_image_into(moving, dims, dz, dy, dx, &mut warped),
        WarpInterpolation::Nearest => {
            warp_image_nearest_into(moving, dims, dz, dy, dx, &mut warped)
        }
    }
    warped
}

/// Compute `mean((fixed − warped)²)` from a pre-warped moving buffer.
///
/// Zero-allocation MSE: takes the already-warped `warped` slice directly
/// instead of re-warping `moving` through the displacement field.  Use this
/// in registration hot loops where `warp_image_into` has already produced
/// `warped` for the next force computation — the redundant warp performed
/// by [`compute_mse_streaming`] is the dominant cost (~50% of the per-iteration
/// MSE step on 256³ fields).
///
/// # Arguments
/// - `fixed`  — reference image (flat `[nz, ny, nx]` Z-major).
/// - `warped` — pre-warped moving image, same length as `fixed`.
///
/// # Returns
/// Mean squared error as `f64`.
///
/// # Panics
/// Panics in debug builds if `fixed.len() != warped.len()`.
pub(crate) fn compute_mse_inplace(fixed: &[f32], warped: &[f32]) -> f64 {
    // Release-safe length check: the parallel reduction below would otherwise
    // read out of bounds on `warped` if the two slices have different lengths.
    assert_eq!(
        fixed.len(),
        warped.len(),
        "compute_mse_inplace: fixed and warped must have the same length"
    );
    let n = fixed.len();
    if n == 0 {
        return 0.0;
    }
    // Parallel reduction over fixed/aligned chunks; partial sums are summed
    // in f64 to bound cross-chunk associativity error well below f32 epsilon.
    //
    // Chunk size: 4096 f32 = 16 KB — fits in L1 cache on every modern x86/ARM
    // core, and yields 4096 (≈ n/16K) chunks on 256³ fields, which is enough
    // parallelism to saturate 4–16 worker threads without per-chunk overhead
    // swamping the per-voxel subtraction.
    const MSE_INPLACE_CHUNK: usize = 4096;
    let chunk = MSE_INPLACE_CHUNK;
    let n_chunks = n.div_ceil(chunk);
    let sum = moirai::reduce_index_with::<moirai::Adaptive, _, _, _>(
        n_chunks,
        0.0_f64,
        |ci| {
            let lo = ci * chunk;
            let hi = (lo + chunk).min(n);
            let mut s = 0.0_f64;
            for i in lo..hi {
                let d = (fixed[i] - warped[i]) as f64;
                s += d * d;
            }
            s
        },
        |a, b| a + b,
    );
    sum / n as f64
}

/// Compute `mean((fixed − warp(moving, D))²)` without materialising a warped buffer.
///
/// Streams trilinear samples of `moving` under displacement `D = (dz, dy, dx)` directly
/// into a squared-error accumulator. No intermediate `Vec<f32>` is allocated.
///
/// Returns the mean squared error as `f64`.
///
/// **Prefer [`compute_mse_inplace`] in registration hot loops** when the warped
/// image is already in hand — it avoids a redundant warp on every iteration.
pub(crate) fn compute_mse_streaming(
    fixed: &[f32],
    moving: &[f32],
    dims: VolumeDims,
    dz: &[f32],
    dy: &[f32],
    dx: &[f32],
) -> f64 {
    let [nz, ny, nx] = dims.0;
    // Parallel reduction over z-slices; each slice accumulates its own
    // partial sum from immutable inputs only. Per-slice sequential summation
    // order is preserved, so results match the sequential implementation up
    // to the associativity of the cross-slice combine (f64 accumulator).
    let slice_len = ny * nx;
    let sum = moirai::reduce_index_with::<moirai::Adaptive, _, _, _>(
        nz,
        0.0_f64,
        |iz| {
            let base = iz * slice_len;
            let mut s = 0.0_f64;
            for iy in 0..ny {
                for ix in 0..nx {
                    let fi = base + iy * nx + ix;
                    let wz = iz as f32 + dz[fi];
                    let wy = iy as f32 + dy[fi];
                    let wx = ix as f32 + dx[fi];
                    let warped = trilinear_interpolate(moving, dims, wz, wy, wx);
                    let diff = (fixed[fi] - warped) as f64;
                    s += diff * diff;
                }
            }
            s
        },
        |a, b| a + b,
    );
    sum / fixed.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Warp of a constant image is constant regardless of displacement.
    #[test]
    fn warp_constant_image() {
        let dims = VolumeDims::new([4, 4, 4]);
        let n = 4 * 4 * 4;
        let data = vec![5.0_f32; n];
        let dz = vec![0.5_f32; n];
        let dy = vec![-0.3_f32; n];
        let dx = vec![1.1_f32; n];
        let out = warp_image(&data, dims, &dz, &dy, &dx, WarpInterpolation::Trilinear);
        for &v in &out {
            assert!((v - 5.0).abs() < 1e-5, "expected 5.0, got {v}");
        }
    }

    /// Nearest-neighbour warp preserves exact label ids (no blending), where a
    /// trilinear warp of the same field would produce fractional values.
    #[test]
    fn warp_nearest_preserves_labels() {
        let dims = VolumeDims::new([1, 1, 6]);
        // Two adjacent label regions: ids 0,0,0, 5,5,5 along x.
        let labels = vec![0.0_f32, 0.0, 0.0, 5.0, 5.0, 5.0];
        let n = 6;
        // Shift by −1 voxel in x: voxel ix samples moving(ix − 1).
        let dz = vec![0.0_f32; n];
        let dy = vec![0.0_f32; n];
        let dx = vec![-1.0_f32; n];

        let nn = warp_image(&labels, dims, &dz, &dy, &dx, WarpInterpolation::Nearest);
        // Every output value must be an exact original id (0 or 5), never blended.
        for &v in &nn {
            assert!(v == 0.0 || v == 5.0, "nearest blended a label: {v}");
        }
        // The 0→5 boundary shifts right by one voxel: index 4 becomes 5 (was the
        // boundary), confirming the shift direction + NN selection.
        assert_eq!(nn, vec![0.0, 0.0, 0.0, 0.0, 5.0, 5.0]);

        // A half-voxel shift trilinearly blends 0 and 5 → fractional; nearest does not.
        let dx_half = vec![-0.5_f32; n];
        let tri = warp_image(
            &labels,
            dims,
            &dz,
            &dy,
            &dx_half,
            WarpInterpolation::Trilinear,
        );
        assert!(
            tri.iter().any(|&v| v != 0.0 && v != 5.0),
            "trilinear should blend at the boundary (control for the NN guarantee)"
        );
        let nn_half = warp_image(
            &labels,
            dims,
            &dz,
            &dy,
            &dx_half,
            WarpInterpolation::Nearest,
        );
        for &v in &nn_half {
            assert!(v == 0.0 || v == 5.0, "nearest blended at half-voxel: {v}");
        }
    }

    /// Warp with zero displacement must return the original image.
    #[test]
    fn warp_identity_displacement() {
        let dims = VolumeDims::new([6, 6, 6]);
        let [nz, ny, nx] = dims.0;
        let n = nz * ny * nx;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let zeros = vec![0.0_f32; n];
        let out = warp_image(
            &data,
            dims,
            &zeros,
            &zeros,
            &zeros,
            WarpInterpolation::Trilinear,
        );
        for (i, (&orig, &warped)) in data.iter().zip(out.iter()).enumerate() {
            assert!(
                (orig - warped).abs() < 1e-5,
                "voxel {i}: expected {orig}, got {warped}"
            );
        }
    }

    /// Streaming MSE of identical images with zero displacement is zero.
    #[test]
    fn streaming_mse_identical_is_zero() {
        let dims = VolumeDims::new([4, 4, 4]);
        let n = 4 * 4 * 4;
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let zeros = vec![0.0_f32; n];
        let mse = compute_mse_streaming(&data, &data, dims, &zeros, &zeros, &zeros);
        assert!(mse < 1e-10, "expected ~0, got {mse}");
    }

    /// `compute_mse_inplace` of identical buffers is zero.
    #[test]
    fn inplace_mse_identical_is_zero() {
        let data: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();
        let mse = compute_mse_inplace(&data, &data);
        assert!(mse < 1e-10, "expected ~0, got {mse}");
    }

    /// `compute_mse_inplace` of a known constant offset matches the analytical
    /// mean squared error, independent of warp path.  Tolerance `1e-6`
    /// accommodates the ~1 ULP-per-add accumulated reduction error across
    /// the parallel `n / 4096` chunks.
    #[test]
    fn inplace_mse_constant_offset() {
        let fixed: Vec<f32> = (0..1000).map(|i| i as f32 * 0.01).collect();
        let warped: Vec<f32> = fixed.iter().map(|&v| v + 0.5).collect();
        let mse = compute_mse_inplace(&fixed, &warped);
        assert!(
            (mse - 0.25).abs() < 1e-6,
            "expected MSE 0.25 for offset 0.5, got {mse}"
        );
    }

    /// `compute_mse_inplace` matches a hand-rolled reference sum at a
    /// non-zero displacement — guards against cross-chunk reduction bugs
    /// that the all-zero test cannot catch.
    #[test]
    fn inplace_matches_reference_at_nonzero_displacement() {
        let dims = VolumeDims::new([8, 8, 8]);
        let n = 8 * 8 * 8;
        let fixed: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        let moving: Vec<f32> = (0..n).map(|i| (i as f32).cos()).collect();
        // Constant non-zero translation along x.
        let dz = vec![0.0_f32; n];
        let dy = vec![0.0_f32; n];
        let dx = vec![1.0_f32; n];
        let warped = warp_image(&moving, dims, &dz, &dy, &dx, WarpInterpolation::Trilinear);
        let mse_inplace = compute_mse_inplace(&fixed, &warped);
        // Hand-rolled reference sum.
        let mse_ref: f64 = fixed
            .iter()
            .zip(warped.iter())
            .map(|(&f, &w)| {
                let d = (f - w) as f64;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        assert!(
            (mse_inplace - mse_ref).abs() < 1e-9,
            "inplace {mse_inplace} != reference {mse_ref}"
        );
    }

    /// `compute_mse_inplace` and `compute_mse_streaming` agree on the
    /// zero-displacement MSE (both reduce to the same squared-difference sum).
    #[test]
    fn inplace_matches_streaming_zero_displacement() {
        let dims = VolumeDims::new([6, 6, 6]);
        let n = 6 * 6 * 6;
        let fixed: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        let moving: Vec<f32> = (0..n).map(|i| (i as f32).cos()).collect();
        let zeros = vec![0.0_f32; n];
        let mse_streaming = compute_mse_streaming(&fixed, &moving, dims, &zeros, &zeros, &zeros);
        // `warped == moving` because displacement is zero, so the two helpers
        // must produce identical results.
        let mse_inplace = compute_mse_inplace(&fixed, &moving);
        assert!(
            (mse_streaming - mse_inplace).abs() < 1e-9,
            "inplace {mse_inplace} != streaming {mse_streaming} at zero displacement"
        );
    }
}
