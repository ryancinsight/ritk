//! Canonical neighbourhood-rank kernel shared by [`RankFilter`](super::RankFilter)
//! and [`PercentileFilter`](super::PercentileFilter).
//!
//! # Why this lives in its own module
//!
//! `RankFilter::apply` and `PercentileFilter::apply` are two thin valuation
//! layers that differ only in how they translate their *public* parameter
//! (`rank : usize` vs `percentile : f32`) into the same internal
//! rank index `k ∈ [0, se.len())`.
//!
//! The selection-and-iteration loop is identical between them at the byte
//! level — promoting it to a shared function means there is exactly one
//! place to evolve the algorithm (clamp hoists, sliding histograms,
//! SIMD interior paths, etc.) and exactly one place to test the inner
//! loop's value semantics.

use ritk_morphology::Offset3D;

/// Compute the element at absolute position `rank_idx` in the sorted
/// order of every voxel's SE neighbourhood on a 3-D `f32` volume stored
/// in row-major `(Z, Y, X)` order.
///
/// # Boundary handling
///
/// Replicate (clamp) padding: out-of-bounds indices are clamped to the
/// nearest valid index along each axis. This matches `MedianFilter`,
/// `GrayscaleDilation`, `GrayscaleErosion`, and `scipy.ndimage` with
/// `mode="nearest"`.
///
/// # Algorithm
///
/// `select_nth_unstable_by` (introselect) is `O(n)` average and avoids
/// the full `O(n log n)` sort. Asymptotically optimal for the rank
/// selection problem.
///
/// Parallelised over z-slices via `moirai`'s adaptive scheduler
/// (canonical pattern, matches `BilateralFilter::compute`,
/// `MedianFilter::median_3d`, and `jacobian_determinant`).
pub(crate) fn neighborhood_rank_3d(
    data: &[f32],
    dims: [usize; 3],
    rank_idx: usize,
    se: &[Offset3D],
) -> Vec<f32> {
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);
    let n = se.len();
    debug_assert!(rank_idx < n, "rank {rank_idx} out of range [0, {n})");

    let mut output = vec![0.0_f32; nz * ny * nx];
    let stride = ny * nx;
    let nz_i32 = nz as i32 - 1;
    let ny_i32 = ny as i32 - 1;
    let nx_i32 = nx as i32 - 1;

    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        &mut output,
        stride,
        |iz, out_slice| {
            let mut scratch: Vec<f32> = Vec::with_capacity(n);
            let iz_i32 = iz as i32;

            for (iy, out_row) in out_slice.chunks_exact_mut(nx).enumerate() {
                let iy_i32 = iy as i32;
                for (ix, out_cell) in out_row.iter_mut().enumerate() {
                    scratch.clear();
                    let ix_i32 = ix as i32;
                    for off in se {
                        let zz = (iz_i32 + off.iz()).clamp(0, nz_i32) as usize;
                        let yy = (iy_i32 + off.iy()).clamp(0, ny_i32) as usize;
                        let xx = (ix_i32 + off.ix()).clamp(0, nx_i32) as usize;
                        scratch.push(data[zz * stride + yy * nx + xx]);
                    }
                    scratch.select_nth_unstable_by(rank_idx, |a, b| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    *out_cell = scratch[rank_idx];
                }
            }
        },
    );

    output
}
