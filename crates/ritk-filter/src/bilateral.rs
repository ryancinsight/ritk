//! Edge-preserving bilateral filter for 3-D volumes.
//!
//! # Algorithm
//! For each centre voxel **p** the output is the weighted average of all
//! voxels **q** inside the axis-aligned cube `[p Â± r]Â³`, where
//! `r = âŒˆ3 Â· Ïƒ_sâŒ‰`:
//!
//! ```text
//! w(p, q) = exp(âˆ’â€–p âˆ’ qâ€–Â² / (2 Ïƒ_sÂ²)) Â· exp(âˆ’(I(p) âˆ’ I(q))Â² / (2 Ïƒ_rÂ²))
//! Output(p) = Î£ w(p,q) Â· I(q)  /  Î£ w(p,q)
//! ```
//!
//! Out-of-bounds neighbours are **skipped** (only in-bounds voxels contribute
//! to numerator and denominator), so the estimator remains unbiased at image
//! boundaries.
//!
//! # Precision
//! All weight accumulation is performed in `f64` to avoid catastrophic
//! cancellation.
//!
//! # Complexity
//! O(n Â· (2r+1)Â³) per image, where `r = âŒˆ3 Â· Ïƒ_sâŒ‰`.
//!
//! # Performance notes (Sprint 377)
//!
//! Per-voxel cost is dominated by the inner neighbour loop, which executes
//! `(2r+1)Â³` iterations and pays one `f64` `exp` plus four table lookups
//! (`data[...]` + `spatial_w[dÂ²]` + the spatial-weight is precomputed).
//!
//! ## Carry-forward: range-domain LUT (PERF-377-02)
//!
//! The remaining `exp` call is `exp(-rdÂ² / (2Ïƒ_rÂ²))` where `rd = I(p)âˆ’I(q)`.
//! A 1-D LUT `range_w[k] = exp(-kÂ² / (2Ïƒ_rÂ²))` over `k = round(rdÂ·qscale)`
//! is the natural mirror of the spatial LUT, but a worst-case
//! quantisation analysis shows it **cannot meet the existing `1e-5`
//! uniform-image test epsilon** without an unworkably large table:
//!
//! - At the kernel knee `|rd| â‰ˆ Ïƒ_r` the absolute weight perturbation per
//!   neighbour is `|Î´w| â‰¤ |dw/drd| Â· Î”rd = (1/Ïƒ_r)Â·exp(-Â½)Â·(Â½/qscale)`.
//! - Propagating through `B(p) = Î£wÂ·I / Î£w` yields
//!   `|Î”B| â‰¤ |Î´w| Â· (|max I| + |B|) / w_avg`. With typical imaging-range
//!   `M â‰ˆ 300` HU and `w_avg â‰ˆ 0.5`:
//!   `Î”B â‰¤ (1/Ïƒ_r)Â·exp(-Â½)Â·(Â½/qscale) Â· 600`
//! - To hold `Î”B < 1e-5` with Ïƒ_r = 50: `qscale â‰³ 728 000` bins/unit,
//!   i.e. **millions of f64 entries per Ïƒ_r** â€” not a real trade.
//!
//! Alternative paths and their trade-offs:
//! 1. **Hybrid**: exact `exp` for `|rd| â‰¤ 3Ïƒ_r` (kernel-sensitive band),
//!    LUT for the heavily-attenuated tail. Saves only the asymptotic
//!    `exp` slab; expected speedup â‰¤ ~2Ã— at typical Ïƒ_r.
//! 2. **Loosen the test tolerance** to a value derived from the analysis
//!    above (~`0.05` HU for bilateral at Ïƒ_r â‰¥ 50, comparable to scipy /
//!    ITK defaults). Behaviour-equivalent to a 5â€“10Ã— faster algorithm
//!    only after the test contract is updated; a `[minor]`-class change.
//! 3. **Drop the LUT** and keep the current `exp`-per-neighbour path.
//!
//! All three are documented as deferred backlog items; PERF-377-02 (range
//! LUT) was committed at the parallelism-only stage (`ca5b49a5`) and the
//! analytical note above is the rationale for leaving the per-neighbour
//! `exp` in place. Reopen the work item when either (a) a workload
//! justifies the 2Ã— from option 1, or (b) a test-tolerance re-baselining
//! is approved by the test contract owner.
//!
//! ## Why no Huang-style median sliding histogram here
//!
//! The BilateralFilter algorithm has a closed-form per-neighbour cost
//! (`exp + two multiply-adds + one table lookup`), so the next dominant
//! cost to amortise is the **neighbour walk itself**, not a sort or
//! selection. The 5-D weights-vs-radius reduction that Huang offers for
//! median has no analogue here; the LUT is the correct optimisation.
//! See `median.rs` (PERF-377-01) for the histogram analysis on the
//! median path, where the per-voxel cost *is* sort-dominated.

use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec, rebuild};
use serde::{Deserialize, Serialize};

/// Spatial-domain sigma for bilateral filtering (Ïƒ_s > 0).
///
/// Controls the spatial extent of influence: larger values â†’ smoother edges.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(transparent)]
pub struct SpatialSigma(f64);

impl SpatialSigma {
    /// Construct with validation. Panics if `v <= 0.0` or not finite.
    pub fn new(v: f64) -> Self {
        assert!(
            v.is_finite() && v > 0.0,
            "SpatialSigma must be positive finite, got {v}"
        );
        Self(v)
    }
    /// Raw value.
    #[inline]
    pub fn get(self) -> f64 {
        self.0
    }
}

/// Intensity-domain sigma for bilateral filtering (Ïƒ_r > 0).
///
/// Controls the intensity extent of influence: larger values â†’ less edge-preserving.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(transparent)]
pub struct RangeSigma(f64);

impl RangeSigma {
    /// Construct with validation. Panics if `v <= 0.0` or not finite.
    pub fn new(v: f64) -> Self {
        assert!(
            v.is_finite() && v > 0.0,
            "RangeSigma must be positive finite, got {v}"
        );
        Self(v)
    }
    /// Raw value.
    #[inline]
    pub fn get(self) -> f64 {
        self.0
    }
}

/// Edge-preserving bilateral filter for 3-D volumes.
///
/// Combines a spatial Gaussian and an intensity-range Gaussian to smooth
/// homogeneous regions while preserving edges.
///
/// # Invariants
/// - `spatial_sigma` and `range_sigma` are clamped to a minimum of `1e-10`
///   before use, preventing division by zero.
/// - The neighbourhood radius is `âŒˆ3 Â· spatial_sigmaâŒ‰` voxels.
/// - Accumulation uses `f64` arithmetic.
pub struct BilateralFilter {
    /// Spatial Gaussian sigma in voxels.
    pub spatial_sigma: SpatialSigma,
    /// Intensity-range Gaussian sigma (same units as voxel values).
    pub range_sigma: RangeSigma,
}

impl BilateralFilter {
    /// Construct a new bilateral filter.
    ///
    /// # Arguments
    /// * `spatial_sigma` â€” standard deviation of the spatial Gaussian (voxels).
    /// * `range_sigma`   â€” standard deviation of the intensity Gaussian.
    pub fn new(spatial_sigma: f64, range_sigma: f64) -> Self {
        Self {
            spatial_sigma: SpatialSigma::new(spatial_sigma),
            range_sigma: RangeSigma::new(range_sigma),
        }
    }

    /// Apply the filter to a 3-D image.
    ///
    /// Returns a new `Image` with identical shape and spatial metadata
    /// (origin, spacing, direction).
    ///
    /// # Errors
    /// Returns `Err` if the tensor data cannot be read as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (data, dims) = extract_vec(image)?;
        let filtered = compute(
            &data,
            dims,
            self.spatial_sigma.get(),
            self.range_sigma.get(),
        );
        Ok(rebuild(filtered, dims, image))
    }

    /// Coeus-native sister of [`BilateralFilter::apply`].
    ///
    /// Runs the identical joint spatial/range kernel via the shared `compute`
    /// host core on the image's contiguous host buffer, so the result is
    /// bitwise-identical to the Burn path. No Burn tensor is constructed.
    /// Spatial metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let spatial = self.spatial_sigma.get();
        let range = self.range_sigma.get();
        crate::native_support::map_flat_image(image, backend, |data, dims| {
            compute(data, dims, spatial, range)
        })
    }
}

/// Minimum sigma value to prevent division-by-zero in bilateral weighting.
const SIGMA_MIN: f64 = 1e-10;

// â”€â”€ bilateral_3d â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Bilateral filter on a 3-D volume stored in flat ZÃ—YÃ—X order.
///
/// # Algorithm
/// For each centre voxel **p**:
/// 1. Neighbourhood radius `r = âŒˆ3 Â· Ïƒ_sâŒ‰`.
/// 2. For each neighbour **q** in `[p Â± r]Â³` (out-of-bounds skipped):
///    `w(p, q) = exp(âˆ’d_sÂ² / (2 Ïƒ_sÂ²)) Â· exp(âˆ’d_rÂ² / (2 Ïƒ_rÂ²))`
///    where `d_s = â€–p âˆ’ qâ€–`, `d_r = |I(p) âˆ’ I(q)|`.
/// 3. `Output(p) = Î£ wÂ·I(q) / Î£ w`.
///
/// # Performance
///
/// - The spatial kernel `exp(âˆ’d_sÂ² / (2 Ïƒ_sÂ²))` depends only on the squared
///   offset, so it is precomputed once into a 1-D table
///   `spatial_w[dÂ²]` for `dÂ² âˆˆ 0..=3Â·rÂ²`. Each neighbour lookup is one
///   table-load instead of three squarings + one mul + one `exp`.
/// - The neighbourhood bounds are clamped once per centre voxel
///   (`iz..=izÂ±r` â†’ `z_lo..z_hi`, same for y, x), removing every
///   per-neighbour `as isize` cast and branch and letting the inner
///   loop walk a simple `usize` range.
/// - The outer Z-loop is parallelised across z-slices via `moirai`'s
///   adaptive work-stealing scheduler (matches the canonical pattern
///   used by `median_3d`, `rank::kernel::neighborhood_rank_3d`,
///   `jacobian_determinant`).
///   Each z-slice writes into a disjoint contiguous range of the output
///   buffer, so no synchronisation is needed across threads.
///
/// Accumulation is `f64` to avoid catastrophic cancellation.
fn compute(data: &[f32], dims: [usize; 3], spatial_sigma: f64, range_sigma: f64) -> Vec<f32> {
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);

    // Guard degenerate sigma values.
    let spatial_sigma = spatial_sigma.max(SIGMA_MIN);
    let range_sigma = range_sigma.max(SIGMA_MIN);

    let r = (3.0 * spatial_sigma).ceil() as usize;
    let inv_two_ss2 = 1.0_f64 / (2.0 * spatial_sigma * spatial_sigma);
    let inv_two_sr2 = 1.0_f64 / (2.0 * range_sigma * range_sigma);

    // Precompute the 1-D spatial-kernel lookup table:
    //   spatial_w[dÂ²] = exp(-dÂ² / (2 Ïƒ_sÂ²)),   0 â‰¤ dÂ² â‰¤ 3Â·rÂ²
    //
    // Index 0 (the centre voxel, d=0) is always 1.0; offsets beyond the
    // neighbourhood footprint are never accessed.
    let table_len = 3 * r * r + 1;
    let spatial_w: Vec<f64> = (0..table_len)
        .map(|d2| (-(d2 as f64) * inv_two_ss2).exp())
        .collect();

    let slab = ny * nx;
    let mut output = vec![0.0_f32; nz * ny * nx];

    // Parallelise over z-slices. Each z-slice produces `ny * nx` output
    // voxels in the contiguous range `[iz * slab, (iz + 1) * slab)`;
    // threads write only into their own disjoint window so no send/sync
    // is required for `output`. The read-only `data` and `spatial_w`
    // tables are `Send` + `Sync`, so the closures can be sent across
    // worker threads without copying.
    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        &mut output,
        slab,
        |iz, iz_out| {
            let z_lo = iz.saturating_sub(r);
            let z_hi = (iz + r + 1).min(nz);
            let center_val_offset = iz * slab;

            // Pre-bake `(dzÂ², dzÂ² + dyÂ²)` for every (z, y) pair we will
            // touch below: the inner x-loop only walks `dxÂ²`. Skipping
            // the `as isize` cast + the squaring per (z, y, x) ticks
            // saves `(2r+1)Â² -- (2r+1)` redundant multiplies per voxel.
            for (iy, out_row) in iz_out.chunks_exact_mut(nx).enumerate() {
                let y_lo = iy.saturating_sub(r);
                let y_hi = (iy + r + 1).min(ny);
                let center_row_offset = center_val_offset + iy * nx;

                for (ix, cell) in out_row.iter_mut().enumerate() {
                    let center_flat = center_row_offset + ix;
                    let center_val = data[center_flat] as f64;

                    let x_lo = ix.saturating_sub(r);
                    let x_hi = (ix + r + 1).min(nx);

                    let mut weighted_sum = 0.0_f64;
                    let mut weight_total = 0.0_f64;

                    // Clamped neighbourhood walk: `(2r+1)Â³` candidates with
                    // single dynamic bound check per axis (already known),
                    // then a `usize` triple-nested loop with one table lookup
                    // and one range `exp` per neighbour. No `as isize` casts,
                    // no `as usize` round-trips, no per-neighbour branches.
                    for z in z_lo..z_hi {
                        let dz = z as isize - iz as isize;
                        let dz2 = (dz * dz) as usize;
                        let z_row_base = z * slab;
                        for y in y_lo..y_hi {
                            let dy = y as isize - iy as isize;
                            let d2_xy = dz2 + (dy * dy) as usize;
                            let row_base = z_row_base + y * nx;
                            for x in x_lo..x_hi {
                                let dx = x as isize - ix as isize;
                                let d2 = d2_xy + (dx * dx) as usize;
                                let sw = spatial_w[d2];
                                let n_val = data[row_base + x] as f64;
                                let rd = center_val - n_val;
                                let w = sw * (-rd * rd * inv_two_sr2).exp();
                                weighted_sum += w * n_val;
                                weight_total += w;
                            }
                        }
                    }

                    *cell = if weight_total > 1e-20 {
                        (weighted_sum / weight_total) as f32
                    } else {
                        data[center_flat]
                    };
                }
            }
        },
    );

    output
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
#[path = "tests_bilateral.rs"]
mod tests;

#[cfg(test)]
mod tests_native {
    use super::BilateralFilter;
    use crate::native_support::{assert_coeus_matches_coeus, make_native_image, native_vals};
    use coeus_core::SequentialBackend;

    #[test]
    fn matches_burn() {
        let vals: Vec<f32> = (0..60).map(|i| ((i * 7) % 11) as f32).collect();
        assert_coeus_matches_coeus(
            vals,
            [3, 4, 5],
            |img| {
                BilateralFilter::new(1.5, 2.0)
                    .apply(img)
                    .expect("burn bilateral")
            },
            |img, backend| BilateralFilter::new(1.5, 2.0).apply_native(img, backend),
        );
    }

    #[test]
    fn oracle_constant_field_preserved() {
        // Weighted average of equal values is that value: a constant is a fixed
        // point of the bilateral filter regardless of the kernel weights.
        let img = make_native_image(vec![6.0f32; 27], [3, 3, 3]);
        let out = BilateralFilter::new(1.5, 2.0)
            .apply_native(&img, &SequentialBackend)
            .expect("native bilateral");
        for &v in &native_vals(&out) {
            assert!(
                (v - 6.0).abs() < 1e-4,
                "constant field must be preserved, got {v}"
            );
        }
    }
}
