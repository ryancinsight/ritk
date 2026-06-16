//! Edge-preserving bilateral filter for 3-D volumes.
//!
//! # Algorithm
//! For each centre voxel **p** the output is the weighted average of all
//! voxels **q** inside the axis-aligned cube `[p ± r]³`, where
//! `r = ⌈3 · σ_s⌉`:
//!
//! ```text
//! w(p, q) = exp(−‖p − q‖² / (2 σ_s²)) · exp(−(I(p) − I(q))² / (2 σ_r²))
//! Output(p) = Σ w(p,q) · I(q)  /  Σ w(p,q)
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
//! O(n · (2r+1)³) per image, where `r = ⌈3 · σ_s⌉`.

use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};
use serde::{Deserialize, Serialize};

/// Spatial-domain sigma for bilateral filtering (σ_s > 0).
///
/// Controls the spatial extent of influence: larger values → smoother edges.
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

/// Intensity-domain sigma for bilateral filtering (σ_r > 0).
///
/// Controls the intensity extent of influence: larger values → less edge-preserving.
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
/// - The neighbourhood radius is `⌈3 · spatial_sigma⌉` voxels.
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
    /// * `spatial_sigma` — standard deviation of the spatial Gaussian (voxels).
    /// * `range_sigma`   — standard deviation of the intensity Gaussian.
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
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (data, dims) = extract_vec(image)?;
        let filtered = compute(
            &data,
            dims,
            self.spatial_sigma.get(),
            self.range_sigma.get(),
        );
        Ok(rebuild(filtered, dims, image))
    }
}

/// Minimum sigma value to prevent division-by-zero in bilateral weighting.
const SIGMA_MIN: f64 = 1e-10;

// ── bilateral_3d ────────────────────────────────────────────────────────────────

/// Bilateral filter on a 3-D volume stored in flat Z×Y×X order.
///
/// # Algorithm
/// For each centre voxel **p**:
/// 1. Neighbourhood radius `r = ⌈3 · σ_s⌉`.
/// 2. For each neighbour **q** in `[p ± r]³` (out-of-bounds skipped):
///    `w(p, q) = exp(−d_s² / (2 σ_s²)) · exp(−d_r² / (2 σ_r²))`
///    where `d_s = ‖p − q‖`, `d_r = |I(p) − I(q)|`.
/// 3. `Output(p) = Σ w·I(q) / Σ w`.
///
/// # Performance
///
/// - The spatial kernel `exp(−d_s² / (2 σ_s²))` depends only on the squared
///   offset, so it is precomputed once into a 1-D table
///   `spatial_w[d²]` for `d² ∈ 0..=3·r²`. Each neighbour lookup is one
///   table-load instead of three squarings + one mul + one `exp`.
/// - The neighbourhood bounds are clamped once per centre voxel
///   (`iz..=iz±r` → `z_lo..z_hi`, same for y, x), removing every
///   per-neighbour `as isize` cast and branch and letting the inner
///   loop walk a simple `usize` range.
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
    //   spatial_w[d²] = exp(-d² / (2 σ_s²)),   0 ≤ d² ≤ 3·r²
    //
    // Index 0 (the centre voxel, d=0) is always 1.0; offsets beyond the
    // neighbourhood footprint are never accessed.
    let table_len = 3 * r * r + 1;
    let mut spatial_w = vec![0.0_f64; table_len];
    for (d2, slot) in spatial_w.iter_mut().enumerate() {
        *slot = (-(d2 as f64) * inv_two_ss2).exp();
    }

    let slab = ny * nx;
    let mut output = vec![0.0_f32; nz * ny * nx];

    for iz in 0..nz {
        let z_lo = iz.saturating_sub(r);
        let z_hi = (iz + r + 1).min(nz);

        for iy in 0..ny {
            let y_lo = iy.saturating_sub(r);
            let y_hi = (iy + r + 1).min(ny);

            for ix in 0..nx {
                let center_flat = iz * slab + iy * nx + ix;
                let center_val = data[center_flat] as f64;

                let x_lo = ix.saturating_sub(r);
                let x_hi = (ix + r + 1).min(nx);

                let mut weighted_sum = 0.0_f64;
                let mut weight_total = 0.0_f64;

                // Clamped neighbourhood walk: `(2r+1)³` candidates with
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
                        let dy2 = (dy * dy) as usize;
                        let row_base = z_row_base + y * nx;
                        let d2_xy = dz2 + dy2;
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

                output[center_flat] = if weight_total > 1e-20 {
                    (weighted_sum / weight_total) as f32
                } else {
                    data[center_flat]
                };
            }
        }
    }

    output
}

// ── Tests ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_bilateral.rs"]
mod tests;
