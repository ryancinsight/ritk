//! Contrast Limited Adaptive Histogram Equalization (CLAHE) filter.
//!
//! # Mathematical Specification
//!
//! CLAHE (Zuiderveld 1994) divides an image into a grid of `n_tiles_y × n_tiles_x`
//! non-overlapping rectangular tiles, computes a clip-limited histogram mapping
//! per tile, and interpolates between neighbouring tile mappings for each pixel.
//!
//! ## Tile CDF computation (per tile T with n_T pixels, bins B):
//!
//! 1. **Histogram**: `H_T[b] = |{p ∈ T : bin(p) = b}|`
//!    where `bin(p) = floor((p - v_min) / span * (B - 1))`, clamped to `[0, B-1]`.
//!
//! 2. **Clip threshold**: `C = max(1, alpha * n_T / B)`
//!    where `alpha = clip_limit` (dimensionless factor; uniform distribution ≡ 1.0).
//!
//! 3. **Redistribution**:
//!    - excess `E = Σ_{b: H_T[b] > C} (H_T[b] - C)`
//!    - `H'_T[b] = min(H_T[b], C) + floor(E / B)` for all b
//!    - `H'_T[b] += 1` for the first `E mod B` bins (distributes the integer remainder)
//!
//! 4. **Normalised CDF**: `F_T[b] = (Σ_{i=0}^{b} H'_T[i]) / n_T`
//!    Domain: `[0.0, 1.0]`.
//!
//! ## Per-pixel mapping:
//!
//! For pixel at image coordinate `(y, x)`:
//!   - compute `bin_v = floor((v - v_min) / span * (B - 1))`, clamped to `[0, B-1]`
//!   - find the four surrounding tile centers `(ty0, tx0), (ty0, tx1), (ty1, tx0), (ty1, tx1)`
//!   - bilinear interpolation weights: `u = (y_f - ty0)`, `t = (x_f - tx0)` (both clamped to `[0, 1]`)
//!     where `y_f = (y - tile_h/2) / tile_h`, `x_f = (x - tile_w/2) / tile_w`
//!   - `mapped = (1-u)*((1-t)*F_c00[bin_v] + t*F_c01[bin_v]) + u*((1-t)*F_c10[bin_v] + t*F_c11[bin_v])`
//!   - `output(y, x) = v_min + mapped * span`
//!
//! ## 3D application:
//!
//! For 3D medical images, CLAHE is applied independently to each axial (Z/depth) slice,
//! which is standard practice in medical image processing toolkits (ITK, ImageJ).
//!
//! ## Output invariant:
//!
//! `output_range ⊆ [v_min, v_max]` where `v_min, v_max` are the per-slice min/max.
//! When `v_min == v_max` (uniform slice), output equals input.
//!
//! # References
//!
//! - Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization.
//!   In *Graphics Gems IV* (pp. 474-485). Academic Press.
//! - FIJI/ImageJ CLAHE plugin: Stephan Saalfeld et al.

use crate::filter::ops::{extract_vec, rebuild};
use crate::image::Image;
use anyhow::Result;
use burn::tensor::backend::Backend;
use rayon::prelude::*;

/// Contrast Limited Adaptive Histogram Equalization (CLAHE) filter.
///
/// Applies the Zuiderveld (1994) algorithm independently to each axial slice
/// of a 3-D image. Spatial metadata (origin, spacing, direction) is preserved.
///
/// # Default parameters
/// - `tile_grid_size = [8, 8]` — 8×8 tile grid per slice
/// - `clip_limit = 40.0` — clip factor relative to uniform distribution
/// - `bins = 256` — histogram bin count
///
/// # Complexity
/// O(depth × rows × cols × (2r+1)) where r = 0 for bin lookups; total O(N × B/T)
/// for T tiles, B bins per tile (amortized over interpolation).
pub struct ClaheFilter {
    /// Number of tiles in [rows, cols] direction per 2D slice.
    pub tile_grid_size: [usize; 2],
    /// Clip limit factor alpha. Clip threshold per tile = max(1, alpha * tile_pixels / bins).
    /// Values near 1.0 approximate uniform distribution (no enhancement).
    /// ImageJ default = 3.0 (slope), Zuiderveld common default = 40.0.
    pub clip_limit: f32,
    /// Histogram bin count. Default 256.
    pub bins: usize,
}

impl ClaheFilter {
    /// Create a new CLAHE filter with explicit parameters.
    ///
    /// # Arguments
    /// * `tile_grid_size` — `[n_tiles_rows, n_tiles_cols]`, minimum 1 along each axis.
    /// * `clip_limit` — clip factor ≥ 1.0 (1.0 = no clipping; higher = more enhancement).
    /// * `bins` — histogram bins ≥ 2.
    pub fn new(tile_grid_size: [usize; 2], clip_limit: f32, bins: usize) -> Self {
        Self {
            tile_grid_size: [tile_grid_size[0].max(1), tile_grid_size[1].max(1)],
            clip_limit: clip_limit.max(1.0),
            bins: bins.max(2),
        }
    }

    /// Default CLAHE filter: 8×8 tiles, clip_limit=40.0, 256 bins.
    ///
    /// Matches common ImageJ/SimpleITK defaults for medical image preprocessing.
    pub fn default_medical() -> Self {
        Self::new([8, 8], 40.0, 256)
    }

    /// Apply CLAHE to a 3-D image.
    ///
    /// Processes each axial (Z=depth) slice independently. Returns a new image
    /// with the same spatial metadata as the input.
    ///
    /// # Errors
    /// Returns `Err` if the tensor data cannot be extracted as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        let shape = image.shape();
        let [depth, rows, cols] = [shape[0], shape[1], shape[2]];

        let (vals_vec, dims) = extract_vec(image)?;
        let vals = &vals_vec;

        // Process each axial slice in parallel via Rayon.
        let n_tiles_y = self.tile_grid_size[0].min(rows).max(1);
        let n_tiles_x = self.tile_grid_size[1].min(cols).max(1);
        let clip_limit = self.clip_limit;
        let bins = self.bins;
        let slice_size = rows * cols;

        let out_slices: Vec<Vec<f32>> = (0..depth)
            .into_par_iter()
            .map(|d| {
                let slice = &vals[d * slice_size..(d + 1) * slice_size];
                clahe_2d(slice, rows, cols, n_tiles_y, n_tiles_x, clip_limit, bins)
            })
            .collect();

        let mut out = Vec::with_capacity(depth * slice_size);
        for s in out_slices {
            out.extend_from_slice(&s);
        }

        Ok(rebuild(out, dims, image))
    }
}

// ── Internal: 2D CLAHE ────────────────────────────────────────────────────────

/// Apply CLAHE to a single 2D slice (flat row-major, `rows × cols`).
///
/// Returns a new flat Vec<f32> of identical length with CLAHE applied.
///
/// # Invariant
/// Output values lie in `[v_min, v_max]` where `v_min/v_max` are the
/// minimum and maximum finite values of `pixels`. When all values are
/// identical (span = 0), output equals input.
fn clahe_2d(
    pixels: &[f32],
    rows: usize,
    cols: usize,
    n_tiles_y: usize,
    n_tiles_x: usize,
    clip_limit: f32,
    bins: usize,
) -> Vec<f32> {
    debug_assert_eq!(pixels.len(), rows * cols);

    // Find global min/max for this slice.
    let (v_min, v_max) = {
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for &v in pixels {
            if v.is_finite() {
                mn = mn.min(v);
                mx = mx.max(v);
            }
        }
        if mn.is_infinite() || mn >= mx {
            // Uniform or all-non-finite slice: identity.
            return pixels.to_vec();
        }
        (mn, mx)
    };

    // Clamp tile counts to valid range.
    let n_ty = n_tiles_y.max(1).min(rows);
    let n_tx = n_tiles_x.max(1).min(cols);

    // Tile dimensions (floating-point for interpolation calculations).
    let tile_h = rows as f32 / n_ty as f32;
    let tile_w = cols as f32 / n_tx as f32;

    // Build CDF lookup table for each tile.
    // Layout: cdfs[ty * n_tx + tx] = Vec<f32> of length `bins`.
    let cdfs: Vec<Vec<f32>> = (0..n_ty)
        .flat_map(|ty| (0..n_tx).map(move |tx| (ty, tx)))
        .map(|(ty, tx)| {
            let y0 = ty * rows / n_ty;
            let y1 = ((ty + 1) * rows / n_ty).min(rows);
            let x0 = tx * cols / n_tx;
            let x1 = ((tx + 1) * cols / n_tx).min(cols);

            let mut tile_vals = Vec::with_capacity((y1 - y0) * (x1 - x0));
            for y in y0..y1 {
                for x in x0..x1 {
                    tile_vals.push(pixels[y * cols + x]);
                }
            }
            build_tile_cdf(&tile_vals, v_min, v_max, bins, clip_limit)
        })
        .collect();

    // Map each pixel using bilinear interpolation between 4 surrounding tile CDFs.
    let span = v_max - v_min;
    let mut output = vec![0.0f32; pixels.len()];

    for y in 0..rows {
        for x in 0..cols {
            let v = pixels[y * cols + x];
            let normalized = ((v - v_min) / span).clamp(0.0, 1.0);
            let bin = ((normalized * (bins as f32 - 1.0)).floor() as usize).min(bins - 1);

            // Fractional tile coordinates centered on tile midpoints.
            // Tile center positions: tile_h*0.5, tile_h*1.5, ..., tile_h*(n_ty-0.5)
            let ty_f = (y as f32 - tile_h * 0.5) / tile_h;
            let tx_f = (x as f32 - tile_w * 0.5) / tile_w;

            // Clamp to valid tile index range.
            let ty0 = (ty_f.floor() as isize).clamp(0, n_ty as isize - 1) as usize;
            let tx0 = (tx_f.floor() as isize).clamp(0, n_tx as isize - 1) as usize;
            let ty1 = (ty0 + 1).min(n_ty - 1);
            let tx1 = (tx0 + 1).min(n_tx - 1);

            // Bilinear interpolation weights (clamped to [0, 1]).
            let u = (ty_f - ty0 as f32).clamp(0.0, 1.0); // row weight
            let t = (tx_f - tx0 as f32).clamp(0.0, 1.0); // col weight

            let f00 = cdfs[ty0 * n_tx + tx0][bin];
            let f01 = cdfs[ty0 * n_tx + tx1][bin];
            let f10 = cdfs[ty1 * n_tx + tx0][bin];
            let f11 = cdfs[ty1 * n_tx + tx1][bin];

            // Bilinear interpolation: (1-u)*lerp_row0 + u*lerp_row1
            let mapped = (1.0 - u) * ((1.0 - t) * f00 + t * f01) + u * ((1.0 - t) * f10 + t * f11);

            output[y * cols + x] = v_min + mapped.clamp(0.0, 1.0) * span;
        }
    }

    output
}

/// Build a normalised CDF lookup table for a single tile.
///
/// # Algorithm
/// 1. Compute histogram with `bins` buckets over `[v_min, v_max]`.
/// 2. Clip each bucket at threshold `C = max(1, clip_limit * n_T / bins)`.
/// 3. Redistribute excess `E` uniformly: each bin gets `floor(E/bins)`,
///    the first `E mod bins` bins each receive one extra count.
/// 4. Compute cumulative sum and normalise to `[0, 1]` by dividing by `n_T`.
///
/// # Returns
/// `Vec<f32>` of length `bins`; entry `b` is the normalised CDF at bin `b`.
///
/// # Edge case
/// When `tile_vals` is empty or `n_T == 0`, returns identity ramp `b/(bins-1)`.
fn build_tile_cdf(
    tile_vals: &[f32],
    v_min: f32,
    v_max: f32,
    bins: usize,
    clip_limit: f32,
) -> Vec<f32> {
    let n = tile_vals.len();
    if n == 0 {
        // Identity ramp: uniform CDF.
        return (0..bins).map(|b| b as f32 / (bins - 1) as f32).collect();
    }

    let span = v_max - v_min;
    let mut hist = vec![0u64; bins];

    if span <= 0.0 {
        // Uniform slice: all pixels map to bin 0 → CDF steps at 0.
        hist[0] = n as u64;
    } else {
        for &v in tile_vals {
            let normalized = ((v - v_min) / span).clamp(0.0, 1.0);
            let bin = ((normalized * (bins as f32 - 1.0)).floor() as usize).min(bins - 1);
            hist[bin] += 1;
        }
    }

    // Clip histogram and compute excess.
    let clip_threshold = ((clip_limit * n as f32 / bins as f32).ceil() as u64).max(1);
    let mut excess = 0u64;
    for h in hist.iter_mut() {
        if *h > clip_threshold {
            excess += *h - clip_threshold;
            *h = clip_threshold;
        }
    }

    // Redistribute excess uniformly.
    let redistribute = excess / bins as u64;
    let remainder = (excess as usize) - (redistribute as usize) * bins;
    for h in hist.iter_mut() {
        *h += redistribute;
    }
    for h in hist.iter_mut().take(remainder) {
        *h += 1;
    }

    // Compute normalised CDF.
    let mut cdf = Vec::with_capacity(bins);
    let mut cumsum = 0u64;
    for &h in &hist {
        cumsum += h;
        cdf.push(cumsum as f32 / n as f32);
    }

    cdf
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_clahe.rs"]
mod tests_clahe;
