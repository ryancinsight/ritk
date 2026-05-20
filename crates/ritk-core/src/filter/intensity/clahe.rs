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
//! - compute `bin_v = floor((v - v_min) / span * (B - 1))`, clamped to `[0, B-1]`
//! - find the four surrounding tile centers `(ty0, tx0), (ty0, tx1), (ty1, tx0), (ty1, tx1)`
//! - bilinear interpolation weights: `u = (y_f - ty0)`, `t = (x_f - tx0)` (both clamped to `[0, 1]`)
//!   where `y_f = (y - tile_h/2) / tile_h`, `x_f = (x - tile_w/2) / tile_w`
//! - `mapped = (1-u)*((1-t)*F_c00[bin_v] + t*F_c01[bin_v]) + u*((1-t)*F_c10[bin_v] + t*F_c11[bin_v])`
//! - `output(y, x) = v_min + mapped * span`
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
//! # Scratch-buffer reuse
//!
//! `ClaheScratch` pre-allocates all per-tile buffers (CDFs, histograms, tile pixel
//! values, output slice) once. `apply_with_scratch` reuses these buffers across
//! repeated CLAHE applications, eliminating per-tile allocations. Each Rayon thread
//! receives its own `ClaheScratch` via `map_with`, so no synchronization is needed.
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

/// Pre-allocated scratch buffers for zero-allocation CLAHE execution.
///
/// All per-tile working memory is allocated once and reused across calls to
/// [`ClaheFilter::apply_with_scratch`]. Each Rayon thread uses its own
/// `ClaheScratch` via `map_with`.
///
/// Layout: `cdfs` and `histograms` are flattened `n_tiles_y * n_tiles_x * bins`
/// elements; access tile `(ty, tx)` at offset `(ty * n_tiles_x + tx) * bins`.
#[derive(Clone)]
pub struct ClaheScratch {
    cdfs: Vec<f32>,
    histograms: Vec<u64>,
    tile_vals: Vec<f32>,
    output: Vec<f32>,
    n_tiles_y: usize,
    n_tiles_x: usize,
    bins: usize,
}

impl ClaheScratch {
    /// Pre-allocate scratch buffers for the given slice and tile dimensions.
    pub fn new(
        rows: usize,
        cols: usize,
        n_tiles_y: usize,
        n_tiles_x: usize,
        bins: usize,
    ) -> Self {
        let nty = n_tiles_y.max(1).min(rows).max(1);
        let ntx = n_tiles_x.max(1).min(cols).max(1);
        let n_tiles = nty * ntx;
        Self {
            cdfs: vec![0.0f32; n_tiles * bins],
            histograms: vec![0u64; n_tiles * bins],
            tile_vals: Vec::with_capacity(rows * cols),
            output: vec![0.0f32; rows * cols],
            n_tiles_y: nty,
            n_tiles_x: ntx,
            bins,
        }
    }

    /// Returns the CDF buffer size in f32 elements.
    pub fn cdf_len(&self) -> usize {
        self.cdfs.len()
    }

    /// Returns the histogram buffer size in u64 elements.
    pub fn histogram_len(&self) -> usize {
        self.histograms.len()
    }

    /// Returns the output buffer size in f32 elements.
    pub fn output_len(&self) -> usize {
        self.output.len()
    }

    /// Returns the cached tile grid dimensions `(n_tiles_y, n_tiles_x)`.
    pub fn tile_grid_dims(&self) -> (usize, usize) {
        (self.n_tiles_y, self.n_tiles_x)
    }

    /// Returns the cached bin count.
    pub fn bins(&self) -> usize {
        self.bins
    }
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

    /// Apply CLAHE to a 3-D image, creating a fresh scratch buffer internally.
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

        let n_tiles_y = self.tile_grid_size[0].min(rows).max(1);
        let n_tiles_x = self.tile_grid_size[1].min(cols).max(1);

        let scratch_init =
                ClaheScratch::new(rows, cols, n_tiles_y, n_tiles_x, self.bins);

            let clip_limit = self.clip_limit;
            let bins = self.bins;
            let slice_size = rows * cols;

            let out_slices: Vec<Vec<f32>> = (0..depth)
                .into_par_iter()
                .map_with(scratch_init, |scratch, d| {
                    let slice = &vals[d * slice_size..(d + 1) * slice_size];
                    clahe_2d_with_scratch(
                        slice,
                        rows,
                        cols,
                        n_tiles_y,
                        n_tiles_x,
                        clip_limit,
                        bins,
                        scratch,
                    )
                })
                .collect();

            let mut out = Vec::with_capacity(depth * slice_size);
            for s in out_slices {
                out.extend_from_slice(&s);
            }
            Ok(rebuild(out, dims, image))
        }

    /// Apply CLAHE to a 3-D image using a caller-provided scratch buffer.
    ///
    /// Each Rayon thread receives its own `ClaheScratch` via `map_with`.
    /// The passed `scratch` is consumed as the init value for one thread;
    /// additional threads clone it. After the call, `scratch` is re-initialized
    /// to the correct dimensions for potential reuse.
    ///
    /// # Errors
    /// Returns `Err` if the tensor data cannot be extracted as `f32`.
    pub fn apply_with_scratch<B: Backend>(
        &self,
        image: &Image<B, 3>,
        scratch: &mut ClaheScratch,
    ) -> Result<Image<B, 3>> {
        let shape = image.shape();
        let [depth, rows, cols] = [shape[0], shape[1], shape[2]];
        let (vals_vec, dims) = extract_vec(image)?;
        let vals = &vals_vec;

        let n_tiles_y = self.tile_grid_size[0].min(rows).max(1);
        let n_tiles_x = self.tile_grid_size[1].min(cols).max(1);

        // Take ownership of the caller's scratch for use as the map_with init.
        // Additional threads will clone this init value.
        let nty = n_tiles_y;
        let ntx = n_tiles_x;
        let bins = self.bins;
        let scratch_init =
            std::mem::replace(scratch, ClaheScratch::new(rows, cols, nty, ntx, bins));

        let clip_limit = self.clip_limit;
        let slice_size = rows * cols;

        let out_slices: Vec<Vec<f32>> = (0..depth)
            .into_par_iter()
            .map_with(scratch_init, |thread_scratch, d| {
                let slice = &vals[d * slice_size..(d + 1) * slice_size];
                clahe_2d_with_scratch(
                    slice,
                    rows,
                    cols,
                    n_tiles_y,
                    n_tiles_x,
                    clip_limit,
                    bins,
                    thread_scratch,
                )
            })
            .collect();

        let mut out = Vec::with_capacity(depth * slice_size);
        for s in out_slices {
            out.extend_from_slice(&s);
        }
        Ok(rebuild(out, dims, image))
    }
}

// ── Internal: 2D CLAHE with scratch reuse ─────────────────────────────────────

/// Apply CLAHE to a single 2D slice (flat row-major, `rows × cols`) using
/// pre-allocated scratch buffers.
///
/// Returns a new flat `Vec<f32>` of identical length with CLAHE applied.
/// The scratch CDFs, histograms, and tile_vals buffers are reused (cleared
/// and refilled each call); the output is freshly allocated for the caller
/// to own after the parallel slice collection.
///
/// # Invariant
/// Output values lie in `[v_min, v_max]` where `v_min/v_max` are the
/// minimum and maximum finite values of `pixels`. When all values are
/// identical (span = 0), output equals input.
fn clahe_2d_with_scratch(
    pixels: &[f32],
    rows: usize,
    cols: usize,
    n_tiles_y: usize,
    n_tiles_x: usize,
    clip_limit: f32,
    bins: usize,
    scratch: &mut ClaheScratch,
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

    // Build CDF lookup table for each tile using scratch buffers.
    // Zero-fill the entire CDF and histogram arrays once.
    let n_tiles = n_ty * n_tx;
    let cdf_len = n_tiles * bins;
    scratch.cdfs[..cdf_len].fill(0.0);
    scratch.histograms[..cdf_len].fill(0);

    for ty in 0..n_ty {
        for tx in 0..n_tx {
            let y0 = ty * rows / n_ty;
            let y1 = ((ty + 1) * rows / n_ty).min(rows);
            let x0 = tx * cols / n_tx;
            let x1 = ((tx + 1) * cols / n_tx).min(cols);

            scratch.tile_vals.clear();
            for y in y0..y1 {
                for x in x0..x1 {
                    scratch.tile_vals.push(pixels[y * cols + x]);
                }
            }

            let tile_offset = (ty * n_tx + tx) * bins;
            let tile_hist = &mut scratch.histograms[tile_offset..tile_offset + bins];
            let tile_cdf = &mut scratch.cdfs[tile_offset..tile_offset + bins];

            build_tile_cdf_into(
                &scratch.tile_vals,
                v_min,
                v_max,
                bins,
                clip_limit,
                tile_hist,
                tile_cdf,
            );
        }
    }

    // Map each pixel using bilinear interpolation between 4 surrounding tile CDFs.
    let span = v_max - v_min;
    scratch.output.clear();
    scratch.output.resize(pixels.len(), 0.0f32);

    for y in 0..rows {
        for x in 0..cols {
            let v = pixels[y * cols + x];
            let normalized = ((v - v_min) / span).clamp(0.0, 1.0);
            let bin = ((normalized * (bins as f32 - 1.0)).floor() as usize).min(bins - 1);

            // Fractional tile coordinates centered on tile midpoints.
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

            let f00 = scratch.cdfs[(ty0 * n_tx + tx0) * bins + bin];
            let f01 = scratch.cdfs[(ty0 * n_tx + tx1) * bins + bin];
            let f10 = scratch.cdfs[(ty1 * n_tx + tx0) * bins + bin];
            let f11 = scratch.cdfs[(ty1 * n_tx + tx1) * bins + bin];

            // Bilinear interpolation: (1-u)*lerp_row0 + u*lerp_row1
            let mapped =
                (1.0 - u) * ((1.0 - t) * f00 + t * f01) + u * ((1.0 - t) * f10 + t * f11);

            scratch.output[y * cols + x] = v_min + mapped.clamp(0.0, 1.0) * span;
        }
    }

    scratch.output.clone()
}

/// Build a normalised CDF for a single tile, writing into pre-allocated slices.
///
/// Writes the clipped histogram into `hist_out` and the normalised CDF into
/// `cdf_out`. Both must have length `bins`.
///
/// # Algorithm
/// 1. Compute histogram with `bins` buckets over `[v_min, v_max]`.
/// 2. Clip each bucket at threshold `C = max(1, clip_limit * n_T / bins)`.
/// 3. Redistribute excess `E` uniformly: each bin gets `floor(E/bins)`,
///    the first `E mod bins` bins each receive one extra count.
/// 4. Compute cumulative sum and normalise to `[0, 1]` by dividing by `n_T`.
///
/// # Edge case
/// When `tile_vals` is empty or `n_T == 0`, writes identity ramp `b/(bins-1)`.
fn build_tile_cdf_into(
    tile_vals: &[f32],
    v_min: f32,
    v_max: f32,
    bins: usize,
    clip_limit: f32,
    hist_out: &mut [u64],
    cdf_out: &mut [f32],
) {
    debug_assert_eq!(hist_out.len(), bins);
    debug_assert_eq!(cdf_out.len(), bins);

    let n = tile_vals.len();
    if n == 0 {
        // Identity ramp: uniform CDF.
        for (b, out) in cdf_out.iter_mut().enumerate() {
            *out = b as f32 / (bins - 1) as f32;
        }
        return;
    }

    hist_out.fill(0);

    let span = v_max - v_min;
    if span <= 0.0 {
        // Uniform slice: all pixels map to bin 0.
        hist_out[0] = n as u64;
    } else {
        for &v in tile_vals {
            let normalized = ((v - v_min) / span).clamp(0.0, 1.0);
            let bin = ((normalized * (bins as f32 - 1.0)).floor() as usize).min(bins - 1);
            hist_out[bin] += 1;
        }
    }

    // Clip histogram and compute excess.
    let clip_threshold = ((clip_limit * n as f32 / bins as f32).ceil() as u64).max(1);
    let mut excess = 0u64;
    for h in hist_out.iter_mut() {
        if *h > clip_threshold {
            excess += *h - clip_threshold;
            *h = clip_threshold;
        }
    }

    // Redistribute excess uniformly.
    let redistribute = excess / bins as u64;
    let remainder = (excess as usize) - (redistribute as usize) * bins;
    for h in hist_out.iter_mut() {
        *h += redistribute;
    }
    for h in hist_out.iter_mut().take(remainder) {
        *h += 1;
    }

    // Compute normalised CDF.
    let mut cumsum = 0u64;
    for (i, &h) in hist_out.iter().enumerate() {
        cumsum += h;
        cdf_out[i] = cumsum as f32 / n as f32;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[path = "tests_clahe.rs"]
mod tests_clahe;
