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

pub mod interpolate;
pub mod tile_cdf;

use interpolate::clahe_2d_with_scratch;

use anyhow::Result;
use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec, rebuild};

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
    pub(crate) cdfs: Vec<f32>,
    pub(crate) histograms: Vec<u64>,
    pub(crate) output: Vec<f32>,
    n_tiles_y: usize,
    n_tiles_x: usize,
    bins: usize,
}

impl ClaheScratch {
    /// Pre-allocate scratch buffers for the given slice and tile dimensions.
    pub fn new(rows: usize, cols: usize, n_tiles_y: usize, n_tiles_x: usize, bins: usize) -> Self {
        let nty = n_tiles_y.max(1).min(rows).max(1);
        let ntx = n_tiles_x.max(1).min(cols).max(1);
        let n_tiles = nty * ntx;
        Self {
            cdfs: vec![0.0f32; n_tiles * bins],
            histograms: vec![0u64; n_tiles * bins],
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
        let (vals_vec, dims) = extract_vec(image)?;
        let out = self.clahe_flat(&vals_vec, dims);
        Ok(rebuild(out, dims, image))
    }

    /// Coeus-native sister of [`ClaheFilter::apply`].
    ///
    /// Runs the identical per-axial-slice CLAHE via the shared
    /// [`clahe_flat`](Self::clahe_flat) host core (per-slice driver over
    /// [`clahe_2d_with_scratch`]) on the image's contiguous host buffer, so the
    /// result is bitwise-identical to the Burn path. No Burn tensor is
    /// constructed. Spatial metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        crate::native_support::map_flat_image(image, backend, |vals, dims| {
            self.clahe_flat(vals, dims)
        })
    }

    /// Substrate-agnostic host core: applies CLAHE independently to each axial
    /// (Z=depth) slice of a flat z-major buffer. Single source of truth for the
    /// Burn [`apply`](Self::apply) and Coeus-native
    /// [`apply_native`](Self::apply_native) paths.
    fn clahe_flat(&self, vals: &[f32], dims: [usize; 3]) -> Vec<f32> {
        let [depth, rows, cols] = dims;
        let n_tiles_y = self.tile_grid_size[0].min(rows).max(1);
        let n_tiles_x = self.tile_grid_size[1].min(cols).max(1);
        let clip_limit = self.clip_limit;
        let bins = self.bins;
        let slice_size = rows * cols;

        moirai::map_collect_index_with::<moirai::Adaptive, _, _>(depth, |depth_index| {
            let mut scratch = ClaheScratch::new(rows, cols, n_tiles_y, n_tiles_x, bins);
            let slice = &vals[depth_index * slice_size..(depth_index + 1) * slice_size];
            clahe_2d_with_scratch(
                slice,
                rows,
                cols,
                n_tiles_y,
                n_tiles_x,
                clip_limit,
                bins,
                &mut scratch,
            )
        })
        .into_iter()
        .flatten()
        .collect()
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
        // Re-initialize the caller's scratch to the correct dimensions for reuse.
        *scratch = ClaheScratch::new(rows, cols, nty, ntx, bins);

        let clip_limit = self.clip_limit;
        let slice_size = rows * cols;

        let out: Vec<f32> = moirai::map_collect_index_with::<moirai::Adaptive, _, _>(depth, |d| {
            let mut thread_scratch = ClaheScratch::new(rows, cols, nty, ntx, bins);
            let slice = &vals[d * slice_size..(d + 1) * slice_size];
            clahe_2d_with_scratch(
                slice,
                rows,
                cols,
                n_tiles_y,
                n_tiles_x,
                clip_limit,
                bins,
                &mut thread_scratch,
            )
        })
        .into_iter()
        .flatten()
        .collect();
        Ok(rebuild(out, dims, image))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[path = "../tests_clahe.rs"]
mod tests_clahe;

#[cfg(test)]
#[path = "../tests_clahe_apply.rs"]
mod tests_clahe_apply;

#[cfg(test)]
mod tests_native {
    use super::ClaheFilter;
    use crate::native_support::{assert_native_matches_burn, make_native_image, native_vals};
    use coeus_core::SequentialBackend;

    #[test]
    fn matches_burn() {
        // Two axial slices, structured intensities so tile histograms differ.
        let vals: Vec<f32> = (0..128).map(|i| ((i * 17) % 64) as f32).collect();
        assert_native_matches_burn(
            vals,
            [2, 8, 8],
            |img| {
                ClaheFilter::new([2, 2], 4.0, 32)
                    .apply(img)
                    .expect("burn clahe")
            },
            |img, backend| ClaheFilter::new([2, 2], 4.0, 32).apply_native(img, backend),
        );
    }

    #[test]
    fn oracle_uniform_slice_unchanged() {
        // A uniform slice (`v_min == v_max`) maps to itself: the CLAHE per-slice
        // invariant `output == input` when the slice has zero intensity span.
        let img = make_native_image(vec![12.0f32; 64], [1, 8, 8]);
        let out = ClaheFilter::new([2, 2], 4.0, 32)
            .apply_native(&img, &SequentialBackend)
            .expect("native clahe");
        for &v in &native_vals(&out) {
            assert_eq!(v, 12.0, "uniform slice must be preserved, got {v}");
        }
    }
}
