//! Bilinear interpolation of tile CDFs for CLAHE.
//!
//! For each pixel `(y, x)`, the mapped value is computed by bilinearly
//! interpolating between the four surrounding tile CDFs. Two variants
//! are provided:
//!
//! - **Allocating** (`clahe_2d`): uses a `Vec<Vec<f32>>` of per-tile CDFs,
//!   preserved for the legacy 2-D unit tests.
//! - **Zero-allocation** (`clahe_2d_with_scratch`): reads CDFs from the
//!   flat scratch buffer, used by the production path.

#[cfg(test)]
use super::tile_cdf::build_tile_cdf;
use super::tile_cdf::build_tile_cdf_into;

/// Apply CLAHE to a single 2D slice (flat row-major, `rows × cols`) using
/// pre-allocated scratch buffers.
///
/// Returns a new flat `Vec<f32>` of identical length with CLAHE applied.
/// The scratch CDFs and histogram buffers are reused (cleared and refilled
/// each call); the output is freshly allocated for the caller
/// to own after the parallel slice collection.
///
/// # Invariant
/// Output values lie in `[v_min, v_max]` where `v_min/v_max` are the
/// minimum and maximum finite values of `pixels`. When all values are
/// identical (span = 0), output equals input.
#[allow(clippy::too_many_arguments)]
pub(super) fn clahe_2d_with_scratch(
    pixels: &[f32],
    rows: usize,
    cols: usize,
    n_tiles_y: usize,
    n_tiles_x: usize,
    clip_limit: f32,
    bins: usize,
    scratch: &mut super::ClaheScratch,
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

            let tile_offset = (ty * n_tx + tx) * bins;
            let tile_hist = &mut scratch.histograms[tile_offset..tile_offset + bins];
            let tile_cdf = &mut scratch.cdfs[tile_offset..tile_offset + bins];
            build_tile_cdf_into(
                pixels, y0, y1, x0, x1, cols, v_min, v_max, bins, clip_limit, tile_hist, tile_cdf,
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
            let mapped = (1.0 - u) * ((1.0 - t) * f00 + t * f01) + u * ((1.0 - t) * f10 + t * f11);

            scratch.output[y * cols + x] = v_min + mapped.clamp(0.0, 1.0) * span;
        }
    }

    std::mem::take(&mut scratch.output)
}

// ── Legacy allocating path (used by 2D unit tests) ──────────────────────────

/// Apply CLAHE to a single 2D slice using allocating per-tile CDFs.
///
/// This is the legacy path preserved for the 2-D unit tests in `tests_clahe.rs`.
/// The production path uses [`clahe_2d_with_scratch`] instead.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn clahe_2d(
    pixels: &[f32],
    rows: usize,
    cols: usize,
    n_tiles_y: usize,
    n_tiles_x: usize,
    clip_limit: f32,
    bins: usize,
) -> Vec<f32> {
    debug_assert_eq!(pixels.len(), rows * cols);

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
            return pixels.to_vec();
        }
        (mn, mx)
    };

    let n_ty = n_tiles_y.max(1).min(rows);
    let n_tx = n_tiles_x.max(1).min(cols);
    let tile_h = rows as f32 / n_ty as f32;
    let tile_w = cols as f32 / n_tx as f32;

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

    let span = v_max - v_min;
    let mut output = vec![0.0f32; pixels.len()];

    for y in 0..rows {
        for x in 0..cols {
            let v = pixels[y * cols + x];
            let normalized = ((v - v_min) / span).clamp(0.0, 1.0);
            let bin = ((normalized * (bins as f32 - 1.0)).floor() as usize).min(bins - 1);

            let ty_f = (y as f32 - tile_h * 0.5) / tile_h;
            let tx_f = (x as f32 - tile_w * 0.5) / tile_w;

            let ty0 = (ty_f.floor() as isize).clamp(0, n_ty as isize - 1) as usize;
            let tx0 = (tx_f.floor() as isize).clamp(0, n_tx as isize - 1) as usize;
            let ty1 = (ty0 + 1).min(n_ty - 1);
            let tx1 = (tx0 + 1).min(n_tx - 1);

            let u = (ty_f - ty0 as f32).clamp(0.0, 1.0);
            let t = (tx_f - tx0 as f32).clamp(0.0, 1.0);

            let f00 = cdfs[ty0 * n_tx + tx0][bin];
            let f01 = cdfs[ty0 * n_tx + tx1][bin];
            let f10 = cdfs[ty1 * n_tx + tx0][bin];
            let f11 = cdfs[ty1 * n_tx + tx1][bin];

            let mapped = (1.0 - u) * ((1.0 - t) * f00 + t * f01) + u * ((1.0 - t) * f10 + t * f11);
            output[y * cols + x] = v_min + mapped.clamp(0.0, 1.0) * span;
        }
    }

    output
}
