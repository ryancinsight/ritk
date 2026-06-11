//! Tile CDF computation for CLAHE.
//!
//! Implements histogram construction, clip-limit redistribution, and normalised
//! CDF generation for a single tile. Two variants are provided:
//!
//! - **Allocating** (`build_tile_cdf`): returns a `Vec<f32>`, used by the legacy
//!   2-D unit tests that validate the algorithm against a reference path.
//! - **Zero-allocation** (`build_tile_cdf_into`): writes into pre-allocated
//!   slices, used by the production `clahe_2d_with_scratch` path.

/// Build a normalised CDF for a single tile, returning a freshly allocated `Vec<f32>`.
///
/// This is the legacy allocating path, preserved for the 2-D unit tests in
/// `tests_clahe.rs`. The production path uses [`build_tile_cdf_into`] instead.
///
/// # Algorithm
/// 1. Compute histogram with `bins` buckets over `[v_min, v_max]`.
/// 2. Clip each bucket at threshold `C = max(1, clip_limit * n / bins)`.
/// 3. Redistribute excess `E` uniformly: each bin gets `floor(E/bins)`,
///    the first `E mod bins` bins each receive one extra count.
/// 4. Compute cumulative sum and normalise to `[0, 1]` by dividing by `n`.
///
/// # Edge case
/// When the tile is empty (`n == 0`), returns identity ramp `b/(bins-1)`.
#[cfg(test)]
pub(super) fn build_tile_cdf(
    tile_vals: &[f32],
    v_min: f32,
    v_max: f32,
    bins: usize,
    clip_limit: f32,
) -> Vec<f32> {
    let n = tile_vals.len();
    if n == 0 {
        return (0..bins).map(|b| b as f32 / (bins - 1) as f32).collect();
    }

    let span = v_max - v_min;
    let mut hist = vec![0u64; bins];

    if span <= 0.0 {
        hist[0] = n as u64;
    } else {
        for &v in tile_vals {
            let normalized = ((v - v_min) / span).clamp(0.0, 1.0);
            let bin = ((normalized * (bins as f32 - 1.0)).floor() as usize).min(bins - 1);
            hist[bin] += 1;
        }
    }

    clip_and_redistribute(&mut hist, n, bins, clip_limit);

    let mut cdf = Vec::with_capacity(bins);
    let mut cumsum = 0u64;
    for &h in &hist {
        cumsum += h;
        cdf.push(cumsum as f32 / n as f32);
    }
    cdf
}

/// Build a normalised CDF for a single tile, writing into pre-allocated slices.
///
/// Writes the clipped histogram into `hist_out` and the normalised CDF into
/// `cdf_out`. Both must have length `bins`.
///
/// This is the zero-allocation variant used by the production
/// `clahe_2d_with_scratch` path.
///
/// # Algorithm
/// 1. Compute histogram with `bins` buckets over `[v_min, v_max]`.
/// 2. Clip each bucket at threshold `C = max(1, clip_limit * n_T / bins)`.
/// 3. Redistribute excess `E` uniformly: each bin gets `floor(E/bins)`,
///    the first `E mod bins` bins each receive one extra count.
/// 4. Compute cumulative sum and normalise to `[0, 1]` by dividing by `n_T`.
///
/// # Edge case
/// When the tile region is empty or `n_T == 0`, writes identity ramp `b/(bins-1)`.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_tile_cdf_into(
    pixels: &[f32],
    y0: usize,
    y1: usize,
    x0: usize,
    x1: usize,
    cols: usize,
    v_min: f32,
    v_max: f32,
    bins: usize,
    clip_limit: f32,
    hist_out: &mut [u64],
    cdf_out: &mut [f32],
) {
    debug_assert_eq!(hist_out.len(), bins);
    debug_assert_eq!(cdf_out.len(), bins);
    let tile_rows = y1 - y0;
    let tile_cols = x1 - x0;
    let n = tile_rows * tile_cols;
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
        for y in y0..y1 {
            for x in x0..x1 {
                let v = pixels[y * cols + x];
                let normalized = ((v - v_min) / span).clamp(0.0, 1.0);
                let bin = ((normalized * (bins as f32 - 1.0)).floor() as usize).min(bins - 1);
                hist_out[bin] += 1;
            }
        }
    }

    clip_and_redistribute(hist_out, n, bins, clip_limit);

    // Compute normalised CDF.
    let mut cumsum = 0u64;
    for (i, &h) in hist_out.iter().enumerate() {
        cumsum += h;
        cdf_out[i] = cumsum as f32 / n as f32;
    }
}

/// Clip histogram bins at the computed threshold and redistribute excess
/// uniformly across all bins.
///
/// Clip threshold: `C = max(1, ceil(clip_limit * n / bins))`.
///
/// Redistribution:
/// - excess `E = Σ_{b: H[b] > C} (H[b] - C)`
/// - Each bin receives `floor(E / bins)` additional counts
/// - The first `E mod bins` bins each receive one extra count
fn clip_and_redistribute(hist: &mut [u64], n: usize, bins: usize, clip_limit: f32) {
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
}
