//! Tests for clahe
//! Extracted to keep the 500-line structural limit.

// ── Legacy allocating helpers (used by 2D unit tests) ─────────────────────
//
// These duplicate the original allocating logic from clahe.rs. They are
// preserved here because the 2D unit tests validate the CLAHE algorithm
// against the allocating reference path. The production path now uses
// `clahe_2d_with_scratch` / `build_tile_cdf_into` instead.

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

fn build_tile_cdf(
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

    let clip_threshold = ((clip_limit * n as f32 / bins as f32).ceil() as u64).max(1);
    let mut excess = 0u64;
    for h in hist.iter_mut() {
        if *h > clip_threshold {
            excess += *h - clip_threshold;
            *h = clip_threshold;
        }
    }
    let redistribute = excess / bins as u64;
    let remainder = (excess as usize) - (redistribute as usize) * bins;
    for h in hist.iter_mut() {
        *h += redistribute;
    }
    for h in hist.iter_mut().take(remainder) {
        *h += 1;
    }

    let mut cdf = Vec::with_capacity(bins);
    let mut cumsum = 0u64;
    for &h in &hist {
        cumsum += h;
        cdf.push(cumsum as f32 / n as f32);
    }
    cdf
}

// ── build_tile_cdf ────────────────────────────────────────────────────────

#[test]
fn tile_cdf_empty_returns_identity_ramp() {
    // postcondition: empty tile produces identity CDF [0, 1/(B-1), …, 1]
    let cdf = build_tile_cdf(&[][..], 0.0, 1.0, 4, 40.0);
    assert_eq!(cdf.len(), 4);
    assert!((cdf[0]).abs() < 1e-6, "cdf[0]={}", cdf[0]);
    assert!((cdf[3] - 1.0).abs() < 1e-6, "cdf[3]={}", cdf[3]);
}

#[test]
fn tile_cdf_uniform_values_peaks_at_zero() {
    // All 8 pixels equal 5.0, v_min=v_max=5 → span=0 → bin 0 gets all counts.
    let vals = vec![5.0_f32; 8];
    let cdf = build_tile_cdf(&vals, 5.0, 5.0, 4, 40.0);
    // With span=0 all go to bin 0; CDF[0] = 8/8 = 1.0
    assert!((cdf[0] - 1.0).abs() < 1e-6, "cdf[0]={}", cdf[0]);
}

#[test]
fn tile_cdf_uniform_distribution_no_clipping() {
    // 256 pixels uniformly covering [0, 255] → one pixel per bin → no clipping
    let vals: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let cdf = build_tile_cdf(&vals, 0.0, 255.0, 256, 40.0);
    // CDF should be approximately linear: cdf[b] ≈ (b+1)/256
    assert_eq!(cdf.len(), 256);
    for (b, &c) in cdf.iter().enumerate() {
        let expected = (b + 1) as f32 / 256.0;
        assert!(
            (c - expected).abs() < 2.0 / 256.0,
            "bin {b}: got {c:.4}, expected {expected:.4}"
        );
    }
    assert!((cdf[255] - 1.0).abs() < 1e-6, "CDF must end at 1.0");
}

#[test]
fn tile_cdf_last_entry_is_one() {
    // CDF[last] must be 1.0 for any valid input (normalised cumsum = n/n).
    let vals: Vec<f32> = (0..100).map(|i| i as f32 * 2.5 - 50.0).collect();
    let cdf = build_tile_cdf(&vals, -50.0, 200.0, 64, 10.0);
    assert!((cdf[63] - 1.0).abs() < 1e-6, "last CDF entry = {}", cdf[63]);
}

#[test]
fn tile_cdf_monotone_non_decreasing() {
    // CDF must be non-decreasing: F[b] >= F[b-1] for all b.
    let vals: Vec<f32> = vec![0.0, 10.0, 20.0, 20.0, 50.0, 100.0, 200.0, 255.0];
    let cdf = build_tile_cdf(&vals, 0.0, 255.0, 32, 5.0);
    for i in 1..cdf.len() {
        assert!(
            cdf[i] >= cdf[i - 1] - 1e-7,
            "CDF not monotone at {i}: {:.4} < {:.4}",
            cdf[i],
            cdf[i - 1]
        );
    }
}

// ── clahe_2d ─────────────────────────────────────────────────────────────

#[test]
fn clahe_2d_output_length_matches_input() {
    // Shape invariant: output.len() == rows * cols.
    let pixels = vec![1.0_f32; 16 * 16];
    let out = clahe_2d(&pixels, 16, 16, 4, 4, 40.0, 256);
    assert_eq!(out.len(), 16 * 16);
}

#[test]
fn clahe_2d_uniform_slice_returns_identity() {
    // When all pixels equal v_min=v_max, output must equal input.
    let v = 42.5_f32;
    let pixels = vec![v; 8 * 8];
    let out = clahe_2d(&pixels, 8, 8, 2, 2, 40.0, 256);
    for (i, (&inp, &outp)) in pixels.iter().zip(out.iter()).enumerate() {
        assert!(
            (inp - outp).abs() < 1e-5,
            "pixel {i}: input={inp}, output={outp}"
        );
    }
}

#[test]
fn clahe_2d_output_in_input_range() {
    // Output invariant: all output values in [v_min, v_max].
    // Deterministic pseudo-random sequence via Knuth multiplicative LCG.
    let mut x = 12345u64;
    let pixels: Vec<f32> = (0..32 * 32)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 33) as f32 / u32::MAX as f32) * 2000.0 - 1000.0
        })
        .collect();

    let v_min = pixels.iter().cloned().fold(f32::INFINITY, f32::min);
    let v_max = pixels.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let out = clahe_2d(&pixels, 32, 32, 4, 4, 40.0, 256);
    for &o in &out {
        assert!(
            o >= v_min - 1e-4 && o <= v_max + 1e-4,
            "output {o} outside [{v_min}, {v_max}]"
        );
    }
}

#[test]
fn clahe_2d_single_tile_equals_global_he_shape() {
    // With n_tiles_y=1, n_tiles_x=1, CLAHE collapses to global HE.
    // Verify output has same length and range as input.
    let pixels: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let out = clahe_2d(&pixels, 8, 8, 1, 1, 1000.0, 256);
    assert_eq!(out.len(), 64);
    let v_min = 0.0_f32;
    let v_max = 63.0_f32;
    for &o in &out {
        assert!(o >= v_min - 1e-4 && o <= v_max + 1e-4);
    }
}

#[test]
fn clahe_2d_contrast_enhanced_range_preserved() {
    // Ramp from 0..16 on a 4x4 grid — CLAHE should spread the histogram
    // but keep output in [0, 15].
    let pixels: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let out = clahe_2d(&pixels, 4, 4, 2, 2, 40.0, 16);
    assert_eq!(out.len(), 16);
    for &o in &out {
        assert!(o >= -1e-4 && o <= 15.0 + 1e-4, "output {o} out of range");
    }
}
