//! Tests for clahe
//! Extracted to keep the 500-line structural limit.

use super::interpolate::clahe_2d;
use super::tile_cdf::build_tile_cdf;

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
        assert!(
            (-1e-4..=15.0 + 1e-4).contains(&o),
            "output {o} out of range"
        );
    }
}
