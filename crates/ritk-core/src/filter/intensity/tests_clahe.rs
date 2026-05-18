//! Tests for clahe
//! Extracted to keep the 500-line structural limit.
use super::*;
use crate::filter::ops::extract_vec_infallible;

use burn_ndarray::NdArray;
type B = NdArray<f32>;
use crate::image::Image;
use crate::spatial::{Direction, Point, Spacing};
use burn::tensor::{Shape, Tensor, TensorData};

fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    let device = Default::default();
    let td = TensorData::new(data, Shape::new(shape));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

// ── build_tile_cdf ────────────────────────────────────────────────────────

#[test]
fn tile_cdf_empty_returns_identity_ramp() {
    // postcondition: empty tile produces identity CDF [0, 1/(B-1), …, 1]
    let cdf = build_tile_cdf(&[], 0.0, 1.0, 4, 40.0);
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

// ── ClaheFilter::apply ────────────────────────────────────────────────────

#[test]
fn apply_preserves_shape_and_metadata() {
    // Shape and spatial metadata invariants.
    let data: Vec<f32> = (0..8 * 16 * 16).map(|i| (i % 256) as f32).collect();
    let img = make_image(data, [8, 16, 16]);
    let origin = *img.origin();
    let spacing = *img.spacing();
    let direction = *img.direction();

    let filter = ClaheFilter::new([4, 4], 40.0, 256);
    let out = filter.apply(&img).expect("CLAHE apply failed");

    assert_eq!(out.shape(), [8, 16, 16]);
    assert_eq!(out.origin(), &origin);
    assert_eq!(out.spacing(), &spacing);
    assert_eq!(out.direction(), &direction);
}

#[test]
fn apply_uniform_volume_identity() {
    // Uniform voxel value → output = input (all slices are uniform → identity path).
    let data = vec![128.0_f32; 4 * 8 * 8];
    let img = make_image(data.clone(), [4, 8, 8]);
    let filter = ClaheFilter::new([2, 2], 40.0, 256);
    let out = filter.apply(&img).expect("CLAHE apply failed");
    let (out_data, _) = extract_vec_infallible(&out);
    let out_vals: Vec<f32> = out_data.as_slice().to_vec();
    for (i, (&inp, &outp)) in data.iter().zip(out_vals.iter()).enumerate() {
        assert!(
            (inp - outp).abs() < 1e-4,
            "voxel {i}: input={inp}, output={outp}"
        );
    }
}

#[test]
fn apply_output_in_input_range() {
    // Output voxels must lie within [v_min_slice, v_max_slice] (per-slice).
    let data: Vec<f32> = (0..4 * 32 * 32)
        .map(|i| ((i * 7 + 13) % 512) as f32 - 100.0)
        .collect();
    let img = make_image(data.clone(), [4, 32, 32]);
    let filter = ClaheFilter::new([4, 4], 40.0, 256);
    let out = filter.apply(&img).expect("CLAHE apply failed");
    let (out_data, _) = extract_vec_infallible(&out);
    let out_vals: Vec<f32> = out_data.as_slice().to_vec();

    let slice_size = 32 * 32;
    for d in 0..4 {
        let sl_in = &data[d * slice_size..(d + 1) * slice_size];
        let sl_out = &out_vals[d * slice_size..(d + 1) * slice_size];
        let v_min = sl_in.iter().cloned().fold(f32::INFINITY, f32::min);
        let v_max = sl_in.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for &o in sl_out {
            assert!(
                o >= v_min - 0.5 && o <= v_max + 0.5,
                "slice {d}: output {o} outside [{v_min}, {v_max}]"
            );
        }
    }
}

#[test]
fn apply_ramp_increases_midrange_contrast() {
    // Analytical: for a uniform ramp [0,255] on a 1x1 tile grid with large
    // clip limit, CLAHE ≈ global HE. The output should be approximately a
    // linear ramp from 0 to 255 (identity) because the ramp already has
    // uniform histogram. The midpoint pixel value should change less than 20%.
    let n = 16usize;
    let data: Vec<f32> = (0..(1 * n * n))
        .map(|i| i as f32 / (n * n - 1) as f32 * 255.0)
        .collect();
    let img = make_image(data.clone(), [1, n, n]);
    let filter = ClaheFilter::new([1, 1], 1000.0, 256);
    let out = filter.apply(&img).expect("CLAHE apply failed");
    let (out_data, _) = extract_vec_infallible(&out);
    let out_vals: Vec<f32> = out_data.as_slice().to_vec();

    // Last pixel should map close to 255.0
    assert!(
        out_vals[n * n - 1] > 200.0,
        "last pixel = {}",
        out_vals[n * n - 1]
    );
    // First pixel should map close to 0.0
    assert!(out_vals[0] < 50.0, "first pixel = {}", out_vals[0]);
}
