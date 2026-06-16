use super::*;
use burn::tensor::{Shape, Tensor, TensorData};

use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
type B = NdArray<f32>;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(data, shape)
}

// ── histogram_equalize_global ─────────────────────────────────────────────

#[test]
fn global_he_empty_input_returns_empty() {
    let out = histogram_equalize_global(&[], 256);
    assert!(out.is_empty());
}

#[test]
fn global_he_uniform_input_is_identity() {
    // All pixels equal 50.0 → span = 0 → identity path → output = input.
    let vals = vec![50.0_f32; 16];
    let out = histogram_equalize_global(&vals, 256);
    for (i, (&inp, &outp)) in vals.iter().zip(out.iter()).enumerate() {
        assert!(
            (inp - outp).abs() < 1e-5,
            "index {i}: input={inp}, output={outp}"
        );
    }
}

#[test]
fn global_he_output_in_input_range() {
    // Analytical: output ∈ [v_min, v_max] for all finite inputs.
    let vals: Vec<f32> = (0..64).map(|i| i as f32 * 3.0 - 50.0).collect();
    let v_min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let v_max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let out = histogram_equalize_global(&vals, 256);
    for &o in &out {
        assert!(
            o >= v_min - 1e-4 && o <= v_max + 1e-4,
            "output {o} outside [{v_min}, {v_max}]"
        );
    }
}

#[test]
fn global_he_last_output_is_vmax() {
    // For a strictly increasing ramp, the last pixel (max value) must map to v_max.
    let vals: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let out = histogram_equalize_global(&vals, 32);
    assert!(
        (out[31] - 31.0).abs() < 1.0,
        "last output = {}, expected ~31.0",
        out[31]
    );
}

#[test]
fn global_he_first_output_is_near_vmin() {
    // The first value (minimum) maps to v_min + cdf[0]*span = v_min + (1/N)*span.
    // For N=16, bins=16, cdf[0] = 1/16 → output[0] = 0 + 1/16 * 15 ≈ 0.9375.
    let vals: Vec<f32> = (0..16).map(|i| i as f32).collect(); // [0, 1, ..., 15]
    let out = histogram_equalize_global(&vals, 16);
    // First pixel (v=0) maps through bin 0; cdf[0] = 1/16 → output ≈ 0.9375
    assert!(
        out[0] >= 0.0 && out[0] < 3.0,
        "first output = {}, expected small positive value",
        out[0]
    );
}

#[test]
fn global_he_monotone_output_for_sorted_input() {
    // If input is sorted ascending, output must also be non-decreasing
    // (since the CDF mapping is non-decreasing).
    let vals: Vec<f32> = (0..64).map(|i| i as f32 * 1.5).collect();
    let out = histogram_equalize_global(&vals, 64);
    for i in 1..out.len() {
        assert!(
            out[i] >= out[i - 1] - 1e-5,
            "output not monotone at {i}: {:.4} < {:.4}",
            out[i],
            out[i - 1]
        );
    }
}

#[test]
fn global_he_preserves_length() {
    let vals: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let out = histogram_equalize_global(&vals, 256);
    assert_eq!(out.len(), 100);
}

// ── HistogramEqualizationFilter::apply ────────────────────────────────────

#[test]
fn apply_preserves_shape_and_metadata() {
    let data: Vec<f32> = (0..4 * 16 * 16).map(|i| (i % 256) as f32).collect();
    let img = make_image(data, [4, 16, 16]);
    let origin = *img.origin();
    let spacing = *img.spacing();
    let direction = *img.direction();

    let filter = HistogramEqualizationFilter::new(256);
    let out = filter.apply(&img).expect("HE apply failed");

    assert_eq!(out.shape(), [4, 16, 16]);
    assert_eq!(out.origin(), &origin);
    assert_eq!(out.spacing(), &spacing);
    assert_eq!(out.direction(), &direction);
}

#[test]
fn apply_uniform_volume_is_identity() {
    let data = vec![75.0_f32; 2 * 8 * 8];
    let img = make_image(data.clone(), [2, 8, 8]);
    let filter = HistogramEqualizationFilter::new(256);
    let out = filter.apply(&img).expect("HE apply failed");
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
fn apply_output_in_global_range() {
    let data: Vec<f32> = (0..2 * 16 * 16).map(|i| (i % 200) as f32 - 50.0).collect();
    let v_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let v_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let img = make_image(data, [2, 16, 16]);
    let filter = HistogramEqualizationFilter::new(256);
    let out = filter.apply(&img).expect("HE apply failed");
    let (out_data, _) = extract_vec_infallible(&out);
    let out_vals: Vec<f32> = out_data.as_slice().to_vec();
    for &o in &out_vals {
        assert!(
            o >= v_min - 0.5 && o <= v_max + 0.5,
            "output {o} outside global range [{v_min}, {v_max}]"
        );
    }
}
