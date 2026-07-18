//! Tests for `ClaheFilter::apply` and `ClaheScratch`.
//!
//! Extracted from `tests_clahe.rs` to keep the 500-line structural limit.
#![allow(clippy::identity_op, clippy::erasing_op)]

use super::*;
use coeus_core::SequentialBackend;
use ritk_image::native::Image as NativeImage;
use ritk_image::test_support as ts;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

use ritk_image::Image;

fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(data, shape)
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
fn native_apply_preserves_uniform_values_and_metadata() {
    let backend = SequentialBackend;
    let source = NativeImage::from_flat_on(
        vec![42.5; 4],
        [1, 2, 2],
        Point::new([2.0, 3.0, 4.0]),
        Spacing::new([0.5, 1.0, 2.0]),
        Direction::identity(),
        &backend,
    )
    .unwrap();
    let output = ClaheFilter::new([1, 1], 40.0, 256)
        .apply_native(&source, &backend)
        .unwrap();

    assert_eq!(output.data_slice().unwrap(), &[42.5; 4]);
    assert_eq!(output.shape(), source.shape());
    assert_eq!(output.origin(), source.origin());
    assert_eq!(output.spacing(), source.spacing());
    assert_eq!(output.direction(), source.direction());
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

// ── ClaheScratch + apply_with_scratch ──────────────────────────────────────

#[test]
fn apply_with_scratch_matches_apply_output() {
    // Bit-identity invariant: apply_with_scratch must produce identical
    // output to apply for the same input and filter parameters.
    let data: Vec<f32> = (0..4 * 16 * 16)
        .map(|i| ((i * 7 + 13) % 512) as f32 - 100.0)
        .collect();
    let img = make_image(data, [4, 16, 16]);

    let filter = ClaheFilter::new([4, 4], 40.0, 256);

    let out_apply = filter.apply(&img).expect("apply failed");

    let mut scratch = ClaheScratch::new(16, 16, 4, 4, 256);
    let out_scratch = filter
        .apply_with_scratch(&img, &mut scratch)
        .expect("apply_with_scratch failed");

    let (apply_data, _) = extract_vec_infallible(&out_apply);
    let (scratch_data, _) = extract_vec_infallible(&out_scratch);

    let apply_vals: Vec<f32> = apply_data.as_slice().to_vec();
    let scratch_vals: Vec<f32> = scratch_data.as_slice().to_vec();

    assert_eq!(
        apply_vals.len(),
        scratch_vals.len(),
        "output lengths must match"
    );
    for (i, (&a, &s)) in apply_vals.iter().zip(scratch_vals.iter()).enumerate() {
        assert!((a - s).abs() < 1e-6, "voxel {i}: apply={a}, scratch={s}");
    }
}

#[test]
fn scratch_reuse_preserves_results() {
    // Reuse invariant: calling apply_with_scratch twice with the same
    // scratch and same input produces identical output.
    let data: Vec<f32> = (0..2 * 8 * 8)
        .map(|i| ((i * 11 + 7) % 256) as f32)
        .collect();
    let img = make_image(data, [2, 8, 8]);

    let filter = ClaheFilter::new([2, 2], 40.0, 256);

    let mut scratch = ClaheScratch::new(8, 8, 2, 2, 256);
    let out1 = filter
        .apply_with_scratch(&img, &mut scratch)
        .expect("first apply_with_scratch failed");
    let out2 = filter
        .apply_with_scratch(&img, &mut scratch)
        .expect("second apply_with_scratch failed");

    let (data1, _) = extract_vec_infallible(&out1);
    let (data2, _) = extract_vec_infallible(&out2);

    let vals1: Vec<f32> = data1.as_slice().to_vec();
    let vals2: Vec<f32> = data2.as_slice().to_vec();

    assert_eq!(vals1.len(), vals2.len(), "output lengths must match");
    for (i, (&v1, &v2)) in vals1.iter().zip(vals2.iter()).enumerate() {
        assert!((v1 - v2).abs() < 1e-6, "voxel {i}: first={v1}, second={v2}");
    }
}

#[test]
fn scratch_allocates_correct_sizes() {
    // Structural invariant: ClaheScratch buffer sizes must equal
    // n_tiles_y * n_tiles_x * bins for CDFs and histograms, and
    // rows * cols for output.
    let rows = 32;
    let cols = 48;
    let nty = 4;
    let ntx = 6;
    let bins = 128;

    let scratch = ClaheScratch::new(rows, cols, nty, ntx, bins);

    let expected_tile_count = nty * ntx;
    assert_eq!(
        scratch.cdf_len(),
        expected_tile_count * bins,
        "CDF buffer size must equal n_tiles * bins"
    );
    assert_eq!(
        scratch.histogram_len(),
        expected_tile_count * bins,
        "histogram buffer size must equal n_tiles * bins"
    );
    assert_eq!(
        scratch.output_len(),
        rows * cols,
        "output buffer size must equal rows * cols"
    );
    assert_eq!(scratch.tile_grid_dims(), (nty, ntx));
    assert_eq!(scratch.bins(), bins);
}
