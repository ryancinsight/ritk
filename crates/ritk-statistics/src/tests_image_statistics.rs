//! Numerical and property tests for the canonical native image-statistics API.

use super::{
    compute_statistics_from_slice,
    native::{compute_statistics, masked_statistics},
};
use coeus_core::MoiraiBackend;
use ritk_image::Image as NativeImage;
use ritk_spatial::{Direction, Point, Spacing};

type Native3DImage = NativeImage<f32, MoiraiBackend, 3>;
type Native1DImage = NativeImage<f32, MoiraiBackend, 1>;

// ── Test helpers ──────────────────────────────────────────────────────────

/// Native construction helper: scalar slice → `NativeImage<f32, MoiraiBackend, D>`.
#[inline]
fn make_native_image<const D: usize>(
    data: Vec<f32>,
    dims: [usize; D],
) -> NativeImage<f32, MoiraiBackend, D> {
    NativeImage::from_flat(
        data,
        dims,
        Point::new([0.0_f64; D]),
        Spacing::new([1.0_f64; D]),
        Direction::identity(),
    )
    .expect("atlas image construction")
}

// ── Positive full-image oracle tests ──────────────────────────────────────

#[test]
fn test_uniform_image() {
    // All voxels = 5.0 → std = 0, all percentiles = 5.
    let image: Native3DImage = make_native_image(vec![5.0_f32; 27], [3, 3, 3]);
    let s = compute_statistics(&image).expect("native extraction");

    assert_eq!(s.min, 5.0);
    assert_eq!(s.max, 5.0);
    assert!((s.mean - 5.0).abs() < 1e-6, "mean={}", s.mean);
    assert!(
        s.std < 1e-6,
        "std must be 0 for uniform image, got {}",
        s.std
    );
    assert_eq!(s.percentiles, [5.0, 5.0, 5.0]);
}

#[test]
fn test_known_sequence() {
    // Values [1,2,3,4,5,6,7,8] (n=8):
    //   mean     = 36/8 = 4.5
    //   variance = 42/8 = 5.25  →  std = √5.25 ≈ 2.2913
    //   p25 = values[8/4]   = values[2] = 3.0
    //   p50 = values[8/2]   = values[4] = 5.0
    //   p75 = values[24/4]  = values[6] = 7.0
    let data: Vec<f32> = (1u8..=8).map(|x| x as f32).collect();
    let image: Native1DImage = make_native_image(data, [8]);
    let s = compute_statistics(&image).expect("native extraction");

    assert_eq!(s.min, 1.0);
    assert_eq!(s.max, 8.0);
    assert!((s.mean - 4.5).abs() < 1e-5, "mean={}", s.mean);
    assert!(
        (s.std - 5.25_f32.sqrt()).abs() < 1e-4,
        "std={} expected={}",
        s.std,
        5.25_f32.sqrt()
    );
    assert_eq!(s.percentiles[0], 3.0, "p25");
    assert_eq!(s.percentiles[1], 5.0, "p50");
    assert_eq!(s.percentiles[2], 7.0, "p75");
}

#[test]
fn test_slice_input_preserves_input_order() {
    // Host-slice input: verify the input is not reordered by
    // `compute_from_values` (which allocates-and-sorts internally; the
    // user's slice must remain untouched).
    let data = vec![4.0, 1.0, 3.0, 2.0];

    let stats = compute_statistics_from_slice(&data, 0);

    assert_eq!(
        data,
        vec![4.0, 1.0, 3.0, 2.0],
        "host-slice input must remain untouched"
    );
    assert_eq!(stats.min, 1.0);
    assert_eq!(stats.max, 4.0);
    assert_eq!(stats.percentiles, [2.0, 3.0, 4.0]);
}

#[test]
fn test_native_image_preserves_values_through_from_flat() {
    // NativeImage::from_flat should be a 1:1 round-trip — the underlying buffer
    // matches the input Vec element-by-element.
    let image: Native1DImage = make_native_image(vec![4.0, 1.0, 3.0, 2.0], [4]);
    let stats = compute_statistics(&image).expect("native extraction");
    assert_eq!(stats.min, 1.0);
    assert_eq!(stats.max, 4.0);
    assert_eq!(stats.percentiles, [2.0, 3.0, 4.0]);

    // The Native extract path uses `extract_image_slice` which returns
    // `&[f32]` slice; verify its length matches the input vec size.
    let (slice, _shape) =
        ritk_tensor_ops::native::extract_image_slice(&image).expect("slice extract");
    assert_eq!(slice.len(), 4);
    assert_eq!(slice.to_vec(), vec![4.0, 1.0, 3.0, 2.0]);
}

#[test]
fn test_single_voxel() {
    // n=1: all statistics collapse to the single value.
    let image: Native1DImage = make_native_image(vec![42.0_f32], [1]);
    let s = compute_statistics(&image).expect("native extraction");

    assert_eq!(s.min, 42.0);
    assert_eq!(s.max, 42.0);
    assert!((s.mean - 42.0).abs() < 1e-6);
    assert!(
        s.std < 1e-6,
        "std must be 0 for single voxel, got {}",
        s.std
    );
    assert_eq!(s.percentiles, [42.0, 42.0, 42.0]);
}

#[test]
fn test_two_values() {
    // n=2, values=[1,2]:
    //   mean = 1.5, variance = 0.25, std = 0.5
    //   p25 = values[2/4=0] = 1.0
    //   p50 = values[2/2=1] = 2.0
    //   p75 = values[6/4=1] = 2.0
    let image: Native1DImage = make_native_image(vec![1.0, 2.0], [2]);
    let s = compute_statistics(&image).expect("native extraction");

    assert_eq!(s.min, 1.0);
    assert_eq!(s.max, 2.0);
    assert!((s.mean - 1.5).abs() < 1e-5);
    assert!((s.std - 0.5).abs() < 1e-5, "std={}", s.std);
    assert_eq!(s.percentiles[0], 1.0, "p25");
    assert_eq!(s.percentiles[1], 2.0, "p50");
    assert_eq!(s.percentiles[2], 2.0, "p75");
}

#[test]
fn test_reverse_order_input_matches_sorted() {
    // Sort order of input must not change the result (statistics are
    // permutation-invariant). Verifies the native API's algorithm is
    // re-order agnostic like the Coeus reference path.
    let sorted: Native1DImage = make_native_image(vec![1.0, 2.0, 3.0, 4.0], [4]);
    let reversed: Native1DImage = make_native_image(vec![4.0, 3.0, 2.0, 1.0], [4]);

    let s_sorted = compute_statistics(&sorted).expect("native extraction");
    let s_reversed = compute_statistics(&reversed).expect("native extraction");

    assert_eq!(s_sorted.min, s_reversed.min);
    assert_eq!(s_sorted.max, s_reversed.max);
    assert!((s_sorted.mean - s_reversed.mean).abs() < 1e-6);
    assert!((s_sorted.std - s_reversed.std).abs() < 1e-6);
    assert_eq!(s_sorted.percentiles, s_reversed.percentiles);
}

// ── Masked-statistics oracle tests ────────────────────────────────────────

#[test]
fn test_masked_subset() {
    // Values [1..8]; mask foreground at indices 2..=5 → [3,4,5,6] (n=4).
    //   mean     = 4.5
    //   variance = 5/4 = 1.25  →  std = √1.25 ≈ 1.1180
    //   p25 = values[4/4=1] = 4.0
    //   p50 = values[4/2=2] = 5.0
    //   p75 = values[12/4=3] = 6.0
    let data: Vec<f32> = (1u8..=8).map(|x| x as f32).collect();
    let mut mask_data = vec![0.0_f32; 8];
    for v in mask_data.iter_mut().take(6).skip(2) {
        *v = 1.0;
    }
    let image: Native1DImage = make_native_image(data, [8]);
    let mask: Native1DImage = make_native_image(mask_data, [8]);
    let s = masked_statistics(&image, &mask).expect("non-empty foreground");

    assert_eq!(s.min, 3.0);
    assert_eq!(s.max, 6.0);
    assert!((s.mean - 4.5).abs() < 1e-5, "mean={}", s.mean);
    assert!(
        (s.std - 1.25_f32.sqrt()).abs() < 1e-4,
        "std={} expected={}",
        s.std,
        1.25_f32.sqrt()
    );
    assert_eq!(s.percentiles[0], 4.0, "p25");
    assert_eq!(s.percentiles[1], 5.0, "p50");
    assert_eq!(s.percentiles[2], 6.0, "p75");
}

#[test]
fn test_masked_all_foreground_matches_unmasked() {
    // mask = all ones → identical result to compute_statistics.
    let data: Vec<f32> = (1u8..=8).map(|x| x as f32).collect();
    let mask_data = vec![1.0_f32; 8];

    let image: Native1DImage = make_native_image(data, [8]);
    let mask: Native1DImage = make_native_image(mask_data, [8]);

    let s_full = compute_statistics(&image).expect("native extraction");
    let s_masked = masked_statistics(&image, &mask).expect("non-empty foreground");

    assert_eq!(s_full.min, s_masked.min);
    assert_eq!(s_full.max, s_masked.max);
    assert!((s_full.mean - s_masked.mean).abs() < 1e-6);
    assert!((s_full.std - s_masked.std).abs() < 1e-6);
    assert_eq!(s_full.percentiles, s_masked.percentiles);
}

#[test]
fn test_masked_single_foreground_voxel() {
    // Only one foreground voxel → std = 0, all percentiles = that value.
    let data = vec![10.0, 20.0, 30.0, 40.0];
    let mut mask_data = vec![0.0_f32; 4];
    mask_data[2] = 1.0; // foreground = value 30.0

    let image: Native1DImage = make_native_image(data, [4]);
    let mask: Native1DImage = make_native_image(mask_data, [4]);
    let s = masked_statistics(&image, &mask).expect("non-empty foreground");

    assert_eq!(s.min, 30.0);
    assert_eq!(s.max, 30.0);
    assert!((s.mean - 30.0).abs() < 1e-6);
    assert!(s.std < 1e-6, "std of single-foreground = 0");
    assert_eq!(s.percentiles, [30.0, 30.0, 30.0]);
}

// ── Error semantics tests (native API `Result::Err` matching legacy panic text) ──

#[test]
fn test_masked_empty_mask_returns_empty_foreground_error() {
    // Native failures retain a diagnostic that identifies the violated mask
    // contract without introducing a parallel error vocabulary.
    let image: Native1DImage = make_native_image(vec![1.0, 2.0, 3.0], [3]);
    let mask: Native1DImage = make_native_image(vec![0.0, 0.0, 0.0], [3]);

    let err = masked_statistics(&image, &mask).unwrap_err();
    assert_eq!(
        err.to_string(),
        "coeus image statistics: mask contains no foreground voxels",
        "Display text must mirror legacy panic message verbatim (no prefix)"
    );
}

#[test]
fn test_masked_shape_mismatch_returns_shape_mismatch_error() {
    // Shape failures report both observed element counts.
    let image: Native1DImage = make_native_image(vec![1.0, 2.0, 3.0], [3]);
    let mask: Native1DImage = make_native_image(vec![1.0, 1.0], [2]);

    let err = masked_statistics(&image, &mask).unwrap_err();
    let display = err.to_string();
    assert!(
        display.contains(
            "coeus image statistics: image element count 3 does not match mask element count 2"
        ),
        "Display text must surface shape-mismatch numeric diagnostic; got: {display}"
    );
}

// ── Large-N f64-accumulation precision ────────────────────────────────────
//
// Sequential f32 summation of n ≈ 10^7 elements with CT-scale values
// (-2048..=3071, mean ≈ -789) produces a running total of ~-85 billion.
// At that magnitude the f32 ULP (≈8192) exceeds typical per-element values
// so additions round to zero; the accumulated sum saturates.  The f64
// accumulator path must return a mean within ε = 1.0 HU of the f64 reference.

#[test]
fn test_large_n_ct_scale_mean_precision() {
    // n = 10,485,760 (10 × 2^20) elements spanning a CT-like range.
    // Pattern: floor(i * 5120 / n) − 2048 produces values uniformly spaced
    // in [−2048, 3071] with mean = 511.5.
    // Analytical mean: (−2048 + 3071) / 2 = 511.5
    let n: usize = 10_485_760;
    let scale = 5120_f64;
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as f64 * scale / n as f64).floor() as f32) - 2048.0)
        .collect();

    // Use the slice-input API for the large-N precision check.
    let s = compute_statistics_from_slice(&data, 0);

    let expected_mean = 511.5_f32;
    assert!(
        (s.mean - expected_mean).abs() < 1.0,
        "large-N mean={} expected≈{} (f64 accumulation required for precision)",
        s.mean,
        expected_mean
    );
    assert_eq!(s.min, -2048.0, "min");
    assert_eq!(s.max, 3071.0, "max");
}

#[test]
fn test_large_n_negative_mean_precision() {
    // n = 10,485,760 elements all equal to a large negative CT value (−789).
    // Sequential f32 sum saturates at n ≈ 17M; at n = 10.5M the error is
    // ~200 HU without f64 accumulation and <0.01 HU with it.
    let n: usize = 10_485_760;
    let constant = -789.0_f32;
    let data = vec![constant; n];

    let s = compute_statistics_from_slice(&data, 0);

    assert!(
        (s.mean - constant).abs() < 1.0,
        "large-N constant mean={} expected={} (precision lost without f64 accumulation)",
        s.mean,
        constant
    );
    assert_eq!(s.min, constant, "min");
    assert_eq!(s.max, constant, "max");
    assert!(
        s.std < 1e-3,
        "std of constant array must be ~0, got {}",
        s.std
    );
}
