use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support::make_image;
use ritk_image::Image;

type TestBackend = NdArray<f32>;

#[cfg(feature = "coeus")]
fn make_coeus_image<const D: usize>(
    data: Vec<f32>,
    dims: [usize; D],
) -> ritk_image::coeus::Image<f32, coeus_core::MoiraiBackend, D> {
    use ritk_spatial::{Direction, Point, Spacing};

    ritk_image::coeus::Image::from_flat(
        data,
        dims,
        Point::new([0.0; D]),
        Spacing::new([1.0; D]),
        Direction::identity(),
    )
    .unwrap()
}

// ── Positive tests ────────────────────────────────────────────────────────

#[test]
fn test_uniform_image() {
    // All voxels = 5.0 → std = 0, all percentiles = 5.
    let image: Image<TestBackend, 3> = make_image(vec![5.0f32; 27], [3, 3, 3]);
    let s = compute_statistics(&image);

    assert_eq!(s.min, 5.0);
    assert_eq!(s.max, 5.0);
    assert!((s.mean - 5.0).abs() < 1e-6, "mean={}", s.mean);
    assert!(s.std < 1e-6, "std={}", s.std);
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
    let image: Image<TestBackend, 1> = make_image(data, [8]);
    let s = compute_statistics(&image);

    assert_eq!(s.min, 1.0);
    assert_eq!(s.max, 8.0);
    assert!((s.mean - 4.5).abs() < 1e-5, "mean={}", s.mean);
    assert!(
        (s.std - 5.25f32.sqrt()).abs() < 1e-4,
        "std={} expected={}",
        s.std,
        5.25f32.sqrt()
    );
    assert_eq!(s.percentiles[0], 3.0, "p25");
    assert_eq!(s.percentiles[1], 5.0, "p50");
    assert_eq!(s.percentiles[2], 7.0, "p75");
}

#[cfg(feature = "coeus")]
#[test]
fn coeus_compute_statistics_matches_burn_path() {
    let data: Vec<f32> = (1u8..=8).map(|x| x as f32).collect();
    let burn_image: Image<TestBackend, 1> = make_image(data.clone(), [8]);
    let coeus_image = make_coeus_image(data, [8]);

    let burn_stats = compute_statistics(&burn_image);
    let coeus_stats = coeus::compute_statistics(&coeus_image).unwrap();

    assert_eq!(coeus_stats, burn_stats);
}

#[test]
fn test_single_voxel() {
    // n=1: all statistics collapse to the single value.
    let image: Image<TestBackend, 1> = make_image(vec![42.0], [1]);
    let s = compute_statistics(&image);

    assert_eq!(s.min, 42.0);
    assert_eq!(s.max, 42.0);
    assert!((s.mean - 42.0).abs() < 1e-6);
    assert!(s.std < 1e-6, "std must be 0 for single voxel");
    assert_eq!(s.percentiles, [42.0, 42.0, 42.0]);
}

#[test]
fn test_two_values() {
    // n=2, values=[1,2]:
    //   mean = 1.5, variance = 0.25, std = 0.5
    //   p25 = values[2/4=0] = 1.0
    //   p50 = values[2/2=1] = 2.0
    //   p75 = values[6/4=1] = 2.0
    let image: Image<TestBackend, 1> = make_image(vec![1.0, 2.0], [2]);
    let s = compute_statistics(&image);

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
    // Sort order of input must not change the result.
    let sorted: Image<TestBackend, 1> = make_image(vec![1.0, 2.0, 3.0, 4.0], [4]);
    let reversed: Image<TestBackend, 1> = make_image(vec![4.0, 3.0, 2.0, 1.0], [4]);

    let s_sorted = compute_statistics(&sorted);
    let s_reversed = compute_statistics(&reversed);

    assert_eq!(s_sorted.min, s_reversed.min);
    assert_eq!(s_sorted.max, s_reversed.max);
    assert!((s_sorted.mean - s_reversed.mean).abs() < 1e-6);
    assert!((s_sorted.std - s_reversed.std).abs() < 1e-6);
    assert_eq!(s_sorted.percentiles, s_reversed.percentiles);
}

// ── Masked statistics ─────────────────────────────────────────────────────

#[test]
fn test_masked_statistics_subset() {
    // Values [1..8]; mask foreground at indices 2..=5 → [3,4,5,6] (n=4).
    //   mean     = 4.5
    //   variance = 5/4 = 1.25  →  std = √1.25 ≈ 1.1180
    //   p25 = values[4/4=1] = 4.0
    //   p50 = values[4/2=2] = 5.0
    //   p75 = values[12/4=3] = 6.0
    let data: Vec<f32> = (1u8..=8).map(|x| x as f32).collect();
    let mut mask_data = vec![0.0f32; 8];
    for v in mask_data.iter_mut().take(6).skip(2) {
        *v = 1.0;
    }

    let image: Image<TestBackend, 1> = make_image(data, [8]);
    let mask: Image<TestBackend, 1> = make_image(mask_data, [8]);
    let s = masked_statistics(&image, &mask);

    assert_eq!(s.min, 3.0);
    assert_eq!(s.max, 6.0);
    assert!((s.mean - 4.5).abs() < 1e-5, "mean={}", s.mean);
    assert!(
        (s.std - 1.25f32.sqrt()).abs() < 1e-4,
        "std={} expected={}",
        s.std,
        1.25f32.sqrt()
    );
    assert_eq!(s.percentiles[0], 4.0, "p25");
    assert_eq!(s.percentiles[1], 5.0, "p50");
    assert_eq!(s.percentiles[2], 6.0, "p75");
}

#[test]
fn test_masked_statistics_all_foreground_matches_full() {
    // mask = all ones → identical result to compute_statistics.
    let data: Vec<f32> = (1u8..=8).map(|x| x as f32).collect();
    let mask_data = vec![1.0f32; 8];

    let image: Image<TestBackend, 1> = make_image(data, [8]);
    let mask: Image<TestBackend, 1> = make_image(mask_data, [8]);

    let s_full = compute_statistics(&image);
    let s_masked = masked_statistics(&image, &mask);

    assert_eq!(s_full.min, s_masked.min);
    assert_eq!(s_full.max, s_masked.max);
    assert!((s_full.mean - s_masked.mean).abs() < 1e-6);
    assert!((s_full.std - s_masked.std).abs() < 1e-6);
    assert_eq!(s_full.percentiles, s_masked.percentiles);
}

#[cfg(feature = "coeus")]
#[test]
fn coeus_masked_statistics_matches_burn_path() {
    let data: Vec<f32> = (1u8..=8).map(|x| x as f32).collect();
    let mut mask_data = vec![0.0f32; 8];
    for v in mask_data.iter_mut().take(6).skip(2) {
        *v = 1.0;
    }
    let burn_image: Image<TestBackend, 1> = make_image(data.clone(), [8]);
    let burn_mask: Image<TestBackend, 1> = make_image(mask_data.clone(), [8]);
    let coeus_image = make_coeus_image(data, [8]);
    let coeus_mask = make_coeus_image(mask_data, [8]);

    let burn_stats = masked_statistics(&burn_image, &burn_mask);
    let coeus_stats = coeus::masked_statistics(&coeus_image, &coeus_mask).unwrap();

    assert_eq!(coeus_stats, burn_stats);
}

#[test]
fn test_masked_statistics_single_foreground_voxel() {
    // Only one foreground voxel → std = 0, all percentiles = that value.
    let data = vec![10.0, 20.0, 30.0, 40.0];
    let mut mask_data = vec![0.0f32; 4];
    mask_data[2] = 1.0; // foreground is value 30.0

    let image: Image<TestBackend, 1> = make_image(data, [4]);
    let mask: Image<TestBackend, 1> = make_image(mask_data, [4]);
    let s = masked_statistics(&image, &mask);

    assert_eq!(s.min, 30.0);
    assert_eq!(s.max, 30.0);
    assert!((s.mean - 30.0).abs() < 1e-6);
    assert!(s.std < 1e-6);
    assert_eq!(s.percentiles, [30.0, 30.0, 30.0]);
}

// ── Negative / boundary ───────────────────────────────────────────────────

#[test]
#[should_panic(expected = "mask contains no foreground voxels")]
fn test_masked_statistics_empty_mask_panics() {
    let image: Image<TestBackend, 1> = make_image(vec![1.0, 2.0, 3.0], [3]);
    let mask: Image<TestBackend, 1> = make_image(vec![0.0, 0.0, 0.0], [3]);
    let _ = masked_statistics(&image, &mask);
}

#[test]
#[should_panic(expected = "identical element count")]
fn test_masked_statistics_shape_mismatch_panics() {
    let image: Image<TestBackend, 1> = make_image(vec![1.0, 2.0, 3.0], [3]);
    let mask: Image<TestBackend, 1> = make_image(vec![1.0, 1.0], [2]);
    let _ = masked_statistics(&image, &mask);
}

#[cfg(feature = "coeus")]
#[test]
fn coeus_masked_statistics_empty_mask_returns_error() {
    let image = make_coeus_image(vec![1.0, 2.0, 3.0], [3]);
    let mask = make_coeus_image(vec![0.0, 0.0, 0.0], [3]);

    let err = coeus::masked_statistics(&image, &mask).unwrap_err();

    assert_eq!(
        err.to_string(),
        "coeus image statistics: mask contains no foreground voxels"
    );
}

// ── Large-N f64-accumulation precision ───────────────────────────────────
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
