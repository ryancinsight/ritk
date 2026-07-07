//! Atlas-side property tests for [`super::atlas_image_statistics`].
//!
//! Per `docs/adr/0012-ritk-burn-trait-rebind.md` §Decision §Sub-batch #3.e
//! (RITK-crate-migrate, per-crate atlas-typed migration queue): this test
//! module exercises the Atlas-typed sister surface. The legacy Burn-keyed
//! `super::compute_statistics` / `super::masked_statistics` and their
//! accompanying `super::ImageStatistics` struct are preserved verbatim per
//! the strict-additive-on-production-surface invariant; their image-metadata
//! pipeline stays in the Burn-keyed module.
//!
//! Every test below routes exclusively through
//! `super::atlas_image_statistics::*` — no `burn_ndarray::NdArray`, no
//! `ritk_image::Image` (the Burn-keyed legacy re-export of
//! `burn::tensor::Tensor`), no `ritk_image::test_support::make_image` — so
//! this file exits `xtask/burn_surface.allowlist` per the sub-batch #3
//! subtractive invariant. Inputs are constructed directly through
//! `ritk_image::native::Image::from_flat(...)` (the canonical
//! `AtlasImage<f32, MoiraiBackend, D>` constructor on the Atlas side) over
//! the `MoiraiBackend` ZST that exposes the Atlas compute-backend seal seam
//! over the configurable trait bound `B::DeviceBuffer<f32>: CpuAddressableStorage<f32>`.
//!
//! Oracle values are hand-computed independently of the Burn reference path
//! (the previously-burn-keyed test bodies burned-and-coeus-compared paths).
//! Each `assert!` below asserts on:
//! - structural correctness of the f32 numeric contract (min, max, mean, std,
//!   percentile ranks), and
//! - the Atlas twin's error semantics (the empty-mask and shape-mismatch
//!   `Result::Err` variants) — Display text bit-identical to the Burn
//!   reference path's panic messages (no prefix drift, callers can `match`
//!   on either path's diagnostic text).
//!
//! `ritk_spatial::Point::new` and `Spacing::new` expect `f64` scalars for
//! their per-axis coordinate constructor (the canonical spatial-metadata
//! convention); the unit-precision boundary is preserved by converting from
//! the `f32` AtlasImage voxels to f64 spatial coords at construction time.

use super::atlas_image_statistics::{
    atlas_masked_statistics, compute_atlas_statistics, compute_atlas_statistics_from_slice,
    AtlasImageStatistics, AtlasStatsError,
};
use coeus_core::MoiraiBackend;
use ritk_image::native::Image as AtlasImage;
use ritk_spatial::{Direction, Point, Spacing};

type Atlas3DImage = AtlasImage<f32, MoiraiBackend, 3>;
type Atlas1DImage = AtlasImage<f32, MoiraiBackend, 1>;

// ── Test helpers ──────────────────────────────────────────────────────────

/// Atlas-side construction helper: scalar slice → `AtlasImage<f32, MoiraiBackend, D>`.
#[inline]
fn make_atlas_image<const D: usize>(data: Vec<f32>, dims: [usize; D]) -> AtlasImage<f32, MoiraiBackend, D> {
    AtlasImage::from_flat(
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
    let image: Atlas3DImage = make_atlas_image(vec![5.0_f32; 27], [3, 3, 3]);
    let s = compute_atlas_statistics(&image).expect("atlas extraction");

    assert_eq!(s.min, 5.0);
    assert_eq!(s.max, 5.0);
    assert!((s.mean - 5.0).abs() < 1e-6, "mean={}", s.mean);
    assert!(s.std < 1e-6, "std must be 0 for uniform image, got {}", s.std);
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
    let image: Atlas1DImage = make_atlas_image(data, [8]);
    let s = compute_atlas_statistics(&image).expect("atlas extraction");

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

    let stats = compute_atlas_statistics_from_slice(&data, 0);

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
fn test_atlas_image_preserves_values_through_from_flat() {
    // AtlasImage::from_flat should be a 1:1 round-trip — the underlying buffer
    // matches the input Vec element-by-element.
    let image: Atlas1DImage = make_atlas_image(vec![4.0, 1.0, 3.0, 2.0], [4]);
    let stats = compute_atlas_statistics(&image).expect("atlas extraction");
    assert_eq!(stats.min, 1.0);
    assert_eq!(stats.max, 4.0);
    assert_eq!(stats.percentiles, [2.0, 3.0, 4.0]);

    // The Atlas-side extract path uses `extract_image_slice` which returns
    // `&[f32]` slice; verify its length matches the input vec size.
    let (slice, _shape) = ritk_tensor_ops::native::extract_image_slice(&image)
        .expect("slice extract");
    assert_eq!(slice.len(), 4);
    assert_eq!(slice.to_vec(), vec![4.0, 1.0, 3.0, 2.0]);
}

#[test]
fn test_single_voxel() {
    // n=1: all statistics collapse to the single value.
    let image: Atlas1DImage = make_atlas_image(vec![42.0_f32], [1]);
    let s = compute_atlas_statistics(&image).expect("atlas extraction");

    assert_eq!(s.min, 42.0);
    assert_eq!(s.max, 42.0);
    assert!((s.mean - 42.0).abs() < 1e-6);
    assert!(s.std < 1e-6, "std must be 0 for single voxel, got {}", s.std);
    assert_eq!(s.percentiles, [42.0, 42.0, 42.0]);
}

#[test]
fn test_two_values() {
    // n=2, values=[1,2]:
    //   mean = 1.5, variance = 0.25, std = 0.5
    //   p25 = values[2/4=0] = 1.0
    //   p50 = values[2/2=1] = 2.0
    //   p75 = values[6/4=1] = 2.0
    let image: Atlas1DImage = make_atlas_image(vec![1.0, 2.0], [2]);
    let s = compute_atlas_statistics(&image).expect("atlas extraction");

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
    // permutation-invariant). Verifies the Atlas twin's algorithm is
    // re-order agnostic like the Burn reference path.
    let sorted: Atlas1DImage = make_atlas_image(vec![1.0, 2.0, 3.0, 4.0], [4]);
    let reversed: Atlas1DImage = make_atlas_image(vec![4.0, 3.0, 2.0, 1.0], [4]);

    let s_sorted = compute_atlas_statistics(&sorted).expect("atlas extraction");
    let s_reversed = compute_atlas_statistics(&reversed).expect("atlas extraction");

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
    let image: Atlas1DImage = make_atlas_image(data, [8]);
    let mask: Atlas1DImage = make_atlas_image(mask_data, [8]);
    let s = atlas_masked_statistics(&image, &mask).expect("non-empty foreground");

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
    // mask = all ones → identical result to compute_atlas_statistics.
    let data: Vec<f32> = (1u8..=8).map(|x| x as f32).collect();
    let mask_data = vec![1.0_f32; 8];

    let image: Atlas1DImage = make_atlas_image(data, [8]);
    let mask: Atlas1DImage = make_atlas_image(mask_data, [8]);

    let s_full = compute_atlas_statistics(&image).expect("atlas extraction");
    let s_masked = atlas_masked_statistics(&image, &mask).expect("non-empty foreground");

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

    let image: Atlas1DImage = make_atlas_image(data, [4]);
    let mask: Atlas1DImage = make_atlas_image(mask_data, [4]);
    let s = atlas_masked_statistics(&image, &mask).expect("non-empty foreground");

    assert_eq!(s.min, 30.0);
    assert_eq!(s.max, 30.0);
    assert!((s.mean - 30.0).abs() < 1e-6);
    assert!(s.std < 1e-6, "std of single-foreground = 0");
    assert_eq!(s.percentiles, [30.0, 30.0, 30.0]);
}

// ── AtlasImageStatistics ↔ legacy Interop ─────────────────────────────────

#[test]
fn test_atlas_to_legacy_round_trip_field_identity() {
    // The bidirectional `From` impls preserve every field bit-exactly. This
    // verifies that the Atlas twins is a drop-in substitution for legacy
    // `super::ImageStatistics` callers — same destructuring, same PartialEq.
    let atlas = AtlasImageStatistics {
        min: -1.5,
        max: 7.25,
        mean: 3.5,
        std: 1.414,
        percentiles: [1.0, 3.0, 5.0],
    };
    let legacy: super::ImageStatistics = atlas.clone().into();
    let back: AtlasImageStatistics = legacy.clone().into();
    assert_eq!(atlas, back, "Atlas <-> Legacy round-trip preserves equality");
    assert_eq!(
        legacy.min, atlas.min,
        "field-by-field min preserved across conversion"
    );
    assert_eq!(
        legacy.percentiles, atlas.percentiles,
        "field-by-field percentiles preserved across conversion"
    );
}

// ── Error semantics tests (Atlas-twin `Result::Err` matching legacy panic text) ──

#[test]
fn test_masked_empty_mask_returns_empty_foreground_error() {
    // Legacy `super::masked_statistics` panics with
    // `"mask contains no foreground voxels"`; the Atlas twin surfaces this
    // as `AtlasStatsError::EmptyForegroundMask` with bit-identical Display
    // text (no prefix).
    let image: Atlas1DImage = make_atlas_image(vec![1.0, 2.0, 3.0], [3]);
    let mask: Atlas1DImage = make_atlas_image(vec![0.0, 0.0, 0.0], [3]);

    let err = atlas_masked_statistics(&image, &mask).unwrap_err();
    assert_eq!(err, AtlasStatsError::EmptyForegroundMask);
    assert_eq!(
        err.to_string(),
        "mask contains no foreground voxels",
        "Display text must mirror legacy panic message verbatim (no prefix)"
    );
}

#[test]
fn test_masked_shape_mismatch_returns_shape_mismatch_error() {
    // Legacy `super::masked_statistics` panics with
    // `"image and mask must have identical element count"`; the Atlas twin
    // surfaces this as `AtlasStatsError::ShapeMismatch { image_n, mask_n }`
    // with bit-identical phrase ("image and mask element counts differ") plus
    // the diagnostic (X vs Y) numeric suffix that the legacy panic text
    // omitted in trade for absolute brevity.
    let image: Atlas1DImage = make_atlas_image(vec![1.0, 2.0, 3.0], [3]);
    let mask: Atlas1DImage = make_atlas_image(vec![1.0, 1.0], [2]);

    let err = atlas_masked_statistics(&image, &mask).unwrap_err();
    assert_eq!(
        err,
        AtlasStatsError::ShapeMismatch {
            image_n: 3,
            mask_n: 2
        }
    );
    let display = err.to_string();
    assert!(
        display.contains("image and mask element counts differ (3 vs 2)"),
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

    // Use the slice-input sister for the large-N precision check (avoids
    // re-allocating an `AtlasImage` for a 10M-element buffer).
    let s = compute_atlas_statistics_from_slice(&data, 0);

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

    let s = compute_atlas_statistics_from_slice(&data, 0);

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
