//! Unit tests for isolated watershed segmentation.
//!
//! The primary test is a value-semantic differential against
//! `sitk.IsolatedWatershed`: the watershed runs on the gradient magnitude and
//! binary-searches the flood level until the two seeds fall in separate basins,
//! labelling seed1's basin `1.0`, seed2's `2.0`, and all other basins `0.0`.

use super::isolated_watershed;
use super::IsolatedWatershed;
use super::IsolatedWatershedConfig;
use burn_ndarray::NdArray;
use ritk_image::test_support::make_image;
use ritk_image::Image;

type B = NdArray<f32>;

/// Build a `[1, ny, nx]` image (z == 1 → 2-D) from a flat row-major slice.
fn image_2d(data: Vec<f32>, ny: usize, nx: usize) -> Image<B, 3> {
    make_image(data, [1, ny, nx])
}

/// Extract labels as a flat `Vec<f32>` from a label image.
fn labels(img: &Image<B, 3>) -> Vec<f32> {
    img.data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .expect("label image must be f32")
        .to_vec()
}

// ── Differential vs sitk.IsolatedWatershed (7×7 two-valley relief) ─────────────
//
// Image: a central high cross (values 5/6/9) separating four low corners, with
// two seeds in the top-centre and bottom-centre low regions. sitk labels each
// seed's gradient-watershed basin (replaceValue1=1, replaceValue2=2) and 0 for
// the rest. Golden captured from
//   sitk.IsolatedWatershed(img, seed1=[3,1], seed2=[3,5], threshold=0.0,
//       upperValueLimit=1.0, isolatedValueTolerance=0.001,
//       replaceValue1=1, replaceValue2=2)
// (sitk seeds are [x, y]; here x=3 is the centre column, y=1 / y=5 the seed rows).

const RELIEF_7X7: [f32; 49] = [
    1.0, 1.0, 2.0, 5.0, 2.0, 1.0, 1.0, //
    1.0, 1.0, 2.0, 5.0, 2.0, 1.0, 1.0, //
    2.0, 2.0, 3.0, 6.0, 3.0, 2.0, 2.0, //
    5.0, 5.0, 6.0, 9.0, 6.0, 5.0, 5.0, //
    2.0, 2.0, 3.0, 6.0, 3.0, 2.0, 2.0, //
    1.0, 1.0, 2.0, 5.0, 2.0, 1.0, 1.0, //
    1.0, 1.0, 2.0, 5.0, 2.0, 1.0, 1.0, //
];

const GOLDEN_7X7: [f32; 49] = [
    0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
    0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, //
    0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, //
];

#[test]
fn test_isolated_watershed_matches_sitk_relief_7x7() {
    let dims = [1_usize, 7, 7];
    // seed1 at (y=1, x=3) → flat 1*7+3 = 10; seed2 at (y=5, x=3) → flat 38.
    let seed1 = 1 * 7 + 3;
    let seed2 = 5 * 7 + 3;
    let config = IsolatedWatershedConfig {
        threshold: 0.0,
        isolated_value_tolerance: 0.001,
        upper_value_limit: 1.0,
    };
    let result = isolated_watershed(&RELIEF_7X7, dims, seed1, seed2, &config);
    assert_eq!(
        result, GOLDEN_7X7,
        "isolated watershed must match sitk.IsolatedWatershed exactly"
    );
}

#[test]
fn test_isolated_watershed_apply_matches_sitk_relief_7x7() {
    let image = image_2d(RELIEF_7X7.to_vec(), 7, 7);
    let filter = IsolatedWatershed {
        seed1: [0, 1, 3],
        seed2: [0, 5, 3],
        threshold: 0.0,
        isolated_value_tolerance: 0.001,
        upper_value_limit: 1.0,
    };
    let result = filter.apply(&image).expect("apply must not error");
    assert_eq!(result.shape(), [1, 7, 7]);
    assert_eq!(labels(&result), GOLDEN_7X7);
}

// ── Edge case: identical seeds → all label 1 ──────────────────────────────────

#[test]
fn test_isolated_watershed_identical_seeds_all_label1() {
    let data = vec![0.2_f32, 0.5, 0.2, 0.8];
    let dims = [1_usize, 1, 4];
    let result = isolated_watershed(&data, dims, 0, 0, &IsolatedWatershedConfig::default());
    assert!(
        result.iter().all(|&v| v == 1.0),
        "identical seeds must yield all-label-1 output, got {result:?}"
    );
}

// ── Spatial metadata preserved through apply ──────────────────────────────────

#[test]
fn test_isolated_watershed_spatial_metadata_preserved() {
    let image = image_2d(RELIEF_7X7.to_vec(), 7, 7);
    let filter = IsolatedWatershed {
        seed1: [0, 1, 3],
        seed2: [0, 5, 3],
        threshold: 0.0,
        isolated_value_tolerance: 0.001,
        upper_value_limit: 1.0,
    };
    let result = filter.apply(&image).unwrap();
    assert_eq!(result.origin(), image.origin());
    assert_eq!(result.spacing(), image.spacing());
    assert_eq!(result.direction(), image.direction());
}
