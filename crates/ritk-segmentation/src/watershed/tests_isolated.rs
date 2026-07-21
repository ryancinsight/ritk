//! Unit tests for isolated watershed segmentation.
//!
//! The primary test is a value-semantic differential for the gradient-descent
//! `IsolatedWatershed` algorithm: each voxel is assigned the basin of the local
//! minimum of `g` it flows to via steepest descent, then seed basins are labeled
//! 1.0 and 2.0, and the rest 0.0.

use super::isolated_watershed_values;
use super::IsolatedWatershed;
use super::IsolatedWatershedConfig;
use coeus_core::SequentialBackend;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_image::test_support::make_image;
use ritk_image::Image as NativeImage;
use ritk_image::Image;

type B = SequentialBackend;

/// Build a `[1, ny, nx]` image (z == 1 → 2-D) from a flat row-major slice.
fn image_2d(data: Vec<f32>, ny: usize, nx: usize) -> Image<f32, B, 3> {
    make_image(data, [1, ny, nx])
}

/// Extract labels as a flat `Vec<f32>` from a label image.
fn labels(img: &Image<f32, B, 3>) -> Vec<f32> {
    img.data().to_vec()
}

// ── Gradient-descent IsolatedWatershed (7×7 two-valley relief) ─────────────────
//
// Image: a central high cross (values 5/6/9) separating four low corners, with
// two seeds in the top-centre and bottom-centre low regions.
//
// Gradient magnitude has local minima at all four corners and at the three
// central saddle points on each axis.  For these seeds:
//
//   seed1 (y=1, x=3): gradient-descent drains to (y=0, x=3) which is the
//     top-centre local minimum; its basin covers the three voxels with low
//     gradient at (y=0–1, x=2–4).
//   seed2 (y=5, x=3): gradient-descent drains to (y=6, x=3) which is the
//     bottom-centre local minimum; its basin covers (y=5–6, x=2–4).
//
// Label 1.0 → top basin (rows 0–1, cols 2–4); label 2.0 → bottom basin
// (rows 5–6, cols 2–4); label 0.0 → remaining voxels.
//
// Derivation: g[1,3]=0.5 → steepest descent to g[0,3]=0 (min); g[5,3]=0.5 →
// steepest descent to g[6,3]=0 (min).  Neighbouring voxels at g≈2–2.8 flow
// to the central saddle minima at (3,3), (3,0), (3,6), not to the seed basins.

const RELIEF_7X7: [f32; 49] = [
    1.0, 1.0, 2.0, 5.0, 2.0, 1.0, 1.0, //
    1.0, 1.0, 2.0, 5.0, 2.0, 1.0, 1.0, //
    2.0, 2.0, 3.0, 6.0, 3.0, 2.0, 2.0, //
    5.0, 5.0, 6.0, 9.0, 6.0, 5.0, 5.0, //
    2.0, 2.0, 3.0, 6.0, 3.0, 2.0, 2.0, //
    1.0, 1.0, 2.0, 5.0, 2.0, 1.0, 1.0, //
    1.0, 1.0, 2.0, 5.0, 2.0, 1.0, 1.0, //
];

// Golden produced by SimpleITK's hierarchical isolated watershed.
// seed1=(y=1, x=3) → basin of local min at (y=0, x=3) → label 1.0 at rows 0–1, cols 2–4.
// seed2=(y=5, x=3) → basin of local min at (y=6, x=3) → label 2.0 at rows 5–6, cols 2–4.
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
    #[allow(clippy::identity_op)]
    let seed1 = 1 * 7 + 3;
    let seed2 = 5 * 7 + 3;
    let config = IsolatedWatershedConfig::default();
    let result = isolated_watershed_values(&RELIEF_7X7, dims, seed1, seed2, &config);
    assert_eq!(
        result, GOLDEN_7X7,
        "isolated watershed must produce correct gradient-descent basin labels"
    );
}

#[test]
fn test_isolated_watershed_apply_matches_sitk_relief_7x7() {
    let image = image_2d(RELIEF_7X7.to_vec(), 7, 7);
    let filter = IsolatedWatershed::new([0, 1, 3], [0, 5, 3], IsolatedWatershedConfig::default());
    let result = filter.apply(&image).expect("apply must not error");
    assert_eq!(result.shape(), [1, 7, 7]);
    assert_eq!(labels(&result), GOLDEN_7X7);
}

// ── Edge case: identical seeds → all label 1 ──────────────────────────────────

#[test]
fn test_isolated_watershed_identical_seeds_all_label1() {
    let data = vec![0.2_f32, 0.5, 0.2, 0.8];
    let dims = [1_usize, 1, 4];
    let result = isolated_watershed_values(&data, dims, 0, 0, &IsolatedWatershedConfig::default());
    assert!(
        result.iter().all(|&v| v == 1.0),
        "identical seeds must yield all-label-1 output, got {result:?}"
    );
}

// ── Spatial metadata preserved through apply ──────────────────────────────────

#[test]
fn test_isolated_watershed_spatial_metadata_preserved() {
    let image = image_2d(RELIEF_7X7.to_vec(), 7, 7);
    let filter = IsolatedWatershed::new([0, 1, 3], [0, 5, 3], IsolatedWatershedConfig::default());
    let result = filter
        .apply(&image)
        .expect("infallible: validated precondition");
    assert_eq!(result.origin(), image.origin());
    assert_eq!(result.spacing(), image.spacing());
    assert_eq!(result.direction(), image.direction());
}

// ── Plateau / flat region validation ──────────────────────────────────────────

#[test]
fn test_isolated_watershed_flat_plateau() {
    // 1-D image with a flat plateau:
    // Intensities: [4.0, 3.0, 2.0, 2.0, 2.0, 1.0, 0.0]
    let data = vec![4.0_f32, 3.0, 2.0, 2.0, 2.0, 1.0, 0.0];
    let dims = [1_usize, 1, 7];
    let config = IsolatedWatershedConfig::default();
    let result = isolated_watershed_values(&data, dims, 0, 6, &config);
    // Verify successful execution (no panic or cycle) and that the seed basins
    // receive the correct labels.
    assert_eq!(result.len(), 7);
    assert_eq!(result[0], 1.0);
    assert_eq!(result[6], 2.0);
}

#[test]
fn test_isolated_watershed_plateau_flow() {
    // Linear ramp with flat gradient magnitude in the middle:
    // Intensities: [6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    // g: [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5]
    // Local minima in g are at 0 and 6.
    // SimpleITK 2.5.0 assigns the connected flat-gradient component to its
    // first lowest boundary, leaving only the final voxel in seed2's basin.
    let data = vec![6.0_f32, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0];
    let dims = [1_usize, 1, 7];
    let config = IsolatedWatershedConfig::default();
    let result = isolated_watershed_values(&data, dims, 0, 6, &config);

    assert_eq!(result.len(), 7);
    assert_eq!(result, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]);
}

#[test]
fn configuration_validation_errors_are_exact() {
    for threshold in [f32::NAN, f32::NEG_INFINITY, -0.1, 1.1] {
        assert_eq!(
            IsolatedWatershedConfig::new(threshold, 0.001, 1.0)
                .unwrap_err()
                .to_string(),
            format!("isolated watershed threshold must be finite and in [0, 1], got {threshold}")
        );
    }
    for tolerance in [f64::NAN, f64::INFINITY, 0.0, -0.1] {
        assert_eq!(
            IsolatedWatershedConfig::new(0.0, tolerance, 1.0)
                .unwrap_err()
                .to_string(),
            format!("isolated watershed tolerance must be finite and positive, got {tolerance}")
        );
    }
    assert_eq!(
        IsolatedWatershedConfig::new(0.5, 0.001, 0.4)
            .unwrap_err()
            .to_string(),
        "isolated watershed upper value limit must be finite and in [0.5, 1], got 0.4"
    );
}

#[test]
fn validation_rejects_invalid_seed_and_sample() {
    let image = image_2d(vec![0.0, f32::NAN], 1, 2);
    let filter = IsolatedWatershed::new([0, 0, 0], [0, 0, 1], IsolatedWatershedConfig::default());
    assert_eq!(
        filter.apply(&image).unwrap_err().to_string(),
        "isolated watershed sample at flat index 1 must be finite, got NaN"
    );
    let image = image_2d(vec![0.0, 1.0], 1, 2);
    let filter = IsolatedWatershed::new([0, 1, 0], [0, 0, 1], IsolatedWatershedConfig::default());
    assert_eq!(
        filter.apply(&image).unwrap_err().to_string(),
        "isolated watershed seed1 [0, 1, 0] is outside shape [1, 1, 2]"
    );
}

#[test]
fn native_and_legacy_execution_are_exact_with_geometry() {
    let values = RELIEF_7X7.to_vec();
    let legacy = image_2d(values.clone(), 7, 7);
    let origin = Point::new([2.0, 3.0, 5.0]);
    let spacing = Spacing::new([0.5, 1.0, 2.0]);
    let direction = Direction::from_rows([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    let native = NativeImage::from_flat_on(
        values,
        [1, 7, 7],
        origin,
        spacing,
        direction,
        &SequentialBackend,
    )
    .expect("infallible: validated precondition");
    let filter = IsolatedWatershed::new([0, 1, 3], [0, 5, 3], IsolatedWatershedConfig::default());
    let expected = filter
        .apply(&legacy)
        .expect("infallible: validated precondition");
    let actual = filter
        .apply_native(&native, &SequentialBackend)
        .expect("infallible: validated precondition");
    assert_eq!(
        actual
            .data_slice()
            .expect("infallible: validated precondition"),
        expected
            .data_slice()
            .expect("invariant: contiguous host storage")
    );
    assert_eq!(*actual.origin(), origin);
    assert_eq!(*actual.spacing(), spacing);
    assert_eq!(*actual.direction(), direction);
}

#[test]
fn binary_search_stops_at_last_evaluated_level() {
    // Fourth deterministic RNG case from seed 20260712. SimpleITK 2.5.0
    // stops at level 0.998046875; evaluating the next unevaluated midpoint
    // 0.9990234375 incorrectly absorbs all 26 background voxels into seed2.
    let relief = vec![
        -0.4217013,
        0.8255467,
        -0.10693295,
        1.2878131,
        -0.66178954,
        0.32409462,
        -1.6825387,
        0.15025051,
        -0.6496092,
        -2.9501545,
        -0.710426,
        -1.3518157,
        1.7783191,
        -0.3269836,
        -1.1650844,
        -0.54003257,
        1.9648689,
        0.8496015,
        -0.79775494,
        -0.72220045,
        0.83873594,
        0.43512583,
        0.8552623,
        -0.20597069,
        -2.0306077,
        -0.96593696,
        1.183625,
        0.592062,
        -0.7258898,
        -2.6461942,
        -0.26070863,
        0.3568976,
        1.2084751,
        -0.83981955,
        -0.203254,
        -1.3480649,
        -0.8894473,
        0.28772372,
        -0.20647214,
        -0.5742853,
        0.2245181,
        -0.5700375,
        0.026643064,
        -1.3193803,
        1.8009243,
        0.74783707,
        -1.8457456,
        0.90884584,
        -1.0550301,
        0.40466174,
        0.26878977,
        -0.078658596,
        0.5079706,
        0.80048037,
        0.23433799,
        -0.8401252,
    ];
    let expected = vec![
        1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 0.0, 1.0,
        0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0,
    ];
    let filter = IsolatedWatershed::new([0, 1, 1], [0, 5, 6], IsolatedWatershedConfig::default());
    let image = image_2d(relief, 7, 8);
    assert_eq!(
        labels(
            &filter
                .apply(&image)
                .expect("infallible: validated precondition")
        ),
        expected
    );

    let label_counts = |config| {
        let output = labels(
            &IsolatedWatershed::new([0, 1, 1], [0, 5, 6], config)
                .apply(&image)
                .expect("infallible: validated precondition"),
        );
        [0.0, 1.0, 2.0].map(|label| output.iter().filter(|&&value| value == label).count())
    };
    assert_eq!(
        label_counts(
            IsolatedWatershedConfig::new(0.2, 0.001, 1.0)
                .expect("infallible: validated precondition")
        ),
        [1, 31, 24]
    );
    assert_eq!(
        label_counts(
            IsolatedWatershedConfig::new(0.0, 0.1, 1.0)
                .expect("infallible: validated precondition")
        ),
        [27, 8, 21]
    );
    assert_eq!(
        label_counts(
            IsolatedWatershedConfig::new(0.0, 0.001, 0.8)
                .expect("infallible: validated precondition")
        ),
        [32, 8, 16]
    );
}
