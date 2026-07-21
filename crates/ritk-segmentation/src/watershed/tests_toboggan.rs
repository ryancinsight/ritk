//! Differential tests for [`TobogganFilter`] against SimpleITK reference output.
//!
//! Expected label images are captured verbatim from `sitk.Toboggan` on the same
//! 2-D reliefs — an external oracle, not a ritk self-comparison.

use super::{validate_relief, TobogganFilter, MAX_SAMPLE_COUNT};
use coeus_core::SequentialBackend;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_image::test_support as ts;
use ritk_image::Image as NativeImage;
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

fn z1(flat: Vec<f32>, rows: usize, cols: usize) -> ritk_image::Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(flat, [1, rows, cols])
}

fn run(img: &ritk_image::Image<f32, B, 3>) -> Vec<f32> {
    extract_vec_infallible(
        &TobogganFilter::new()
            .apply(img)
            .expect("infallible: validated precondition"),
    )
    .0
}

fn f(v: &[i32]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

#[test]
fn single_basin_all_one_label() {
    // Bowl with a single minimum → every voxel slides to label 2.
    let img = z1(
        f(&[5, 4, 3, 4, 5, 4, 3, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4]),
        4,
        5,
    );
    assert_eq!(run(&img), f(&[2; 20]));
}

#[test]
fn four_corner_basins_match_sitk() {
    // Four corner minima separated by a central ridge.
    let img = z1(
        f(&[
            1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4, 5, 4, 3, 2, 3, 4, 3, 2, 1, 2, 3, 2, 1,
        ]),
        5,
        5,
    );
    let expect = f(&[
        2, 2, 3, 3, 3, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 4, 4, 5, 5, 5, 4, 4, 5, 5, 5,
    ]);
    assert_eq!(run(&img), expect);
}

#[test]
fn ridge_split_matches_sitk() {
    let img = z1(f(&[1, 5, 1, 5, 5, 5, 1, 5, 1]), 3, 3);
    assert_eq!(run(&img), f(&[2, 3, 3, 4, 3, 3, 4, 3, 3]));
}

#[test]
fn native_and_legacy_execution_are_exact_with_geometry() {
    let values = f(&[1, 5, 1, 5, 5, 5, 1, 5, 1]);
    let legacy = z1(values.clone(), 3, 3);
    let origin = Point::new([2.0, 3.0, 5.0]);
    let spacing = Spacing::new([0.5, 1.0, 2.0]);
    let direction = Direction::from_rows([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    let native = NativeImage::from_flat_on(
        values,
        [1, 3, 3],
        origin,
        spacing,
        direction,
        &SequentialBackend,
    )
    .expect("infallible: validated precondition");
    let filter = TobogganFilter::new();
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
fn validation_errors_are_exact() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            TobogganFilter::new()
                .apply(&z1(vec![0.0, value], 1, 2))
                .unwrap_err()
                .to_string(),
            format!("Toboggan relief at flat index 1 must be finite, got {value}")
        );
    }
    assert_eq!(
        validate_relief(&[], [1, 0, 2]).unwrap_err().to_string(),
        "Toboggan requires nonzero dimensions, got [1, 0, 2]"
    );
    assert_eq!(
        validate_relief(&[], [usize::MAX, 2, 1])
            .unwrap_err()
            .to_string(),
        format!(
            "Toboggan shape product overflows usize: [{}, 2, 1]",
            usize::MAX
        )
    );
    assert_eq!(
        validate_relief(&[], [1, 4096, 4096])
            .unwrap_err()
            .to_string(),
        format!(
            "Toboggan supports at most {MAX_SAMPLE_COUNT} samples for exact f32 labels starting at 2, got 16777216"
        )
    );
    assert_eq!(
        validate_relief(&[0.0], [1, 1, 2]).unwrap_err().to_string(),
        "Toboggan shape [1, 1, 2] requires 2 samples, got 1"
    );
}
