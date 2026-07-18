use super::*;
use ritk_image::test_support as ts;

type B = coeus_core::SequentialBackend;

/// Positive and negative operands use C/C++ truncated-toward-zero remainder,
/// matching ITK/sitk: `7%3=1, 8%3=2, 9%3=0, 10%3=1, −7%3=−1, −8%3=−2`.
#[test]
fn modulus_matches_itk_truncated_remainder() {
    let img = ts::make_image::<f32, B, 3>(vec![7.0, 8.0, 9.0, 10.0, -7.0, -8.0], [1, 1, 6]);
    let out = ModulusImageFilter::new(3).apply(&img);
    assert_eq!(
        out.data_slice().into_owned(),
        vec![1.0, 2.0, 0.0, 1.0, -1.0, -2.0]
    );
}

/// Dividend of 1 maps every integral voxel to 0.
#[test]
fn modulus_by_one_is_zero() {
    let img = ts::make_image::<f32, B, 3>(vec![3.0, 17.0, -42.0, 0.0], [1, 1, 4]);
    let out = ModulusImageFilter::new(1).apply(&img);
    assert_eq!(out.data_slice().into_owned(), vec![0.0; 4]);
}

#[test]
#[should_panic(expected = "dividend must be non-zero")]
fn modulus_by_zero_panics() {
    let _ = ModulusImageFilter::new(0);
}
