//! Construction, validation and metadata contracts of [`Image`].
#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]

use coeus_core::SequentialBackend;

use coeus_tensor::Tensor;

use crate::test_support::metadata_2d;
use crate::types::Image;

type TensorImage<const D: usize> = Image<f32, SequentialBackend, D>;

#[test]
fn construction_preserves_shape_and_metadata() {
    let data =
        Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let (origin, spacing, direction) = metadata_2d();

    let image = TensorImage::<2>::new(data, origin, spacing, direction).unwrap();

    assert_eq!(image.shape(), [2, 3]);
    assert_eq!(image.origin(), &origin);
    assert_eq!(image.spacing(), &spacing);
    assert_eq!(image.direction(), &direction);
    assert_eq!(image.data_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn from_flat_preserves_shape_values_and_metadata() {
    let (origin, spacing, direction) = metadata_2d();

    let image =
        TensorImage::<2>::from_flat(vec![1.0, 2.0, 3.0, 4.0], [2, 2], origin, spacing, direction)
            .unwrap();

    assert_eq!(image.shape(), [2, 2]);
    assert_eq!(image.origin(), &origin);
    assert_eq!(image.spacing(), &spacing);
    assert_eq!(image.direction(), &direction);
    assert_eq!(image.data_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn from_flat_rejects_shape_product_mismatch() {
    let (origin, spacing, direction) = metadata_2d();

    let err = TensorImage::<2>::from_flat(vec![1.0, 2.0, 3.0], [2, 2], origin, spacing, direction)
        .unwrap_err();

    assert_eq!(
        err.to_string(),
        "image flat data length 3 does not match shape [2, 2] product 4"
    );
}

#[test]
fn from_flat_rejects_shape_product_overflow() {
    let (origin, spacing, direction) = metadata_2d();

    let err = TensorImage::<2>::from_flat(Vec::new(), [usize::MAX, 2], origin, spacing, direction)
        .unwrap_err();

    assert_eq!(
        err.to_string(),
        "image shape [18446744073709551615, 2] product overflows usize"
    );
}

#[test]
fn construction_rejects_rank_mismatch() {
    let data = Tensor::<f32, SequentialBackend>::from_slice([2, 3, 1], &[0.0; 6]);
    let (origin, spacing, direction) = metadata_2d();

    let err = TensorImage::<2>::new(data, origin, spacing, direction).unwrap_err();

    assert_eq!(
        err.to_string(),
        "image tensor rank mismatch: expected 2, got 3"
    );
}

#[test]
fn into_parts_returns_exact_tensor_and_metadata() {
    let data = Tensor::<f32, SequentialBackend>::from_slice([2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let (origin, spacing, direction) = metadata_2d();
    let image = TensorImage::<2>::new(data, origin, spacing, direction).unwrap();

    let (data, returned_origin, returned_spacing, returned_direction, returned_map) =
        image.into_parts();
    assert!(returned_map.is_cartesian());

    assert_eq!(data.shape(), &[2, 2]);
    assert_eq!(data.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(returned_origin, origin);
    assert_eq!(returned_spacing, spacing);
    assert_eq!(returned_direction, direction);
}
