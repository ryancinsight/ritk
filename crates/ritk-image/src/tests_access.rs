//! Host-access contracts: when a view borrows and when it materialises.
#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]

use coeus_core::SequentialBackend;

use coeus_tensor::Tensor;

use crate::test_support::metadata_2d;
use crate::types::Image;

type TensorImage<const D: usize> = Image<f32, SequentialBackend, D>;

#[test]
fn data_cow_borrows_when_contiguous() {
    let (origin, spacing, direction) = metadata_2d();
    let image = TensorImage::<2>::from_flat(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [2, 3],
        origin,
        spacing,
        direction,
    )
    .unwrap();

    let cow = image.data_cow();
    assert!(
        matches!(cow, std::borrow::Cow::Borrowed(_)),
        "contiguous image must borrow (zero-copy)"
    );
    assert_eq!(cow.as_ref(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(image.data_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn data_cow_materializes_logical_order_for_permuted_view() {
    // Build a [2, 3] image, permute the tensor to [3, 2] (non-contiguous
    // strided view), and re-wrap. Logical row-major order of the permuted
    // view is the host transpose — the oracle the extraction must match.
    let (origin, spacing, direction) = metadata_2d();
    let base =
        Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let permuted = base.permute(&[1, 0]); // shape [3, 2], strides non-contiguous
    let image = TensorImage::<2>::new(permuted, origin, spacing, direction).unwrap();

    // The strict borrow API must refuse the strided view (existing contract).
    assert!(
        image.data_slice().is_err(),
        "data_slice must reject non-contiguous"
    );

    // Host transpose oracle: [[1,4],[2,5],[3,6]] row-major.
    let expected = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let cow = image.data_cow();
    assert!(
        matches!(cow, std::borrow::Cow::Owned(_)),
        "non-contiguous image must materialize (owned)"
    );
    assert_eq!(cow.as_ref(), &expected);
    assert_eq!(image.data_vec(), expected.to_vec());
}

#[test]
fn data_slice_rejects_non_contiguous_layout() {
    let data =
        Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let column_view = Tensor::from_raw_parts(
        data.storage().clone(),
        data.layout().slice(&[(0, 2), (1, 2)]),
    );
    let (origin, spacing, direction) = metadata_2d();
    let image = TensorImage::<2>::new(column_view, origin, spacing, direction).unwrap();

    let err = image.data_slice().unwrap_err();

    assert_eq!(
        err.to_string(),
        "image data is not contiguous: shape=[2, 1], strides=[3, 1]"
    );
}
