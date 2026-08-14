//! Borrowed-view contracts: that the view borrows, and what it borrows.
//!
//! The claim these tests exist to falsify is that [`crate::view`] removes
//! copies. Value equality cannot show that — a materializing implementation
//! would satisfy it too. Provenance can: a view that borrows the source buffer
//! has a `data()` pointer *inside* the source allocation, and a view that
//! copied would not. Every zero-copy assertion below is a pointer-identity
//! assertion for that reason.

use coeus_core::{CpuAddressableStorage, SequentialBackend};
use coeus_tensor::Tensor;

use crate::test_support::metadata_2d;
use crate::types::Image;
use crate::view::tensor_view;

type TensorImage<const D: usize> = Image<f32, SequentialBackend, D>;
type HostTensor = Tensor<f32, SequentialBackend>;

fn base_tensor() -> HostTensor {
    HostTensor::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
}

/// Address of a tensor's storage, the identity a borrow must preserve.
fn storage_addr(tensor: &HostTensor) -> usize {
    tensor.storage().as_slice().as_ptr() as usize
}

#[test]
fn view_of_contiguous_tensor_borrows_the_source_buffer() {
    let tensor = base_tensor();
    let view = tensor_view::<f32, SequentialBackend, 2>(&tensor).unwrap();

    assert_eq!(
        view.data().as_ptr() as usize,
        storage_addr(&tensor),
        "view must borrow the source allocation, not a copy of it"
    );
    assert_eq!(view.shape(), [2, 3]);
    assert_eq!(view.strides(), [3, 1]);
    assert_eq!(*view.get([1, 2]).unwrap(), 6.0);
}

#[test]
fn view_of_permuted_tensor_borrows_and_indexes_through_strides() {
    // The case `data_slice` rejects outright and `data_cow` answers with a
    // whole-buffer copy: a permuted (strided) view.
    let permuted = base_tensor().permute(&[1, 0]);
    let view = tensor_view::<f32, SequentialBackend, 2>(&permuted).unwrap();

    assert_eq!(
        view.data().as_ptr() as usize,
        storage_addr(&permuted),
        "a strided layout is expressed by strides, not by materializing"
    );
    assert_eq!(view.shape(), [3, 2]);

    // Transpose oracle: element [row, column] of the permuted view is
    // [column, row] of the base.
    let base = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    for (column, base_row) in base.iter().enumerate() {
        for (row, expected) in base_row.iter().enumerate() {
            assert_eq!(*view.get([row, column]).unwrap(), *expected);
        }
    }
}

#[test]
fn view_of_offset_tensor_borrows_where_to_contiguous_copies() {
    // The sharp edge this seam exists for. `Layout::is_contiguous` inspects
    // strides only, so an offset row slice reports *contiguous* — and
    // `to_contiguous` still allocates and copies it, because its fast path
    // additionally requires `offset == 0`. The view pays neither cost.
    let base = base_tensor();
    let row = Tensor::from_raw_parts(
        base.storage().clone(),
        base.layout().slice(&[(1, 2), (0, 3)]),
    );
    assert!(
        row.is_contiguous(),
        "fixture must be the offset-but-contiguous case"
    );
    assert_eq!(
        row.layout().offset(),
        3,
        "fixture must carry a nonzero offset"
    );

    let view = tensor_view::<f32, SequentialBackend, 2>(&row).unwrap();
    assert_eq!(
        view.data().as_ptr() as usize,
        storage_addr(&base),
        "the view must borrow the original allocation"
    );
    assert_eq!(view.offset(), 3, "the offset rides in the layout");
    assert_eq!(*view.get([0, 0]).unwrap(), 4.0);
    assert_eq!(*view.get([0, 2]).unwrap(), 6.0);

    // Differential evidence for the copy the view avoids: the same tensor
    // through `to_contiguous` lands in a different allocation.
    let materialized = row.to_contiguous();
    assert_ne!(
        storage_addr(&materialized),
        storage_addr(&base),
        "to_contiguous copies an offset layout — the cost the view removes"
    );
}

#[test]
fn image_view_borrows_voxels_without_copying() {
    let (origin, spacing, direction) = metadata_2d();
    let image = TensorImage::<2>::from_flat(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [2, 3],
        origin,
        spacing,
        direction,
    )
    .unwrap();

    let view = image.view().unwrap();
    assert_eq!(
        view.data().as_ptr() as usize,
        storage_addr(image.data()),
        "Image::view must borrow the image's own buffer"
    );
    assert_eq!(view.shape(), image.shape());
    assert_eq!(*view.get([0, 1]).unwrap(), 2.0);
}

#[test]
fn image_view_succeeds_on_the_layout_data_slice_rejects() {
    // `data_slice` and `view` disagree by design: the flat contract cannot
    // express a strided layout, the indexed one can.
    let (origin, spacing, direction) = metadata_2d();
    let image =
        TensorImage::<2>::new(base_tensor().permute(&[1, 0]), origin, spacing, direction).unwrap();

    assert!(
        image.data_slice().is_err(),
        "flat borrow rejects the layout"
    );

    let view = image.view().expect("indexed borrow accepts it");
    assert_eq!(view.shape(), [3, 2]);
    assert_eq!(*view.get([2, 1]).unwrap(), 6.0);
}
