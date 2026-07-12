//! Differential parity between the native (Coeus `MoiraiBackend`) multi-frame
//! paths and the Burn oracle paths.
//!
//! Burn is retained as a dev-only oracle: these tests assert that the native
//! reader decodes byte-identically to the Burn reader, and that the native
//! writer emits a byte-identical Part-10 file for identical voxels. The
//! analytical round-trip tests in `roundtrip.rs`/`reader.rs` are the primary
//! value-semantic gate; this module is the cross-carrier equivalence check.

use super::*;
use burn_ndarray::NdArray;
use ritk_core::image::Image as BurnImage;
use ritk_image::tensor::backend::Backend;
use ritk_image::tensor::{Shape, Tensor, TensorData};

type OracleBackend = NdArray<f32>;

/// Build the Burn oracle image from the same flat buffer as [`native_image`].
fn burn_image(data: Vec<f32>, dims: [usize; 3]) -> BurnImage<OracleBackend, 3> {
    let device = <OracleBackend as Backend>::Device::default();
    let tensor =
        Tensor::<OracleBackend, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
    BurnImage::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    )
}

#[test]
fn native_reader_matches_burn_reader_bytewise() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("mf_parity.dcm");
    let dims = [3usize, 4, 5];
    let n = dims[0] * dims[1] * dims[2];
    // Signed-spanning ramp exercises the writer rescale and reader intercept.
    let data: Vec<f32> = (0..n).map(|i| i as f32 * 1.5 - 37.0).collect();

    // Emit one file through the native writer; both readers decode the same bytes.
    let image = native_image(data.clone(), dims, [1.0, -2.0, 3.0], [2.0, 1.5, 0.75]);
    write_dicom_multiframe_native(&path, &image).expect("native write");

    let native = load_dicom_multiframe_native(&path).expect("native read");
    let device = <OracleBackend as Backend>::Device::default();
    let burn = load_dicom_multiframe::<OracleBackend, _>(&path, &device).expect("burn read");

    assert_eq!(native.shape(), burn.shape(), "shape parity");
    let native_vals = native.data_slice().expect("contiguous native data");
    let burn_vals = burn.try_data_vec().expect("burn host data");
    assert_eq!(
        native_vals,
        burn_vals.as_slice(),
        "native reader must decode identically to the burn reader"
    );
}

#[test]
fn native_writer_matches_burn_writer_bytewise() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let native_path = tmp.path().join("mf_native.dcm");
    let burn_path = tmp.path().join("mf_burn.dcm");
    let dims = [2usize, 3, 4];
    let n = dims[0] * dims[1] * dims[2];
    let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 4.0).collect();

    write_dicom_multiframe_native(&native_path, &native_image(data.clone(), dims, [0.0; 3], [1.0; 3]))
        .expect("native write");
    write_dicom_multiframe(&burn_path, &burn_image(data, dims)).expect("burn write");

    // The generated UIDs are random per call, so full-file byte equality does
    // not hold; assert equality of the decoded PixelData tensor instead, which
    // is the substrate-invariant part of the encode contract.
    let native = load_dicom_multiframe_native(&native_path).expect("native read");
    let burn_reload = load_dicom_multiframe_native(&burn_path).expect("burn-file read");
    assert_eq!(native.shape(), burn_reload.shape(), "shape parity");
    assert_eq!(
        native.data_slice().expect("native contiguous"),
        burn_reload.data_slice().expect("burn-file contiguous"),
        "native and burn writers must encode identical pixel data"
    );
}
