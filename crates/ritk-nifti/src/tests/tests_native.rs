//! Value-semantic coverage for the native NIfTI reader path.

use crate::header::{
    write_single_file_bytes, HeaderDims, HeaderSpatial, NiftiDatatype, NiftiHeader,
};
use coeus_core::SequentialBackend;

#[test]
fn read_nifti_native_preserves_shape_and_voxels() {
    // 2Ã—2Ã—2 cube: file order (x-fastest) equals output [z, y, x] order, so the
    // decoded voxels equal the input sequence 0..8 element-for-element.
    let header = NiftiHeader::new_3d(
        HeaderDims {
            nx: 2,
            ny: 2,
            nz: 2,
        },
        NiftiDatatype::Float32,
        HeaderSpatial {
            pixdim: [1.0, 0.75, 1.5, 2.0, 1.0, 1.0, 1.0, 1.0],
            srow_x: [-0.75, 0.0, 0.0, -11.0],
            srow_y: [0.0, -1.5, 0.0, 7.5],
            srow_z: [0.0, 0.0, 2.0, 3.25],
        },
    )
    .expect("valid header");

    let data: Vec<u8> = (0..8u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let bytes = write_single_file_bytes(&header, &data);

    let backend = SequentialBackend;
    let image = crate::read_nifti_from_bytes(&bytes, &backend).expect("native NIfTI read");

    assert_eq!(
        image.shape(),
        [2, 2, 2],
        "coeus image shape is [nz, ny, nx]"
    );
    let loaded = image.data_slice().expect("contiguous host voxel data");
    let expected: Vec<f32> = (0..8u32).map(|i| i as f32).collect();
    assert_eq!(loaded, expected.as_slice());
}

use ritk_spatial::{Direction, Point, Spacing};

/// Anisotropic, non-trivially-oriented test volume shared by both writers.
fn test_volume() -> (Vec<f32>, [usize; 3], Point<3>, Spacing<3>, Direction<3>) {
    let dims = [2usize, 3, 4];
    let n = dims[0] * dims[1] * dims[2];
    let voxels: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 3.0).collect();
    (
        voxels,
        dims,
        Point::new([-11.0, 7.5, 3.25]),
        Spacing::new([2.0, 1.5, 0.75]),
        Direction::identity(),
    )
}

#[test]
fn native_writer_round_trips_through_native_reader() {
    let (voxels, dims, origin, spacing, direction) = test_volume();
    let backend = SequentialBackend;
    let image =
        ritk_image::Image::from_flat_on(voxels.clone(), dims, origin, spacing, direction, &backend)
            .expect("native image");

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("coeus_roundtrip.nii");
    crate::write_nifti(&path, &image, &backend).expect("native NIfTI write");

    let loaded = crate::read_nifti(&path, &backend).expect("native NIfTI read");
    assert_eq!(loaded.shape(), dims);
    assert_eq!(
        loaded.data_slice().expect("contiguous"),
        voxels.as_slice(),
        "voxels must round-trip exactly (f32 LE both ways)"
    );
    let sp = loaded.spacing();
    for k in 0..3 {
        assert!(
            (sp[k] - spacing[k]).abs() < 1e-5,
            "spacing[{k}] must round-trip: {} vs {}",
            sp[k],
            spacing[k]
        );
    }
    let og = loaded.origin();
    for k in 0..3 {
        assert!(
            (og[k] - origin[k]).abs() < 1e-4,
            "origin[{k}] must round-trip: {} vs {}",
            og[k],
            origin[k]
        );
    }
}
