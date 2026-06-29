//! Value-semantic coverage for the Coeus-backed NIfTI reader path.

use crate::header::{
    write_single_file_bytes, HeaderDims, HeaderSpatial, NiftiDatatype, NiftiHeader,
};
use crate::read_nifti_coeus_from_bytes;
use coeus_core::SequentialBackend;

#[test]
fn read_nifti_coeus_preserves_shape_and_voxels() {
    // 2×2×2 cube: file order (x-fastest) equals output [z, y, x] order, so the
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
    let image = read_nifti_coeus_from_bytes(&bytes, &backend).expect("coeus NIfTI read");

    assert_eq!(
        image.shape(),
        [2, 2, 2],
        "coeus image shape is [nz, ny, nx]"
    );
    let loaded = image.data_slice().expect("contiguous host voxel data");
    let expected: Vec<f32> = (0..8u32).map(|i| i as f32).collect();
    assert_eq!(loaded, expected.as_slice());
}
