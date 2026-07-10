use crate::HEADER_SIZE;
use coeus_core::SequentialBackend;
use ritk_image::native::Image;
use ritk_spatial::{Direction, Point, Spacing};

pub(crate) type TestBackend = SequentialBackend;

pub(crate) const IDENTITY_DIR: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

pub(crate) fn build_mgh_bytes(
    version: i32,
    dims: [i32; 3],
    mri_type: i32,
    spacing: [f32; 3],
    dir_cols: [[f32; 3]; 3],
    c_ras: [f32; 3],
    data: &[u8],
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(HEADER_SIZE + data.len());
    buf.extend_from_slice(&version.to_be_bytes());
    buf.extend_from_slice(&dims[0].to_be_bytes());
    buf.extend_from_slice(&dims[1].to_be_bytes());
    buf.extend_from_slice(&dims[2].to_be_bytes());
    buf.extend_from_slice(&1_i32.to_be_bytes());
    buf.extend_from_slice(&mri_type.to_be_bytes());
    buf.extend_from_slice(&0_i32.to_be_bytes());
    buf.extend_from_slice(&1_i16.to_be_bytes());
    for &s in &spacing {
        buf.extend_from_slice(&s.to_be_bytes());
    }
    for col in &dir_cols {
        for &v in col {
            buf.extend_from_slice(&v.to_be_bytes());
        }
    }
    for &c in &c_ras {
        buf.extend_from_slice(&c.to_be_bytes());
    }
    debug_assert_eq!(buf.len(), 90);
    buf.resize(HEADER_SIZE, 0u8);
    buf.extend_from_slice(data);
    buf
}

pub(crate) fn make_image(
    data: Vec<f32>,
    nz: usize,
    ny: usize,
    nx: usize,
) -> Image<f32, TestBackend, 3> {
    make_image_with_spatial(
        data,
        nz,
        ny,
        nx,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

pub(crate) fn make_image_with_spatial(
    data: Vec<f32>,
    nz: usize,
    ny: usize,
    nx: usize,
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
) -> Image<f32, TestBackend, 3> {
    Image::from_flat_on(
        data,
        [nz, ny, nx],
        origin,
        spacing,
        direction,
        &SequentialBackend,
    )
    .expect("invariant: test voxel count matches the declared shape")
}
