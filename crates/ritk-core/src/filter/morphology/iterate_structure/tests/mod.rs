//! Tests for `iterate_structure` and `BoolStructure`.

use super::*;

mod bool_structure;
mod edge_cases;
mod iterate;

/// Build a 2-D BoolStructure from a row-major vec of 0/1.
pub(crate) fn struct_2d<R: AsRef<[u8]>>(rows: &[R]) -> BoolStructure<2> {
    let shape = [rows.len(), rows[0].as_ref().len()];
    let data: Vec<bool> = rows
        .iter()
        .flat_map(|r| r.as_ref().iter())
        .map(|&v| v != 0)
        .collect();
    BoolStructure::from_data(shape, data)
}

/// Build a 3-D BoolStructure from a flat row-major vec of 0/1.
pub(crate) fn struct_3d<Y: AsRef<[u8]>, Z: AsRef<[Y]>>(zs: &[Z]) -> BoolStructure<3> {
    let nz = zs.len();
    let ny = zs[0].as_ref().len();
    let nx = zs[0].as_ref()[0].as_ref().len();
    let shape = [nz, ny, nx];
    let data: Vec<bool> = zs
        .iter()
        .flat_map(|z| z.as_ref().iter())
        .flat_map(|y| y.as_ref().iter())
        .map(|&v| v != 0)
        .collect();
    BoolStructure::from_data(shape, data)
}
