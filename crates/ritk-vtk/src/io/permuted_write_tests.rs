//! The legacy writer stores every signed axis permutation on the file axes.
//!
//! Fixtures use power-of-two spacings and integer origins, so world points are
//! exact in `f64` and the oracle compares their bits.

use super::{VtkReader, VtkWriter};
use coeus_core::SequentialBackend;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

/// Distinct extents per tensor axis, so a lost relabeling changes the shape.
const DIMS: [usize; 3] = [2, 3, 4];

/// All 48 signed 3-D permutation matrices: 6 column orders × 8 sign masks.
fn signed_permutations() -> Vec<Direction<3>> {
    let orders = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut directions = Vec::with_capacity(48);
    for order in orders {
        for mask in 0..8u8 {
            let mut rows = [[0.0; 3]; 3];
            for (column, &row) in order.iter().enumerate() {
                rows[row][column] = if mask & (1 << column) == 0 { 1.0 } else { -1.0 };
            }
            directions.push(Direction::from_rows(rows));
        }
    }
    directions
}

fn ramp_image(direction: Direction<3>) -> Image<f32, SequentialBackend, 3> {
    Image::from_flat_on(
        (0..24u8).map(f32::from).collect(),
        DIMS,
        Point::new([3.0, -5.0, 7.0]),
        Spacing::new([0.5, 2.0, 4.0]),
        direction,
        &SequentialBackend,
    )
    .expect("fixture image")
}

/// World point bits → voxel value, sorted, for every voxel of `image`.
fn world_map(image: &Image<f32, SequentialBackend, 3>) -> Vec<([u64; 3], u32)> {
    let [nz, ny, nx] = image.shape();
    let values = image.data_slice().expect("contiguous image");
    let mut map = Vec::with_capacity(values.len());
    let mut flat = 0;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let index = [z, y, x].map(|i| {
                    f64::from(u32::try_from(i).expect("invariant: fixture index fits u32"))
                });
                let point = image.continuous_index_to_physical_point(&Point::new(index));
                map.push((point.to_array().map(f64::to_bits), values[flat].to_bits()));
                flat += 1;
            }
        }
    }
    map.sort_unstable();
    map
}

/// Each of the 48 signed permutations writes and reads back with every voxel
/// at its original world point.
#[test]
fn every_signed_permutation_round_trips_world_positions() {
    let directory = tempdir().expect("temporary directory");
    let path = directory.path().join("permuted.vtk");
    for direction in signed_permutations() {
        let image = ramp_image(direction);

        VtkWriter::new(SequentialBackend)
            .write(&path, &image)
            .expect("a signed permutation is writable");
        let loaded = VtkReader::new(SequentialBackend)
            .read(&path)
            .expect("read the written file");

        assert_eq!(
            world_map(&loaded),
            world_map(&image),
            "{direction:?}: world point → value map changed"
        );
    }
}

/// An identity-tagged image advances world X along tensor axis 0, so the file
/// must list that axis first: `DIMENSIONS 2 3 4` and `SPACING 0.5 2 4`, the
/// reverse of what a file-axes image of the same shape emits.
#[test]
fn an_identity_image_writes_its_depth_axis_as_file_x() {
    let directory = tempdir().expect("temporary directory");
    let path = directory.path().join("identity.vtk");

    VtkWriter::new(SequentialBackend)
        .write(&path, &ramp_image(Direction::identity()))
        .expect("identity is a signed permutation");

    let bytes = std::fs::read(&path).expect("read written file");
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("DIMENSIONS 2 3 4"), "got:\n{text}");
    assert!(text.contains("SPACING 0.5 2 4"), "got:\n{text}");
    assert!(text.contains("ORIGIN 3 -5 7"), "got:\n{text}");
}
