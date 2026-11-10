//! Analyze file axes against world axes, from a header this crate did not write.

use anyhow::Result;
use coeus_core::SequentialBackend;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

use super::{analyze_header, write_analyze_fixture};
use crate::codec::{read_le, write_le};
use crate::{read_analyze, write_analyze, DT_FLOAT};

/// Distinct extents per tensor axis, so a lost relabeling changes the shape.
const DIMS: [usize; 3] = [2, 3, 4];
/// Powers of two, exact in the header's `f32` `pixdim`.
const SPACING: [f64; 3] = [0.5, 2.0, 4.0];
/// `originator` stores `round(origin / spacing)` per file axis, and any of the
/// three spacings can end up on any world axis, so each component is a
/// multiple of all of them and the origin round-trips exactly.
const ORIGIN: [f64; 3] = [4.0, -8.0, 12.0];

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

fn ramp_image(direction: Direction<3>) -> Result<Image<f32, SequentialBackend, 3>> {
    Image::from_flat_on(
        (0..24u8).map(f32::from).collect(),
        DIMS,
        Point::new(ORIGIN),
        Spacing::new(SPACING),
        direction,
        &SequentialBackend,
    )
}

/// World point bits → voxel value, sorted, for every voxel of `image`.
fn world_map(image: &Image<f32, SequentialBackend, 3>) -> Result<Vec<([u64; 3], u32)>> {
    let [nz, ny, nx] = image.shape();
    let values = image.data_slice()?;
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
    Ok(map)
}

/// Each of the 48 signed permutations writes and reads back with every voxel
/// at its original world point.
#[test]
fn every_signed_permutation_round_trips_world_positions() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("permuted.hdr");
    for direction in signed_permutations() {
        let image = ramp_image(direction)?;

        write_analyze(&path, &image, &SequentialBackend)?;
        let loaded = read_analyze(&path, &SequentialBackend)?;

        assert_eq!(
            world_map(&loaded)?,
            world_map(&image)?,
            "{direction:?}: world point → value map changed"
        );
    }
    Ok(())
}

/// An identity-tagged image advances world X along tensor axis 0, so the
/// header lists that axis first: `dim = 2 3 4`, `pixdim = 0.5 2 4`,
/// `originator = 8 -4 3`, and file X varies fastest through the depth index.
#[test]
fn an_identity_image_writes_its_depth_axis_as_file_x() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("identity.hdr");

    write_analyze(
        &path,
        &ramp_image(Direction::identity())?,
        &SequentialBackend,
    )?;

    let header = std::fs::read(&path)?;
    let dims = [42, 44, 46].map(|offset| read_le::<i16>(&header, offset));
    let pixdim = [80, 84, 88].map(|offset| read_le::<f32>(&header, offset));
    let originator = [253, 255, 257].map(|offset| read_le::<i16>(&header, offset));
    assert_eq!(dims, [2, 3, 4], "dim[1..=3]");
    assert_eq!(pixdim, [0.5, 2.0, 4.0], "pixdim[1..=3]");
    assert_eq!(originator, [8, -4, 3], "originator[0..3]");

    // File voxel (fx, fy, fz) is tensor voxel (fx, fy, fz) = fx·12 + fy·4 + fz.
    let payload = std::fs::read(path.with_extension("img"))?;
    let stored: Vec<f32> = payload
        .chunks_exact(4)
        .map(|bytes| read_le::<f32>(bytes, 0))
        .collect();
    let expected: Vec<f32> = (0..4u8)
        .flat_map(|fz| (0..3u8).flat_map(move |fy| (0..2u8).map(move |fx| fx * 12 + fy * 4 + fz)))
        .map(f32::from)
        .collect();
    assert_eq!(stored, expected);
    Ok(())
}

/// File X advances world X and file Z advances world Z.
///
/// The oracle is a header this crate did not write: `dim = 2 3 4`,
/// `pixdim = 1 10 100`. Tensor index `(depth, row, col)` is `(z, y, x)`, so
/// `(0, 0, 1)` is the second X sample at world `(1, 0, 0)` and `(3, 0, 0)` is
/// the fourth Z slice at world `(0, 0, 300)`. An identity direction instead
/// makes the depth index advance world X.
#[test]
fn a_hand_built_analyze_axes_advance_their_world_axes() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("axes-oracle.hdr");
    let mut header = analyze_header(DT_FLOAT, 32, [2, 3, 4]);
    write_le::<f32>(&mut header, 80, 1.0);
    write_le::<f32>(&mut header, 84, 10.0);
    write_le::<f32>(&mut header, 88, 100.0);
    let payload: Vec<u8> = (0..24u8)
        .flat_map(|voxel| f32::from(voxel).to_le_bytes())
        .collect();
    write_analyze_fixture(&path, &header, &payload)?;

    let loaded = read_analyze(&path, &SequentialBackend)?;

    assert_eq!(loaded.shape(), [4, 3, 2], "shape is [depth, row, col]");
    assert_eq!(
        loaded.continuous_index_to_physical_point(&Point::new([0.0, 0.0, 1.0])),
        Point::new([1.0, 0.0, 0.0])
    );
    assert_eq!(
        loaded.continuous_index_to_physical_point(&Point::new([0.0, 1.0, 0.0])),
        Point::new([0.0, 10.0, 0.0])
    );
    assert_eq!(
        loaded.continuous_index_to_physical_point(&Point::new([3.0, 0.0, 0.0])),
        Point::new([0.0, 0.0, 300.0])
    );
    let ramp: Vec<f32> = (0..24u8).map(f32::from).collect();
    assert_eq!(loaded.data_slice()?, ramp.as_slice());

    Ok(())
}
