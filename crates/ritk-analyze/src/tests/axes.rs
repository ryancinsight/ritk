//! Analyze file axes against world axes, from a header this crate did not write.

use anyhow::Result;
use coeus_core::SequentialBackend;
use ritk_spatial::Point;
use tempfile::tempdir;

use super::{analyze_header, write_analyze_fixture};
use crate::codec::write_le;
use crate::{read_analyze, DT_FLOAT};

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
