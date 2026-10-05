use anyhow::Result;
use ritk_image_io::VolumeError;
use ritk_spatial::{CoordinateMap, Direction, Point, SliceSeries, SliceTransform, Spacing};
use tempfile::tempdir;

use super::write_nrrd_flat;

#[test]
fn unrepresentable_geometry_preserves_existing_destination() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("existing.nrrd");
    std::fs::write(&path, b"existing destination")?;
    let direction = Direction::from_rows([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);

    let error = write_nrrd_flat(
        &path,
        [1, 1, 1],
        &Spacing::new([f64::MAX, 1.0, 1.0]),
        &Point::origin(),
        &direction,
        &[7.0],
        &CoordinateMap::Cartesian,
    )
    .expect_err("unrepresentable physical axis is rejected before file creation");

    assert!(matches!(
        error.downcast_ref::<VolumeError>(),
        Some(VolumeError::UnrepresentablePhysicalAxisNorm { column: 0 })
    ));
    assert_eq!(std::fs::read(path)?, b"existing destination");
    Ok(())
}

#[test]
fn oversized_nrrd_header_preserves_existing_destination() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("oversized-header.nrrd");
    std::fs::write(&path, b"existing destination")?;
    let direction = Direction::identity();
    let transforms = (0..18_000)
        .map(|_| SliceTransform::new(direction, [f64::MAX; 3]))
        .collect();
    let coordinate_map = CoordinateMap::SliceSeries(
        SliceSeries::try_new(transforms).expect("slice series has transforms"),
    );
    let samples = vec![7.0; 18_000];

    let error = write_nrrd_flat(
        &path,
        [18_000, 1, 1],
        &Spacing::new([1.0; 3]),
        &Point::origin(),
        &Direction::from_rows([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        &samples,
        &coordinate_map,
    )
    .expect_err("writer header limit matches the reader limit");

    assert!(error.to_string().contains("NRRD output header is"));
    assert_eq!(std::fs::read(&path)?, b"existing destination");
    Ok(())
}
