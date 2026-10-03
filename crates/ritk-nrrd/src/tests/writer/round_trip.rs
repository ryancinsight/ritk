use super::*;

#[test]
fn test_round_trip_nrrd() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("round_trip.nrrd");
    let backend = SequentialBackend;

    // RITK [Z,Y,X] shape [2, 3, 4] with analytically known values 0..23.
    let data_vec: Vec<f32> = (0..24).map(sample_value).collect();
    let origin = Point::new([10.0, 20.0, 30.0]);
    let spacing = Spacing::new([0.9, 0.75, 1.5]);
    let direction = Direction::identity();
    let image = make_image(data_vec.clone(), [2, 3, 4], origin, spacing, direction);

    crate::write_nrrd(&path, &image, &backend)?;
    let loaded = crate::read_nrrd(&path, &backend)?;

    // Shape
    assert_eq!(loaded.shape(), [2, 3, 4]);

    // Origin
    assert!((loaded.origin()[0] - 10.0).abs() < 1e-6, "origin[0]");
    assert!((loaded.origin()[1] - 20.0).abs() < 1e-6, "origin[1]");
    assert!((loaded.origin()[2] - 30.0).abs() < 1e-6, "origin[2]");

    // Spacing
    assert!((loaded.spacing()[0] - 0.9).abs() < 1e-6, "spacing[0]");
    assert!((loaded.spacing()[1] - 0.75).abs() < 1e-6, "spacing[1]");
    assert!((loaded.spacing()[2] - 1.5).abs() < 1e-6, "spacing[2]");

    // Voxel values: every element must equal its original value.
    {
        let loaded_vals = loaded.data_slice().expect("contiguous host data");
        for (i, (&got, &expected)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-5,
                "voxel[{}]: expected {}, got {}",
                i,
                expected,
                got
            );
        }
    }
    Ok(())
}

/// Native writer round-trip: write and read back, verifying bit-perfect content.
#[test]
fn native_writer_produces_valid_nrrd() -> Result<()> {
    let nx = 4usize;
    let ny = 3usize;
    let nz = 2usize;
    let data: Vec<f32> = (0..(nx * ny * nz))
        .map(|i| sample_value(i) * 0.5 - 3.0)
        .collect();
    let origin = Point::new([5.0, -10.0, 15.0]);
    let spacing = Spacing::new([1.5, 0.75, 0.9]);
    let direction = axial_direction();

    let dir = tempdir()?;
    let path = dir.path().join("native.nrrd");
    let backend = SequentialBackend;

    let image = make_image(data.clone(), [nz, ny, nx], origin, spacing, direction);
    crate::write_nrrd(&path, &image, &backend)?;

    let loaded = crate::read_nrrd(&path, &backend)?;
    assert_eq!(loaded.shape(), [nz, ny, nx]);
    assert_eq!(*loaded.origin(), origin);
    assert_eq!(*loaded.spacing(), spacing);
    let vox = loaded.data_slice().expect("contiguous voxels");
    for (i, (&got, &expected)) in vox.iter().zip(data.iter()).enumerate() {
        assert_eq!(got.to_bits(), expected.to_bits(), "voxel[{i}] mismatch");
    }
    Ok(())
}

/// The end-to-end contract of the coordinate-map field: a beam-space image
/// written to a real file and read back must still be beam-space.
///
/// This is what the codec unit tests cannot prove — that the writer actually
/// emits the field and the reader actually attaches it. Without this wiring an
/// ultrasound acquisition silently reloads as a Cartesian raster, and every
/// downstream measurement then refers to the wrong physical points.
#[test]
fn curvilinear_geometry_survives_a_file_round_trip() -> Result<()> {
    use ritk_spatial::{CoordinateMap, CurvilinearArray};

    let geometry =
        CurvilinearArray::try_new(1.0e-4, 0.06, 0.5_f64.to_radians(), (-16.0_f64).to_radians())
            .expect("geometry");
    let map = CoordinateMap::CurvilinearArray(geometry);

    let dims = [1, 4, 5];
    let data: Vec<f32> = (0..20).map(sample_value).collect();
    let image = make_image(
        data.clone(),
        dims,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
    .with_coordinate_map(map.clone())
    .expect("2-D image accepts a curvilinear map");

    let directory = tempdir()?;
    let path = directory.path().join("beam_space.nrrd");
    crate::write_nrrd(&path, &image, &SequentialBackend)?;

    let loaded: Image<f32, TestBackend, 3> = crate::read_nrrd(&path, &SequentialBackend)?;
    assert_eq!(
        *loaded.coordinate_map(),
        map,
        "acquisition geometry must survive the round trip"
    );

    // The map must not disturb the payload or the affine metadata.
    assert_eq!(loaded.shape(), dims);
    let voxels = loaded.data_cow_on(&SequentialBackend);
    assert_eq!(voxels.as_ref(), data.as_slice(), "voxels must be unchanged");
    Ok(())
}

/// A Cartesian image must produce a header with no coordinate-map field, so
/// ordinary volumes stay byte-identical and remain readable by tools that know
/// nothing about the key.
#[test]
fn cartesian_images_write_no_coordinate_map_field() -> Result<()> {
    let image = make_image(
        vec![0.0_f32; 8],
        [2, 2, 2],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );
    let directory = tempdir()?;
    let path = directory.path().join("plain.nrrd");
    crate::write_nrrd(&path, &image, &SequentialBackend)?;

    let header = std::fs::read_to_string(&path).unwrap_or_default();
    let header = header.split("\n\n").next().unwrap_or_default().to_string();
    assert!(
        !header.contains(crate::coordinate_map::COORDINATE_MAP_KEY),
        "a Cartesian image must not emit the key; header was:\n{header}"
    );

    let loaded: Image<f32, TestBackend, 3> = crate::read_nrrd(&path, &SequentialBackend)?;
    assert!(loaded.coordinate_map().is_cartesian());
    Ok(())
}
