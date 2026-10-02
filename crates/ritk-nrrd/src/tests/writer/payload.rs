use super::{axial_direction, make_image, nrrd_payload, payload_samples, zeros_image, TestBackend};
use crate::write_nrrd_with_data;
use anyhow::Result;
use coeus_core::SequentialBackend;
use ritk_codecs::sample::Exact;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

/// The binary payload size must equal `nx * ny * nz * 4` bytes
/// (one 4-byte little-endian f32 per voxel) following the blank terminator.
#[test]
fn test_payload_size_correct() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("payload.nrrd");
    let backend = SequentialBackend;

    let nz = 3usize;
    let ny = 4usize;
    let nx = 5usize;
    let n_voxels = nz * ny * nx;

    let image = make_image(
        vec![1.0f32; n_voxels],
        [nz, ny, nx],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );

    crate::write_nrrd(&path, &image, &backend)?;

    let bytes = std::fs::read(&path)?;
    let expected_payload = n_voxels * 4;

    let actual_payload = nrrd_payload(&bytes).len();
    assert_eq!(
        actual_payload, expected_payload,
        "payload is {} bytes; expected {} ({} voxels × 4)",
        actual_payload, expected_payload, n_voxels
    );

    Ok(())
}

/// RITK [Z,Y,X] flat tensor values must be written directly because NRRD
/// raw payload order is X-fastest.
#[test]
fn test_payload_written_in_x_fastest_order() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("payload_order.nrrd");
    let backend = SequentialBackend;

    let nz = 2usize;
    let ny = 2usize;
    let nx = 3usize;
    let mut data_vec = Vec::with_capacity(nz * ny * nx);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                data_vec.push((100 * x + 10 * y + z) as f32);
            }
        }
    }

    let image = make_image(
        data_vec.clone(),
        [nz, ny, nx],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        axial_direction(),
    );

    crate::write_nrrd(&path, &image, &backend)?;

    let bytes = std::fs::read(&path)?;
    let payload_values = payload_samples(nrrd_payload(&bytes));
    assert_eq!(
        payload_values, data_vec,
        "NRRD payload order must match X-fastest RITK ZYX flat storage"
    );

    Ok(())
}

#[test]
fn test_caller_payload_length_must_match_shape() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("wrong_payload.nrrd");
    let image = make_image(
        vec![0.0; 8],
        [2, 2, 2],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );

    let error =
        write_nrrd_with_data(&path, &image, &[0.0; 7]).expect_err("short payload must be rejected");
    assert!(
        error.to_string().contains("requires 8"),
        "error must report the required voxel count: {error}"
    );
    assert!(!path.exists(), "invalid payload must not create a file");
    Ok(())
}

/// NrrdWriter struct delegates correctly to `write_nrrd`.
#[test]
fn test_writer_struct_creates_file() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("writer_struct.nrrd");
    let backend = SequentialBackend;

    let image = zeros_image(
        [2, 2, 2],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );

    crate::write_nrrd(&path, &image, &backend)?;

    assert!(path.exists(), "output file must exist after write");
    assert!(
        std::fs::metadata(&path)?.len() > 0,
        "output file must be non-empty"
    );

    Ok(())
}

/// Write an Image via `write_nrrd` and read it back; verify shape,
/// spatial metadata, and every voxel value.
#[test]
fn test_round_trip_nrrd() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("round_trip.nrrd");
    let backend = SequentialBackend;

    // RITK [Z,Y,X] shape [2, 3, 4] with analytically known values 0..23.
    let data_vec: Vec<f32> = (0u32..24).map(|i| i as f32).collect();
    let origin = Point::new([10.0, 20.0, 30.0]);
    let spacing = Spacing::new([0.9, 0.75, 1.5]);
    let direction = Direction::identity();
    let image = make_image(data_vec.clone(), [2, 3, 4], origin, spacing, direction);

    crate::write_nrrd(&path, &image, &backend)?;
    let loaded = crate::read_nrrd::<f32, _, _, _>(&path, &backend, Exact)?;

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
fn writer_produces_valid_nrrd() -> Result<()> {
    let nx = 4usize;
    let ny = 3usize;
    let nz = 2usize;
    let data: Vec<f32> = (0..(nx * ny * nz)).map(|i| i as f32 * 0.5 - 3.0).collect();
    let origin = Point::new([5.0, -10.0, 15.0]);
    let spacing = Spacing::new([1.5, 0.75, 0.9]);
    let direction = axial_direction();

    let dir = tempdir()?;
    let path = dir.path().join("native.nrrd");
    let backend = SequentialBackend;

    let image = make_image(data.clone(), [nz, ny, nx], origin, spacing, direction);
    crate::write_nrrd(&path, &image, &backend)?;

    let loaded = crate::read_nrrd::<f32, _, _, _>(&path, &backend, Exact)?;
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
    let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
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

    let loaded: Image<f32, TestBackend, 3> =
        crate::read_nrrd::<f32, _, _, _>(&path, &SequentialBackend, Exact)?;
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
