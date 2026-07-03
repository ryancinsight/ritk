use crate::{read_metaimage, write_metaimage};
use anyhow::Result;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

type TestBackend = NdArray<f32>;

// ── Helpers ───────────────────────────────────────────────────────────

/// Write a minimal `.mha` file with MET_FLOAT data and identity metadata.
fn write_minimal_mha(
    path: &std::path::Path,
    data: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    spacing: [f64; 3],
    origin: [f64; 3],
) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();
    writeln!(f, "ObjectType = Image").unwrap();
    writeln!(f, "NDims = 3").unwrap();
    writeln!(f, "BinaryData = True").unwrap();
    writeln!(f, "BinaryDataByteOrderMSB = False").unwrap();
    writeln!(f, "CompressedData = False").unwrap();
    writeln!(f, "TransformMatrix = 1 0 0 0 1 0 0 0 1").unwrap();
    writeln!(f, "Offset = {} {} {}", origin[0], origin[1], origin[2]).unwrap();
    writeln!(f, "CenterOfRotation = 0 0 0").unwrap();
    writeln!(
        f,
        "ElementSpacing = {} {} {}",
        spacing[0], spacing[1], spacing[2]
    )
    .unwrap();
    writeln!(f, "DimSize = {} {} {}", nx, ny, nz).unwrap();
    writeln!(f, "ElementType = MET_FLOAT").unwrap();
    writeln!(f, "ElementDataFile = LOCAL").unwrap();
    for &v in data {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
}

/// Write a compressed `.mha` whose `DimSize` claims `nx*ny*nz` voxels but whose
/// zlib payload inflates to `raw_payload`. Used to exercise the bounded
/// decompression capacity hint against a hostile header.
fn write_compressed_mha_with_dims(
    path: &std::path::Path,
    raw_payload: &[u8],
    nx: usize,
    ny: usize,
    nz: usize,
) {
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();
    writeln!(f, "ObjectType = Image").unwrap();
    writeln!(f, "NDims = 3").unwrap();
    writeln!(f, "BinaryData = True").unwrap();
    writeln!(f, "BinaryDataByteOrderMSB = False").unwrap();
    writeln!(f, "CompressedData = True").unwrap();
    writeln!(f, "TransformMatrix = 1 0 0 0 1 0 0 0 1").unwrap();
    writeln!(f, "Offset = 0 0 0").unwrap();
    writeln!(f, "CenterOfRotation = 0 0 0").unwrap();
    writeln!(f, "ElementSpacing = 1 1 1").unwrap();
    writeln!(f, "DimSize = {} {} {}", nx, ny, nz).unwrap();
    writeln!(f, "ElementType = MET_FLOAT").unwrap();
    writeln!(f, "ElementDataFile = LOCAL").unwrap();
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
    enc.write_all(raw_payload).unwrap();
    let compressed = enc.finish().unwrap();
    f.write_all(&compressed).unwrap();
}

#[test]
fn test_compressed_hostile_dimsize_errors_without_oom() {
    // DimSize claims a 1024^3 float volume (~4.3 GiB) but the zlib payload
    // inflates to 16 bytes. The capped capacity hint must avoid a multi-GiB
    // reservation; the post-inflation length check then rejects the file.
    let dir = tempdir().unwrap();
    let path = dir.path().join("hostile_compressed.mha");
    write_compressed_mha_with_dims(&path, &[0u8; 16], 1024, 1024, 1024);

    let device: <TestBackend as Backend>::Device = Default::default();
    let result = read_metaimage::<TestBackend, _>(&path, &device);
    assert!(
        result.is_err(),
        "Hostile compressed DimSize must fail, not OOM"
    );
}

// ── Shape and metadata ─────────────────────────────────────────────────

/// The reader must shape X-fastest MetaImage payloads directly as RITK [Z,Y,X].
/// A header with `DimSize = 4 3 2` (nx=4, ny=3, nz=2) must produce
/// an Image with shape [2, 3, 4] = [nz, ny, nx].
#[test]
fn test_shape_mapped_to_zyx_without_permutation() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("shape.mha");

    let nx = 4usize;
    let ny = 3usize;
    let nz = 2usize;
    let data: Vec<f32> = (0..(nx * ny * nz)).map(|i| i as f32).collect();
    write_minimal_mha(&path, &data, nx, ny, nz, [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]);

    let device: <TestBackend as Backend>::Device = Default::default();
    let image = read_metaimage::<TestBackend, _>(&path, &device)?;

    assert_eq!(image.shape(), [nz, ny, nx], "shape must be [nz, ny, nx]");
    Ok(())
}

#[test]
fn test_x_fastest_payload_values_are_not_permuted() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("payload_values.mha");

    let nx = 3usize;
    let ny = 2usize;
    let nz = 2usize;
    let data: Vec<f32> = (0..(nx * ny * nz)).map(|i| i as f32).collect();
    write_minimal_mha(&path, &data, nx, ny, nz, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]);

    let device: <TestBackend as Backend>::Device = Default::default();
    let image = read_metaimage::<TestBackend, _>(&path, &device)?;
    image.with_data_slice(|values| {
        assert_eq!(values, data.as_slice());
    });
    Ok(())
}

/// File spacing [x,y,z] maps to internal image-axis spacing [z,y,x].
#[test]
fn test_spacing_metadata_reordered_to_internal_axes() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("spacing.mha");
    let data = vec![0.0f32; 4 * 3 * 2];
    write_minimal_mha(&path, &data, 4, 3, 2, [0.9, 0.8, 1.5], [5.0, 6.0, 7.0]);

    let device: <TestBackend as Backend>::Device = Default::default();
    let image = read_metaimage::<TestBackend, _>(&path, &device)?;

    assert!((image.spacing()[0] - 1.5).abs() < 1e-9);
    assert!((image.spacing()[1] - 0.8).abs() < 1e-9);
    assert!((image.spacing()[2] - 0.9).abs() < 1e-9);

    assert!((image.origin()[0] - 5.0).abs() < 1e-9);
    assert!((image.origin()[1] - 6.0).abs() < 1e-9);
    assert!((image.origin()[2] - 7.0).abs() < 1e-9);
    Ok(())
}

/// Identity file-axis TransformMatrix maps to internal [z,y,x] direction columns.
#[test]
fn test_file_identity_direction_reordered_to_internal_axes() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("dir.mha");
    let data = vec![0.0f32; 2 * 2 * 2];
    write_minimal_mha(&path, &data, 2, 2, 2, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]);

    let device: <TestBackend as Backend>::Device = Default::default();
    let image = read_metaimage::<TestBackend, _>(&path, &device)?;

    let d = image.direction().0;
    let expected = Direction::from_row_major([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
    for row in 0..3usize {
        for col in 0..3usize {
            assert!(
                (d[(row, col)] - expected[(row, col)]).abs() < 1e-9,
                "Direction[{},{}] = {} != {}",
                row,
                col,
                d[(row, col)],
                expected[(row, col)]
            );
        }
    }
    Ok(())
}

// ── Round-trip ─────────────────────────────────────────────────────────

/// Write an Image via `write_metaimage` and read it back; verify shape,
/// spatial metadata, and every voxel value.
#[test]
fn test_round_trip_mha() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("round_trip.mha");
    let device: <TestBackend as Backend>::Device = Default::default();

    // RITK [Z,Y,X] shape [2, 3, 4] with analytically known values 0..23.
    let data_vec: Vec<f32> = (0u32..24).map(|i| i as f32).collect();
    let tensor = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(data_vec.clone(), Shape::new([2, 3, 4])),
        &device,
    );
    let origin = Point::new([10.0, 20.0, 30.0]);
    let spacing = Spacing::new([0.9, 0.8, 1.5]);
    let direction = Direction::identity();
    let image = Image::new(tensor, origin, spacing, direction);

    write_metaimage(&path, &image)?;
    let loaded = read_metaimage::<TestBackend, _>(&path, &device)?;

    // Shape
    assert_eq!(loaded.shape(), [2, 3, 4]);

    // Origin
    assert!((loaded.origin()[0] - 10.0).abs() < 1e-5);
    assert!((loaded.origin()[1] - 20.0).abs() < 1e-5);
    assert!((loaded.origin()[2] - 30.0).abs() < 1e-5);

    // Spacing
    assert!((loaded.spacing()[0] - 0.9).abs() < 1e-5);
    assert!((loaded.spacing()[1] - 0.8).abs() < 1e-5);
    assert!((loaded.spacing()[2] - 1.5).abs() < 1e-5);

    // Voxel values: every element must equal its original value.
    loaded.with_data_slice(|loaded_vals| {
        for (i, (&got, &expected)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-5,
                "voxel[{}]: expected {}, got {}",
                i,
                expected,
                got
            );
        }
    });
    Ok(())
}

// ── Error paths ────────────────────────────────────────────────────────

/// Reading a non-existent file must return an error (not panic).
#[test]
fn test_missing_file_returns_error() {
    let device: <TestBackend as Backend>::Device = Default::default();
    let result = read_metaimage::<TestBackend, _>("/nonexistent/path/file.mha", &device);
    let msg = match result {
        Ok(_) => panic!("missing file must fail"),
        Err(err) => format!("{err:?}"),
    };
    assert!(
        msg.contains("Cannot open MetaImage file"),
        "error must preserve file-open context, got: {msg}"
    );
}

/// A file that is missing a required header field must return an error.
#[test]
fn test_missing_required_field_returns_error() -> Result<()> {
    use std::io::Write;
    let dir = tempdir()?;
    let path = dir.path().join("bad_header.mha");
    {
        let mut f = std::fs::File::create(&path)?;
        // Intentionally omits DimSize.
        writeln!(f, "ObjectType = Image")?;
        writeln!(f, "NDims = 3")?;
        writeln!(f, "ElementSpacing = 1 1 1")?;
        writeln!(f, "Offset = 0 0 0")?;
        writeln!(f, "ElementType = MET_FLOAT")?;
        writeln!(f, "ElementDataFile = LOCAL")?;
    }
    let device: <TestBackend as Backend>::Device = Default::default();
    let result = read_metaimage::<TestBackend, _>(&path, &device);
    let msg = match result {
        Ok(_) => panic!("missing DimSize must fail"),
        Err(err) => format!("{err:?}"),
    };
    assert!(
        msg.contains("DimSize"),
        "error must name the missing field, got: {msg}"
    );
    Ok(())
}

/// Unsupported ElementType must return a descriptive error.
#[test]
fn test_unsupported_element_type_returns_error() -> Result<()> {
    use std::io::Write;
    let dir = tempdir()?;
    let path = dir.path().join("bad_type.mha");
    {
        let mut f = std::fs::File::create(&path)?;
        writeln!(f, "ObjectType = Image")?;
        writeln!(f, "NDims = 3")?;
        writeln!(f, "DimSize = 2 2 2")?;
        writeln!(f, "ElementSpacing = 1 1 1")?;
        writeln!(f, "Offset = 0 0 0")?;
        writeln!(f, "TransformMatrix = 1 0 0 0 1 0 0 0 1")?;
        writeln!(f, "ElementType = MET_LONG")?; // not supported
        writeln!(f, "ElementDataFile = LOCAL")?;
        // Write 8*8 = 64 bytes of dummy data (MET_LONG = 8 bytes each, 8 voxels)
        let dummy = vec![0u8; 64];
        f.write_all(&dummy)?;
    }
    let device: <TestBackend as Backend>::Device = Default::default();
    let result = read_metaimage::<TestBackend, _>(&path, &device);
    assert!(result.is_err(), "Expected Err for unsupported ElementType");
    let msg = format!("{:?}", result.unwrap_err());
    assert!(
        msg.contains("MET_LONG"),
        "Error message must name the unsupported type; got: {}",
        msg
    );
    Ok(())
}

/// Extra trailing payload bytes are malformed input, not ignored data.
#[test]
fn test_extra_payload_bytes_return_error() -> Result<()> {
    use std::io::Write;
    let dir = tempdir()?;
    let path = dir.path().join("extra_payload.mha");
    {
        let mut f = std::fs::File::create(&path)?;
        writeln!(f, "ObjectType = Image")?;
        writeln!(f, "NDims = 3")?;
        writeln!(f, "BinaryData = True")?;
        writeln!(f, "BinaryDataByteOrderMSB = False")?;
        writeln!(f, "CompressedData = False")?;
        writeln!(f, "TransformMatrix = 1 0 0 0 1 0 0 0 1")?;
        writeln!(f, "Offset = 0 0 0")?;
        writeln!(f, "ElementSpacing = 1 1 1")?;
        writeln!(f, "DimSize = 2 2 1")?;
        writeln!(f, "ElementType = MET_FLOAT")?;
        writeln!(f, "ElementDataFile = LOCAL")?;
        for value in [1.0_f32, 2.0, 3.0, 4.0, 5.0] {
            f.write_all(&value.to_le_bytes())?;
        }
    }

    let device: <TestBackend as Backend>::Device = Default::default();
    let result = read_metaimage::<TestBackend, _>(&path, &device);
    assert!(result.is_err(), "extra payload bytes must fail");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("payload length mismatch") && msg.contains("expected") && msg.contains("16"),
        "error must name the exact payload length violation; got: {msg}"
    );
    Ok(())
}

/// `DimSize` multiplication must be checked before it can drive allocation or
/// byte-count arithmetic.
#[test]
fn test_dim_size_overflow_returns_error() -> Result<()> {
    use std::io::Write;
    let dir = tempdir()?;
    let path = dir.path().join("overflow_dimsize.mha");
    {
        let mut f = std::fs::File::create(&path)?;
        writeln!(f, "ObjectType = Image")?;
        writeln!(f, "NDims = 3")?;
        writeln!(f, "BinaryData = True")?;
        writeln!(f, "BinaryDataByteOrderMSB = False")?;
        writeln!(f, "CompressedData = False")?;
        writeln!(f, "TransformMatrix = 1 0 0 0 1 0 0 0 1")?;
        writeln!(f, "Offset = 0 0 0")?;
        writeln!(f, "ElementSpacing = 1 1 1")?;
        writeln!(f, "DimSize = {} 2 1", usize::MAX)?;
        writeln!(f, "ElementType = MET_FLOAT")?;
        writeln!(f, "ElementDataFile = LOCAL")?;
    }

    let device: <TestBackend as Backend>::Device = Default::default();
    let result = read_metaimage::<TestBackend, _>(&path, &device);
    assert!(result.is_err(), "overflowing DimSize must fail");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("voxel count overflow") && msg.contains("DimSize"),
        "error must name the DimSize overflow; got: {msg}"
    );
    Ok(())
}

/// External `.raw` file referenced from a `.mhd` header must be read.
#[test]
fn test_mhd_external_raw_file() -> Result<()> {
    use std::io::Write;
    let dir = tempdir()?;
    let mhd_path = dir.path().join("volume.mhd");
    let raw_path = dir.path().join("volume.raw");

    let nx = 2usize;
    let ny = 2usize;
    let nz = 2usize;
    let data: Vec<f32> = (0..8).map(|i| i as f32).collect();

    // Write raw binary
    {
        let mut f = std::fs::File::create(&raw_path)?;
        for &v in &data {
            f.write_all(&v.to_le_bytes())?;
        }
    }

    // Write header referencing the raw file by name
    {
        let mut f = std::fs::File::create(&mhd_path)?;
        writeln!(f, "ObjectType = Image")?;
        writeln!(f, "NDims = 3")?;
        writeln!(f, "BinaryData = True")?;
        writeln!(f, "BinaryDataByteOrderMSB = False")?;
        writeln!(f, "CompressedData = False")?;
        writeln!(f, "TransformMatrix = 1 0 0 0 1 0 0 0 1")?;
        writeln!(f, "Offset = 0 0 0")?;
        writeln!(f, "CenterOfRotation = 0 0 0")?;
        writeln!(f, "ElementSpacing = 1 1 1")?;
        writeln!(f, "DimSize = {} {} {}", nx, ny, nz)?;
        writeln!(f, "ElementType = MET_FLOAT")?;
        writeln!(f, "ElementDataFile = volume.raw")?;
    }

    let device: <TestBackend as Backend>::Device = Default::default();
    let image = read_metaimage::<TestBackend, _>(&mhd_path, &device)?;

    // Shape must be [nz, ny, nx] = [2, 2, 2]
    assert_eq!(image.shape(), [nz, ny, nx]);

    // Voxels: total = 8; verify X-fastest raw order is kept as RITK flat order.
    image.with_data_slice(|loaded_vals| {
        assert_eq!(loaded_vals, data.as_slice());
    });
    Ok(())
}

#[test]
fn native_read_metaimage_preserves_shape_and_voxels() {
    use coeus_core::SequentialBackend;

    let dir = tempdir().unwrap();
    let path = dir.path().join("coeus.mha");
    let nx = 2usize;
    let ny = 2usize;
    let nz = 2usize;
    let data: Vec<f32> = (0..(nx * ny * nz)).map(|i| i as f32).collect();
    write_minimal_mha(&path, &data, nx, ny, nz, [1.5, 2.0, 2.5], [0.0, 0.0, 0.0]);

    let backend = SequentialBackend;
    let image = crate::native::read_metaimage(&path, &backend).expect("coeus MetaImage read");

    assert_eq!(
        image.shape(),
        [nz, ny, nx],
        "coeus image shape is [nz, ny, nx]"
    );
    let loaded = image.data_slice().expect("contiguous host voxel data");
    assert_eq!(loaded, data.as_slice());
}
