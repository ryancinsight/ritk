use anyhow::Result;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use nalgebra::SMatrix;
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

use crate::{write_nrrd, NrrdWriter};

type TestBackend = NdArray<f32>;

// ── Helpers ───────────────────────────────────────────────────────────

/// Scan `haystack` for the ASCII byte pattern `needle`.
fn bytes_contain(haystack: &[u8], needle: &str) -> bool {
    let nb = needle.as_bytes();
    haystack.windows(nb.len()).any(|w| w == nb)
}

fn axial_direction() -> Direction<3> {
    Direction(SMatrix::<f64, 3, 3>::from_row_slice(&[
        0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
    ]))
}

fn nrrd_payload(bytes: &[u8]) -> &[u8] {
    let terminator = b"\n\n";
    let header_end = bytes
        .windows(terminator.len())
        .position(|w| w == terminator)
        .map(|p| p + terminator.len())
        .expect("blank-line terminator not found in NRRD file");
    &bytes[header_end..]
}

fn decode_le_f32_payload(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

// ── Header content ─────────────────────────────────────────────────────

/// A written NRRD file must contain the mandatory header fields.
#[test]
fn test_mandatory_header_fields_present() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("mandatory.nrrd");
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

    let tensor = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![1.0f32; 2 * 3 * 4], Shape::new([2, 3, 4])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction(SMatrix::identity()),
    );

    write_nrrd(&path, &image)?;

    let bytes = std::fs::read(&path)?;
    assert!(bytes_contain(&bytes, "NRRD0004"), "missing NRRD magic");
    assert!(bytes_contain(&bytes, "type: float"), "missing type");
    assert!(bytes_contain(&bytes, "dimension: 3"), "missing dimension");
    assert!(bytes_contain(&bytes, "encoding: raw"), "missing encoding");
    assert!(bytes_contain(&bytes, "endian: little"), "missing endian");
    assert!(
        bytes_contain(&bytes, "space directions:"),
        "missing space directions"
    );
    assert!(
        bytes_contain(&bytes, "space origin:"),
        "missing space origin"
    );

    Ok(())
}

/// `sizes` must be written as `nx ny nz` — the NRRD [X,Y,Z] order, which
/// is the reverse of RITK's [Z,Y,X] convention.
/// An Image with RITK shape [nz=2, ny=3, nx=4] must produce `sizes: 4 3 2`.
#[test]
fn test_sizes_written_in_xyz_order() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("sizes.nrrd");
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

    // RITK shape [nz=2, ny=3, nx=4]
    let tensor = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![0.0f32; 2 * 3 * 4], Shape::new([2, 3, 4])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction(SMatrix::identity()),
    );

    write_nrrd(&path, &image)?;

    let bytes = std::fs::read(&path)?;
    assert!(
        bytes_contain(&bytes, "sizes: 4 3 2"),
        "sizes must be nx ny nz = 4 3 2 for RITK shape [2, 3, 4]"
    );

    Ok(())
}

/// For the canonical axial RITK direction, NRRD `space directions` must
/// encode file axes `[x,y,z]` as internal columns `[col,row,depth]`.
#[test]
fn test_space_directions_encodes_spacing_on_diagonal() -> Result<()> {
    let dir_tmp = tempdir()?;
    let path = dir_tmp.path().join("diag_sd.nrrd");
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

    let tensor = Tensor::<TestBackend, 3>::zeros([2, 2, 2], &device);
    let image = Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([0.9, 0.75, 1.5]),
        axial_direction(),
    );

    write_nrrd(&path, &image)?;

    let bytes = std::fs::read(&path)?;
    // For axial internal columns depth=Z, row=Y, col=X:
    //   sd0(file x) = internal col   * spacing[2] = (1.5,0,0)
    //   sd1(file y) = internal row   * spacing[1] = (0,0.75,0)
    //   sd2(file z) = internal depth * spacing[0] = (0,0,0.9)
    assert!(
        bytes_contain(&bytes, "(1.5,0,0)"),
        "sd0 must encode internal column spacing on file X"
    );
    assert!(
        bytes_contain(&bytes, "(0,0.75,0)"),
        "sd1 must encode internal row spacing on file Y"
    );
    assert!(
        bytes_contain(&bytes, "(0,0,0.9)"),
        "sd2 must encode internal depth spacing on file Z"
    );

    Ok(())
}

/// `space origin` must contain the image origin coordinates.
#[test]
fn test_space_origin_written_correctly() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("origin.nrrd");
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

    let tensor = Tensor::<TestBackend, 3>::zeros([2, 2, 2], &device);
    let image = Image::new(
        tensor,
        Point::new([10.5, 20.25, 30.125]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction(SMatrix::identity()),
    );

    write_nrrd(&path, &image)?;

    let bytes = std::fs::read(&path)?;
    assert!(
        bytes_contain(&bytes, "10.5"),
        "origin[0] not found in header"
    );
    assert!(
        bytes_contain(&bytes, "20.25"),
        "origin[1] not found in header"
    );
    assert!(
        bytes_contain(&bytes, "30.125"),
        "origin[2] not found in header"
    );

    Ok(())
}

/// The binary payload size must equal `nx * ny * nz * 4` bytes
/// (one 4-byte little-endian f32 per voxel) following the blank terminator.
#[test]
fn test_payload_size_correct() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("payload.nrrd");
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

    let nz = 3usize;
    let ny = 4usize;
    let nx = 5usize;
    let n_voxels = nz * ny * nx;

    let tensor = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![1.0f32; n_voxels], Shape::new([nz, ny, nx])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction(SMatrix::identity()),
    );

    write_nrrd(&path, &image)?;

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
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

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

    let tensor = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(data_vec.clone(), Shape::new([nz, ny, nx])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        axial_direction(),
    );

    write_nrrd(&path, &image)?;

    let bytes = std::fs::read(&path)?;
    let payload_values = decode_le_f32_payload(nrrd_payload(&bytes));
    assert_eq!(
        payload_values, data_vec,
        "NRRD payload order must match X-fastest RITK ZYX flat storage"
    );

    Ok(())
}

/// NrrdWriter struct delegates correctly to `write_nrrd`.
#[test]
fn test_writer_struct_creates_file() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("writer_struct.nrrd");
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

    let tensor = Tensor::<TestBackend, 3>::zeros([2, 2, 2], &device);
    let image = Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction(SMatrix::identity()),
    );

    let writer = NrrdWriter;
    writer.write(&path, &image)?;

    assert!(path.exists(), "output file must exist after write");
    assert!(
        std::fs::metadata(&path)?.len() > 0,
        "output file must be non-empty"
    );

    Ok(())
}

/// Non-identity direction matrix must appear in `space directions` as
/// correctly scaled NRRD file-axis vectors.
///
/// Test: internal columns depth=Z, row=-X, col=Y with spacing [2,3,4].
///   file x = internal col   →  sd0 = (0,4,0)
///   file y = internal row   →  sd1 = (-3,0,0)
///   file z = internal depth →  sd2 = (0,0,2)
#[test]
fn test_rotated_direction_in_space_directions() -> Result<()> {
    let dir_tmp = tempdir()?;
    let path = dir_tmp.path().join("rotated.nrrd");
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

    let tensor = Tensor::<TestBackend, 3>::zeros([2, 2, 2], &device);

    let mut mat = SMatrix::<f64, 3, 3>::zeros();
    // Column 0 = internal depth axis = physical Z.
    mat[(0, 0)] = 0.0;
    mat[(1, 0)] = 0.0;
    mat[(2, 0)] = 1.0;
    // Column 1 = internal row axis = physical -X.
    mat[(0, 1)] = -1.0;
    mat[(1, 1)] = 0.0;
    mat[(2, 1)] = 0.0;
    // Column 2 = internal column axis = physical Y.
    mat[(0, 2)] = 0.0;
    mat[(1, 2)] = 1.0;
    mat[(2, 2)] = 0.0;

    let image = Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([2.0, 3.0, 4.0]),
        Direction(mat),
    );

    write_nrrd(&path, &image)?;

    let bytes = std::fs::read(&path)?;

    assert!(
        bytes_contain(&bytes, "(0,4,0)"),
        "sd0 must be (0,4,0); file content does not match"
    );
    assert!(
        bytes_contain(&bytes, "(-3,0,0)"),
        "sd1 must be (-3,0,0); file content does not match"
    );
    assert!(
        bytes_contain(&bytes, "(0,0,2)"),
        "sd2 must be (0,0,2); file content does not match"
    );

    Ok(())
}

/// Write an Image via `write_nrrd` and read it back; verify shape,
/// spatial metadata, and every voxel value.
#[test]
fn test_round_trip_nrrd() -> Result<()> {
    use crate::read_nrrd;

    let dir = tempdir()?;
    let path = dir.path().join("round_trip.nrrd");
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

    // RITK [Z,Y,X] shape [2, 3, 4] with analytically known values 0..23.
    let data_vec: Vec<f32> = (0u32..24).map(|i| i as f32).collect();
    let tensor = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(data_vec.clone(), Shape::new([2, 3, 4])),
        &device,
    );
    let origin = Point::new([10.0, 20.0, 30.0]);
    let spacing = Spacing::new([0.9, 0.75, 1.5]);
    let direction = Direction(SMatrix::identity());
    let image = Image::new(tensor, origin, spacing, direction);

    write_nrrd(&path, &image)?;
    let loaded = read_nrrd::<TestBackend, _>(&path, &device)?;

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
    let loaded_td = loaded.data().clone().to_data();
    let loaded_vals = loaded_td.as_slice::<f32>().unwrap();
    for (i, (&got, &expected)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-5,
            "voxel[{}]: expected {}, got {}",
            i,
            expected,
            got
        );
    }

    Ok(())
}
