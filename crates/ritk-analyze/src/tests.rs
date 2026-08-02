use anyhow::Result;
use coeus_core::SequentialBackend;
use ritk_spatial::{Direction, Point, Spacing};
use std::path::Path;
use tempfile::tempdir;

use crate::codec::{write_le, HDR_SIZE};
use crate::{
    read_analyze, write_analyze, DT_DOUBLE, DT_FLOAT, DT_SIGNED_INT, DT_SIGNED_SHORT,
    DT_UNSIGNED_CHAR,
};

fn analyze_header(datatype: i16, bitpix: i16, shape_xyz: [i16; 3]) -> [u8; HDR_SIZE] {
    let mut header = [0u8; HDR_SIZE];
    write_le::<i32>(&mut header, 0, HDR_SIZE as i32);
    write_le::<i32>(&mut header, 32, 16_384);
    header[38] = b'r';
    write_le::<i16>(&mut header, 40, 4);
    write_le::<i16>(&mut header, 42, shape_xyz[0]);
    write_le::<i16>(&mut header, 44, shape_xyz[1]);
    write_le::<i16>(&mut header, 46, shape_xyz[2]);
    write_le::<i16>(&mut header, 48, 1);
    write_le::<i16>(&mut header, 70, datatype);
    write_le::<i16>(&mut header, 72, bitpix);
    write_le::<f32>(&mut header, 80, 1.0);
    write_le::<f32>(&mut header, 84, 1.0);
    write_le::<f32>(&mut header, 88, 1.0);
    write_le::<f32>(&mut header, 108, 0.0);
    write_le::<f32>(&mut header, 112, 1.0);
    header
}

fn write_analyze_fixture(path: &Path, header: &[u8; HDR_SIZE], payload: &[u8]) -> Result<()> {
    std::fs::write(path, header)?;
    std::fs::write(path.with_extension("img"), payload)?;
    Ok(())
}

fn read_error(header: &[u8; HDR_SIZE], payload: &[u8]) -> Result<String> {
    let directory = tempdir()?;
    let path = directory.path().join("malformed.hdr");
    write_analyze_fixture(&path, header, payload)?;
    Ok(read_analyze(&path, &SequentialBackend)
        .expect_err("malformed Analyze fixture must be rejected")
        .to_string())
}

fn make_image(
    values: Vec<f32>,
    shape: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    backend: &SequentialBackend,
) -> Result<ritk_image::Image<f32, SequentialBackend, 3>> {
    ritk_image::Image::from_flat_on(
        values,
        shape,
        origin,
        spacing,
        Direction::identity(),
        backend,
    )
}

#[test]
fn analyze_roundtrip_preserves_shape_spacing_origin_and_values() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("volume.hdr");
    let backend = SequentialBackend;
    let values: Vec<f32> = (0..24).map(|v| v as f32 + 0.25).collect();
    // Core spacing is tensor-axis order [sz, sy, sx]; the file stores file-axis
    // [sx, sy, sz] = [3.75, 2.5, 1.25]. The `originator` field encodes the
    // origin as integer voxel coordinates, so a faithful round-trip requires
    // each world-space [x, y, z] origin component to be an exact integer
    // multiple of its per-axis spacing.
    let image = make_image(
        values.clone(),
        [2, 3, 4],
        Point::new([7.5, 5.0, 3.75]),
        Spacing::new([1.25, 2.5, 3.75]),
        &backend,
    )?;

    write_analyze(&path, &image, &backend)?;
    let loaded = read_analyze(&path, &backend)?;

    assert_eq!(loaded.shape(), [2, 3, 4]);
    assert_eq!(*loaded.spacing(), Spacing::new([1.25, 2.5, 3.75]));
    assert_eq!(*loaded.origin(), Point::new([7.5, 5.0, 3.75]));
    assert_eq!(*loaded.direction(), Direction::identity());
    assert_eq!(loaded.data_slice()?, values.as_slice());

    Ok(())
}

#[test]
fn analyze_writer_emits_pixdim_in_file_axis_order() -> Result<()> {
    // The Analyze header stores spacing in file-axis order pixdim[1..3] =
    // [sx, sy, sz], the reverse of RITK's tensor-axis spacing [sz, sy, sx].
    let dir = tempdir()?;
    let path = dir.path().join("axis.hdr");
    let backend = SequentialBackend;
    let image = make_image(
        vec![0.0_f32; 24],
        [2, 3, 4],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.25, 2.5, 3.75]),
        &backend,
    )?;
    write_analyze(&path, &image, &backend)?;

    let hdr = std::fs::read(&path)?;
    let read_f32 =
        |off: usize| f32::from_le_bytes([hdr[off], hdr[off + 1], hdr[off + 2], hdr[off + 3]]);
    assert_eq!(read_f32(80), 3.75, "pixdim[1] must be sx (file-axis X)");
    assert_eq!(read_f32(84), 2.5, "pixdim[2] must be sy (file-axis Y)");
    assert_eq!(read_f32(88), 1.25, "pixdim[3] must be sz (file-axis Z)");

    Ok(())
}

#[test]
fn analyze_reader_accepts_img_path_and_rejects_invalid_header() -> Result<()> {
    let dir = tempdir()?;
    let hdr_path = dir.path().join("volume.hdr");
    let img_path = dir.path().join("volume.img");
    let backend = SequentialBackend;
    let image = make_image(
        vec![1.0, 2.0],
        [1, 1, 2],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        &backend,
    )?;

    write_analyze(&hdr_path, &image, &backend)?;
    let loaded = read_analyze(&img_path, &backend)?;
    assert_eq!(loaded.shape(), [1, 1, 2]);
    assert_eq!(loaded.data_slice()?, &[1.0, 2.0]);

    std::fs::write(&hdr_path, [0u8; 348])?;
    let err = read_analyze(&hdr_path, &backend).unwrap_err();
    assert!(
        err.to_string().contains("sizeof_hdr"),
        "error must identify invalid Analyze header, got: {err:#}"
    );

    Ok(())
}

#[test]
fn analyze_writer_output_is_byte_stable_for_native_image() -> Result<()> {
    let values: Vec<f32> = (0..24).map(|v| v as f32 * 0.5 - 3.0).collect();
    let origin = Point::new([7.5, 5.0, 3.75]);
    let spacing = Spacing::new([1.25, 2.5, 3.75]);
    let backend = SequentialBackend;

    let dir = tempdir()?;
    let first_path = dir.path().join("first.hdr");
    let second_path = dir.path().join("second.hdr");

    let first_image = make_image(values.clone(), [2, 3, 4], origin, spacing, &backend)?;
    let second_image = make_image(values, [2, 3, 4], origin, spacing, &backend)?;
    write_analyze(&first_path, &first_image, &backend)?;
    write_analyze(&second_path, &second_image, &backend)?;

    assert_eq!(
        std::fs::read(&first_path)?,
        std::fs::read(&second_path)?,
        "Analyze .hdr output must be byte-stable for the same logical image"
    );
    assert_eq!(
        std::fs::read(first_path.with_extension("img"))?,
        std::fs::read(second_path.with_extension("img"))?,
        "Analyze .img output must be byte-stable for the same logical image"
    );
    Ok(())
}

#[test]
fn analyze_reader_decodes_every_supported_scalar_with_scale() -> Result<()> {
    let cases = [
        (DT_UNSIGNED_CHAR, 8, vec![1, 7, 127], vec![2.0, 14.0, 254.0]),
        (
            DT_SIGNED_SHORT,
            16,
            [-2_i16, 0, 321]
                .into_iter()
                .flat_map(i16::to_le_bytes)
                .collect(),
            vec![-4.0, 0.0, 642.0],
        ),
        (
            DT_SIGNED_INT,
            32,
            [-100_000_i32, 0, 100_000]
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect(),
            vec![-200_000.0, 0.0, 200_000.0],
        ),
        (
            DT_FLOAT,
            32,
            [-1.25_f32, 0.0, 2.5]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
            vec![-2.5, 0.0, 5.0],
        ),
        (
            DT_DOUBLE,
            64,
            [-1.25_f64, 0.0, 2.5]
                .into_iter()
                .flat_map(f64::to_le_bytes)
                .collect(),
            vec![-2.5, 0.0, 5.0],
        ),
    ];
    let directory = tempdir()?;

    for (index, (datatype, bitpix, payload, expected)) in cases.into_iter().enumerate() {
        let path = directory.path().join(format!("scalar-{index}.hdr"));
        let mut header = analyze_header(datatype, bitpix, [3, 1, 1]);
        write_le::<f32>(&mut header, 112, 2.0);
        write_analyze_fixture(&path, &header, &payload)?;
        let image = read_analyze(&path, &SequentialBackend)?;
        assert_eq!(image.shape(), [1, 1, 3]);
        assert_eq!(image.data_slice()?, expected.as_slice());
    }

    Ok(())
}

#[test]
fn analyze_reader_rejects_invalid_geometry_and_bit_depth_before_allocation() -> Result<()> {
    let mut negative_dimension = analyze_header(DT_FLOAT, 32, [-1, 1, 1]);
    let error = read_error(&negative_dimension, &[])?;
    assert!(error.contains("nx=-1"), "unexpected error: {error}");

    write_le::<i16>(&mut negative_dimension, 42, 1);
    write_le::<i16>(&mut negative_dimension, 40, 5);
    let error = read_error(&negative_dimension, &[])?;
    assert!(
        error.contains("dimension count 5"),
        "unexpected error: {error}"
    );

    let mut multiple_volumes = analyze_header(DT_FLOAT, 32, [1, 1, 1]);
    write_le::<i16>(&mut multiple_volumes, 48, 2);
    let error = read_error(&multiple_volumes, &[])?;
    assert!(
        error.contains("volume count 2"),
        "unexpected error: {error}"
    );

    let wrong_bitpix = analyze_header(DT_SIGNED_SHORT, 32, [1, 1, 1]);
    let error = read_error(&wrong_bitpix, &[])?;
    assert!(error.contains("expected 16"), "unexpected error: {error}");

    let unsupported_datatype = analyze_header(32, 64, [1, 1, 1]);
    let error = read_error(&unsupported_datatype, &[])?;
    assert!(
        error.contains("Unsupported Analyze datatype 32"),
        "unexpected error: {error}"
    );

    let huge_header = analyze_header(DT_DOUBLE, 64, [i16::MAX; 3]);
    let error = read_error(&huge_header, &[])?;
    assert!(
        error.contains("length mismatch"),
        "huge declared geometry must be rejected from file length before allocation: {error}"
    );

    let mut big_endian = [0u8; HDR_SIZE];
    big_endian[0..4].copy_from_slice(&(HDR_SIZE as i32).to_be_bytes());
    let error = read_error(&big_endian, &[])?;
    assert!(error.contains("big-endian"), "unexpected error: {error}");

    Ok(())
}

#[test]
fn analyze_reader_requires_exact_header_length() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("header-tail.hdr");
    let header = analyze_header(DT_FLOAT, 32, [1, 1, 1]);
    let mut extended = header.to_vec();
    extended.push(0);
    std::fs::write(&path, extended)?;
    std::fs::write(path.with_extension("img"), 1.0_f32.to_le_bytes())?;

    let error = read_analyze(&path, &SequentialBackend)
        .expect_err("extended Analyze header must be rejected")
        .to_string();
    assert!(error.contains("found 349"), "unexpected error: {error}");

    Ok(())
}

#[test]
fn analyze_reader_rejects_non_finite_metadata_and_invalid_offsets() -> Result<()> {
    for (offset, value, field) in [
        (80, f32::NAN, "pixdim[1]"),
        (84, f32::INFINITY, "pixdim[2]"),
        (112, f32::NEG_INFINITY, "funused1 scale"),
    ] {
        let mut header = analyze_header(DT_FLOAT, 32, [1, 1, 1]);
        write_le::<f32>(&mut header, offset, value);
        let error = read_error(&header, &[])?;
        assert!(error.contains(field), "unexpected error: {error}");
    }

    for offset in [-1.0_f32, 1.5] {
        let mut header = analyze_header(DT_FLOAT, 32, [1, 1, 1]);
        write_le::<f32>(&mut header, 108, offset);
        let error = read_error(&header, &[])?;
        assert!(error.contains("vox_offset"), "unexpected error: {error}");
    }

    Ok(())
}

#[test]
fn analyze_reader_requires_exact_payload_and_honors_offset() -> Result<()> {
    let header = analyze_header(DT_SIGNED_SHORT, 16, [2, 1, 1]);
    let error = read_error(&header, &[1, 0, 2])?;
    assert!(error.contains("expected 4"), "unexpected error: {error}");
    let error = read_error(&header, &[1, 0, 2, 0, 3])?;
    assert!(error.contains("found 5"), "unexpected error: {error}");

    let directory = tempdir()?;
    let path = directory.path().join("offset.hdr");
    let mut offset_header = header;
    write_le::<f32>(&mut offset_header, 108, 4.0);
    write_analyze_fixture(&path, &offset_header, &[9, 8, 7, 6, 1, 0, 2, 0])?;
    let image = read_analyze(&path, &SequentialBackend)?;
    assert_eq!(image.data_slice()?, &[1.0, 2.0]);

    Ok(())
}

#[test]
fn analyze_reader_preserves_values_across_decode_chunk_boundaries() -> Result<()> {
    const VOXELS: i16 = 2_049;
    let directory = tempdir()?;
    let path = directory.path().join("chunk-boundary.hdr");
    let header = analyze_header(DT_FLOAT, 32, [VOXELS, 1, 1]);
    let expected: Vec<f32> = (0..VOXELS).map(f32::from).collect();
    let payload: Vec<u8> = expected
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    write_analyze_fixture(&path, &header, &payload)?;

    let image = read_analyze(&path, &SequentialBackend)?;
    assert_eq!(image.shape(), [1, 1, VOXELS as usize]);
    assert_eq!(image.data_slice()?, expected.as_slice());

    Ok(())
}
