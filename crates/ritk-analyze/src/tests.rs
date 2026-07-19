use anyhow::Result;
use coeus_core::SequentialBackend;
use ritk_spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

use crate::{read_analyze, write_analyze};

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
