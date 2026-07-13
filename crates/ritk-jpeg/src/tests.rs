use super::{read_jpeg, write_jpeg, JpegReader, JpegWriter};
use anyhow::Result;
use coeus_core::SequentialBackend;
use ritk_image::native::Image;
use ritk_spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

fn image_from_values(
    shape: [usize; 3],
    values: Vec<f32>,
) -> Result<Image<f32, SequentialBackend, 3>> {
    Image::from_flat_on(
        values,
        shape,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &SequentialBackend,
    )
}

#[test]
fn grayscale_round_trip_matches_independent_decoder() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("roundtrip.jpg");
    let backend = SequentialBackend;
    let input = image_from_values([1, 2, 3], vec![0.0, 32.0, 96.0, 160.0, 224.0, 255.0])?;
    write_jpeg(&path, &input, &backend)?;
    let output = read_jpeg(&path, &backend)?;
    assert_eq!(output.shape(), [1, 2, 3]);
    let expected: Vec<f32> = image::open(&path)?
        .to_luma8()
        .into_raw()
        .into_iter()
        .map(f32::from)
        .collect();
    assert_eq!(output.data_cow_on(&backend).as_ref(), expected.as_slice());
    Ok(())
}

#[test]
fn reader_and_writer_delegate_to_canonical_operations() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("delegation.jpg");
    let input = image_from_values([1, 1, 3], vec![16.0, 128.0, 240.0])?;
    let writer = JpegWriter::new(SequentialBackend);
    writer.write_image(&path, &input)?;
    let reader = JpegReader::new(SequentialBackend);
    let output = reader.read_image(&path)?;
    assert_eq!(output.shape(), [1, 1, 3]);
    assert_eq!(output.data_cow_on(&SequentialBackend).len(), 3);
    Ok(())
}

#[test]
fn writer_rejects_non_planar_and_mismatched_images() -> Result<()> {
    let image = image_from_values([2, 1, 1], vec![0.0, 1.0])?;
    let path = tempdir()?.path().join("invalid.jpg");
    let error = write_jpeg(&path, &image, &SequentialBackend).unwrap_err();
    assert!(error.to_string().contains("depth=1"));
    Ok(())
}

#[test]
fn reader_reports_missing_files() {
    let error = read_jpeg("missing/ritk-image.jpg", &SequentialBackend).unwrap_err();
    assert!(error.to_string().contains("failed to open JPEG file"));
}
