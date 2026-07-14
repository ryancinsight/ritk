use super::{read_tiff, TiffReader};
use crate::write_tiff;
use anyhow::Result;
use coeus_core::SequentialBackend;
use ritk_image::native::Image;
use ritk_spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

fn image(shape: [usize; 3], values: Vec<f32>) -> Result<Image<f32, SequentialBackend, 3>> {
    Image::from_flat_on(
        values,
        shape,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &SequentialBackend,
    )
}

fn assert_round_trip(shape: [usize; 3], values: Vec<f32>) -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("roundtrip.tiff");
    let input = image(shape, values)?;
    write_tiff(&input, &path, &SequentialBackend)?;
    let output = read_tiff(&path, &SequentialBackend)?;
    assert_eq!(output.shape(), shape);
    assert_eq!(
        output.data_cow_on(&SequentialBackend).as_ref(),
        input.data_cow_on(&SequentialBackend).as_ref()
    );
    Ok(())
}

#[test]
fn single_slice_round_trip_is_exact() -> Result<()> {
    assert_round_trip([1, 2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
}

#[test]
fn multiple_slices_preserve_page_order() -> Result<()> {
    assert_round_trip([3, 1, 2], vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0])
}

#[test]
fn negative_values_survive_round_trip() -> Result<()> {
    assert_round_trip([1, 2, 2], vec![-100.5, -1.0, 0.0, 42.25])
}

#[test]
fn reader_struct_delegates_to_canonical_operation() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("reader.tiff");
    let input = image([1, 1, 2], vec![7.0, 9.0])?;
    write_tiff(&input, &path, &SequentialBackend)?;
    let output = TiffReader::new(SequentialBackend).read_image(&path)?;
    assert_eq!(output.data_cow_on(&SequentialBackend).as_ref(), &[7.0, 9.0]);
    Ok(())
}

#[test]
fn missing_file_reports_open_failure() {
    let error = read_tiff("missing/volume.tiff", &SequentialBackend).unwrap_err();
    assert!(error.to_string().contains("Cannot open TIFF file"));
}

#[test]
fn invalid_file_is_rejected() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("invalid.tiff");
    std::fs::write(&path, b"not a TIFF")?;
    let error = read_tiff(&path, &SequentialBackend).unwrap_err();
    assert!(error.to_string().contains("decoder"));
    Ok(())
}
