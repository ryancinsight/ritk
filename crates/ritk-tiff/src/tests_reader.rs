#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
use super::{read_tiff, TiffReader};
use crate::write_tiff;
use anyhow::Result;
use coeus_core::SequentialBackend;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use tempfile::tempdir;
use tiff::decoder::DecodingResult;
use tiff::encoder::{colortype, TiffEncoder};

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

fn write_mixed_gray_rgb_pages(path: &Path) -> Result<()> {
    let writer = BufWriter::new(File::create(path)?);
    let mut encoder = TiffEncoder::new(writer)?;
    encoder.write_image::<colortype::Gray8>(2, 1, &[7, 9])?;
    encoder.write_image::<colortype::RGB8>(2, 1, &[1, 2, 3, 4, 5, 6])?;
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

#[test]
fn grayscale_reader_rejects_later_rgb_page() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("mixed.tiff");
    write_mixed_gray_rgb_pages(&path)?;

    let error = read_tiff(&path, &SequentialBackend).unwrap_err();
    assert!(
        error.to_string().contains("page 1 decoded as RGB(8)"),
        "unexpected mixed-page error: {error:#}"
    );
    Ok(())
}

#[test]
fn hostile_rgb_geometry_is_rejected_before_sample_allocation() {
    let error = super::checked_page_sample_count::<3>(u32::MAX, u32::MAX).unwrap_err();
    assert!(
        error.to_string().contains("overflows usize"),
        "unexpected geometry error: {error:#}"
    );
}

#[test]
fn integer_page_appends_directly_in_value_order() -> Result<()> {
    let mut output = vec![5.0];
    super::append_page_to_scalar(
        &mut output,
        DecodingResult::U16(vec![0, 32_767, u16::MAX]),
        3,
        4,
    )?;
    assert_eq!(output, vec![5.0, 0.0, 32_767.0, 65_535.0]);
    Ok(())
}

#[test]
fn first_float_page_becomes_the_output_allocation() -> Result<()> {
    let page = vec![-1.0, 0.0, 42.5];
    let page_pointer = page.as_ptr();
    let mut output = Vec::new();

    super::append_page_to_scalar(&mut output, DecodingResult::F32(page), 3, 0)?;

    assert_eq!(output, vec![-1.0, 0.0, 42.5]);
    assert_eq!(output.as_ptr(), page_pointer);
    Ok(())
}
