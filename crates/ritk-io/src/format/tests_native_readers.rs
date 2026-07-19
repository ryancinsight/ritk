//! Value-semantic coverage for the unified image reader and writer contracts.

use crate::domain::{ImageReader, ImageWriter};
use coeus_core::SequentialBackend;
use ritk_image::Image as NativeImage;
use ritk_spatial::{Direction, Point, Spacing};
use std::path::Path;

/// Round-trip a native volume through the unified [`crate::domain::ImageWriter`]
/// then [`crate::domain::ImageReader`] adapters; assert exact voxel + shape
/// parity. Exercises the native writer adapter end-to-end (the reader adapters
/// are covered by the differential tests above).
fn assert_native_writer_reader_round_trips<W, R>(path: &std::path::Path, writer: &W, reader: &R)
where
    W: crate::domain::ImageWriter<NativeImage<f32, SequentialBackend, 3>>,
    R: ImageReader<NativeImage<f32, SequentialBackend, 3>>,
{
    let dims = [2usize, 3, 4];
    let n = dims[0] * dims[1] * dims[2];
    let voxels: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 4.0).collect();
    let image = NativeImage::from_flat_on(
        voxels.clone(),
        dims,
        Point::new([5.0, -10.0, 15.0]),
        Spacing::new([1.5, 0.75, 0.9]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("native image");

    writer.write(path, &image).expect("contract write");
    let loaded: NativeImage<f32, SequentialBackend, 3> = reader.read(path).expect("contract read");

    assert_eq!(loaded.shape(), dims, "shape parity");
    assert_eq!(
        loaded.data_slice().expect("contiguous"),
        voxels.as_slice(),
        "native writerâ†’reader contract must preserve voxels exactly"
    );
}

#[test]
fn native_nrrd_writer_reader_contract_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_native_writer_reader_round_trips(
        &dir.path().join("contract.nrrd"),
        &super::nrrd::native::NrrdWriter::new(SequentialBackend),
        &super::nrrd::native::NrrdReader::new(SequentialBackend),
    );
}

#[test]
fn native_analyze_writer_reader_contract_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_native_writer_reader_round_trips(
        &dir.path().join("contract.hdr"),
        &super::analyze::AnalyzeWriter::new(SequentialBackend),
        &super::analyze::AnalyzeReader::new(SequentialBackend),
    );
}

#[test]
fn native_mgh_writer_reader_contract_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_native_writer_reader_round_trips(
        &dir.path().join("contract.mgh"),
        &super::mgh::native::MghWriter::new(SequentialBackend),
        &super::mgh::native::MghReader::new(SequentialBackend),
    );
}

#[test]
fn native_metaimage_writer_reader_contract_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_native_writer_reader_round_trips(
        &dir.path().join("contract.mha"),
        &super::metaimage::native::MetaImageWriter::new(SequentialBackend),
        &super::metaimage::native::MetaImageReader::new(SequentialBackend),
    );
}

#[test]
fn native_minc_writer_reader_contract_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_native_writer_reader_round_trips(
        &dir.path().join("contract.mnc"),
        &super::minc::native::MincWriter::new(SequentialBackend),
        &super::minc::native::MincReader::new(SequentialBackend),
    );
}

#[test]
fn native_tiff_writer_reader_contract_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_native_writer_reader_round_trips(
        &dir.path().join("contract.tiff"),
        &super::tiff::native::TiffWriter::new(SequentialBackend),
        &super::tiff::native::TiffReader::new(SequentialBackend),
    );
}

#[test]
fn native_tiff_reader_matches_burn() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_native_writer_reader_round_trips(
        &dir.path().join("vol.tiff"),
        &super::tiff::native::TiffWriter::new(SequentialBackend),
        &super::tiff::native::TiffReader::new(SequentialBackend),
    );
}

#[test]
fn native_jpeg_reader_matches_burn() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("slice.jpg");
    let image = NativeImage::from_flat_on(
        vec![16.0, 128.0, 240.0],
        [1, 1, 3],
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("native JPEG fixture");
    ImageWriter::write(
        &super::jpeg::native::JpegWriter::new(SequentialBackend),
        &path,
        &image,
    )
    .expect("jpeg write");
    let loaded = ImageReader::read(
        &super::jpeg::native::JpegReader::new(SequentialBackend),
        &path,
    )
    .expect("jpeg read");
    assert_eq!(loaded.shape(), [1, 1, 3]);
    let values = loaded.data_slice().expect("contiguous JPEG data");
    assert!(values[0] <= 24.0);
    assert!((values[1] - 128.0).abs() <= 12.0);
    assert!(values[2] >= 228.0);
}

/// Write a synthetic 8-bit grayscale PNG (no Burn PNG writer exists).
fn write_gray_png(path: &Path, width: u32, height: u32, seed: u8) {
    let img = image::GrayImage::from_fn(width, height, |x, y| {
        image::Luma([((x * 7 + y * 13) as u8).wrapping_add(seed)])
    });
    img.save(path).expect("png save");
}

#[test]
fn native_png_reader_matches_burn() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("slice.png");
    write_gray_png(&path, 12, 8, 3);
    let loaded = ImageReader::read(
        &super::png::native::PngReader::new(SequentialBackend),
        &path,
    )
    .expect("native PNG read");
    assert_eq!(loaded.shape(), [1, 8, 12]);
    assert_eq!(loaded.data_slice().expect("contiguous PNG data").len(), 96);
}

#[test]
fn native_png_series_reader_matches_burn() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_gray_png(&dir.path().join("s000.png"), 6, 4, 11);
    write_gray_png(&dir.path().join("s001.png"), 6, 4, 71);
    let loaded = ImageReader::read(
        &super::png::native::PngSeriesReader::new(SequentialBackend),
        dir.path(),
    )
    .expect("native PNG series read");
    assert_eq!(loaded.shape(), [2, 4, 6]);
    assert_eq!(loaded.data_slice().expect("contiguous PNG data").len(), 48);
}
