//! JPEG tests migrated to the Atlas-native (Coeus) path — ADR 0002.
#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]

use anyhow::Result;
use coeus_core::SequentialBackend;
use ritk_core::rejection::assert_rejects;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

use crate::{read_jpeg, write_jpeg, JpegReader, JpegWriter};

type TestBackend = SequentialBackend;

pub(crate) fn dct_twelve_midpoint(component_ids: &[u8]) -> Vec<u8> {
    let mut bytes = vec![0xff, 0xd8, 0xff, 0xdb, 0x00, 0x83, 0x10];
    for _ in 0..64 {
        bytes.extend_from_slice(&1_u16.to_be_bytes());
    }
    let frame_length =
        u16::try_from(8 + 3 * component_ids.len()).expect("invariant: test frame length fits u16");
    bytes.extend_from_slice(&[0xff, 0xc1]);
    bytes.extend_from_slice(&frame_length.to_be_bytes());
    bytes.extend_from_slice(&[12, 0, 8, 0, 8]);
    bytes.push(u8::try_from(component_ids.len()).expect("invariant: test component count fits u8"));
    for &id in component_ids {
        bytes.extend_from_slice(&[id, 0x11, 0]);
    }
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 0x26, 0x00, 0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x10,
        0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    let scan_length =
        u16::try_from(6 + 2 * component_ids.len()).expect("invariant: test scan length fits u16");
    bytes.extend_from_slice(&[0xff, 0xda]);
    bytes.extend_from_slice(&scan_length.to_be_bytes());
    bytes.push(u8::try_from(component_ids.len()).expect("invariant: test component count fits u8"));
    for &id in component_ids {
        bytes.extend_from_slice(&[id, 0]);
    }
    bytes.extend_from_slice(&[0, 63, 0]);
    bytes.push(u8::MAX >> (component_ids.len() * 2));
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

fn image_from_values(shape: [usize; 3], values: Vec<f32>) -> Image<f32, TestBackend, 3> {
    Image::from_flat_on(
        values,
        shape,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("valid image dimensions")
}

fn reference_luma_values(path: &std::path::Path) -> Vec<f32> {
    image::open(path)
        .expect("independent image decoder must read the JPEG fixture")
        .to_luma8()
        .into_raw()
        .into_iter()
        .map(f32::from)
        .collect()
}

#[test]
fn reader_matches_independent_decoder_for_gradient() {
    let backend = SequentialBackend;
    let (nz, ny, nx) = (1usize, 32usize, 32usize);
    let total = nz * ny * nx;

    let mut data_vec: Vec<f32> = Vec::with_capacity(total);
    let max_idx = (ny * nx - 1) as f32;
    for y in 0..ny {
        for x in 0..nx {
            let idx = (y * nx + x) as f32;
            let val = if max_idx > 0.0 {
                idx / max_idx * 255.0
            } else {
                0.0
            };
            data_vec.push(val);
        }
    }

    let image = image_from_values([nz, ny, nx], data_vec.clone());
    let dir = tempdir().expect("failed to create tempdir");
    let path = dir.path().join("gradient.jpg");

    crate::write_jpeg(&path, &image, &backend).expect("write_jpeg failed");
    let loaded = crate::read_jpeg(&path, &backend).expect("read_jpeg failed");

    assert_eq!(loaded.shape(), [nz, ny, nx]);
    assert_eq!(
        loaded.data_slice().expect("contiguous host data"),
        reference_luma_values(&path)
    );
}

#[test]
fn spatial_metadata_defaults() {
    let backend = SequentialBackend;
    let image = image_from_values([1usize, 4, 4], vec![128.0f32; 16]);
    let dir = tempdir().expect("failed to create tempdir");
    let path = dir.path().join("meta.jpg");

    crate::write_jpeg(&path, &image, &backend).expect("write failed");
    let loaded = crate::read_jpeg(&path, &backend).expect("read failed");

    assert_eq!(
        [loaded.origin()[0], loaded.origin()[1], loaded.origin()[2]],
        [0.0, 0.0, 0.0]
    );
    assert_eq!(
        [
            loaded.spacing()[0],
            loaded.spacing()[1],
            loaded.spacing()[2]
        ],
        [1.0, 1.0, 1.0]
    );
    assert_eq!(loaded.direction(), &Direction::<3>::identity());
}

#[test]
fn reader_matches_independent_decoder_for_non_square_image() {
    let backend = SequentialBackend;
    let (nz, ny, nx) = (1usize, 16usize, 48usize);
    let total = nz * ny * nx;

    let mut data_vec: Vec<f32> = Vec::with_capacity(total);
    for _y in 0..ny {
        for x in 0..nx {
            let val = (x as f32) / (nx as f32 - 1.0) * 255.0;
            data_vec.push(val);
        }
    }

    let image = image_from_values([nz, ny, nx], data_vec.clone());
    let dir = tempdir().expect("failed to create tempdir");
    let path = dir.path().join("rect.jpeg");

    crate::write_jpeg(&path, &image, &backend).expect("write failed");
    let loaded = crate::read_jpeg(&path, &backend).expect("read failed");

    assert_eq!(loaded.shape(), [nz, ny, nx]);
    assert_eq!(
        loaded.data_slice().expect("contiguous host data"),
        reference_luma_values(&path)
    );
}

#[test]
fn write_rejects_nz_not_one() {
    let backend = SequentialBackend;
    let image = image_from_values([2usize, 4, 4], vec![0.0f32; 2 * 4 * 4]);
    let dir = tempdir().expect("failed to create tempdir");
    let path = dir.path().join("bad.jpg");

    let result = crate::write_jpeg(&path, &image, &backend);
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("nz=2") || msg.contains("depth=2"),
        "error message should mention rejected depth, got: {}",
        msg
    );
}

#[test]
fn read_nonexistent_file_errors() {
    let backend = SequentialBackend;
    let result = crate::read_jpeg("/nonexistent/path/to/image.jpg", &backend);
    assert_rejects(result, "failed to open JPEG file");
}

#[test]
fn reader_matches_independent_decoder_for_single_pixel() {
    let backend = SequentialBackend;
    let original_val = 137.0f32;
    let image = image_from_values([1usize, 1, 1], vec![original_val]);
    let dir = tempdir().expect("failed to create tempdir");
    let path = dir.path().join("pixel.jpg");

    crate::write_jpeg(&path, &image, &backend).expect("write failed");
    let loaded = crate::read_jpeg(&path, &backend).expect("read failed");

    assert_eq!(loaded.shape(), [1, 1, 1]);
    assert_eq!(
        loaded.data_slice().expect("contiguous host data"),
        reference_luma_values(&path)
    );
}

#[test]
fn grayscale_reader_scales_twelve_bit_samples_to_display_range() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("wide-gray.jpg");
    std::fs::write(&path, dct_twelve_midpoint(&[1]))?;

    let loaded = read_jpeg(&path, &SequentialBackend)?;

    assert_eq!(loaded.shape(), [1, 8, 8]);
    assert_eq!(
        loaded.data_slice().expect("contiguous host data"),
        &[128.0; 64]
    );
    Ok(())
}

#[test]
fn grayscale_reader_scales_wide_rgb_before_luminance() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("wide-rgb.jpg");
    std::fs::write(&path, dct_twelve_midpoint(b"RGB"))?;

    let loaded = read_jpeg(&path, &SequentialBackend)?;

    assert_eq!(loaded.shape(), [1, 8, 8]);
    assert_eq!(
        loaded.data_slice().expect("contiguous host data"),
        &[128.0; 64]
    );
    Ok(())
}

#[test]
fn writer_preserves_rounded_clamped_constant_blocks() -> Result<()> {
    const HEIGHT: usize = 8;
    const BLOCK_WIDTH: usize = 8;
    const WIDTH: usize = 3 * BLOCK_WIDTH;

    let mut values = Vec::with_capacity(HEIGHT * WIDTH);
    for _ in 0..HEIGHT {
        values.extend(std::iter::repeat_n(-20.0, BLOCK_WIDTH));
        values.extend(std::iter::repeat_n(127.6, BLOCK_WIDTH));
        values.extend(std::iter::repeat_n(300.0, BLOCK_WIDTH));
    }
    let image = image_from_values([1, HEIGHT, WIDTH], values);
    let directory = tempdir()?;
    let path = directory.path().join("constant-blocks.jpg");

    write_jpeg(&path, &image, &SequentialBackend)?;

    // At quality 75 the luminance DC quantizer is 8. An 8x8 constant block
    // has DC coefficient 8 * (sample - 128), so quantization is exact and all
    // AC coefficients are zero. The decoded samples therefore equal the
    // writer's documented round-and-clamp result.
    let mut expected = Vec::with_capacity(HEIGHT * WIDTH);
    for _ in 0..HEIGHT {
        expected.extend(std::iter::repeat_n(0.0, BLOCK_WIDTH));
        expected.extend(std::iter::repeat_n(128.0, BLOCK_WIDTH));
        expected.extend(std::iter::repeat_n(255.0, BLOCK_WIDTH));
    }
    assert_eq!(reference_luma_values(&path), expected);
    Ok(())
}

#[test]
fn reader_and_writer_delegate_to_canonical_operations() -> Result<()> {
    let dir = tempdir()?;
    let wrapped_path = dir.path().join("wrapped.jpg");
    let direct_path = dir.path().join("direct.jpg");
    let input = image_from_values([1, 1, 3], vec![0.0, 128.0, 255.0]);
    let writer = JpegWriter::new(SequentialBackend);
    writer.write_image(&wrapped_path, &input)?;
    write_jpeg(&direct_path, &input, &SequentialBackend)?;
    assert_eq!(std::fs::read(&wrapped_path)?, std::fs::read(&direct_path)?);

    let reader = JpegReader::new(SequentialBackend);
    let wrapped = reader.read_image(&wrapped_path)?;
    let direct = read_jpeg(&wrapped_path, &SequentialBackend)?;
    assert_eq!(wrapped.shape(), direct.shape());
    assert_eq!(
        wrapped.data_cow_on(&SequentialBackend).as_ref(),
        direct.data_cow_on(&SequentialBackend).as_ref()
    );
    Ok(())
}

#[test]
fn writer_rejects_non_planar_and_mismatched_images() -> Result<()> {
    let image = image_from_values([2, 1, 1], vec![0.0, 1.0]);
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

#[test]
fn grayscale_reader_converts_rgb_to_cie_luminance() -> Result<()> {
    use std::fs::File;
    use std::io::BufWriter;

    use image::codecs::jpeg::JpegEncoder;
    use image::RgbImage;

    let directory = tempdir()?;
    let path = directory.path().join("rgb-as-gray.jpg");
    let rgb = RgbImage::from_raw(1, 1, vec![120, 64, 32])
        .expect("invariant: one RGB pixel has three samples");
    JpegEncoder::new_with_quality(BufWriter::new(File::create(&path)?), 100).encode_image(&rgb)?;
    let decoded_rgb = crate::decode::decode_file(&path)?;
    let channels = decoded_rgb.pixels();
    let expected = f32::from(
        u8::try_from(
            (2126 * u32::from(channels[0])
                + 7152 * u32::from(channels[1])
                + 722 * u32::from(channels[2]))
                / 10_000,
        )
        .expect("invariant: weighted average of u8 channels fits in u8"),
    );

    let image = read_jpeg(&path, &SequentialBackend)?;

    assert_eq!(image.shape(), [1, 1, 1]);
    assert_eq!(image.data_cow_on(&SequentialBackend).as_ref(), &[expected]);
    Ok(())
}

#[test]
fn reader_rejects_truncated_jpeg() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("truncated.jpg");
    let mut encoded = consus_raster::jpeg::encode_gray(&[128], 1, 1, 90)?;
    encoded.truncate(encoded.len() - 2);
    std::fs::write(&path, encoded)?;

    let error = read_jpeg(&path, &SequentialBackend).unwrap_err();

    let source = error
        .downcast_ref::<consus_raster::DecodeError>()
        .expect("decode error must preserve the provider cause");
    assert_eq!(source.kind(), consus_raster::DecodeErrorKind::Malformed);
    Ok(())
}

#[test]
fn reader_preserves_encoded_grid_when_exif_requests_rotation() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("oriented.jpg");
    let encoded = consus_raster::jpeg::encode_gray(&[32, 224], 2, 1, 100)?;
    let tiff = [
        b'M', b'M', 0x00, 0x2A, 0x00, 0x00, 0x00, 0x08, 0x00, 0x01, 0x01, 0x12, 0x00, 0x03, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    let payload_len = 6_usize + tiff.len();
    let segment_len =
        u16::try_from(payload_len + 2).expect("invariant: compact EXIF fixture length fits in u16");
    let mut oriented = Vec::with_capacity(encoded.len() + payload_len + 4);
    oriented.extend_from_slice(&encoded[..2]);
    oriented.extend_from_slice(&[0xFF, 0xE1]);
    oriented.extend_from_slice(&segment_len.to_be_bytes());
    oriented.extend_from_slice(b"Exif\0\0");
    oriented.extend_from_slice(&tiff);
    oriented.extend_from_slice(&encoded[2..]);
    std::fs::write(&path, oriented)?;

    let image = read_jpeg(&path, &SequentialBackend)?;

    assert_eq!(image.shape(), [1, 1, 2]);
    assert_eq!(
        image.data_cow_on(&SequentialBackend).as_ref(),
        reference_luma_values(&path)
    );
    Ok(())
}

#[test]
fn wide_grayscale_reader_preserves_luma8_scaling_contract() -> Result<()> {
    let encoded = [
        0xFF, 0xD8, 0xFF, 0xC3, 0x00, 0x0B, 0x10, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00,
        0xFF, 0xC4, 0x00, 0x14, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0F, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x01,
        0x00, 0x00, 0x12, 0x33, 0xFF, 0xD9,
    ];
    let directory = tempdir()?;
    let path = directory.path().join("wide-lossless.jpg");
    std::fs::write(&path, encoded)?;

    let image = read_jpeg(&path, &SequentialBackend)?;

    assert_eq!(image.shape(), [1, 1, 1]);
    assert_eq!(image.data_cow_on(&SequentialBackend).as_ref(), &[18.0]);
    Ok(())
}
