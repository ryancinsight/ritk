//! Native JPEG frame decoding for encapsulated DICOM fragments.
//!
//! # Contract
//! The JPEG decoder produces integer sample values in image raster order. RITK
//! validates that the decoded raster shape and sample representation match the
//! DICOM metadata, then applies the same linear modality LUT used by native
//! uncompressed pixel data: `output = sample * slope + intercept`. RGB24 output
//! is preserved as interleaved samples in raster order.
//!
//! # Backend boundary
//! `backend::JpegDecodeBackend` is a sealed, static-dispatch boundary.
//! `RitkJpegDecoder` is the authoritative implementation; all external
//! `jpeg-decoder` crate dependency has been removed.

mod backend;
pub(crate) mod color;
pub(crate) mod huffman;
pub(crate) mod idct;
pub(crate) mod marker;
pub(crate) mod ritk_decoder;
pub(crate) mod scan_dct;
pub(crate) mod scan_lossless;

use anyhow::{bail, Context, Result};

use self::backend::{JpegDecodeBackend, JpegPixelFormat};
use self::ritk_decoder::RitkJpegDecoder;
use crate::{decode_native_pixel_bytes_checked, PixelLayout};

pub fn decode_jpeg_fragment(fragment: &[u8], layout: PixelLayout) -> Result<Vec<f32>> {
    decode_jpeg_fragment_with::<RitkJpegDecoder>(fragment, layout)
}

#[inline]
fn decode_jpeg_fragment_with<B: JpegDecodeBackend>(
    fragment: &[u8],
    layout: PixelLayout,
) -> Result<Vec<f32>> {
    let decoded = B::decode(fragment)?;
    validate_jpeg_layout(
        decoded.width,
        decoded.height,
        decoded.pixel_format,
        decoded.pixels.len(),
        layout,
    )?;

    match decoded.pixel_format {
        JpegPixelFormat::L8 | JpegPixelFormat::Rgb24 => {
            decode_native_pixel_bytes_checked(&decoded.pixels, layout)
        }
        JpegPixelFormat::L16 => decode_l16_native_endian(&decoded.pixels, layout),
    }
}

fn validate_jpeg_layout(
    width: usize,
    height: usize,
    pixel_format: JpegPixelFormat,
    decoded_len: usize,
    layout: PixelLayout,
) -> Result<()> {
    if width != layout.cols || height != layout.rows {
        bail!(
            "JPEG dimensions {}x{} do not match DICOM layout {}x{}",
            width,
            height,
            layout.cols,
            layout.rows
        );
    }
    let expected_samples_per_pixel = pixel_format.samples_per_pixel();
    if layout.samples_per_pixel != expected_samples_per_pixel {
        bail!(
            "JPEG decoded format {:?} requires samples_per_pixel={}; layout declares {}",
            pixel_format,
            expected_samples_per_pixel,
            layout.samples_per_pixel
        );
    }

    let expected_bytes = layout
        .pixels_per_frame()?
        .checked_mul(pixel_format.pixel_bytes())
        .context("JPEG decoded byte length overflow")?;
    if layout.bits_allocated != pixel_format.bits_allocated() {
        bail!(
            "JPEG decoded format {:?} is incompatible with DICOM BitsAllocated={}",
            pixel_format,
            layout.bits_allocated
        );
    }

    let expected_layout_bytes = layout.bytes_per_frame()?;
    if expected_bytes != expected_layout_bytes {
        bail!(
            "JPEG decoded byte length {} does not match DICOM layout byte length {}",
            expected_bytes,
            expected_layout_bytes
        );
    }
    if decoded_len != expected_bytes {
        bail!(
            "JPEG decoder returned {} bytes; expected {} bytes for decoded format {:?}",
            decoded_len,
            expected_bytes,
            pixel_format
        );
    }
    Ok(())
}

fn decode_l16_native_endian(bytes: &[u8], layout: PixelLayout) -> Result<Vec<f32>> {
    layout.validate_rescale_parameters()?;
    if !bytes.len().is_multiple_of(2) {
        bail!("L16 JPEG decoder returned odd byte length {}", bytes.len());
    }
    let pixels = bytes
        .chunks_exact(2)
        .map(|sample| match layout.pixel_representation {
            crate::PixelSignedness::Signed => i16::from_ne_bytes([sample[0], sample[1]]) as f32,
            crate::PixelSignedness::Unsigned => u16::from_ne_bytes([sample[0], sample[1]]) as f32,
        })
        .map(|sample| sample * layout.rescale_slope + layout.rescale_intercept)
        .collect();
    Ok(pixels)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::jpeg::backend::JpegDecodeBackend;
    use crate::jpeg::ritk_decoder::RitkJpegDecoder;
    use crate::PixelSignedness;

    fn layout(rows: usize, cols: usize, slope: f32, intercept: f32) -> PixelLayout {
        layout_with_bits(rows, cols, 8, PixelSignedness::Unsigned, slope, intercept)
    }

    fn layout_with_bits(
        rows: usize,
        cols: usize,
        bits_allocated: u16,
        pixel_representation: PixelSignedness,
        slope: f32,
        intercept: f32,
    ) -> PixelLayout {
        PixelLayout {
            rows,
            cols,
            samples_per_pixel: 1,
            bits_allocated,
            pixel_representation,
            rescale_slope: slope,
            rescale_intercept: intercept,
        }
    }

    fn encode_grayscale_jpeg(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
        use image::{DynamicImage, GrayImage};

        let gray = GrayImage::from_raw(width, height, pixels.to_vec())
            .expect("test image dimensions must match sample count");
        let mut jpeg = Vec::with_capacity((width as usize * height as usize) / 8);
        DynamicImage::ImageLuma8(gray)
            .write_to(&mut Cursor::new(&mut jpeg), image::ImageFormat::Jpeg)
            .expect("test JPEG encode must succeed");
        jpeg
    }

    fn encode_rgb_jpeg(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
        use image::{DynamicImage, RgbImage};

        let rgb = RgbImage::from_raw(width, height, pixels.to_vec())
            .expect("test RGB image dimensions must match sample count");
        let mut jpeg = Vec::with_capacity((width as usize * height as usize * 3) / 8);
        DynamicImage::ImageRgb8(rgb)
            .write_to(&mut Cursor::new(&mut jpeg), image::ImageFormat::Jpeg)
            .expect("test JPEG encode must succeed");
        jpeg
    }

    fn lossless_single_pixel_jpeg_8bit_gray_128() -> Vec<u8> {
        crate::jpeg::scan_lossless::tests::lossless_8bit_fixture()
    }

    fn lossless_single_pixel_jpeg_16bit_gray_0x1234() -> Vec<u8> {
        crate::jpeg::scan_lossless::tests::lossless_16bit_fixture()
    }

    #[test]
    fn jpeg_baseline_grayscale_fragment_decodes_with_modality_lut() {
        let source = [32u8, 32, 32, 32];
        let jpeg = encode_grayscale_jpeg(2, 2, &source);

        let decoded = decode_jpeg_fragment(&jpeg, layout(2, 2, 2.0, -10.0)).unwrap();

        assert_eq!(decoded.len(), 4);
        for value in decoded {
            assert!(
                (value - 54.0).abs() <= 2.0,
                "expected JPEG decoded sample near 32 with rescale result near 54, got {value}"
            );
        }
    }

    #[test]
    fn jpeg_dimension_mismatch_is_rejected() {
        let jpeg = encode_grayscale_jpeg(2, 2, &[0, 64, 128, 255]);

        let err = decode_jpeg_fragment(&jpeg, layout(1, 4, 1.0, 0.0)).unwrap_err();

        assert!(
            err.to_string().contains("dimensions"),
            "expected dimension validation error, got {err:#}"
        );
    }

    #[test]
    fn jpeg_rgb24_fragment_decodes_interleaved_samples() {
        let source = [120u8, 64, 32, 120, 64, 32];
        let jpeg = encode_rgb_jpeg(2, 1, &source);
        let layout = PixelLayout {
            rows: 1,
            cols: 2,
            samples_per_pixel: 3,
            bits_allocated: 8,
            pixel_representation: PixelSignedness::Unsigned,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
        };

        let decoded = decode_jpeg_fragment(&jpeg, layout).unwrap();

        assert_eq!(decoded.len(), source.len());
        for (i, (actual, expected)) in decoded.iter().zip(source).enumerate() {
            assert!(
                (*actual - f32::from(expected)).abs() <= 16.0,
                "RGB JPEG sample {i}: expected near {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn jpeg_rgb24_rejects_grayscale_layout() {
        let jpeg = encode_rgb_jpeg(1, 1, &[120, 64, 32]);

        let err = decode_jpeg_fragment(&jpeg, layout(1, 1, 1.0, 0.0)).unwrap_err();

        assert!(
            err.to_string().contains("samples_per_pixel"),
            "expected samples-per-pixel validation error, got {err:#}"
        );
    }

    #[test]
    fn jpeg_lossless_grayscale_fragment_decodes_exact_sample() {
        let jpeg = lossless_single_pixel_jpeg_8bit_gray_128();

        let decoded = decode_jpeg_fragment(&jpeg, layout(1, 1, 1.5, -2.0)).unwrap();

        assert_eq!(decoded, vec![190.0]);
    }

    #[test]
    fn jpeg_lossless_signed_l8_fragment_decodes_exact_sample() {
        let jpeg = lossless_single_pixel_jpeg_8bit_gray_128();
        let layout = layout_with_bits(1, 1, 8, PixelSignedness::Signed, 2.0, 5.0);

        let decoded = decode_jpeg_fragment(&jpeg, layout).unwrap();

        assert_eq!(decoded, vec![-251.0]);
    }

    #[test]
    fn jpeg_backend_l16_output_uses_native_endian_contract() {
        let jpeg = lossless_single_pixel_jpeg_16bit_gray_0x1234();

        let decoded = RitkJpegDecoder::decode(&jpeg).unwrap();

        assert_eq!(decoded.pixel_format, JpegPixelFormat::L16);
        assert_eq!(decoded.pixels, 0x1234u16.to_ne_bytes());
    }

    #[test]
    fn jpeg_lossless_l16_fragment_decodes_exact_unsigned_sample() {
        let jpeg = lossless_single_pixel_jpeg_16bit_gray_0x1234();
        let layout = layout_with_bits(1, 1, 16, PixelSignedness::Unsigned, 2.0, -4.0);

        let decoded = decode_jpeg_fragment(&jpeg, layout).unwrap();

        assert_eq!(decoded, vec![9316.0]);
    }
}
