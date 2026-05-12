//! Native JPEG frame decoding for encapsulated DICOM fragments.
//!
//! # Contract
//! The JPEG decoder produces integer sample values in image raster order. RITK
//! validates that the decoded raster shape and sample representation match the
//! DICOM metadata, then applies the same linear modality LUT used by native
//! uncompressed pixel data: `output = sample * slope + intercept`.
//!
//! # Backend boundary
//! `backend::JpegDecodeBackend` is a sealed, static-dispatch boundary. The
//! initial `JpegDecoderCrate` implementation constrains the external
//! `jpeg-decoder` dependency to this module; a RITK-owned decoder can replace
//! it by implementing the same sample-raster contract.

mod backend;

use anyhow::{bail, Context, Result};

use self::backend::{JpegDecodeBackend, JpegDecoderCrate, JpegPixelFormat};
use crate::{decode_native_pixel_bytes_checked, PixelLayout};

pub fn decode_jpeg_fragment(fragment: &[u8], layout: PixelLayout) -> Result<Vec<f32>> {
    decode_jpeg_fragment_with::<JpegDecoderCrate>(fragment, layout)
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
        JpegPixelFormat::L8 => decode_native_pixel_bytes_checked(&decoded.pixels, layout),
        JpegPixelFormat::L16 => decode_l16_native_endian(&decoded.pixels, layout),
        JpegPixelFormat::Rgb24 | JpegPixelFormat::Cmyk32 => bail!(
            "native JPEG decoder only accepts grayscale DICOM pixel data; decoded format was {:?}",
            decoded.pixel_format
        ),
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
    if layout.samples_per_pixel != 1 {
        bail!(
            "native JPEG decoder accepts one DICOM sample per pixel; layout declares {}",
            layout.samples_per_pixel
        );
    }

    let expected_bytes = layout
        .pixels_per_frame()?
        .checked_mul(pixel_format.pixel_bytes())
        .context("JPEG decoded byte length overflow")?;
    if !matches!(
        (pixel_format, layout.bits_allocated),
        (JpegPixelFormat::L8, 8) | (JpegPixelFormat::L16, 16)
    ) {
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
    layout.validate_pixel_representation()?;
    layout.validate_rescale_parameters()?;
    if bytes.len() % 2 != 0 {
        bail!("L16 JPEG decoder returned odd byte length {}", bytes.len());
    }
    let pixels = bytes
        .chunks_exact(2)
        .map(|sample| match layout.pixel_representation {
            1 => i16::from_ne_bytes([sample[0], sample[1]]) as f32,
            _ => u16::from_ne_bytes([sample[0], sample[1]]) as f32,
        })
        .map(|sample| sample * layout.rescale_slope + layout.rescale_intercept)
        .collect();
    Ok(pixels)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::jpeg::backend::{JpegDecodeBackend, JpegDecoderCrate};

    fn layout(rows: usize, cols: usize, slope: f32, intercept: f32) -> PixelLayout {
        layout_with_bits(rows, cols, 8, 0, slope, intercept)
    }

    fn layout_with_bits(
        rows: usize,
        cols: usize,
        bits_allocated: u16,
        pixel_representation: u16,
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
        let mut jpeg = Vec::new();
        DynamicImage::ImageLuma8(gray)
            .write_to(&mut Cursor::new(&mut jpeg), image::ImageFormat::Jpeg)
            .expect("test JPEG encode must succeed");
        jpeg
    }

    fn lossless_single_pixel_jpeg_8bit_gray_128() -> Vec<u8> {
        vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xC3, // SOF3: lossless Huffman
            0x00, 0x0B, // segment length
            0x08, // precision
            0x00, 0x01, // height
            0x00, 0x01, // width
            0x01, // components
            0x01, 0x11, 0x00, // component id, sampling, quant table
            0xFF, 0xC4, // DHT
            0x00, 0x14, // segment length
            0x00, // DC table 0
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // one 1-bit code
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // no longer codes
            0x00, // symbol: category 0
            0xFF, 0xDA, // SOS
            0x00, 0x08, // segment length
            0x01, // components
            0x01, 0x00, // component id, DC/AC table selectors
            0x01, // predictor selection Ra
            0x00, // spectral selection end, must be zero for lossless
            0x00, // point transform
            0x7F, // Huffman code 0 padded with ones: diff=0, prediction=128
            0xFF, 0xD9, // EOI
        ]
    }

    fn lossless_single_pixel_jpeg_16bit_gray_0x1234() -> Vec<u8> {
        vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xC3, // SOF3: lossless Huffman
            0x00, 0x0B, // segment length
            0x10, // precision
            0x00, 0x01, // height
            0x00, 0x01, // width
            0x01, // components
            0x01, 0x11, 0x00, // component id, sampling, quant table
            0xFF, 0xC4, // DHT
            0x00, 0x14, // segment length
            0x00, // DC table 0
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // one 1-bit code
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // no longer codes
            0x0F, // symbol: category 15
            0xFF, 0xDA, // SOS
            0x00, 0x08, // segment length
            0x01, // components
            0x01, 0x00, // component id, DC/AC table selectors
            0x01, // predictor selection Ra
            0x00, // spectral selection end, must be zero for lossless
            0x00, // point transform
            0x12, 0x33, // code 0 + bits for diff -28108 from prediction 32768
            0xFF, 0xD9, // EOI
        ]
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
    fn jpeg_lossless_grayscale_fragment_decodes_exact_sample() {
        let jpeg = lossless_single_pixel_jpeg_8bit_gray_128();

        let decoded = decode_jpeg_fragment(&jpeg, layout(1, 1, 1.5, -2.0)).unwrap();

        assert_eq!(decoded, vec![190.0]);
    }

    #[test]
    fn jpeg_backend_l16_output_uses_native_endian_contract() {
        let jpeg = lossless_single_pixel_jpeg_16bit_gray_0x1234();

        let decoded = JpegDecoderCrate::decode(&jpeg).unwrap();

        assert_eq!(decoded.pixel_format, JpegPixelFormat::L16);
        assert_eq!(decoded.pixels, 0x1234u16.to_ne_bytes());
    }

    #[test]
    fn jpeg_lossless_l16_fragment_decodes_exact_unsigned_sample() {
        let jpeg = lossless_single_pixel_jpeg_16bit_gray_0x1234();
        let layout = layout_with_bits(1, 1, 16, 0, 2.0, -4.0);

        let decoded = decode_jpeg_fragment(&jpeg, layout).unwrap();

        assert_eq!(decoded, vec![9316.0]);
    }
}
