//! Native JPEG frame decoding for encapsulated DICOM fragments.
//!
//! # Contract
//! The JPEG decoder produces integer sample values in image raster order. RITK
//! validates that the decoded raster shape and sample representation match the
//! DICOM metadata, then applies the same linear modality LUT used by native
//! uncompressed pixel data: `output = sample * slope + intercept`.

use std::io::Cursor;

use anyhow::{bail, Context, Result};
use jpeg_decoder::{Decoder, PixelFormat};

use crate::pixel::{decode_native_pixel_bytes_checked, PixelLayout};

pub fn decode_jpeg_fragment(fragment: &[u8], layout: PixelLayout) -> Result<Vec<f32>> {
    let mut decoder = Decoder::new(Cursor::new(fragment));
    let decoded = decoder
        .decode()
        .context("native JPEG decoder failed to decode fragment")?;
    let info = decoder
        .info()
        .context("native JPEG decoder did not expose image metadata")?;

    validate_jpeg_layout(
        info.width as usize,
        info.height as usize,
        info.pixel_format,
        decoded.len(),
        layout,
    )?;

    match info.pixel_format {
        PixelFormat::L8 => decode_native_pixel_bytes_checked(&decoded, layout),
        PixelFormat::L16 => decode_l16_native_endian(&decoded, layout),
        PixelFormat::RGB24 | PixelFormat::CMYK32 => bail!(
            "native JPEG decoder only accepts grayscale DICOM pixel data; decoded format was {:?}",
            info.pixel_format
        ),
    }
}

fn validate_jpeg_layout(
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
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
        (PixelFormat::L8, 8) | (PixelFormat::L16, 16)
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
    use super::*;

    fn layout(rows: usize, cols: usize, slope: f32, intercept: f32) -> PixelLayout {
        PixelLayout {
            rows,
            cols,
            samples_per_pixel: 1,
            bits_allocated: 8,
            pixel_representation: 0,
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
}
