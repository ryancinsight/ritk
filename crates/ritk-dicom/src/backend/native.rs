//! RITK-native frame decode backend.
//!
//! This backend selects a decoder for each supported encapsulated syntax.

use anyhow::{bail, Result};
use ritk_codecs::ImageReadBudget;

use crate::backend::{
    DecodeFrameRequest, DecodedFrame, EncapsulatedFrameSource, PixelDecodeBackend,
};
use crate::codec::{
    decode_jpeg2000_fragment, decode_jpeg_fragment, decode_jpeg_ls_fragment,
    decode_rle_lossless_fragment,
};
use crate::syntax::Codec;

#[derive(Debug, Clone, Copy, Default)]
pub struct NativeCodecBackend;

impl<O> PixelDecodeBackend<O> for NativeCodecBackend
where
    O: EncapsulatedFrameSource,
{
    fn decode_frame(object: &O, request: DecodeFrameRequest) -> Result<DecodedFrame> {
        let syntax = &request.transfer_syntax;
        let Some(codec) = syntax.codec() else {
            bail!(
                "transfer syntax {} is not implemented by NativeCodecBackend",
                syntax.uid()
            );
        };
        let fragment = object.encapsulated_frame(request.frame_index)?;
        let pixels = match codec {
            Codec::Jpeg => decode_jpeg_fragment(&fragment, request.layout)?,
            Codec::JpegLs => decode_jpeg_ls_fragment(&fragment, request.layout)?,
            Codec::Jpeg2000 => {
                decode_jpeg2000_fragment(&fragment, request.layout, ImageReadBudget::DEFAULT)?
            }
            Codec::Rle => decode_rle_lossless_fragment(&fragment, request.layout)?,
            Codec::JpegXl => crate::codec::jpeg_xl::decode(&fragment, request.layout)?,
        };
        Ok(DecodedFrame { pixels })
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
    use anyhow::{bail, Result};

    use super::*;
    use crate::pixel::PixelLayout;
    use crate::TransferSyntaxKind;

    #[derive(Debug)]
    struct SingleFragment(Vec<u8>);

    impl EncapsulatedFrameSource for SingleFragment {
        fn encapsulated_frame(&self, frame_index: u32) -> Result<std::borrow::Cow<'_, [u8]>> {
            if frame_index == 0 {
                Ok(std::borrow::Cow::Borrowed(&self.0))
            } else {
                bail!("test frame index {frame_index} out of range")
            }
        }
    }

    #[derive(Debug)]
    struct NoFragmentRead;

    impl EncapsulatedFrameSource for NoFragmentRead {
        fn encapsulated_frame(&self, _frame_index: u32) -> Result<std::borrow::Cow<'_, [u8]>> {
            bail!("encapsulated frame must not be read for unsupported native syntax")
        }
    }

    fn rle_fragment_8bit(pixels: &[u8]) -> Vec<u8> {
        let mut fragment = vec![0u8; 64];
        fragment[0..4].copy_from_slice(&1u32.to_le_bytes());
        fragment[4..8].copy_from_slice(&64u32.to_le_bytes());
        fragment.push((pixels.len() - 1) as u8);
        fragment.extend_from_slice(pixels);
        fragment
    }

    fn layout(rows: usize, cols: usize) -> PixelLayout {
        PixelLayout {
            rows,
            cols,
            samples_per_pixel: 1,
            bits_allocated: 8,
            bits_stored: 8,
            pixel_representation: crate::pixel::PixelSignedness::Unsigned,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
        }
    }

    fn encode_grayscale_jpeg(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
        use image::{DynamicImage, GrayImage};

        let image = GrayImage::from_raw(width, height, pixels.to_vec())
            .expect("test image dimensions must match pixel count");
        let mut jpeg = Vec::with_capacity((width as usize * height as usize) / 8);
        DynamicImage::ImageLuma8(image)
            .write_to(
                &mut std::io::Cursor::new(&mut jpeg),
                image::ImageFormat::Jpeg,
            )
            .expect("test JPEG encode must succeed");
        jpeg
    }

    fn encode_rgb_jpeg(width: u32, height: u32, samples: &[u8]) -> Vec<u8> {
        use image::{DynamicImage, RgbImage};

        let image = RgbImage::from_raw(width, height, samples.to_vec())
            .expect("test RGB image dimensions must match sample count");
        let mut jpeg = Vec::with_capacity((width as usize * height as usize * 3) / 8);
        DynamicImage::ImageRgb8(image)
            .write_to(
                &mut std::io::Cursor::new(&mut jpeg),
                image::ImageFormat::Jpeg,
            )
            .expect("test JPEG encode must succeed");
        jpeg
    }

    #[test]
    fn native_backend_decodes_rle() {
        let source = SingleFragment(rle_fragment_8bit(&[1, 2, 3, 4]));
        let decoded = NativeCodecBackend::decode_frame(
            &source,
            DecodeFrameRequest {
                frame_index: 0,
                transfer_syntax: TransferSyntaxKind::RleLossless,
                layout: layout(2, 2),
            },
        )
        .unwrap();

        assert_eq!(decoded.pixels, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn native_backend_decodes_jpeg_baseline() {
        let source = SingleFragment(encode_grayscale_jpeg(2, 2, &[32, 32, 32, 32]));
        let decoded = NativeCodecBackend::decode_frame(
            &source,
            DecodeFrameRequest {
                frame_index: 0,
                transfer_syntax: TransferSyntaxKind::JpegBaseline,
                layout: PixelLayout {
                    rescale_slope: 2.0,
                    rescale_intercept: -10.0,
                    ..layout(2, 2)
                },
            },
        )
        .unwrap();

        assert_eq!(decoded.pixels.len(), 4);
        for value in decoded.pixels {
            assert!(
                (value - 54.0).abs() <= 2.0,
                "expected JPEG decoded sample near 32 with modality LUT result near 54, got {value}"
            );
        }
    }

    #[test]
    fn native_backend_decodes_rgb_jpeg_baseline() {
        let original = [120u8, 64, 32, 120, 64, 32];
        let source = SingleFragment(encode_rgb_jpeg(2, 1, &original));
        let decoded = NativeCodecBackend::decode_frame(
            &source,
            DecodeFrameRequest {
                frame_index: 0,
                transfer_syntax: TransferSyntaxKind::JpegBaseline,
                layout: PixelLayout {
                    rows: 1,
                    cols: 2,
                    samples_per_pixel: 3,
                    bits_allocated: 8,
                    bits_stored: 8,
                    pixel_representation: crate::pixel::PixelSignedness::Unsigned,
                    rescale_slope: 1.0,
                    rescale_intercept: 0.0,
                },
            },
        )
        .unwrap();

        assert_eq!(decoded.pixels.len(), original.len());
        for (i, (actual, expected)) in decoded.pixels.iter().zip(original).enumerate() {
            assert!(
                (*actual - f32::from(expected)).abs() <= 16.0,
                "native RGB JPEG sample {i}: expected near {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn native_backend_rejects_non_native_codec_syntax() {
        let source = NoFragmentRead;
        let err = NativeCodecBackend::decode_frame(
            &source,
            DecodeFrameRequest {
                frame_index: 0,
                transfer_syntax: TransferSyntaxKind::ExplicitVrLittleEndian,
                layout: layout(2, 2),
            },
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("NativeCodecBackend"),
            "expected NativeCodecBackend rejection, got {err:#}"
        );
    }
}
