//! RITK-native frame decode backend.
//!
//! This backend owns dispatch for transfer syntaxes implemented inside
//! `ritk-dicom`. External DICOM object libraries stay outside this module.

use anyhow::{bail, Result};

use crate::backend::{
    DecodeFrameRequest, DecodedFrame, EncapsulatedFrameSource, PixelDecodeBackend,
};
use crate::codec::{
    decode_jpeg2000_fragment, decode_jpeg_fragment, decode_jpeg_ls_fragment,
    decode_rle_lossless_fragment,
};
use crate::syntax::TransferSyntaxKind;

#[derive(Debug, Clone, Copy, Default)]
pub struct NativeCodecBackend;

impl<O> PixelDecodeBackend<O> for NativeCodecBackend
where
    O: EncapsulatedFrameSource,
{
    fn decode_frame(object: &O, request: DecodeFrameRequest) -> Result<DecodedFrame> {
        let pixels = match &request.transfer_syntax {
            syntax if syntax.is_native_jpeg_codec() => {
                let fragment = object.encapsulated_frame(request.frame_index)?;
                decode_jpeg_fragment(&fragment, request.layout)?
            }
            TransferSyntaxKind::JpegLsLossless => {
                let fragment = object.encapsulated_frame(request.frame_index)?;
                decode_jpeg_ls_fragment(&fragment, request.layout)?
            }
            TransferSyntaxKind::Jpeg2000Lossless | TransferSyntaxKind::Jpeg2000Lossy => {
                let fragment = object.encapsulated_frame(request.frame_index)?;
                decode_jpeg2000_fragment(&fragment, request.layout)?
            }
            TransferSyntaxKind::RleLossless => {
                let fragment = object.encapsulated_frame(request.frame_index)?;
                decode_rle_lossless_fragment(&fragment, request.layout)?
            }
            syntax => bail!(
                "transfer syntax {} is not implemented by NativeCodecBackend",
                syntax.uid()
            ),
        };
        Ok(DecodedFrame { pixels })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{bail, Result};

    use super::*;
    use crate::pixel::PixelLayout;

    #[derive(Debug)]
    struct SingleFragment(Vec<u8>);

    impl EncapsulatedFrameSource for SingleFragment {
        fn encapsulated_frame(&self, frame_index: u32) -> Result<Vec<u8>> {
            if frame_index == 0 {
                Ok(self.0.clone())
            } else {
                bail!("test frame index {frame_index} out of range")
            }
        }
    }

    #[derive(Debug)]
    struct NoFragmentRead;

    impl EncapsulatedFrameSource for NoFragmentRead {
        fn encapsulated_frame(&self, _frame_index: u32) -> Result<Vec<u8>> {
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
            pixel_representation: 0,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
        }
    }

    fn encode_grayscale_jpeg(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
        use image::{DynamicImage, GrayImage};

        let image = GrayImage::from_raw(width, height, pixels.to_vec())
            .expect("test image dimensions must match pixel count");
        let mut jpeg = Vec::new();
        DynamicImage::ImageLuma8(image)
            .write_to(
                &mut std::io::Cursor::new(&mut jpeg),
                image::ImageFormat::Jpeg,
            )
            .expect("test JPEG encode must succeed");
        jpeg
    }

    #[test]
    fn native_backend_decodes_rle_without_dicom_rs_object() {
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
    fn native_backend_decodes_jpeg_baseline_without_dicom_rs_object() {
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
    fn native_backend_rejects_non_native_codec_syntax() {
        let source = NoFragmentRead;
        let err = NativeCodecBackend::decode_frame(
            &source,
            DecodeFrameRequest {
                frame_index: 0,
                transfer_syntax: TransferSyntaxKind::JpegLsLossy,
                layout: layout(2, 2),
            },
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("NativeCodecBackend"),
            "expected NativeCodecBackend rejection, got {err:#}"
        );
    }

    #[test]
    fn native_backend_accepts_jpeg_ls_lossless() {
        // JPEG-LS Lossless (1.2.840.10008.1.2.4.80) should now be handled by NativeCodecBackend
        let source = NoFragmentRead; // We don't have valid JPEG-LS data, but the syntax should be recognized
        let result = NativeCodecBackend::decode_frame(
            &source,
            DecodeFrameRequest {
                frame_index: 0,
                transfer_syntax: TransferSyntaxKind::JpegLsLossless,
                layout: layout(2, 2),
            },
        );

        // Should NOT get "not implemented by NativeCodecBackend" error
        // It will fail because NoFragmentRead doesn't provide data, but with a different error
        if let Err(e) = &result {
            let err_str = e.to_string();
            assert!(
                !err_str.contains("not implemented by NativeCodecBackend"),
                "JPEG-LS Lossless should be handled by NativeCodecBackend, got: {err_str}"
            );
        }
    }
}
