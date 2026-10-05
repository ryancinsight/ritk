//! Encapsulated pixel encoders for multi-frame DICOM output.

use anyhow::{bail, Context, Result};
use dicom::core::smallvec::SmallVec;
use dicom::core::value::PixelFragmentSequence;
use ritk_codecs::encode_jpeg_fragment;
use ritk_codecs::encode_rle_lossless_fragment_u16_grayscale;
use ritk_codecs::jpeg::lossless::{encode_grayscale_jpeg_lossless, JpegLosslessPrediction};
use ritk_codecs::jpeg_2000::encoder::{encode_grayscale_j2k, Jpeg2000Encoding};
use ritk_codecs::jpeg_ls::encoder::encode_grayscale_jpeg_ls;
use ritk_codecs::{PixelLayout, PixelSignedness};

use crate::format::dicom::writer::pixel_encoding::{
    DicomImageShape, JPEG_2000_QUANTIZATION_STEP, JPEG_BASELINE_QUALITY, JPEG_LS_NEAR,
};
use crate::format::dicom::writer::DicomWriteError;
use crate::format::dicom::TransferSyntaxKind;

/// Encode eight-bit samples into one baseline JPEG fragment per frame.
pub(super) fn encode_baseline_jpeg_frames(
    pixels: &[u8],
    dimensions: DicomImageShape,
) -> Result<PixelFragmentSequence<Vec<u8>>> {
    validate_pixel_count(pixels.len(), dimensions)?;
    let layout = PixelLayout {
        rows: dimensions.rows,
        cols: dimensions.columns,
        samples_per_pixel: 1,
        bits_allocated: 8,
        bits_stored: 8,
        pixel_representation: PixelSignedness::Unsigned,
        rescale_slope: 1.0,
        rescale_intercept: 0.0,
    };
    let mut fragments = Vec::new();
    fragments
        .try_reserve_exact(dimensions.depth)
        .map_err(|_| DicomWriteError::PixelAllocationFailed)?;
    for frame_index in 0..dimensions.depth {
        let frame_pixels = frame_pixels(pixels, frame_index, dimensions)?;
        let mut frame = Vec::new();
        frame
            .try_reserve_exact(frame_pixels.len())
            .map_err(|_| DicomWriteError::PixelAllocationFailed)?;
        frame.extend(frame_pixels.iter().copied().map(f32::from));
        fragments.push(
            encode_jpeg_fragment(&frame, layout, JPEG_BASELINE_QUALITY)
                .with_context(|| format!("JPEG baseline encode failed for frame {frame_index}"))?,
        );
    }
    Ok(PixelFragmentSequence::new_fragments(SmallVec::from_vec(
        fragments,
    )))
}

/// Encode unsigned sixteen-bit samples using the selected compressed syntax.
pub(super) fn encode_compressed_frames(
    pixels: &[u16],
    dimensions: DicomImageShape,
    syntax: &TransferSyntaxKind,
) -> Result<PixelFragmentSequence<Vec<u8>>> {
    validate_pixel_count(pixels.len(), dimensions)?;
    let rows = u32::from(dimensions.rows_attribute);
    let columns = u32::from(dimensions.columns_attribute);
    let mut fragments = Vec::new();
    fragments
        .try_reserve_exact(dimensions.depth)
        .map_err(|_| DicomWriteError::PixelAllocationFailed)?;
    for frame_index in 0..dimensions.depth {
        let frame = frame_pixels(pixels, frame_index, dimensions)?;
        let encoded = match syntax {
            TransferSyntaxKind::JpegLsLossless => {
                encode_grayscale_jpeg_ls(frame, rows, columns, 16, 0).with_context(|| {
                    format!("JPEG-LS lossless encode failed for frame {frame_index}")
                })?
            }
            TransferSyntaxKind::JpegLsLossy => {
                encode_grayscale_jpeg_ls(frame, rows, columns, 16, JPEG_LS_NEAR).with_context(
                    || format!("JPEG-LS near-lossless encode failed for frame {frame_index}"),
                )?
            }
            TransferSyntaxKind::Jpeg2000Lossless | TransferSyntaxKind::Jpeg2000Lossy => {
                let mut frame_i32 = Vec::new();
                frame_i32
                    .try_reserve_exact(frame.len())
                    .map_err(|_| DicomWriteError::PixelAllocationFailed)?;
                frame_i32.extend(frame.iter().copied().map(i32::from));
                let encoding = match syntax {
                    TransferSyntaxKind::Jpeg2000Lossless => Jpeg2000Encoding::Lossless {
                        decomposition_levels: 1,
                    },
                    TransferSyntaxKind::Jpeg2000Lossy => Jpeg2000Encoding::Lossy {
                        decomposition_levels: 1,
                        quantization_step: JPEG_2000_QUANTIZATION_STEP,
                    },
                    _ => bail!("JPEG 2000 encoder received another transfer syntax"),
                };
                encode_grayscale_j2k(
                    &frame_i32,
                    rows,
                    columns,
                    16,
                    PixelSignedness::Unsigned,
                    encoding,
                )
                .with_context(|| format!("JPEG 2000 encode failed for frame {frame_index}"))?
            }
            TransferSyntaxKind::JpegLosslessFirstOrderPrediction => encode_grayscale_jpeg_lossless(
                frame,
                dimensions.rows,
                dimensions.columns,
                16,
                JpegLosslessPrediction::Left,
            )
            .with_context(|| format!("JPEG lossless encode failed for frame {frame_index}"))?,
            TransferSyntaxKind::JpegLosslessNonHierarchical => encode_grayscale_jpeg_lossless(
                frame,
                dimensions.rows,
                dimensions.columns,
                16,
                JpegLosslessPrediction::AboveOnly,
            )
            .with_context(|| {
                format!("JPEG lossless (non-hierarchical) encode failed for frame {frame_index}")
            })?,
            TransferSyntaxKind::RleLossless => encode_rle_lossless_fragment_u16_grayscale(frame),
            _ => bail!(
                "compressed frame encoder received unsupported transfer syntax '{}'",
                syntax.uid()
            ),
        };
        fragments.push(encoded);
    }
    Ok(PixelFragmentSequence::new_fragments(SmallVec::from_vec(
        fragments,
    )))
}

fn validate_pixel_count(actual: usize, dimensions: DicomImageShape) -> Result<()> {
    if actual == dimensions.total_samples {
        Ok(())
    } else {
        Err(DicomWriteError::PixelCountMismatch {
            expected: dimensions.total_samples,
            actual,
        }
        .into())
    }
}

fn frame_pixels<T>(samples: &[T], frame_index: usize, dimensions: DicomImageShape) -> Result<&[T]> {
    let start = frame_index
        .checked_mul(dimensions.frame_samples)
        .ok_or(DicomWriteError::PixelCountOverflow)?;
    let end = start
        .checked_add(dimensions.frame_samples)
        .ok_or(DicomWriteError::PixelCountOverflow)?;
    samples.get(start..end).ok_or_else(|| {
        DicomWriteError::PixelCountMismatch {
            expected: dimensions.total_samples,
            actual: samples.len(),
        }
        .into()
    })
}
