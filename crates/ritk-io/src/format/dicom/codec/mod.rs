//! DICOM pixel data codec regression helpers.
//!
//! # Architecture
//!
//! Production frame decode is owned by `ritk-dicom` through
//! `decode_frame_with::<DicomRsBackend>`. This module keeps the historical
//! compressed-frame regression helper under `cfg(test)` so codec fixtures still
//! validate the backend contract.
//!
//! - **RLE Lossless**: dispatched to `decode_rle_lossless_frame`, a native
//!   implementation of DICOM PS3.5 Annex G (PackBits per byte-plane + correct
//!   LE reassembly). This bypasses `dicom-transfer-syntax-registry v0.8.2`,
//!   which has an off-by-one write-start offset (`start = 1` instead of `0`)
//!   for 8-bit grayscale images, silently corrupting `pixel[0]` and losing
//!   `pixel[N−1]` for any file where `pixel[0] ≠ 0`.
//! - **JPEG-LS (lossless + near-lossless), JPEG 2000, and RLE Lossless**:
//!   dispatched through `ritk-dicom::NativeCodecBackend`.
//! - **Remaining external compressed transfer syntaxes**: calls the configured
//!   `dicom-rs` backend and then applies the linear modality LUT.
//!
//! # Supported codecs (all pure Rust; no C/C++ FFI)
//!
//! | Transfer Syntax                        | UID                      | Codec          |
//! |----------------------------------------|--------------------------|----------------|
//! | JPEG Baseline (Process 1)              | 1.2.840.10008.1.2.4.50   | jpeg-decoder   |
//! | JPEG Extended (Process 2 & 4)          | 1.2.840.10008.1.2.4.51   | jpeg-decoder   |
//! | JPEG Lossless Non-Hierarchical (P14)   | 1.2.840.10008.1.2.4.57   | jpeg-decoder   |
//! | JPEG Lossless First-Order Prediction   | 1.2.840.10008.1.2.4.70   | jpeg-decoder   |
//! | JPEG-LS Lossless                       | 1.2.840.10008.1.2.4.80   | RITK-native    |
//! | JPEG-LS Near-Lossless                  | 1.2.840.10008.1.2.4.81   | RITK-native    |
//! | JPEG 2000 Lossless                     | 1.2.840.10008.1.2.4.90   | RITK-native    |
//! | JPEG 2000 Lossy (lossless-coded)       | 1.2.840.10008.1.2.4.91   | RITK-native    |
//! | RLE Lossless                           | 1.2.840.10008.1.2.5      | RITK-native    |
//! | JPEG XL Lossless                       | 1.2.840.10008.1.2.4.110  | jxl-oxide      |
//! | JPEG XL JPEG Recompression             | 1.2.840.10008.1.2.4.111  | jxl-oxide      |
//! | JPEG XL                                | 1.2.840.10008.1.2.4.112  | jxl-oxide      |
//!
//! # Mathematical contract
//!
//! `decode_compressed_frame(obj, f, bits, repr, slope, intercept)`:
//!   `Output[i] = codec_sample[i] × slope + intercept`
//!
//! where `codec_sample[i]` is the integer produced by the codec for pixel i.
//! Identical semantics to `decode_pixel_bytes` (DICOM PS3.3 C.7.6.3.1.4).
//! - JPEG Extended tolerance: `|decoded[i] − original[i]| ≤ 16` (same Q75 bound as Baseline).
//! - RLE Lossless exact fidelity: `max|decoded[i] − original[i]| = 0` (lossless by spec).
//!
//! # Invariants
//!
//! - `is_codec_supported() ⟹ is_compressed()`: codec path only for compressed TS.
//! - `is_natively_supported() ⟹ !is_codec_supported()`: disjoint decode paths.
//! - Output length equals `rows × cols` for a single-frame decode.
//! - Rescale is always applied; identity rescale (slope=1, intercept=0) is valid.

#[cfg(test)]
use anyhow::{Context, Result};
#[cfg(test)]
use dicom::object::DefaultDicomObject;
#[cfg(test)]
use ritk_dicom::{
    decode_frame_with, DecodeFrameRequest, DicomRsBackend, PixelLayout, PixelSignedness,
    TransferSyntaxKind,
};

/// Decode one frame from a compressed DICOM object using the registered codec.
///
/// # Arguments
///
/// - `obj`: open Part 10 DICOM object with compressed transfer syntax in file meta.
/// - `frame_idx`: zero-based frame index (0 for single-frame objects).
/// - `bits_allocated`: from (0028,0100); drives byte interpretation in `decode_pixel_bytes`.
/// - `pixel_representation`: from (0028,0103); unsigned or signed.
/// - `slope`: RescaleSlope from (0028,1053); absent ⇒ 1.0.
/// - `intercept`: RescaleIntercept from (0028,1052); absent ⇒ 0.0.
///
/// # Returns
///
/// `Vec<f32>` of length `rows × cols` with modality LUT applied.
///
/// # Errors
///
/// Returns `Err` when the codec fails: unsupported TS, malformed compressed data,
/// or missing codec (feature not enabled).
#[cfg(test)]
pub(super) fn decode_compressed_frame(
    obj: &DefaultDicomObject,
    frame_idx: u32,
    bits_allocated: u16,
    pixel_representation: PixelSignedness,
    slope: f32,
    intercept: f32,
) -> Result<Vec<f32>> {
    let rows: usize = obj
        .element(dicom::core::Tag(0x0028, 0x0010))
        .context("DICOM codec: missing Rows (0028,0010)")?
        .to_str()
        .context("DICOM codec: Rows not string-readable")?
        .trim()
        .parse()
        .context("DICOM codec: Rows not a valid integer")?;
    let cols: usize = obj
        .element(dicom::core::Tag(0x0028, 0x0011))
        .context("DICOM codec: missing Columns (0028,0011)")?
        .to_str()
        .context("DICOM codec: Columns not string-readable")?
        .trim()
        .parse()
        .context("DICOM codec: Columns not a valid integer")?;
    let samples_per_pixel: usize = obj
        .element(dicom::core::Tag(0x0028, 0x0002))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(1);
    let request = DecodeFrameRequest {
        frame_index: frame_idx,
        transfer_syntax: TransferSyntaxKind::from_uid(obj.meta().transfer_syntax()),
        layout: PixelLayout {
            rows,
            cols,
            samples_per_pixel,
            bits_allocated,
            pixel_representation,
            rescale_slope: slope,
            rescale_intercept: intercept,
        },
    };
    Ok(decode_frame_with::<DicomRsBackend>(obj, request)
        .with_context(|| format!("codec decode failed for frame {frame_idx}"))?
        .pixels)
}

#[cfg(test)]
mod tests;
