//! Per-slice pixel decoding for the DICOM series reader.

use std::fmt;

use anyhow::{bail, Context, Result};
use dicom::core::Tag;
use dicom::object::DefaultDicomObject;
use ritk_dicom::{
    decode_frame_with, parse_bytes_with_budget, parse_file_with_budget, DecodeFrameRequest,
    DicomRsBackend, ParseBudget, PixelLayout, TransferSyntaxKind,
};
#[cfg(test)]
use ritk_dicom::{decode_native_pixel_bytes_checked, PixelSignedness};

use super::types::DicomSliceMetadata;

/// Decode raw pixel bytes into f32 values applying per-slice rescale LUT.
///
/// # Invariants
/// - `bits_allocated=8`: each byte is one unsigned sample.
/// - `bits_allocated` selects the byte container width.
/// - `bits_stored` selects the meaningful magnitude and two's-complement sign bit.
///
/// Mathematical derivation: F(x) = x × RescaleSlope + RescaleIntercept
/// per DICOM PS3.3 C.7.6.3.1.4.
#[cfg(test)]
pub(super) fn decode_pixel_bytes(
    bytes: &[u8],
    bits_allocated: u16,
    bits_stored: u16,
    pixel_representation: PixelSignedness,
    slope: f32,
    intercept: f32,
) -> Vec<f32> {
    let bytes_per_sample = usize::from(bits_allocated / 8);
    decode_native_pixel_bytes_checked(
        bytes,
        PixelLayout {
            rows: 1,
            cols: bytes.len() / bytes_per_sample,
            samples_per_pixel: 1,
            bits_allocated,
            bits_stored,
            pixel_representation,
            rescale_slope: slope,
            rescale_intercept: intercept,
        },
    )
    .expect("invariant: test helper receives complete byte-addressable samples")
}

pub(super) fn ensure_scalar_samples_per_pixel(
    samples_per_pixel: usize,
    source: impl fmt::Display,
) -> Result<()> {
    if samples_per_pixel == 1 {
        return Ok(());
    }
    bail!(
        "DICOM scalar volume loader supports only SamplesPerPixel=1; {source} declares \
         SamplesPerPixel={samples_per_pixel}. Decode RGB/color frames through the codec \
         boundary or a color-volume loader"
    )
}

pub(super) fn read_slice_pixels(
    slice: &DicomSliceMetadata,
    budget: &ParseBudget,
) -> Result<Vec<f32>> {
    let obj = parse_file_with_budget::<DicomRsBackend, _>(&slice.path, budget)
        .with_context(|| format!("failed to open DICOM slice {:?}", slice.path))?;
    decode_pixels_from_object(&obj, slice)
}

/// Decode pixels from in-memory Part-10 bytes (zero-disk path for SCP-received instances).
pub(super) fn read_slice_pixels_from_bytes(
    part10_bytes: &[u8],
    slice: &DicomSliceMetadata,
    budget: &ParseBudget,
) -> Result<Vec<f32>> {
    let obj = parse_bytes_with_budget::<DicomRsBackend>(part10_bytes, budget)
        .with_context(|| format!("failed to parse DICOM bytes for {:?}", slice.path))?;
    decode_pixels_from_object(&obj, slice)
}

/// Shared pixel-decode logic operating on an already-parsed DICOM object.
fn decode_pixels_from_object(
    obj: &DefaultDicomObject,
    slice: &DicomSliceMetadata,
) -> Result<Vec<f32>> {
    let ts = slice
        .transfer_syntax_uid
        .as_deref()
        .map(TransferSyntaxKind::from_uid)
        .unwrap_or(TransferSyntaxKind::ImplicitVrLittleEndian);

    let rows = obj
        .element(Tag(0x0028, 0x0010))
        .with_context(|| format!("Rows (0028,0010) absent in {:?}", slice.path))?
        .to_str()
        .with_context(|| format!("Rows (0028,0010) unreadable in {:?}", slice.path))?
        .trim()
        .parse::<usize>()
        .with_context(|| format!("Rows (0028,0010) invalid in {:?}", slice.path))?;

    let cols = obj
        .element(Tag(0x0028, 0x0011))
        .with_context(|| format!("Columns (0028,0011) absent in {:?}", slice.path))?
        .to_str()
        .with_context(|| format!("Columns (0028,0011) unreadable in {:?}", slice.path))?
        .trim()
        .parse::<usize>()
        .with_context(|| format!("Columns (0028,0011) invalid in {:?}", slice.path))?;

    let samples_per_pixel = obj
        .element(Tag(0x0028, 0x0002))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(1);

    ensure_scalar_samples_per_pixel(samples_per_pixel, slice.path.display())?;

    let data = decode_frame_with::<DicomRsBackend>(
        obj,
        DecodeFrameRequest {
            frame_index: 0,
            transfer_syntax: ts,
            layout: PixelLayout {
                rows,
                cols,
                samples_per_pixel,
                bits_allocated: slice.bits_allocated,
                bits_stored: slice.bits_stored,
                pixel_representation: slice.pixel_representation,
                rescale_slope: slice.rescale_slope,
                rescale_intercept: slice.rescale_intercept,
            },
        },
    )
    .with_context(|| format!("DICOM backend decode failed for slice {:?}", slice.path))?
    .pixels;

    if data.is_empty() {
        bail!(
            "DICOM slice contained no decodable pixel data in {:?}",
            slice.path
        );
    }

    Ok(data)
}
