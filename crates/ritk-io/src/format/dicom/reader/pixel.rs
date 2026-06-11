//! Per-slice pixel decoding for the DICOM series reader.

use std::fmt;

use anyhow::{bail, Context, Result};
use dicom::core::Tag;
use dicom::object::DefaultDicomObject;
use ritk_dicom::{
    decode_frame_with, parse_bytes_with, parse_file_with, DecodeFrameRequest, DicomRsBackend,
    PixelLayout, TransferSyntaxKind,
};

use super::types::DicomSliceMetadata;

/// Decode raw pixel bytes into f32 values applying per-slice rescale LUT.
///
/// # Invariants
/// - `bits_allocated=8`: each byte is one unsigned sample.
/// - `bits_allocated=16`, `pixel_representation=1`: each LE i16 pair is one sample.
/// - Any other combination: each LE u16 pair is one sample (unsigned default).
///
/// Mathematical derivation: F(x) = x × RescaleSlope + RescaleIntercept
/// per DICOM PS3.3 C.7.6.3.1.4.
#[cfg(test)]
pub(super) fn decode_pixel_bytes(
    bytes: &[u8],
    bits_allocated: u16,
    pixel_representation: u16,
    slope: f32,
    intercept: f32,
) -> Vec<f32> {
    match (bits_allocated, pixel_representation) {
        (8, 1) => bytes
            .iter()
            .map(|&b| (b as i8) as f32 * slope + intercept)
            .collect(),
        (8, _) => bytes
            .iter()
            .map(|&b| b as f32 * slope + intercept)
            .collect(),
        (16, 1) => bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 * slope + intercept)
            .collect(),
        _ => bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as f32 * slope + intercept)
            .collect(),
    }
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

pub(super) fn read_slice_pixels(slice: &DicomSliceMetadata) -> Result<Vec<f32>> {
    println!("read_slice_pixels: before parse_file_with: {:?}", slice.path);
    let obj = parse_file_with::<DicomRsBackend, _>(&slice.path)
        .with_context(|| format!("failed to open DICOM slice {:?}", slice.path))?;
    println!("read_slice_pixels: after parse_file_with: {:?}", slice.path);
    let res = decode_pixels_from_object(&obj, slice);
    println!("read_slice_pixels: after decode_pixels_from_object: {:?}", slice.path);
    res
}

/// Decode pixels from in-memory Part-10 bytes (zero-disk path for SCP-received instances).
pub(super) fn read_slice_pixels_from_bytes(
    part10_bytes: &[u8],
    slice: &DicomSliceMetadata,
) -> Result<Vec<f32>> {
    let obj = parse_bytes_with::<DicomRsBackend>(part10_bytes)
        .with_context(|| format!("failed to parse DICOM bytes for {:?}", slice.path))?;
    decode_pixels_from_object(&obj, slice)
}

/// Shared pixel-decode logic operating on an already-parsed DICOM object.
fn decode_pixels_from_object(
    obj: &DefaultDicomObject,
    slice: &DicomSliceMetadata,
) -> Result<Vec<f32>> {
    println!("decode_pixels_from_object: start: {:?}", slice.path);
    let ts = slice
        .transfer_syntax_uid
        .as_deref()
        .map(TransferSyntaxKind::from_uid)
        .unwrap_or(TransferSyntaxKind::ImplicitVrLittleEndian);
    println!("decode_pixels_from_object: transfer syntax: {:?}", ts);

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
    println!("decode_pixels_from_object: rows = {}, cols = {}", rows, cols);

    let samples_per_pixel = obj
        .element(Tag(0x0028, 0x0002))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(1);

    ensure_scalar_samples_per_pixel(samples_per_pixel, slice.path.display())?;

    println!("decode_pixels_from_object: before decode_frame_with");
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
                pixel_representation: slice.pixel_representation,
                rescale_slope: slice.rescale_slope,
                rescale_intercept: slice.rescale_intercept,
            },
        },
    )
    .with_context(|| format!("DICOM backend decode failed for slice {:?}", slice.path))?
    .pixels;
    println!("decode_pixels_from_object: after decode_frame_with, pixels.len = {}", data.len());

    if data.is_empty() {
        bail!(
            "DICOM slice contained no decodable pixel data in {:?}",
            slice.path
        );
    }

    Ok(data)
}
