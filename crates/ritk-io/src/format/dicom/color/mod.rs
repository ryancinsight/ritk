//! DICOM color volume loading.
//!
//! This module is the color-volume counterpart to the scalar DICOM loaders.
//! It preserves RGB samples in a typed `ColorVolume<B, 3>` instead of forcing
//! multi-sample frames through scalar `Image<B, 3>`.

use std::path::Path;

use anyhow::{bail, Context, Result};
use dicom::core::Tag;
use dicom::object::DefaultDicomObject;
use ritk_dicom::{
    decode_frame_with, parse_bytes_with, parse_file_with, DecodeFrameRequest, DicomRsBackend,
    PixelLayout, PixelSignedness, TransferSyntaxKind,
};

use super::color_common::{read_optional, read_required, required_string, RGB_CHANNELS};
use super::reader::{self, DicomReadMetadata, DicomSliceMetadata};

/// Check whether a directory contains a DICOM RGB colour series.
///
/// Iterates through the files in `path`, attempting to parse each as DICOM
/// until one succeeds (skipping non-DICOM files like `DICOMDIR`, thumbnails,
/// or hidden files). Inspects `PhotometricInterpretation` (0028,0004) and
/// `SamplesPerPixel` (0028,0002) on the first successfully parsed file.
///
/// Returns `true` when both `PhotometricInterpretation` is `"RGB"` (case-
/// insensitive) and `SamplesPerPixel` equals 3.
///
/// Returns `false` when the first parseable file is not RGB, or `Err` when
/// no DICOM file could be found or parsed.
pub fn is_rgb_dicom_series<P: AsRef<Path>>(path: P) -> Result<bool> {
    let dir = path.as_ref();
    let mut entries = std::fs::read_dir(dir)
        .with_context(|| format!("cannot read DICOM directory '{}'", dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .peekable();

    if entries.peek().is_none() {
        bail!("no files found in '{}'", dir.display());
    }

    // Try each file until one parses as valid DICOM — skips DICOMDIR,
    // thumbnails, and other non-DICOM files.
    for entry in entries {
        let obj = match parse_file_with::<DicomRsBackend, _>(entry.path()) {
            Ok(o) => o,
            Err(_) => continue,
        };

        let samples = read_optional::<usize>(&obj, Tag(0x0028, 0x0002)).unwrap_or(1);
        if samples != RGB_CHANNELS {
            return Ok(false);
        }
        let photometric = required_string(&obj, Tag(0x0028, 0x0004), "PhotometricInterpretation")?;
        return Ok(photometric.trim().eq_ignore_ascii_case("RGB"));
    }

    bail!("no parseable DICOM files found in '{}'", dir.display())
}

/// Read a DICOM RGB series directory into a flat interleaved-RGB buffer.
///
/// Substrate-agnostic counterpart of [`load_color_volume_flat`] that scans a
/// directory first. The returned buffer holds `depth·rows·cols·3` `f32`
/// samples in `[depth, rows, cols, 3]` row-major order (channel fastest),
/// paired with that shape and the resolved series metadata. Only
/// byte-addressable unsigned interleaved RGB data is accepted; scalar,
/// palette, YBR, CMYK, planar, and signed color data are rejected.
pub fn load_color_volume_flat_from_path<P: AsRef<Path>>(
    path: P,
) -> Result<(Vec<f32>, [usize; 4], DicomReadMetadata)> {
    let series = reader::scan::scan_dicom_directory(path)?;
    load_color_volume_flat(series.metadata)
}

/// Decode a pre-scanned DICOM RGB series into a flat interleaved-RGB buffer.
///
/// This is the shared, substrate-free core of the DICOM colour read path: it
/// performs the pixel decode, interleaves the RGB samples into a flat `f32`
/// buffer, and validates per-slice geometry, without constructing any tensor
/// carrier. Callers wrap the buffer in their chosen image container
/// (`ritk_image::native::Image::from_flat`). Pixel decode uses `part10_bytes`
/// from the slice metadata when present, falling back to file-path I/O.
///
/// Returns `(flat, [depth, rows, cols, 3], metadata)` with `metadata.dimensions`
/// normalised to `[rows, cols, depth]`.
pub fn load_color_volume_flat(
    mut metadata: DicomReadMetadata,
) -> Result<(Vec<f32>, [usize; 4], DicomReadMetadata)> {
    let slices = metadata.slices.clone();

    if slices.is_empty() {
        bail!("DICOM color series is empty");
    }

    let rows = metadata.dimensions[0];
    let cols = metadata.dimensions[1];
    let depth = metadata.dimensions[2];

    if rows == 0 || cols == 0 || depth == 0 {
        bail!("DICOM color series has invalid zero dimensions");
    }

    if slices.len() != depth {
        bail!(
            "DICOM color series slice count {} does not match metadata depth {}",
            slices.len(),
            depth
        );
    }

    let frame_samples = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(RGB_CHANNELS))
        .context("DICOM color frame sample count overflow")?;

    let total_samples = frame_samples
        .checked_mul(depth)
        .context("DICOM color volume sample count overflow")?;

    // `rows`/`cols` are header-derived (DICOM Rows/Columns), so a hostile or
    // corrupt file could otherwise force an up-front multi-gigabyte zero-fill
    // before any slice is decoded. Cap the speculative reservation and grow
    // the buffer by appending each validated, sequentially-decoded slice
    // instead of pre-sizing and indexing into it.
    let mut volume = Vec::with_capacity(consus_io::bounded_capacity(
        total_samples,
        std::mem::size_of::<f32>(),
    ));

    for slice in slices.iter() {
        let frame = if let Some(ref bytes) = slice.part10_bytes {
            read_rgb_slice_samples_from_bytes(bytes, slice, rows, cols)
        } else {
            read_rgb_slice_samples(slice, rows, cols)
        }
        .with_context(|| format!("failed to decode DICOM RGB slice {:?}", slice.path))?;

        if frame.len() != frame_samples {
            bail!(
                "DICOM RGB slice {:?} returned {} samples; expected {}",
                slice.path,
                frame.len(),
                frame_samples
            );
        }

        volume.extend_from_slice(&frame);
    }

    metadata.dimensions = [rows, cols, depth];

    Ok((volume, [depth, rows, cols, RGB_CHANNELS], metadata))
}

/// Decode RGB pixel samples from a file-based DICOM slice.
fn read_rgb_slice_samples(
    slice: &DicomSliceMetadata,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<Vec<f32>> {
    let obj = parse_file_with::<DicomRsBackend, _>(&slice.path)
        .with_context(|| format!("failed to open DICOM slice {:?}", slice.path))?;

    let transfer_syntax = obj.meta().transfer_syntax();
    let ts = TransferSyntaxKind::from_uid(transfer_syntax);

    validate_and_decode_rgb_slice(&obj, slice, ts, expected_rows, expected_cols)
}

/// Decode RGB pixel samples from in-memory Part-10 bytes (zero-disk path
/// for SCP-received instances).
fn read_rgb_slice_samples_from_bytes(
    part10_bytes: &[u8],
    slice: &DicomSliceMetadata,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<Vec<f32>> {
    let obj = parse_bytes_with::<DicomRsBackend>(part10_bytes)
        .with_context(|| format!("failed to parse DICOM bytes for {:?}", slice.path))?;

    let transfer_syntax = obj.meta().transfer_syntax();
    let ts = TransferSyntaxKind::from_uid(transfer_syntax);

    validate_and_decode_rgb_slice(&obj, slice, ts, expected_rows, expected_cols)
}

/// Shared RGB validation and pixel-decode logic operating on an already-parsed
/// DICOM object.
///
/// Both [`read_rgb_slice_samples`] and [`read_rgb_slice_samples_from_bytes`]
/// delegate to this function after obtaining a `DefaultDicomObject` from
/// their respective parse paths.
fn validate_and_decode_rgb_slice(
    obj: &DefaultDicomObject,
    slice: &DicomSliceMetadata,
    ts: TransferSyntaxKind,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<Vec<f32>> {
    if ts.is_compressed() && !ts.is_codec_supported() {
        bail!(
            "DICOM RGB series: compressed transfer syntax in slice {:?} is not supported",
            slice.path
        );
    }
    if ts.is_big_endian() {
        bail!(
            "DICOM RGB series: big-endian transfer syntax in slice {:?} is not supported",
            slice.path
        );
    }

    let rows = read_required::<usize>(obj, Tag(0x0028, 0x0010), "Rows")?;
    let cols = read_required::<usize>(obj, Tag(0x0028, 0x0011), "Columns")?;

    if rows != expected_rows || cols != expected_cols {
        bail!(
            "DICOM RGB slice {:?} dimensions {}x{} do not match series {}x{}",
            slice.path,
            rows,
            cols,
            expected_rows,
            expected_cols
        );
    }

    let samples_per_pixel = read_optional::<usize>(obj, Tag(0x0028, 0x0002)).unwrap_or(1);
    if samples_per_pixel != RGB_CHANNELS {
        bail!(
            "DICOM color volume loader supports only RGB SamplesPerPixel=3; {:?} declares SamplesPerPixel={}",
            slice.path,
            samples_per_pixel
        );
    }

    let photometric = required_string(obj, Tag(0x0028, 0x0004), "PhotometricInterpretation")?;
    if !photometric.trim().eq_ignore_ascii_case("RGB") {
        bail!(
            "DICOM color volume loader supports only PhotometricInterpretation=RGB; {:?} declares {}",
            slice.path,
            photometric.trim()
        );
    }

    let planar_configuration = read_optional::<u16>(obj, Tag(0x0028, 0x0006)).unwrap_or(0);
    if planar_configuration != 0 {
        bail!(
            "DICOM RGB color volume loader supports only interleaved PlanarConfiguration=0; {:?} declares {}",
            slice.path,
            planar_configuration
        );
    }

    let bits_allocated =
        read_optional::<u16>(obj, Tag(0x0028, 0x0100)).unwrap_or(slice.bits_allocated);
    if bits_allocated != 8 {
        bail!(
            "DICOM RGB color volume loader supports only BitsAllocated=8; {:?} declares {}",
            slice.path,
            bits_allocated
        );
    }

    let pixel_representation: PixelSignedness = read_optional::<u16>(obj, Tag(0x0028, 0x0103))
        .and_then(|v| PixelSignedness::try_from(v).ok())
        .unwrap_or(slice.pixel_representation);
    if pixel_representation != PixelSignedness::Unsigned {
        bail!(
            "DICOM RGB color volume loader supports only unsigned samples; {:?} declares PixelRepresentation={}",
            slice.path,
            u16::from(pixel_representation)
        );
    }

    let decoded = decode_frame_with::<DicomRsBackend>(
        obj,
        DecodeFrameRequest {
            frame_index: 0,
            transfer_syntax: ts,
            layout: PixelLayout {
                rows,
                cols,
                samples_per_pixel,
                bits_allocated,
                pixel_representation,
                rescale_slope: 1.0,
                rescale_intercept: 0.0,
            },
        },
    )
    .with_context(|| format!("DICOM backend decode failed for RGB slice {:?}", slice.path))?;

    Ok(decoded.pixels)
}

#[cfg(test)]
mod tests;
