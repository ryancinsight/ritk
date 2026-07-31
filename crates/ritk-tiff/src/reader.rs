//! TIFF / BigTIFF reader for 3-D volumetric images.
//!
//! Each IFD (Image File Directory) page represents one Z-slice.  Pages are
//! stacked in IFD order to form the Z dimension of the returned
//! `Image<f32, B, 3>` tensor with shape `[nz, ny, nx]`.
//!
//! # Axis convention
//! No axis permutation is applied.  TIFF page data is stored in row-major
//! order (Y outer, X inner), mapping directly to `[ny, nx]` per Z-slice.
//!
//! # Spatial metadata
//! TIFF has no standardized physical-space metadata fields.  The returned
//! image uses default values: origin `[0,0,0]`, spacing `[1,1,1]`,
//! direction identity.
//!
//! # Supported pixel types
//! u8, u16, u32, u64, i8, i16, i32, i64, f32, f64 — all converted to f32.
//! Only single-channel (grayscale) pages are supported.
//!
//! # BigTIFF
//! Both classic TIFF and BigTIFF are handled transparently by the `tiff`
//! crate decoder.

use anyhow::{anyhow, Context, Result};
use coeus_core::ComputeBackend;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::io::{BufReader, Read, Seek};
use std::path::Path;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::ColorType;

/// Read a multi-page TIFF / BigTIFF file into a 3-D `Image`.
///
/// # Algorithm
/// 1. Open the file and create a `tiff::decoder::Decoder`.
/// 2. Read the first page to obtain `(width, height)`.
/// 3. Iterate through all IFD pages; each page becomes one Z-slice.
/// 4. Convert pixel data to `f32` (see `decode_page_to_scalar`).
/// 5. Validate that every page has the same `(width, height)`.
/// 6. Append page samples into one flat tensor buffer with shape `[nz, ny, nx]`.
/// 7. Return `Image` with default spatial metadata.
///
/// # Errors
/// - File cannot be opened or is not a valid TIFF.
/// - Pages have inconsistent dimensions.
/// - Page pixel count does not equal `width * height` (e.g. multi-channel).
pub fn read_tiff<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> Result<Image<f32, B, 3>> {
    let path = path.as_ref();
    let file =
        std::fs::File::open(path).with_context(|| format!("Cannot open TIFF file {:?}", path))?;
    let reader = BufReader::new(file);
    let (data, dims) = decode_tiff_from_reader(reader, path)?;
    Image::from_flat_on(
        data,
        dims,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
        backend,
    )
}

/// Decode all TIFF pages into row-major `f32` voxels and `[nz, ny, nx]` dims.
///
/// `display_path` is used only for error messages.
fn decode_tiff_from_reader<R: Read + Seek>(
    reader: R,
    display_path: &Path,
) -> Result<(Vec<f32>, [usize; 3])> {
    let mut decoder = Decoder::new(reader).map_err(|e| {
        anyhow!(
            "Failed to create TIFF decoder for {:?}: {}",
            display_path,
            e
        )
    })?;

    let (width, height) = decoder
        .dimensions()
        .map_err(|e| anyhow!("Failed to read TIFF dimensions: {}", e))?;
    let nx = usize::try_from(width).context("TIFF width exceeds usize")?;
    let ny = usize::try_from(height).context("TIFF height exceeds usize")?;
    let pixels_per_page = checked_page_sample_count::<1>(width, height)?;

    let mut data = Vec::new();
    let mut nz = 0usize;

    loop {
        let page_index = nz;
        validate_grayscale_page(&mut decoder, page_index)?;

        let result = decoder
            .read_image()
            .map_err(|e| anyhow!("Failed to decode TIFF page {}: {}", page_index, e))?;
        append_page_to_scalar(&mut data, result, pixels_per_page, page_index)?;
        nz = nz.checked_add(1).context("TIFF page count exceeds usize")?;

        if !decoder.more_images() {
            break;
        }

        decoder
            .next_image()
            .map_err(|e| anyhow!("Failed to advance to TIFF page {}: {}", nz, e))?;

        let (w, h) = decoder
            .dimensions()
            .map_err(|e| anyhow!("Failed to read TIFF page {} dimensions: {}", nz, e))?;

        if w != width || h != height {
            return Err(anyhow!(
                "TIFF page {} has dimensions {}x{}, expected {}x{} (must match first page)",
                nz,
                w,
                h,
                width,
                height,
            ));
        }
    }

    Ok((data, [nz, ny, nx]))
}

/// Return the representable sample count for one decoded TIFF page.
///
/// This guard rejects zero geometry and `usize` arithmetic overflow. Large
/// dimensions that remain representable are bounded separately by the TIFF
/// decoder's configured per-decode limits.
pub(crate) fn checked_page_sample_count<const CHANNELS: usize>(
    width: u32,
    height: u32,
) -> Result<usize> {
    let width_usize = usize::try_from(width).context("TIFF width exceeds usize")?;
    let height_usize = usize::try_from(height).context("TIFF height exceeds usize")?;
    let pixels = width_usize
        .checked_mul(height_usize)
        .with_context(|| format!("TIFF page dimensions {width}x{height} overflow usize"))?;
    if pixels == 0 {
        return Err(anyhow!("TIFF page dimensions are zero ({width}x{height})"));
    }
    pixels.checked_mul(CHANNELS).with_context(|| {
        format!("TIFF page sample count {width}x{height}x{CHANNELS} overflows usize")
    })
}

fn validate_grayscale_page<R: Read + Seek>(
    decoder: &mut Decoder<R>,
    page_index: usize,
) -> Result<()> {
    let color_type = decoder
        .colortype()
        .map_err(|e| anyhow!("Failed to read TIFF page {page_index} color type: {e}"))?;
    match color_type {
        ColorType::Gray(_) => Ok(()),
        other => Err(anyhow!(
            "TIFF grayscale loader supports only Gray pages; page {page_index} decoded as {other:?}"
        )),
    }
}

/// Append one decoded page directly to the final `f32` volume.
///
/// Integer and `f64` values convert according to Rust's numeric-cast rules.
/// Large integer magnitudes and finite `f64` values may round because binary32
/// has a 24-bit significand and a narrower exponent range.
///
/// The first `F32` page becomes the volume's backing allocation directly;
/// later `F32` pages append to that allocation. The
/// `first_float_page_becomes_the_output_allocation` regression pins this
/// no-copy first-page invariant.
pub(crate) fn append_page_to_scalar(
    target: &mut Vec<f32>,
    result: DecodingResult,
    expected: usize,
    page_index: usize,
) -> Result<()> {
    match result {
        DecodingResult::U8(values) => {
            append_converted(target, values, expected, page_index, |value| value as f32)
        }
        DecodingResult::U16(values) => {
            append_converted(target, values, expected, page_index, |value| value as f32)
        }
        DecodingResult::U32(values) => {
            append_converted(target, values, expected, page_index, |value| value as f32)
        }
        DecodingResult::U64(values) => {
            append_converted(target, values, expected, page_index, |value| value as f32)
        }
        DecodingResult::I8(values) => {
            append_converted(target, values, expected, page_index, |value| value as f32)
        }
        DecodingResult::I16(values) => {
            append_converted(target, values, expected, page_index, |value| value as f32)
        }
        DecodingResult::I32(values) => {
            append_converted(target, values, expected, page_index, |value| value as f32)
        }
        DecodingResult::I64(values) => {
            append_converted(target, values, expected, page_index, |value| value as f32)
        }
        DecodingResult::F32(values) => {
            validate_page_length(values.len(), expected, page_index)?;
            if target.is_empty() {
                *target = values;
            } else {
                target
                    .try_reserve_exact(values.len())
                    .context("TIFF volume pixel allocation failed")?;
                target.extend(values);
            }
            Ok(())
        }
        DecodingResult::F64(values) => {
            append_converted(target, values, expected, page_index, |value| value as f32)
        }
    }
}

fn append_converted<T>(
    target: &mut Vec<f32>,
    values: Vec<T>,
    expected: usize,
    page_index: usize,
    convert: impl FnMut(T) -> f32,
) -> Result<()> {
    validate_page_length(values.len(), expected, page_index)?;
    target
        .try_reserve_exact(values.len())
        .context("TIFF volume pixel allocation failed")?;
    target.extend(values.into_iter().map(convert));
    Ok(())
}

fn validate_page_length(actual: usize, expected: usize, page_index: usize) -> Result<()> {
    if actual != expected {
        return Err(anyhow!(
            "TIFF page {page_index} has {actual} values, expected {expected}"
        ));
    }
    Ok(())
}

// ── Reader struct ─────────────────────────────────────────────────────────────

/// Backend-bound reader for TIFF / BigTIFF files.
///
/// Carries the compute device so it can implement the `ImageReader<B, 3>`
/// trait from `ritk-io`.
pub struct TiffReader<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> TiffReader<B> {
    /// Create a reader bound to `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Read the TIFF file at `path` into a 3-D `Image`.
    ///
    /// See [`read_tiff`] for full documentation.
    pub fn read_image<P: AsRef<Path>>(&self, path: P) -> Result<Image<f32, B, 3>> {
        read_tiff(path, &self.backend)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_reader.rs"]
mod tests;
