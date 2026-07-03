//! TIFF / BigTIFF reader for 3-D volumetric images.
//!
//! Each IFD (Image File Directory) page represents one Z-slice.  Pages are
//! stacked in IFD order to form the Z dimension of the returned
//! `Image<B, 3>` tensor with shape `[nz, ny, nx]`.
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
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use std::io::{BufReader, Read, Seek};
use std::path::Path;
use tiff::decoder::{Decoder, DecodingResult};

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
pub fn read_tiff<B: Backend, P: AsRef<Path>>(path: P, device: &B::Device) -> Result<Image<B, 3>> {
    let path = path.as_ref();
    let file =
        std::fs::File::open(path).with_context(|| format!("Cannot open TIFF file {:?}", path))?;
    let reader = BufReader::new(file);
    read_tiff_from_reader::<B, _>(reader, device, path)
}


/// Core reader operating on any `Read + Seek` stream.
///
/// `display_path` is used only for error messages.
fn read_tiff_from_reader<B: Backend, R: Read + Seek>(
    reader: R,
    device: &B::Device,
    display_path: &Path,
) -> Result<Image<B, 3>> {
    let (data, dims) = decode_tiff_from_reader(reader, display_path)?;
    let tensor = Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(dims)), device);
    Ok(Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    ))
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
    let nx = width as usize;
    let ny = height as usize;
    let pixels_per_page = nx * ny;

    if pixels_per_page == 0 {
        return Err(anyhow!(
            "TIFF page dimensions are zero ({}x{})",
            width,
            height
        ));
    }

    let mut data = Vec::with_capacity(pixels_per_page);
    let mut nz = 0usize;

    loop {
        let page_index = nz;

        let result = decoder
            .read_image()
            .map_err(|e| anyhow!("Failed to decode TIFF page {}: {}", page_index, e))?;

        let page_data = decode_page_to_scalar(result)?;

        if page_data.len() != pixels_per_page {
            return Err(anyhow!(
                "TIFF page {} has {} values, expected {} ({}x{} single-channel); \
                 multi-channel images are not supported",
                page_index,
                page_data.len(),
                pixels_per_page,
                nx,
                ny,
            ));
        }

        data.extend(page_data);
        nz += 1;

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

/// Convert a [`DecodingResult`] variant to `Vec<f32>`.
///
/// Every integer and float variant is converted losslessly where the source
/// fits in f32.  For u32/i32/u64/i64, large magnitudes may lose precision
/// due to the 24-bit f32 significand.
pub(crate) fn decode_page_to_scalar(result: DecodingResult) -> Result<Vec<f32>> {
    match result {
        DecodingResult::U8(v) => Ok(v.into_iter().map(|x| x as f32).collect()),
        DecodingResult::U16(v) => Ok(v.into_iter().map(|x| x as f32).collect()),
        DecodingResult::U32(v) => Ok(v.into_iter().map(|x| x as f32).collect()),
        DecodingResult::U64(v) => Ok(v.into_iter().map(|x| x as f32).collect()),
        DecodingResult::I8(v) => Ok(v.into_iter().map(|x| x as f32).collect()),
        DecodingResult::I16(v) => Ok(v.into_iter().map(|x| x as f32).collect()),
        DecodingResult::I32(v) => Ok(v.into_iter().map(|x| x as f32).collect()),
        DecodingResult::I64(v) => Ok(v.into_iter().map(|x| x as f32).collect()),
        DecodingResult::F32(v) => Ok(v),
        DecodingResult::F64(v) => Ok(v.into_iter().map(|x| x as f32).collect()),
    }
}

// ── Reader struct ─────────────────────────────────────────────────────────────

/// Backend-bound reader for TIFF / BigTIFF files.
///
/// Carries the compute device so it can implement the `ImageReader<B, 3>`
/// trait from `ritk-io`.
pub struct TiffReader<B: Backend> {
    device: B::Device,
}

impl<B: Backend> TiffReader<B> {
    /// Create a reader bound to `device`.
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    /// Read the TIFF file at `path` into a 3-D `Image`.
    ///
    /// See [`read_tiff`] for full documentation.
    pub fn read_image<P: AsRef<Path>>(&self, path: P) -> Result<Image<B, 3>> {
        read_tiff(path, &self.device)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_reader.rs"]
mod tests;

/// Atlas-native-substrate entry points (transitional module: plain
/// end-state names, disambiguated from the Burn functions by module
/// path only; folds away when the Burn path is deleted — ADR 0002 A1).
#[cfg(feature = "coeus")]
pub mod native {
    #[allow(unused_imports)]
    use super::*;

    /// Read a multi-page TIFF / BigTIFF file into a Coeus-backed 3-D image.
    ///
    /// The Atlas-tensor counterpart to [`read_tiff`]: shares the page decode and
    /// dimension validation via `decode_tiff_from_reader`, differing only in the
    /// final image construction.
    pub fn read_tiff<B, P>(path: P, backend: &B) -> Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        P: AsRef<Path>,
    {
        let path = path.as_ref();
        let file =
            std::fs::File::open(path).with_context(|| format!("Cannot open TIFF file {:?}", path))?;
        let (data, dims) = decode_tiff_from_reader(BufReader::new(file), path)?;
        ritk_image::native::Image::from_flat_on(
            data,
            dims,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
            backend,
        )
    }
}
