//! TIFF / BigTIFF reader for 3-D volumetric images.
//!
//! Each IFD (Image File Directory) page in the TIFF file represents one
//! Z-slice.  Pages are stacked in IFD order to form the Z dimension of the
//! returned `Image<B, 3>` tensor with shape `[nz, ny, nx]`.
//!
//! # Axis convention
//! No axis permutation is applied.  TIFF page data is stored in row-major
//! order (Y outer, X inner), which maps directly to the `[ny, nx]` layout
//! of each Z-slice in the RITK tensor.
//!
//! # Spatial metadata
//! TIFF has no standardized physical-space metadata fields.  The returned
//! image uses default values:
//! - `origin  = [0, 0, 0]`
//! - `spacing = [1, 1, 1]`
//! - `direction = identity`
//!
//! Users must set these externally when physical coordinates are known.
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
/// 4. Convert pixel data to `f32` (see [`decoding_result_to_f32`]).
/// 5. Validate that every page has the same `(width, height)`.
/// 6. Stack slices into tensor shape `[nz, ny, nx]`.
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
    let mut decoder = Decoder::new(reader).map_err(|e| {
        anyhow!(
            "Failed to create TIFF decoder for {:?}: {}",
            display_path,
            e
        )
    })?;

    // First-page dimensions define the expected (width, height) for all pages.
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

    let mut slices: Vec<Vec<f32>> = Vec::new();

    loop {
        let page_index = slices.len();

        let result = decoder
            .read_image()
            .map_err(|e| anyhow!("Failed to decode TIFF page {}: {}", page_index, e))?;

        let page_data = decoding_result_to_f32(result, page_index)?;

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

        slices.push(page_data);

        if !decoder.more_images() {
            break;
        }

        decoder
            .next_image()
            .map_err(|e| anyhow!("Failed to advance to TIFF page {}: {}", slices.len(), e))?;

        // Verify subsequent pages have consistent dimensions.
        let (w, h) = decoder.dimensions().map_err(|e| {
            anyhow!(
                "Failed to read TIFF page {} dimensions: {}",
                slices.len(),
                e
            )
        })?;

        if w != width || h != height {
            return Err(anyhow!(
                "TIFF page {} has dimensions {}x{}, expected {}x{} (must match first page)",
                slices.len(),
                w,
                h,
                width,
                height,
            ));
        }
    }

    let nz = slices.len();
    let total_voxels = nz * ny * nx;

    let mut data = Vec::with_capacity(total_voxels);
    for slice in &slices {
        data.extend_from_slice(slice);
    }

    let tensor_data = TensorData::new(data, Shape::new([nz, ny, nx]));
    let tensor = Tensor::<B, 3>::from_data(tensor_data, device);

    // TIFF has no standard physical-space metadata — use defaults.
    let origin = Point::new([0.0, 0.0, 0.0]);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let direction = Direction::identity();

    Ok(Image::new(tensor, origin, spacing, direction))
}

/// Convert a [`DecodingResult`] variant to `Vec<f32>`.
///
/// Every integer and float variant is converted losslessly where the source
/// fits in f32.  For u32, i32, u64, i64 sources, large magnitudes may lose
/// precision due to the 24-bit f32 significand — this is acceptable for the
/// medical-imaging use case where intensities rarely exceed 2^24.
fn decoding_result_to_f32(result: DecodingResult, page_index: usize) -> Result<Vec<f32>> {
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
        #[allow(unreachable_patterns)]
        _ => Err(anyhow!(
            "Unsupported TIFF sample format on page {}",
            page_index
        )),
    }
}

// ── Public reader struct ──────────────────────────────────────────────────────

/// Stateless reader for TIFF / BigTIFF files.
///
/// Provides a struct-based API that delegates to [`read_tiff`].
pub struct TiffReader;

impl TiffReader {
    /// Read the TIFF file at `path` into a 3-D `Image`.
    ///
    /// See [`read_tiff`] for full documentation.
    pub fn read<B: Backend, P: AsRef<Path>>(path: P, device: &B::Device) -> Result<Image<B, 3>> {
        read_tiff(path, device)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::format::tiff::write_tiff;
    use burn::tensor::backend::Backend;
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use tempfile::tempdir;

    use super::{read_tiff, TiffReader};

    type TestBackend = NdArray<f32>;

    /// Helper: build a 3-D Image from a flat f32 vec and shape [nz, ny, nx].
    fn make_image(data: Vec<f32>, nz: usize, ny: usize, nx: usize) -> Image<TestBackend, 3> {
        let device: <TestBackend as Backend>::Device = Default::default();
        let tensor_data = TensorData::new(data, Shape::new([nz, ny, nx]));
        let tensor = Tensor::<TestBackend, 3>::from_data(tensor_data, &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    // ── Round-trip: single slice ──────────────────────────────────────────

    /// Write a single-slice (nz=1) f32 image and read it back.
    /// Verify shape and every voxel value.
    #[test]
    fn test_round_trip_single_slice() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("single_slice.tiff");
        let device: <TestBackend as Backend>::Device = Default::default();

        let nz = 1usize;
        let ny = 4usize;
        let nx = 5usize;
        let data_vec: Vec<f32> = (0u32..(nz * ny * nx) as u32)
            .map(|i| i as f32 * 0.5)
            .collect();

        let image = make_image(data_vec.clone(), nz, ny, nx);
        write_tiff(&image, &path)?;

        let loaded = read_tiff::<TestBackend, _>(&path, &device)?;

        // Shape
        assert_eq!(loaded.shape(), [nz, ny, nx], "shape mismatch");

        // Spatial defaults
        assert_eq!(loaded.origin(), &Point::new([0.0, 0.0, 0.0]));
        assert_eq!(loaded.spacing(), &Spacing::new([1.0, 1.0, 1.0]));
        assert_eq!(loaded.direction(), &Direction::identity());

        // Voxel values
        let loaded_td = loaded.data().clone().to_data();
        let loaded_vals = loaded_td.as_slice::<f32>().unwrap();
        assert_eq!(
            loaded_vals.len(),
            data_vec.len(),
            "total voxel count mismatch"
        );
        for (i, (&got, &expected)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-6,
                "voxel[{}]: expected {}, got {}",
                i,
                expected,
                got,
            );
        }

        Ok(())
    }

    // ── Round-trip: multi-slice ───────────────────────────────────────────

    /// Write a multi-slice (nz=3) image with analytically-known values and
    /// read it back.  Verify shape and per-voxel correctness.
    #[test]
    fn test_round_trip_multi_slice() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("multi_slice.tiff");
        let device: <TestBackend as Backend>::Device = Default::default();

        let nz = 3usize;
        let ny = 4usize;
        let nx = 5usize;
        // Each voxel's value encodes its linear index: 0.0, 1.0, 2.0, ...
        let data_vec: Vec<f32> = (0u32..(nz * ny * nx) as u32).map(|i| i as f32).collect();

        let image = make_image(data_vec.clone(), nz, ny, nx);
        write_tiff(&image, &path)?;

        let loaded = read_tiff::<TestBackend, _>(&path, &device)?;

        assert_eq!(loaded.shape(), [nz, ny, nx], "shape mismatch");

        let loaded_td = loaded.data().clone().to_data();
        let loaded_vals = loaded_td.as_slice::<f32>().unwrap();
        assert_eq!(loaded_vals.len(), data_vec.len());

        for (i, (&got, &expected)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-6,
                "voxel[{}]: expected {}, got {}",
                i,
                expected,
                got,
            );
        }

        Ok(())
    }

    // ── Verify distinct per-slice values survive round-trip ───────────────

    /// Each Z-slice is filled with a distinct constant so we can verify
    /// slice ordering is preserved through write → read.
    #[test]
    fn test_slice_ordering_preserved() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("slice_order.tiff");
        let device: <TestBackend as Backend>::Device = Default::default();

        let nz = 4usize;
        let ny = 3usize;
        let nx = 2usize;
        let pixels_per_slice = ny * nx;
        let mut data_vec = Vec::with_capacity(nz * pixels_per_slice);
        for z in 0..nz {
            let fill_value = (z + 1) as f32 * 100.0; // 100, 200, 300, 400
            data_vec.extend(std::iter::repeat(fill_value).take(pixels_per_slice));
        }

        let image = make_image(data_vec.clone(), nz, ny, nx);
        write_tiff(&image, &path)?;

        let loaded = read_tiff::<TestBackend, _>(&path, &device)?;
        assert_eq!(loaded.shape(), [nz, ny, nx]);

        let loaded_td = loaded.data().clone().to_data();
        let loaded_vals = loaded_td.as_slice::<f32>().unwrap();

        for z in 0..nz {
            let expected = (z + 1) as f32 * 100.0;
            let offset = z * pixels_per_slice;
            for px in 0..pixels_per_slice {
                let got = loaded_vals[offset + px];
                assert!(
                    (got - expected).abs() < 1e-6,
                    "slice {} pixel {}: expected {}, got {}",
                    z,
                    px,
                    expected,
                    got,
                );
            }
        }

        Ok(())
    }

    // ── TiffReader struct delegates correctly ─────────────────────────────

    #[test]
    fn test_reader_struct_delegates() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("struct_read.tiff");
        let device: <TestBackend as Backend>::Device = Default::default();

        let data_vec: Vec<f32> = vec![42.0; 2 * 3 * 4];
        let image = make_image(data_vec, 2, 3, 4);
        write_tiff(&image, &path)?;

        let loaded = TiffReader::read::<TestBackend, _>(&path, &device)?;
        assert_eq!(loaded.shape(), [2, 3, 4]);

        let loaded_td = loaded.data().clone().to_data();
        let loaded_vals = loaded_td.as_slice::<f32>().unwrap();
        for (i, &v) in loaded_vals.iter().enumerate() {
            assert!(
                (v - 42.0).abs() < 1e-6,
                "voxel[{}]: expected 42.0, got {}",
                i,
                v,
            );
        }

        Ok(())
    }

    // ── Missing file produces error ──────────────────────────────────────

    #[test]
    fn test_missing_file_returns_error() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let result = read_tiff::<TestBackend, _>("/nonexistent/path.tiff", &device);
        assert!(result.is_err(), "missing file must produce an error");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("Cannot open TIFF file"),
            "error message must mention file open failure, got: {}",
            msg,
        );
    }

    // ── Invalid (non-TIFF) file produces error ───────────────────────────

    #[test]
    fn test_invalid_file_returns_error() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("not_a_tiff.tiff");
        std::fs::write(&path, b"this is not a tiff file")?;

        let device: <TestBackend as Backend>::Device = Default::default();
        let result = read_tiff::<TestBackend, _>(&path, &device);
        assert!(result.is_err(), "invalid TIFF must produce an error");

        Ok(())
    }

    // ── Negative f32 values survive round-trip ───────────────────────────

    #[test]
    fn test_negative_values_round_trip() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("negative.tiff");
        let device: <TestBackend as Backend>::Device = Default::default();

        let nz = 2usize;
        let ny = 3usize;
        let nx = 4usize;
        let data_vec: Vec<f32> = (0..(nz * ny * nx) as i32)
            .map(|i| (i as f32) - 12.0) // range: -12.0 .. +11.0
            .collect();

        let image = make_image(data_vec.clone(), nz, ny, nx);
        write_tiff(&image, &path)?;

        let loaded = read_tiff::<TestBackend, _>(&path, &device)?;
        assert_eq!(loaded.shape(), [nz, ny, nx]);

        let loaded_td = loaded.data().clone().to_data();
        let loaded_vals = loaded_td.as_slice::<f32>().unwrap();
        for (i, (&got, &expected)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-6,
                "voxel[{}]: expected {}, got {}",
                i,
                expected,
                got,
            );
        }

        Ok(())
    }
}
