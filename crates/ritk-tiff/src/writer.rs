//! TIFF writer for 3-D volumetric images.
//!
//! Writes an `Image<B, 3>` as a multi-page TIFF file where each IFD page
//! contains one Z-slice encoded as 32-bit IEEE 754 float samples
//! (`Gray32Float`).
//!
//! # Axis convention
//! Input tensor shape `[nz, ny, nx]`.  Each Z-slice is a contiguous
//! `[ny, nx]` sub-array in row-major order, mapping to a TIFF page with
//! `width = nx` and `height = ny`.
//!
//! # Spatial metadata
//! TIFF has no standardized physical-space metadata fields.  The image's
//! `origin`, `spacing`, and `direction` are **not** written to the file.
//!
//! # BigTIFF
//! Current implementation writes classic TIFF.  For volumes exceeding 4 GiB,
//! switch to `TiffEncoder::new_big`.

use anyhow::{anyhow, Context, Result};
use ritk_core::image::Image;
use ritk_image::tensor::backend::Backend;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;
use tiff::encoder::colortype;
use tiff::encoder::TiffEncoder;

/// Write a 3-D `Image` as a multi-page TIFF file.
///
/// # Algorithm
/// 1. Extract tensor data as a flat `&[f32]` slice.
/// 2. Read `[nz, ny, nx]` from the tensor shape.
/// 3. For each Z-slice, write one TIFF page with `width = nx`, `height = ny`,
///    and `Gray32Float` sample type.
///
/// # Errors
/// - File cannot be created.
/// - Tensor data cannot be extracted as `f32`.
/// - The `tiff` encoder fails to write a page.
pub fn write_tiff<B: Backend, P: AsRef<Path>>(image: &Image<B, 3>, path: P) -> Result<()> {
    let f32_vec = image.try_data_vec()?;
    write_tiff_stream(path.as_ref(), image.shape(), &f32_vec)
}

/// Substrate-agnostic TIFF file entry: creates the file and delegates to
/// [`write_tiff_flat`]. TIFF carries no physical-space metadata, so only the
/// shape and flat voxels are needed. The shared SSOT the Burn and Atlas-native
/// writers both wrap.
fn write_tiff_stream(path: &Path, shape: [usize; 3], f32_slice: &[f32]) -> Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("Cannot create TIFF file {:?}", path))?;
    let writer = BufWriter::new(file);
    write_tiff_flat(writer, shape, f32_slice, path)
}

/// Core writer operating on any `Write + Seek` stream from flat `[Z, Y, X]`
/// voxels: one `Gray32Float` IFD page per Z-slice.
fn write_tiff_flat<W: Write + Seek>(
    writer: W,
    shape: [usize; 3],
    f32_slice: &[f32],
    display_path: &Path,
) -> Result<()> {
    let nz = shape[0];
    let ny = shape[1];
    let nx = shape[2];
    let pixels_per_page = ny * nx;

    if pixels_per_page == 0 {
        return Err(anyhow!(
            "Cannot write TIFF with zero-area pages (ny={}, nx={})",
            ny,
            nx,
        ));
    }

    if f32_slice.len() != nz * pixels_per_page {
        return Err(anyhow!(
            "Tensor data length {} does not match shape [{}, {}, {}] = {} voxels",
            f32_slice.len(),
            nz,
            ny,
            nx,
            nz * pixels_per_page,
        ));
    }

    let mut encoder = TiffEncoder::new(writer).map_err(|e| {
        anyhow!(
            "Failed to create TIFF encoder for {:?}: {}",
            display_path,
            e
        )
    })?;

    for z in 0..nz {
        let offset = z * pixels_per_page;
        let slice_data = &f32_slice[offset..offset + pixels_per_page];

        encoder
            .write_image::<colortype::Gray32Float>(nx as u32, ny as u32, slice_data)
            .map_err(|e| {
                anyhow!(
                    "Failed to write TIFF page {} of {:?}: {}",
                    z,
                    display_path,
                    e
                )
            })?;
    }

    Ok(())
}

// ── Writer struct ─────────────────────────────────────────────────────────────

/// Stateless writer for TIFF files.
///
/// Provides a struct-based API that delegates to [`write_tiff`].
pub struct TiffWriter;

impl TiffWriter {
    /// Write `image` to the TIFF file at `path`.
    ///
    /// See [`write_tiff`] for full documentation.
    pub fn write<B: Backend, P: AsRef<Path>>(image: &Image<B, 3>, path: P) -> Result<()> {
        write_tiff(image, path)
    }
}

/// Atlas-native-substrate TIFF writers (plain end-state names, disambiguated
/// from the Burn functions by module path only; folds away when the Burn path
/// is deleted — ADR 0002 A1).
pub mod native {
    use super::write_tiff_stream;
    use anyhow::Result;
    use std::path::Path;

    /// Write an Atlas-native 3-D image as a multi-page TIFF file.
    ///
    /// Host data is extracted layout-independently via `data_cow_on`, then
    /// serialized through the same
    /// [`write_tiff_stream`](super::write_tiff_stream) core as the Burn
    /// [`write_tiff`](super::write_tiff) — byte-identical output for the same
    /// logical image.
    pub fn write_tiff<B, P>(
        image: &ritk_image::native::Image<f32, B, 3>,
        path: P,
        backend: &B,
    ) -> Result<()>
    where
        B: coeus_core::ComputeBackend + Default,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
        P: AsRef<Path>,
    {
        let voxels = image.data_cow_on(backend);
        write_tiff_stream(path.as_ref(), image.shape(), &voxels)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::read_tiff;
    use burn_ndarray::NdArray;
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use ritk_image::tensor::backend::Backend;
    use ritk_image::tensor::{Shape, Tensor, TensorData};
    use tempfile::tempdir;

    use super::{write_tiff, TiffWriter};

    type TestBackend = NdArray<f32>;

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

    #[test]
    fn write_creates_nonempty_file() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("basic.tiff");

        let image = make_image(vec![1.0f32; 2 * 3 * 4], 2, 3, 4);
        write_tiff(&image, &path)?;

        assert!(path.exists(), "output file must exist after write");
        let meta = std::fs::metadata(&path)?;
        assert!(meta.len() > 0, "output file must be non-empty");

        Ok(())
    }

    #[test]
    fn tiff_writer_struct_delegates_to_write_tiff() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("struct_write.tiff");

        let image = make_image(vec![7.5f32; 2 * 2 * 2], 2, 2, 2);
        TiffWriter::write(&image, &path)?;

        assert!(path.exists(), "output file must exist");
        assert!(
            std::fs::metadata(&path)?.len() > 0,
            "output file must be non-empty",
        );

        Ok(())
    }

    #[test]
    fn round_trip_scalar_values_are_bitwise_identical() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("roundtrip_f32.tiff");
        let device: <TestBackend as Backend>::Device = Default::default();

        let nz = 2usize;
        let ny = 3usize;
        let nx = 5usize;
        let data_vec: Vec<f32> = (0..(nz * ny * nx) as u32)
            .map(|i| i as f32 * std::f32::consts::PI / 7.0)
            .collect();

        let image = make_image(data_vec.clone(), nz, ny, nx);
        write_tiff(&image, &path)?;

        let loaded = read_tiff::<TestBackend, _>(&path, &device)?;
        assert_eq!(loaded.shape(), [nz, ny, nx], "shape mismatch");
        loaded.with_data_slice(|loaded_vals| {
            for (i, (&got, &expected)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
                assert!(
                    (got - expected).abs() < 1e-6,
                    "voxel[{}]: expected {}, got {}",
                    i,
                    expected,
                    got,
                );
            }
        });
        Ok(())
    }

    #[test]
    fn multi_page_file_is_larger_than_single_page() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path_1 = dir.path().join("one_page.tiff");
        let path_3 = dir.path().join("three_pages.tiff");

        let ny = 4usize;
        let nx = 5usize;

        let image_1 = make_image(vec![0.0f32; ny * nx], 1, ny, nx);
        write_tiff(&image_1, &path_1)?;

        let image_3 = make_image(vec![0.0f32; 3 * ny * nx], 3, ny, nx);
        write_tiff(&image_3, &path_3)?;

        let size_1 = std::fs::metadata(&path_1)?.len();
        let size_3 = std::fs::metadata(&path_3)?.len();

        let min_extra = (2 * ny * nx * 4) as u64;
        assert!(
            size_3 >= size_1 + min_extra,
            "3-page file ({} bytes) must be at least {} bytes larger than 1-page file ({} bytes)",
            size_3,
            min_extra,
            size_1,
        );

        Ok(())
    }

    #[test]
    fn file_contains_full_scalar_payload() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("payload.tiff");

        let nz = 2usize;
        let ny = 3usize;
        let nx = 4usize;
        let n_voxels = nz * ny * nx;

        let image = make_image(vec![1.0f32; n_voxels], nz, ny, nx);
        write_tiff(&image, &path)?;

        let file_size = std::fs::metadata(&path)?.len();
        let min_payload = (n_voxels * 4) as u64;
        assert!(
            file_size >= min_payload,
            "file size {} must be >= minimum payload {} ({} voxels × 4 bytes)",
            file_size,
            min_payload,
            n_voxels,
        );

        Ok(())
    }

    #[test]
    fn edge_case_values_survive_round_trip() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("edge_values.tiff");
        let device: <TestBackend as Backend>::Device = Default::default();

        let values: Vec<f32> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::MIN,
            1.0e-38,
            std::f32::consts::PI,
            std::f32::consts::E,
            123_456.79,
            -987_654.3,
        ];
        assert_eq!(values.len(), 12);

        let image = make_image(values.clone(), 1, 3, 4);
        write_tiff(&image, &path)?;

        let loaded = read_tiff::<TestBackend, _>(&path, &device)?;
        assert_eq!(loaded.shape(), [1, 3, 4]);
        loaded.with_data_slice(|loaded_vals| {
            for (i, (&got, &expected)) in loaded_vals.iter().zip(values.iter()).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    expected.to_bits(),
                    "voxel[{}]: expected {} (bits {:#010x}), got {} (bits {:#010x})",
                    i,
                    expected,
                    expected.to_bits(),
                    got,
                    got.to_bits(),
                );
            }
        });
        Ok(())
    }
}
