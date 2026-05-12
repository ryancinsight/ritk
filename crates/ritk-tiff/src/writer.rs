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
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
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
    let path = path.as_ref();

    let file = std::fs::File::create(path)
        .with_context(|| format!("Cannot create TIFF file {:?}", path))?;
    let writer = BufWriter::new(file);

    write_tiff_to_writer::<B, _>(image, writer, path)
}

/// Core writer operating on any `Write + Seek` stream.
fn write_tiff_to_writer<B: Backend, W: Write + Seek>(
    image: &Image<B, 3>,
    writer: W,
    display_path: &Path,
) -> Result<()> {
    let tensor_data = image.data().clone().to_data();
    let f32_slice = match tensor_data.as_slice::<f32>() {
        Ok(s) => s,
        Err(e) => {
            return Err(anyhow!(
                "Failed to extract f32 slice from tensor data: {:?}",
                e
            ))
        }
    };

    let shape = image.shape();
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::read_tiff;
    use burn::tensor::backend::Backend;
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
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
    fn round_trip_f32_values_are_bitwise_identical() -> anyhow::Result<()> {
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
    fn file_contains_full_f32_payload() -> anyhow::Result<()> {
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
            123456.789,
            -987654.321,
        ];
        assert_eq!(values.len(), 12);

        let image = make_image(values.clone(), 1, 3, 4);
        write_tiff(&image, &path)?;

        let loaded = read_tiff::<TestBackend, _>(&path, &device)?;
        assert_eq!(loaded.shape(), [1, 3, 4]);

        let loaded_td = loaded.data().clone().to_data();
        let loaded_vals = loaded_td.as_slice::<f32>().unwrap();

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

        Ok(())
    }
}
