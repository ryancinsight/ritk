//! Native multi-page TIFF writer.

use anyhow::{anyhow, Context, Result};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::native::Image;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;
use tiff::encoder::{colortype, TiffEncoder};

/// Writes `[depth, rows, columns]` as one `Gray32Float` TIFF page per slice.
///
/// TIFF does not carry the image's physical-space metadata.
pub fn write_tiff<B, P>(image: &Image<f32, B, 3>, path: P, backend: &B) -> Result<()>
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    P: AsRef<Path>,
{
    let voxels = image.data_cow_on(backend);
    write_tiff_stream(path.as_ref(), image.shape(), &voxels)
}

fn write_tiff_stream(path: &Path, shape: [usize; 3], values: &[f32]) -> Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("cannot create TIFF file {}", path.display()))?;
    write_tiff_flat(BufWriter::new(file), shape, values, path)
}

fn write_tiff_flat<W: Write + Seek>(
    writer: W,
    shape: [usize; 3],
    values: &[f32],
    display_path: &Path,
) -> Result<()> {
    let [depth, rows, columns] = shape;
    let pixels_per_page = rows
        .checked_mul(columns)
        .context("TIFF page dimensions overflow usize")?;
    if pixels_per_page == 0 {
        return Err(anyhow!(
            "cannot write TIFF with zero-area pages: shape={shape:?}"
        ));
    }
    let expected = depth
        .checked_mul(pixels_per_page)
        .context("TIFF volume dimensions overflow usize")?;
    if values.len() != expected {
        return Err(anyhow!(
            "TIFF data length {} does not match shape {shape:?} ({expected} voxels)",
            values.len()
        ));
    }
    let width = u32::try_from(columns).context("TIFF width exceeds u32")?;
    let height = u32::try_from(rows).context("TIFF height exceeds u32")?;
    let mut encoder = TiffEncoder::new(writer).map_err(|error| {
        anyhow!(
            "failed to create TIFF encoder for {}: {error}",
            display_path.display()
        )
    })?;
    for page in 0..depth {
        let offset = page * pixels_per_page;
        encoder
            .write_image::<colortype::Gray32Float>(
                width,
                height,
                &values[offset..offset + pixels_per_page],
            )
            .map_err(|error| {
                anyhow!(
                    "failed to write TIFF page {page} of {}: {error}",
                    display_path.display()
                )
            })?;
    }
    Ok(())
}

/// Backend-bound native TIFF writer.
pub struct TiffWriter<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> TiffWriter<B> {
    /// Creates a writer that extracts image storage through `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }
}

impl<B> TiffWriter<B>
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    /// Writes `image` to `path`.
    pub fn write<P: AsRef<Path>>(&self, image: &Image<f32, B, 3>, path: P) -> Result<()> {
        write_tiff(image, path, &self.backend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read_tiff;
    use coeus_core::SequentialBackend;
    use ritk_spatial::{Direction, Point, Spacing};
    use tempfile::tempdir;

    fn image(shape: [usize; 3], values: Vec<f32>) -> Result<Image<f32, SequentialBackend, 3>> {
        Image::from_flat_on(
            values,
            shape,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
            &SequentialBackend,
        )
    }

    #[test]
    fn multipage_round_trip_is_exact() -> Result<()> {
        let directory = tempdir()?;
        let path = directory.path().join("volume.tiff");
        let input = image([2, 2, 3], (0..12).map(|value| value as f32 - 3.0).collect())?;
        write_tiff(&input, &path, &SequentialBackend)?;
        let output = read_tiff(&path, &SequentialBackend)?;
        assert_eq!(output.shape(), input.shape());
        assert_eq!(
            output.data_cow_on(&SequentialBackend).as_ref(),
            input.data_cow_on(&SequentialBackend).as_ref()
        );
        Ok(())
    }

    #[test]
    fn writer_struct_creates_nonempty_file() -> Result<()> {
        let directory = tempdir()?;
        let path = directory.path().join("writer.tiff");
        let input = image([1, 1, 2], vec![3.0, 5.0])?;
        TiffWriter::new(SequentialBackend).write(&input, &path)?;
        assert!(std::fs::metadata(path)?.len() > 0);
        Ok(())
    }

    #[test]
    fn special_float_bit_patterns_survive_round_trip() -> Result<()> {
        let directory = tempdir()?;
        let path = directory.path().join("special.tiff");
        let values = vec![f32::NEG_INFINITY, -0.0, f32::MIN_POSITIVE, f32::INFINITY];
        let input = image([1, 2, 2], values.clone())?;
        write_tiff(&input, &path, &SequentialBackend)?;
        let output = read_tiff(&path, &SequentialBackend)?;
        let actual: Vec<u32> = output
            .data_cow_on(&SequentialBackend)
            .iter()
            .map(|value| value.to_bits())
            .collect();
        assert_eq!(
            actual,
            values.into_iter().map(f32::to_bits).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn writer_rejects_zero_area_pages() -> Result<()> {
        let input = image([1, 0, 2], Vec::new())?;
        let directory = tempdir()?;
        let path = directory.path().join("invalid.tiff");
        let error = write_tiff(&input, path, &SequentialBackend).unwrap_err();
        assert!(error.to_string().contains("zero-area"));
        Ok(())
    }
}
