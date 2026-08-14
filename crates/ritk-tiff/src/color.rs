//! TIFF RGB reader for channel-explicit 3-D volumes.
//!
//! Each RGB IFD page contributes one Z-slice to `RgbVolume<f32, B>` with tensor
//! shape `[depth, height, width, 3]`.

use std::io::{BufReader, Read, Seek};
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use coeus_core::ComputeBackend;
use ritk_image::RgbVolume;
use ritk_spatial::{Direction, Point, Spacing};
use tiff::decoder::Decoder;
use tiff::ColorType;

use crate::reader::{append_page_to_scalar, checked_page_sample_count};

const RGB_CHANNELS: usize = 3;

/// Read a multi-page RGB TIFF / BigTIFF file into a channel-explicit volume.
///
/// All pages must be `ColorType::RGB(_)`, have identical dimensions, and
/// decode to exactly `width * height * 3` samples.
pub fn read_tiff_color_to_volume<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> Result<RgbVolume<f32, B>> {
    let path = path.as_ref();
    let file =
        std::fs::File::open(path).with_context(|| format!("Cannot open TIFF file {:?}", path))?;
    let reader = BufReader::new(file);
    read_tiff_color_from_reader(reader, backend, path)
}

fn read_tiff_color_from_reader<B: ComputeBackend, R: Read + Seek>(
    reader: R,
    backend: &B,
    display_path: &Path,
) -> Result<RgbVolume<f32, B>> {
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
    let samples_per_page = checked_page_sample_count::<RGB_CHANNELS>(width, height)?;

    let mut data = Vec::new();
    let mut depth = 0usize;

    loop {
        let page_index = depth;
        validate_rgb_page(&mut decoder, page_index)?;

        let result = decoder
            .read_image()
            .map_err(|e| anyhow!("Failed to decode TIFF page {}: {}", page_index, e))?;
        append_page_to_scalar(&mut data, result, samples_per_page, page_index)?;
        depth = depth
            .checked_add(1)
            .context("TIFF page count exceeds usize")?;

        if !decoder.more_images() {
            break;
        }

        decoder
            .next_image()
            .map_err(|e| anyhow!("Failed to advance to TIFF page {}: {}", depth, e))?;

        let (w, h) = decoder
            .dimensions()
            .map_err(|e| anyhow!("Failed to read TIFF page {} dimensions: {}", depth, e))?;

        if w != width || h != height {
            return Err(anyhow!(
                "TIFF page {} has dimensions {}x{}, expected {}x{} (must match first page)",
                depth,
                w,
                h,
                width,
                height,
            ));
        }
    }

    RgbVolume::from_flat_on(
        data,
        [depth, ny, nx],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
        backend,
    )
}

fn validate_rgb_page<R: Read + Seek>(decoder: &mut Decoder<R>, page_index: usize) -> Result<()> {
    let color_type = decoder
        .colortype()
        .map_err(|e| anyhow!("Failed to read TIFF page {} color type: {}", page_index, e))?;
    match color_type {
        ColorType::RGB(_) => Ok(()),
        other => Err(anyhow!(
            "TIFF RGB color loader supports only RGB pages; page {} decoded as {:?}",
            page_index,
            other
        )),
    }
}

/// Backend-bound reader for RGB TIFF / BigTIFF files.
pub struct TiffColorReader<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> TiffColorReader<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn read_volume<P: AsRef<Path>>(&self, path: P) -> Result<RgbVolume<f32, B>> {
        read_tiff_color_to_volume(path, &self.backend)
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
    use super::*;
    use coeus_core::SequentialBackend;
    use std::fs::File;
    use std::io::BufWriter;
    use tempfile::tempdir;
    use tiff::encoder::{colortype, TiffEncoder};

    type TestBackend = SequentialBackend;

    fn write_rgb8_pages(path: &Path, width: u32, height: u32, pages: &[Vec<u8>]) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        let mut encoder = TiffEncoder::new(writer)?;
        for page in pages {
            encoder.write_image::<colortype::RGB8>(width, height, page)?;
        }
        Ok(())
    }

    fn write_gray8_page(path: &Path, width: u32, height: u32, pixels: &[u8]) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        let mut encoder = TiffEncoder::new(writer)?;
        encoder.write_image::<colortype::Gray8>(width, height, pixels)?;
        Ok(())
    }

    fn replace_ifd_long(path: &Path, target_tag: u16, value: u32) -> Result<()> {
        let mut bytes = std::fs::read(path)?;
        if !matches!(bytes.get(0..2), Some(b"II")) {
            return Err(anyhow!("test TIFF encoder did not emit little-endian data"));
        }
        let ifd_offset_bytes: [u8; 4] = bytes
            .get(4..8)
            .context("test TIFF header has no first-IFD offset")?
            .try_into()
            .context("test TIFF first-IFD offset has wrong width")?;
        let ifd_offset = usize::try_from(u32::from_le_bytes(ifd_offset_bytes))?;
        let entry_count_bytes: [u8; 2] = bytes
            .get(ifd_offset..ifd_offset + 2)
            .context("test TIFF has no IFD entry count")?
            .try_into()
            .context("test TIFF IFD entry count has wrong width")?;
        let entry_count = usize::from(u16::from_le_bytes(entry_count_bytes));

        for entry in 0..entry_count {
            let entry_offset = ifd_offset + 2 + entry * 12;
            let tag_bytes: [u8; 2] = bytes
                .get(entry_offset..entry_offset + 2)
                .context("test TIFF IFD entry is truncated")?
                .try_into()
                .context("test TIFF tag has wrong width")?;
            if u16::from_le_bytes(tag_bytes) == target_tag {
                bytes
                    .get_mut(entry_offset + 8..entry_offset + 12)
                    .context("test TIFF LONG value is truncated")?
                    .copy_from_slice(&value.to_le_bytes());
                std::fs::write(path, bytes)?;
                return Ok(());
            }
        }
        Err(anyhow!("test TIFF does not contain tag {target_tag}"))
    }

    fn volume_values(volume: &RgbVolume<f32, TestBackend>) -> Vec<f32> {
        volume.data_cow_on(&SequentialBackend).into_owned()
    }

    #[test]
    fn read_tiff_color_to_volume_preserves_rgb_page_stack() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("rgb_stack.tiff");
        let page0 = vec![1, 2, 3, 4, 5, 6];
        let page1 = vec![10, 20, 30, 40, 50, 60];
        write_rgb8_pages(&path, 2, 1, &[page0.clone(), page1.clone()])?;
        let volume = read_tiff_color_to_volume(&path, &SequentialBackend)?;

        assert_eq!(volume.shape(), [2, 1, 2, 3]);
        assert_eq!(volume.spatial_shape(), [2, 1, 2]);
        assert_eq!(
            volume_values(&volume),
            page0
                .iter()
                .chain(page1.iter())
                .map(|&v| v as f32)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            [volume.origin()[0], volume.origin()[1], volume.origin()[2]],
            [0.0, 0.0, 0.0]
        );
        assert_eq!(
            [
                volume.spacing()[0],
                volume.spacing()[1],
                volume.spacing()[2]
            ],
            [1.0, 1.0, 1.0]
        );
        Ok(())
    }

    #[test]
    fn read_tiff_color_to_volume_rejects_grayscale_tiff() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("gray.tiff");
        write_gray8_page(&path, 2, 1, &[7, 9])?;
        let err = read_tiff_color_to_volume(&path, &SequentialBackend).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("supports only RGB pages"),
            "expected RGB page rejection, got {msg}"
        );
        Ok(())
    }

    #[test]
    fn tiff_color_reader_delegates_to_rgb_loader() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("reader.tiff");
        write_rgb8_pages(&path, 1, 1, &[vec![32, 128, 224]])?;
        let reader = TiffColorReader::new(SequentialBackend);

        let volume = reader.read_volume(&path)?;

        assert_eq!(volume.shape(), [1, 1, 1, 3]);
        assert_eq!(volume_values(&volume), vec![32.0, 128.0, 224.0]);
        Ok(())
    }

    #[test]
    fn rgb_loader_rejects_hostile_declared_geometry_before_allocation() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("hostile_rgb.tiff");
        write_rgb8_pages(&path, 1, 1, &[vec![1, 2, 3]])?;
        replace_ifd_long(&path, 256, u32::MAX)?;
        replace_ifd_long(&path, 257, u32::MAX)?;

        let error = read_tiff_color_to_volume(&path, &SequentialBackend).unwrap_err();
        let message = format!("{error:#}");
        assert!(
            message.contains("Format error: Inconsistent sizes encountered")
                || message.contains("TIFF page sample count")
                    && message.contains("overflows usize"),
            "unexpected hostile-geometry error: {message}"
        );
        Ok(())
    }
}
