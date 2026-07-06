//! TIFF RGB reader for channel-explicit 3-D volumes.
//!
//! Each RGB IFD page contributes one Z-slice to `RgbVolume<B>` with tensor
//! shape `[depth, height, width, 3]`.

use std::io::{BufReader, Read, Seek};
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use ritk_image::tensor::backend::Backend;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_core::image::RgbVolume;
use ritk_core::spatial::{Direction, Point, Spacing};
use tiff::decoder::Decoder;
use tiff::ColorType;

use crate::reader::decode_page_to_scalar;

const RGB_CHANNELS: usize = 3;

/// Read a multi-page RGB TIFF / BigTIFF file into a channel-explicit volume.
///
/// All pages must be `ColorType::RGB(_)`, have identical dimensions, and
/// decode to exactly `width * height * 3` samples.
pub fn read_tiff_color_to_volume<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<RgbVolume<B>> {
    let path = path.as_ref();
    let file =
        std::fs::File::open(path).with_context(|| format!("Cannot open TIFF file {:?}", path))?;
    let reader = BufReader::new(file);
    read_tiff_color_from_reader::<B, _>(reader, device, path)
}

fn read_tiff_color_from_reader<B: Backend, R: Read + Seek>(
    reader: R,
    device: &B::Device,
    display_path: &Path,
) -> Result<RgbVolume<B>> {
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
    let samples_per_page = pixels_per_page * RGB_CHANNELS;

    if pixels_per_page == 0 {
        return Err(anyhow!(
            "TIFF page dimensions are zero ({}x{})",
            width,
            height
        ));
    }

    let mut data = Vec::with_capacity(samples_per_page);
    let mut depth = 0usize;

    loop {
        let page_index = depth;
        validate_rgb_page(&mut decoder, page_index)?;

        let result = decoder
            .read_image()
            .map_err(|e| anyhow!("Failed to decode TIFF page {}: {}", page_index, e))?;
        let page_data = decode_page_to_scalar(result)?;

        if page_data.len() != samples_per_page {
            return Err(anyhow!(
                "TIFF RGB page {} has {} values, expected {} ({}x{}x3)",
                page_index,
                page_data.len(),
                samples_per_page,
                nx,
                ny,
            ));
        }

        data.extend(page_data);
        depth += 1;

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

    let tensor_data = TensorData::new(data, Shape::new([depth, ny, nx, RGB_CHANNELS]));
    let tensor = Tensor::<B, 4>::from_data(tensor_data, device);

    RgbVolume::try_new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
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
pub struct TiffColorReader<B: Backend> {
    device: B::Device,
}

impl<B: Backend> TiffColorReader<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    pub fn read_volume<P: AsRef<Path>>(&self, path: P) -> Result<RgbVolume<B>> {
        read_tiff_color_to_volume(path, &self.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ritk_image::tensor::backend::Backend;
    use burn_ndarray::NdArray;
    use std::fs::File;
    use std::io::BufWriter;
    use tempfile::tempdir;
    use tiff::encoder::{colortype, TiffEncoder};

    type TestBackend = NdArray<f32>;

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

    fn volume_values(volume: &RgbVolume<TestBackend>) -> Vec<f32> {
        volume.with_data_slice(|s| s.to_vec())
    }

    #[test]
    fn read_tiff_color_to_volume_preserves_rgb_page_stack() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("rgb_stack.tiff");
        let page0 = vec![1, 2, 3, 4, 5, 6];
        let page1 = vec![10, 20, 30, 40, 50, 60];
        write_rgb8_pages(&path, 2, 1, &[page0.clone(), page1.clone()])?;
        let device = <TestBackend as Backend>::Device::default();

        let volume = read_tiff_color_to_volume::<TestBackend, _>(&path, &device)?;

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
        let device = <TestBackend as Backend>::Device::default();

        let err = read_tiff_color_to_volume::<TestBackend, _>(&path, &device).unwrap_err();
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
        let reader = TiffColorReader::<TestBackend>::new(Default::default());

        let volume = reader.read_volume(&path)?;

        assert_eq!(volume.shape(), [1, 1, 1, 3]);
        assert_eq!(volume_values(&volume), vec![32.0, 128.0, 224.0]);
        Ok(())
    }
}
