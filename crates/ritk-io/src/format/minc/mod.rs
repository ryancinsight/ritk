//! MINC2 I/O adapters over Coeus-backed images.

use crate::domain::{to_io_err, ImageReader, ImageWriter};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::native::Image;
use std::path::Path;

pub use ritk_minc::{read_minc, write_minc};

/// Backend-bound MINC2 reader.
pub struct MincReader<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> MincReader<B> {
    /// Create a reader that constructs images on `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }
}

impl<B: ComputeBackend> ImageReader<Image<f32, B, 3>> for MincReader<B> {
    fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
        read_minc(path, &self.backend).map_err(to_io_err)
    }
}

/// Backend-bound MINC2 writer.
pub struct MincWriter<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> MincWriter<B> {
    /// Create a writer that extracts host data through `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }
}

impl<B> ImageWriter<Image<f32, B, 3>> for MincWriter<B>
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> std::io::Result<()> {
        write_minc(image, path, &self.backend).map_err(to_io_err)
    }
}

#[cfg(test)]
mod tests {
    use coeus_core::SequentialBackend;
    use ritk_spatial::{Direction, Point, Spacing};

    use super::*;

    #[test]
    fn reader_writer_adapters_round_trip_exact_values() {
        let image = Image::from_flat_on(
            (0..8).map(|value| value as f32).collect(),
            [2, 2, 2],
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
            &SequentialBackend,
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adapter.mnc");
        let writer = MincWriter::new(SequentialBackend);
        let reader = MincReader::new(SequentialBackend);

        ImageWriter::write(&writer, &path, &image).unwrap();
        let loaded = ImageReader::read(&reader, path).unwrap();

        assert_eq!(loaded.data_slice().unwrap(), image.data_slice().unwrap());
        assert_eq!(loaded.shape(), image.shape());
    }
}
