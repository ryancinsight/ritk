pub use ritk_tiff::{
    read_tiff, read_tiff_color_to_volume, write_tiff, TiffColorReader, TiffReader, TiffWriter,
};

/// Atlas-native-substrate implementors of [`crate::domain::ImageReader`].
///
/// Transitional module: names inside are the plain end-state names; the
/// module itself disambiguates from the Burn types during coexistence and
/// folds away when the Burn path is deleted (ADR 0002).
pub mod native {
    use crate::domain::{to_io_err, ImageReader, ImageWriter};
    use coeus_core::{ComputeBackend, CpuAddressableStorage};
    use ritk_image::native::Image;
    use std::path::Path;

    /// Backend-bound Atlas-native reader (counterpart of the Burn [`super::TiffReader`]).
    pub struct TiffReader<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> TiffReader<B> {
        /// Create a reader that constructs images on `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B: ComputeBackend> ImageReader<Image<f32, B, 3>> for TiffReader<B> {
        fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
            ritk_tiff::native::read_tiff(path, &self.backend).map_err(to_io_err)
        }
    }

    /// Backend-bound Atlas-native writer (counterpart of the Burn writer).
    pub struct TiffWriter<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> TiffWriter<B> {
        /// Create a writer that extracts host data via `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B> ImageWriter<Image<f32, B, 3>> for TiffWriter<B>
    where
        B: ComputeBackend + Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> std::io::Result<()> {
            ritk_tiff::native::write_tiff(image, path, &self.backend).map_err(to_io_err)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use coeus_core::SequentialBackend;
        use ritk_spatial::{Direction, Point, Spacing};
        use tempfile::tempdir;

        /// Trait-dispatched round trip: write through `ImageWriter`, read back
        /// through `ImageReader`, exact voxel + shape parity — the unified
        /// contract is usable end-to-end on the Atlas substrate.
        #[test]
        fn native_contract_round_trips_tiff() {
            let dims = [2usize, 3, 4];
            let n = dims[0] * dims[1] * dims[2];
            let voxels: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 3.0).collect();
            let image = Image::from_flat_on(
                voxels.clone(),
                dims,
                Point::new([0.0, 0.0, 0.0]),
                Spacing::new([1.0, 1.0, 1.0]),
                Direction::identity(),
                &SequentialBackend,
            )
            .expect("coeus image");

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("contract.tiff");

            let writer = TiffWriter::new(SequentialBackend);
            ImageWriter::write(&writer, &path, &image).expect("contract write");

            let reader = TiffReader::new(SequentialBackend);
            let loaded = ImageReader::read(&reader, &path).expect("contract read");

            assert_eq!(loaded.shape(), dims, "shape round-trip");
            assert_eq!(
                loaded.data_slice().expect("contiguous"),
                voxels.as_slice(),
                "voxels must round-trip exactly through the trait contract"
            );
        }
    }
}
