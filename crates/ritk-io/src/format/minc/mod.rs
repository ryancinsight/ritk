//! MINC2 I/O adapters over Coeus-backed images.

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

    /// Backend-bound Atlas-native reader (counterpart of the Burn `MincReader`).
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
            ritk_minc::read_minc(path, &self.backend).map_err(to_io_err)
        }
    }

    /// Backend-bound Atlas-native writer (counterpart of the Burn writer).
    pub struct MincWriter<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> MincWriter<B> {
        /// Create a writer that extracts host data via `backend`.
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
            ritk_minc::write_minc(image, path, &self.backend).map_err(to_io_err)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use coeus_core::SequentialBackend;
        use ritk_spatial::{Direction, Point, Spacing};
        use tempfile::tempdir;

        /// The native MINC writer emits a valid HDF5 container through the
        /// unified `ImageWriter` contract.
        #[test]
        fn native_writer_produces_hdf5_signature() {
            let image = Image::from_flat_on(
                vec![1.0f32; 8],
                [2usize, 2, 2],
                Point::new([0.0, 0.0, 0.0]),
                Spacing::new([1.0, 1.0, 1.0]),
                Direction::identity(),
                &SequentialBackend,
            )
            .expect("coeus image");

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("adapter.mnc");

            let writer = MincWriter::new(SequentialBackend);
            ImageWriter::write(&writer, &path, &image).expect("contract write");

            let bytes = std::fs::read(&path).expect("read back");
            assert_eq!(
                &bytes[0..8],
                b"\x89HDF\r\n\x1a\n",
                "output must start with HDF5 signature"
            );
        }

        /// The native MINC reader rejects a non-HDF5 payload with a typed error
        /// through the unified `ImageReader` contract.
        #[test]
        fn native_reader_requires_valid_hdf5() {
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("bad.mnc");
            std::fs::write(&path, b"not an hdf5 file").expect("write bad file");

            let reader = MincReader::new(SequentialBackend);
            let result = ImageReader::read(&reader, &path);
            assert!(result.is_err(), "reading invalid HDF5 must fail");
        }
    }
}
