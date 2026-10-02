//! MINC2 through the native provider at the `f32` surfaces of this crate.
//!
//! These surfaces return `f32` images until the dispatch reads in the caller's
//! type (ADR 0053 decision 6), so they read under [`Cast`]: an integer or `f64`
//! image whose stored values `f32` cannot all hold is cast with a warning
//! rather than refused. A stored integer is then mapped through the
//! `image-min` / `image-max` real range, from its `valid_range`, into `f32`, so
//! the returned intensities are real values. The result is exact only when
//! that map is the identity, as for files the RITK writer produced, and `f32`
//! holds the stored type. The writer stores the `f32` samples as `f32`.

/// Atlas-native-substrate implementors of [`crate::domain::ImageReader`].
///
/// Transitional module: names inside are the plain end-state names; the
/// module itself disambiguates from the Coeus types during coexistence and
/// folds away when the Coeus path is deleted (ADR 0002).
pub mod native {
    use crate::domain::{to_io_err, ImageReader, ImageWriter};
    use coeus_core::{ComputeBackend, CpuAddressableStorage};
    use ritk_codecs::sample::Cast;
    use ritk_image::Image;
    use std::path::Path;

    /// Backend-bound Atlas-native reader (counterpart of the Coeus `MincReader`).
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
            ritk_minc::read_minc(path, &self.backend, Cast).map_err(to_io_err)
        }
    }

    /// Backend-bound Atlas-native writer (counterpart of the Coeus writer).
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
            let error =
                ImageReader::read(&reader, &path).expect_err("reading invalid HDF5 must fail");
            assert!(
                error.to_string().contains("HDF5 open failed"),
                "invalid MINC payload must report an HDF5-open failure, got: {error}"
            );
        }
    }
}
