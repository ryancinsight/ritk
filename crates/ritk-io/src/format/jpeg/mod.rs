pub use ritk_jpeg::{
    read_jpeg, read_jpeg_color_to_volume, write_jpeg, JpegColorReader, JpegReader, JpegWriter,
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

    /// Backend-bound Atlas-native reader (counterpart of the Burn [`super::JpegReader`]).
    pub struct JpegReader<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> JpegReader<B> {
        /// Create a reader that constructs images on `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B: ComputeBackend> ImageReader<Image<f32, B, 3>> for JpegReader<B> {
        fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
            ritk_jpeg::native::read_jpeg(path, &self.backend).map_err(to_io_err)
        }
    }

    /// Backend-bound Atlas-native writer (counterpart of the Burn writer).
    pub struct JpegWriter<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> JpegWriter<B> {
        /// Create a writer that extracts host data via `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B> ImageWriter<Image<f32, B, 3>> for JpegWriter<B>
    where
        B: ComputeBackend + Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> std::io::Result<()> {
            ritk_jpeg::native::write_jpeg(path, image, &self.backend).map_err(to_io_err)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use coeus_core::SequentialBackend;
        use ritk_spatial::{Direction, Point, Spacing};
        use tempfile::tempdir;

        /// Trait-dispatched round trip through the unified contract. JPEG is
        /// lossy, so tolerances bound the DCT quantization error per intensity
        /// band rather than asserting exact equality.
        #[test]
        fn native_contract_round_trips_jpeg() {
            let image = Image::from_flat_on(
                vec![16.0f32, 128.0, 240.0],
                [1usize, 1, 3],
                Point::new([0.0, 0.0, 0.0]),
                Spacing::new([1.0, 1.0, 1.0]),
                Direction::identity(),
                &SequentialBackend,
            )
            .expect("coeus image");

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("contract.jpg");

            let writer = JpegWriter::new(SequentialBackend);
            ImageWriter::write(&writer, &path, &image).expect("contract write");

            let reader = JpegReader::new(SequentialBackend);
            let loaded = ImageReader::read(&reader, &path).expect("contract read");

            assert_eq!(loaded.shape(), [1, 1, 3], "shape round-trip");
            let values = loaded.data_slice().expect("contiguous");
            assert_eq!(values.len(), 3);
            assert!(values[0] <= 24.0, "expected dark pixel, got {}", values[0]);
            assert!(
                (values[1] - 128.0).abs() <= 12.0,
                "expected mid pixel near 128, got {}",
                values[1]
            );
            assert!(values[2] >= 228.0, "expected bright pixel, got {}", values[2]);
        }
    }
}
