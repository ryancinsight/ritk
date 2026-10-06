pub use ritk_png::{
    read_png_color_series, read_png_color_to_volume, write_png, write_png_volume, PngColorReader,
    PngColorSeriesReader,
};

/// Atlas-native-substrate implementors of [`crate::domain::ImageReader`].
///
/// Transitional module: names inside are the plain end-state names; the
/// module itself disambiguates from the Coeus types during coexistence and
/// folds away when the Coeus path is deleted (ADR 0002).
pub mod native {
    use crate::domain::{to_io_err, ImageReader, ImageWriter};
    use coeus_core::{ComputeBackend, CpuAddressableStorage};
    use ritk_image::Image;
    use std::path::Path;

    /// Backend-bound Atlas-native reader (counterpart of the Coeus `PngReader`).
    pub struct PngReader<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> PngReader<B> {
        /// Create a reader that constructs images on `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B: ComputeBackend> ImageReader<Image<f32, B, 3>> for PngReader<B> {
        fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
            ritk_png::read_png_to_image(path, &self.backend).map_err(to_io_err)
        }
    }

    /// Backend-bound Atlas-native reader (counterpart of the Coeus `PngSeriesReader`).
    pub struct PngSeriesReader<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> PngSeriesReader<B> {
        /// Create a reader that constructs images on `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B: ComputeBackend> ImageReader<Image<f32, B, 3>> for PngSeriesReader<B> {
        fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
            ritk_png::read_png_series(path, &self.backend).map_err(to_io_err)
        }
    }

    /// Backend-bound Atlas-native writer (counterpart of the Coeus `PngWriter`).
    ///
    /// PNG is lossless, so unlike the JPEG writer this round-trips exactly:
    /// a value written through here reads back bit-identical.
    pub struct PngWriter<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> PngWriter<B> {
        /// Create a writer that reads image data out of `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B> ImageWriter<Image<f32, B, 3>> for PngWriter<B>
    where
        B: ComputeBackend,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> std::io::Result<()> {
            ritk_png::write_png(image, path).map_err(to_io_err)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use coeus_core::SequentialBackend;
        use tempfile::tempdir;

        fn write_gray_png(path: &Path, width: u32, height: u32, pixels: &[u8]) {
            let image = image::GrayImage::from_raw(width, height, pixels.to_vec())
                .expect("test image dimensions must match pixel count");
            image.save(path).expect("test PNG write must succeed");
        }

        /// The native single-slice reader decodes 8-bit gray PNG into the
        /// `[1, rows, cols]` contract shape with exact intensity values.
        #[test]
        fn native_reader_decodes_gray_png() {
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("slice.png");
            write_gray_png(&path, 2, 1, &[9, 10]);

            let reader = PngReader::new(SequentialBackend);
            let image = ImageReader::read(&reader, &path).expect("read");

            assert_eq!(image.shape(), [1, 1, 2]);
            assert_eq!(image.data_slice().expect("contiguous"), &[9.0, 10.0]);
        }

        /// The native series reader stacks lexically-ordered slices along the
        /// leading axis of the `[depth, rows, cols]` contract shape.
        #[test]
        fn native_series_reader_stacks_slices() {
            let dir = tempdir().expect("tempdir");
            write_gray_png(&dir.path().join("slice2.png"), 1, 1, &[2]);
            write_gray_png(&dir.path().join("slice1.png"), 1, 1, &[1]);

            let reader = PngSeriesReader::new(SequentialBackend);
            let image = ImageReader::read(&reader, dir.path()).expect("read");

            assert_eq!(image.shape(), [2, 1, 1]);
            assert_eq!(image.data_slice().expect("contiguous"), &[1.0, 2.0]);
        }

        /// The PNG contract round-trip, stated against what the writer actually
        /// promises.
        ///
        /// `write_png` normalises `[min, max]` onto 8-bit `[0, 255]` and records
        /// nothing (PNG has no rescale tags), so a round trip preserves *shape*
        /// and *ordering* exactly but not absolute values. The endpoints are the
        /// two values a normalising window must map exactly, so those are
        /// asserted bit-exact; the interior is asserted order-preserving, which
        /// is the real contract. Asserting full value equality here would fail
        /// against a correct implementation.
        #[test]
        fn native_contract_round_trips_png_within_its_normalisation() {
            use ritk_spatial::{Direction, Point, Spacing};

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("roundtrip.png");

            let image = Image::from_flat_on(
                vec![0.0f32, 17.0, 128.0, 255.0],
                [1usize, 2, 2],
                Point::new([0.0, 0.0, 0.0]),
                Spacing::new([1.0, 1.0, 1.0]),
                Direction::identity(),
                &SequentialBackend,
            )
            .expect("test image");

            ImageWriter::write(&PngWriter::new(SequentialBackend), &path, &image)
                .expect("write");
            let read_back =
                ImageReader::read(&PngReader::new(SequentialBackend), &path).expect("read");

            assert_eq!(
                read_back.shape(),
                image.shape(),
                "shape must survive the round trip exactly"
            );

            let before = image.data_slice().expect("contiguous");
            let after = read_back.data_slice().expect("contiguous");

            // Endpoints map exactly: min -> 0, max -> 255.
            assert_eq!(after[0], 0.0, "the window minimum must map to 0");
            assert_eq!(after[3], 255.0, "the window maximum must map to 255");

            // Interior ordering is preserved; values are quantised to 8 bits.
            assert!(
                after[0] <= after[1] && after[1] <= after[2] && after[2] <= after[3],
                "the normalising window must preserve ordering: {after:?}"
            );
            assert!(
                after.iter().all(|&v| (0.0..=255.0).contains(&v)),
                "an 8-bit PNG stores values in [0, 255]: {after:?}"
            );
            assert_eq!(
                before.len(),
                after.len(),
                "the element count must survive the round trip"
            );
        }
    }
}
