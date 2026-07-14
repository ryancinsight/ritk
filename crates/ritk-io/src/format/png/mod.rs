pub use ritk_png::{
    read_png_color_series, read_png_color_to_volume, PngColorReader, PngColorSeriesReader,
};

/// Atlas-native-substrate implementors of [`crate::domain::ImageReader`].
///
/// Transitional module: names inside are the plain end-state names; the
/// module itself disambiguates from the Burn types during coexistence and
/// folds away when the Burn path is deleted (ADR 0002).
pub mod native {
    use crate::domain::{to_io_err, ImageReader};
    use coeus_core::ComputeBackend;
    use ritk_image::native::Image;
    use std::path::Path;

    /// Backend-bound Atlas-native reader (counterpart of the Burn [`super::PngReader`]).
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

    /// Backend-bound Atlas-native reader (counterpart of the Burn [`super::PngSeriesReader`]).
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
    }
}
