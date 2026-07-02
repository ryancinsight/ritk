pub use ritk_png::{
    read_png_color_series, read_png_color_to_volume, read_png_series, read_png_to_image,
    PngColorReader, PngColorSeriesReader, PngReader, PngSeriesReader,
};

use crate::domain::ImageReader;
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use std::path::Path;

impl<B: Backend> ImageReader<B, 3> for PngReader<B> {
    fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<B, 3>> {
        self.read_image(path)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }
}

impl<B: Backend> ImageReader<B, 3> for PngSeriesReader<B> {
    fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<B, 3>> {
        self.read_image(path)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::{PngReader, PngSeriesReader};
    use crate::domain::ImageReader;
    use burn::tensor::backend::Backend;
    use burn_ndarray::NdArray;
    use std::path::Path;
    use tempfile::tempdir;

    type TestBackend = NdArray<f32>;

    fn write_gray_png(path: &Path, width: u32, height: u32, pixels: &[u8]) {
        let image = image::GrayImage::from_raw(width, height, pixels.to_vec())
            .expect("test image dimensions must match pixel count");
        image.save(path).expect("test PNG write must succeed");
    }

    fn tensor_values(image: &ritk_core::image::Image<TestBackend, 3>) -> Vec<f32> {
        image.data_slice().into_owned()
    }

    #[test]
    fn png_reader_adapter_delegates_to_authoritative_crate() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("slice.png");
        write_gray_png(&path, 2, 1, &[9, 10]);

        let device: <TestBackend as Backend>::Device = Default::default();
        let reader = PngReader::<TestBackend>::new(device);
        let image = ImageReader::<TestBackend, 3>::read(&reader, &path)?;

        assert_eq!(image.shape(), [1, 1, 2]);
        assert_eq!(tensor_values(&image), vec![9.0, 10.0]);
        Ok(())
    }

    #[test]
    fn png_series_reader_adapter_delegates_to_authoritative_crate() -> anyhow::Result<()> {
        let dir = tempdir()?;
        write_gray_png(&dir.path().join("slice2.png"), 1, 1, &[2]);
        write_gray_png(&dir.path().join("slice1.png"), 1, 1, &[1]);

        let device: <TestBackend as Backend>::Device = Default::default();
        let reader = PngSeriesReader::<TestBackend>::new(device);
        let image = ImageReader::<TestBackend, 3>::read(&reader, dir.path())?;

        assert_eq!(image.shape(), [2, 1, 1]);
        assert_eq!(tensor_values(&image), vec![1.0, 2.0]);
        Ok(())
    }
}

#[cfg(feature = "coeus")]
pub use coeus::{CoeusPngReader, CoeusPngSeriesReader};

/// Coeus-typed reader implementors for the [`crate::domain::coeus`] contract
/// (ADR 0002 cutover step 2 — format coverage).
#[cfg(feature = "coeus")]
mod coeus {
    use crate::domain::coeus::{to_io_err, CoeusImageReader};
    use coeus_core::ComputeBackend;
    use ritk_image::coeus::Image;
    use std::path::Path;

    /// Backend-bound Coeus reader (counterpart of [`super::PngReader`]).
    pub struct CoeusPngReader<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> CoeusPngReader<B> {
        /// Create a reader that constructs images on `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B: ComputeBackend> CoeusImageReader<f32, B, 3> for CoeusPngReader<B> {
        fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
            ritk_png::read_png_to_image_coeus(path, &self.backend).map_err(to_io_err)
        }
    }

    /// Backend-bound Coeus reader (counterpart of [`super::PngSeriesReader`]).
    pub struct CoeusPngSeriesReader<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> CoeusPngSeriesReader<B> {
        /// Create a reader that constructs images on `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B: ComputeBackend> CoeusImageReader<f32, B, 3> for CoeusPngSeriesReader<B> {
        fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
            ritk_png::read_png_series_coeus(path, &self.backend).map_err(to_io_err)
        }
    }
}
